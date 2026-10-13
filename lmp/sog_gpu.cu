// sog_gpu.cu — plugin-internal GPU SOG kspace (raw CUDA + cuFFT). Tier-2 full-GPU pipeline:
//   spread(q@x) -> cuFFT fwd -> green-multiply (grad meshes + potential + energy/6-virial reductions)
//   -> one cuFFT inverse of the mesh potential -> adjoint weight-derivative gather.
// Two splines: separable QuadS (PRIMARY — 3D tensor-product 1D weights, GPU-friendly scatter once
// cuFFT makes the FFT ~free and spreading dominates) and CubeS2-4 (BASELINE — bit-matches CPU sog.cpp).
// Math ported from sog.cpp::compute_single. STATUS: nvcc-syntax-compiled; NOT yet GPU-bit-validated
// against the CPU path (cuFFT is unnormalized vs LAMMPS FFT3d 1/ngrid — the normalization constants
// below must be confirmed by the run-0 bit-match once the GPU frees). See sog-progress.md §8.
#include "sog_gpu.cuh"
#include "sog_spline.h"     // CubeS2 node tables + xi constants (single source of truth)
#include <cufft.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
  fprintf(stderr,"sog_gpu CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); }}while(0)

// ── device state ──
struct SogGpuState {
  int nx=0, ny=0, nz=0; size_t ngrid=0;
  cufftHandle plan=0;                 // Z2Z, 3D
  cufftDoubleComplex *d_rho=nullptr;  // charge mesh / rho_hat (in place)
  cufftDoubleComplex *d_gx=nullptr, *d_gy=nullptr, *d_gz=nullptr, *d_pot=nullptr;
  double *d_ge=nullptr, *d_gf=nullptr, *d_gv=nullptr, *d_gs=nullptr;  // green energy/force/virial/self tables
  double *d_gsv=nullptr;             // bare K_v(k²) table for the self-energy strain-derivative virial
  double *d_x=nullptr, *d_q=nullptr, *d_force=nullptr, *d_vpot=nullptr;  // per-atom
  double *d_red=nullptr;             // energy + 6 fv_local + diag_sum + 6 self-virial + 6 r⊗F (20 doubles)
  int cap_atoms=0;

  // GPU timing (cudaEvent-based, accumulated per N steps)
  cudaEvent_t evt_spread_start=0, evt_spread_stop=0;
  cudaEvent_t evt_fftfwd_start=0, evt_fftfwd_stop=0;
  cudaEvent_t evt_kspace_start=0, evt_kspace_stop=0;
  cudaEvent_t evt_fftinv_start=0, evt_fftinv_stop=0;
  cudaEvent_t evt_gather_start=0, evt_gather_stop=0;
  double acc_spread=0, acc_fftfwd=0, acc_kspace=0, acc_fftinv=0, acc_gather=0;
  int timing_count=0, timing_interval=100;  // report every N steps
};

// ── helpers ──
__device__ __forceinline__ double periodic_frac(double xi, double lo, double L){
  double t=(xi-lo)/L; t-=floor(t); return t;
}
__device__ __forceinline__ int wrap(int i,int n){ i%=n; return i<0? i+n : i; }
__device__ __forceinline__ size_t midx(int ix,int iy,int iz,int nx,int ny){
  return (size_t)iz*ny*nx + (size_t)iy*nx + ix;   // matches sog.cpp mesh_index (x fastest)
}

// ── separable QuadS 1D weights — correct port of sog_spline.h / quads_spline.py ──
// True midtown quadrature splines (§A.2/§A.3): ξ○ = 1/√3 (order 4), 0.72879488 (order 6),
// weights c₁…c_ν with reflection c_i(θ)=c_{1-i}(1-θ). Offsets 1−ν…ν. (The earlier port used
// CubeS₂'s L and ξ=2/√15 — WRONG; that has been replaced.)
__device__ const double kQuadsXi6_d = 0.72879488;   // 0.72879488² etc computed inline
// order 4 (ξ○=1/√3 ⇒ ξ²=1/3): c₁ = -½t³+½t²-(3ξ²-2)/2·t+ξ²/2 ; c₂ = ⅙t³+(3ξ²-1)/6·t
__device__ __forceinline__ double q_c1_4(double t){ const double x2=1.0/3.0;
  return -0.5*t*t*t + 0.5*t*t - (3.0*x2-2.0)/2.0*t + 0.5*x2; }
__device__ __forceinline__ double q_c2_4(double t){ const double x2=1.0/3.0;
  return (1.0/6.0)*t*t*t + (3.0*x2-1.0)/6.0*t; }
__device__ __forceinline__ void quads4_w(double t, double *w /*[4]*/){   // offsets -1,0,1,2
  w[0]=q_c2_4(1.0-t); w[1]=q_c1_4(1.0-t); w[2]=q_c1_4(t); w[3]=q_c2_4(t);
}
__device__ __forceinline__ double q_dc1_4(double t){ const double x2=1.0/3.0;
  return -1.5*t*t+t-(3.0*x2-2.0)/2.0; }
__device__ __forceinline__ double q_dc2_4(double t){ const double x2=1.0/3.0;
  return 0.5*t*t+(3.0*x2-1.0)/6.0; }
__device__ __forceinline__ void quads4_wd(double t,double*w,double*dw){
  quads4_w(t,w); dw[0]=-q_dc2_4(1.0-t); dw[1]=-q_dc1_4(1.0-t);
  dw[2]=q_dc1_4(t); dw[3]=q_dc2_4(t);
}
// order 6 (ξ○=0.72879488): c₁,c₂,c₃ degree-5 (§A.3)
__device__ __forceinline__ double q_c1_6(double t,double x2,double x4){
  double t2=t*t,t3=t2*t,t4=t3*t,t5=t4*t;
  return (1.0/12.0)*t5 - (1.0/6.0)*t4 + (10.0*x2-7.0)/12.0*t3
       - (3.0*x2-2.0)/3.0*t2 + (5.0*x4-7.0*x2+4.0)/4.0*t - (3.0*x4-4.0*x2)/6.0; }
__device__ __forceinline__ double q_c2_6(double t,double x2,double x4){
  double t2=t*t,t3=t2*t,t4=t3*t,t5=t4*t;
  return -(1.0/24.0)*t5 + (1.0/24.0)*t4 - (10.0*x2-7.0)/24.0*t3
       + (6.0*x2-1.0)/24.0*t2 - (5.0*x4-7.0*x2+2.0)/8.0*t + (3.0*x4-x2)/24.0; }
__device__ __forceinline__ double q_c3_6(double t,double x2,double x4){
  double t3=t*t*t,t5=t3*t*t;
  return (1.0/120.0)*t5 + (2.0*x2-1.0)/24.0*t3 + (15.0*x4-15.0*x2+4.0)/120.0*t; }
__device__ __forceinline__ void quads6_w(double t, double *w /*[6]*/){   // offsets -2,-1,0,1,2,3
  const double x2=kQuadsXi6_d*kQuadsXi6_d, x4=x2*x2;
  w[0]=q_c3_6(1.0-t,x2,x4); w[1]=q_c2_6(1.0-t,x2,x4); w[2]=q_c1_6(1.0-t,x2,x4);
  w[3]=q_c1_6(t,x2,x4);     w[4]=q_c2_6(t,x2,x4);     w[5]=q_c3_6(t,x2,x4);
}
__device__ __forceinline__ double q_dc1_6(double t,double x2,double x4){
  double t2=t*t,t3=t2*t,t4=t3*t;
  return (5.0/12.0)*t4-(2.0/3.0)*t3+(10.0*x2-7.0)/4.0*t2
       -2.0*(3.0*x2-2.0)/3.0*t+(5.0*x4-7.0*x2+4.0)/4.0; }
__device__ __forceinline__ double q_dc2_6(double t,double x2,double x4){
  double t2=t*t,t3=t2*t,t4=t3*t;
  return -(5.0/24.0)*t4+(1.0/6.0)*t3-(10.0*x2-7.0)/8.0*t2
       +(6.0*x2-1.0)/12.0*t-(5.0*x4-7.0*x2+2.0)/8.0; }
__device__ __forceinline__ double q_dc3_6(double t,double x2,double x4){
  double t2=t*t,t4=t2*t2;
  return (1.0/24.0)*t4+(2.0*x2-1.0)/8.0*t2+(15.0*x4-15.0*x2+4.0)/120.0; }
__device__ __forceinline__ void quads6_wd(double t,double*w,double*dw){
  const double x2=kQuadsXi6_d*kQuadsXi6_d,x4=x2*x2; quads6_w(t,w);
  dw[0]=-q_dc3_6(1.0-t,x2,x4); dw[1]=-q_dc2_6(1.0-t,x2,x4);
  dw[2]=-q_dc1_6(1.0-t,x2,x4); dw[3]=q_dc1_6(t,x2,x4);
  dw[4]=q_dc2_6(t,x2,x4); dw[5]=q_dc3_6(t,x2,x4);
}

// ── CubeS₂ device port — EXACT port of sog_spline.h (bit-matches CPU sog.cpp) ──
// Node tables live in __constant__ memory (uploaded once at create). GpuNode layout == CubeS2Node4.
typedef LAMMPS_NS::CubeS2Node4 GpuNode;
__constant__ GpuNode c_nodes4[32];
__constant__ GpuNode c_nodes6[88];
__device__ const double kXi4 = 0.5773502691896258;   // 1/√3  (kCubes2Xi4)
__device__ const double kXi6 = 0.6503998764035732;   // kCubes2Xi6

__device__ __forceinline__ double cs2_L(double th,double xi){ double x2=xi*xi;
  return -0.5*th*th*th + 0.5*th*th - (9.0*x2-2.0)/6.0*th + 0.5*x2; }
__device__ __forceinline__ double cs2_R(double th,double xi){ double x2=xi*xi;
  return (1.0/6.0)*th*th*th + (3.0*x2-1.0)/6.0*th; }
__device__ __forceinline__ double eta_pos_off(int off,double c){ return (off==1)?c:(1.0-c); }

__device__ __forceinline__ double cs2_w4(double tx,double ty,double tz,const GpuNode&nd,double xi){
  if(nd.cls==0){
    double ex=(nd.dx==1)?tx:(1.0-tx), ey=(nd.dy==1)?ty:(1.0-ty), ez=(nd.dz==1)?tz:(1.0-tz);
    return cs2_L(ex,xi)*ey*ez + cs2_L(ey,xi)*ex*ez + cs2_L(ez,xi)*ex*ey;
  } else {
    double es,e1,e2;
    if(nd.sp_axis==0){ es=nd.sp_is_neg?(1.0-tx):tx; e1=(nd.dy==1)?ty:(1.0-ty); e2=(nd.dz==1)?tz:(1.0-tz); }
    else if(nd.sp_axis==1){ es=nd.sp_is_neg?(1.0-ty):ty; e1=(nd.dx==1)?tx:(1.0-tx); e2=(nd.dz==1)?tz:(1.0-tz); }
    else { es=nd.sp_is_neg?(1.0-tz):tz; e1=(nd.dx==1)?tx:(1.0-tx); e2=(nd.dy==1)?ty:(1.0-ty); }
    return cs2_R(es,xi)*e1*e2;
  }
}

__device__ __forceinline__ double cs2_L111_6(double t,double xi){ double x2=xi*xi,x4=x2*x2;
  return (1.0/12.0)*t*t*t*t*t - (1.0/6.0)*t*t*t*t + (10.0*x2-1.0)/12.0*t*t*t
       - (6.0*x2-1.0)/6.0*t*t + (5.0*x4-x2)/4.0*t - (3.0*x4-x2)/6.0; }
__device__ __forceinline__ double cs2_L311_6(double t,double xi){ double x2=xi*xi,x4=x2*x2;
  return (1.0/120.0)*t*t*t*t*t + (2.0*x2-1.0)/24.0*t*t*t + (15.0*x4-15.0*x2+4.0)/120.0*t; }
__device__ __forceinline__ double cs2_L211_6(double t,double xi){ return -0.25*cs2_L111_6(t,xi)-2.5*cs2_L311_6(t,xi); }
__device__ __forceinline__ double cs2_S6(double t,double xi){
  return -0.5*t*t*t + 0.5*t*t - (3.0*xi*xi-2.0)/2.0*t + 0.5*xi*xi; }

__device__ __forceinline__ double cs2_w6(double tx,double ty,double tz,const GpuNode&nd,double xi){
  double coords[3]={tx,ty,tz}; int offs[3]={nd.dx,nd.dy,nd.dz};
  if(nd.cls==0){
    double ex=(nd.dx==1)?tx:(1.0-tx), ey=(nd.dy==1)?ty:(1.0-ty), ez=(nd.dz==1)?tz:(1.0-tz);
    return cs2_L111_6(ex,xi)*ey*ez + cs2_L111_6(ey,xi)*ez*ex + cs2_L111_6(ez,xi)*ex*ey
         + cs2_S6(ex,xi)*cs2_S6(ey,xi)*cs2_S6(ez,xi);
  } else if(nd.cls==1){
    int sp=nd.sp_axis; double es=nd.sp_is_neg?(1.0-coords[sp]):coords[sp];
    int n1=(sp+1)%3,n2=(sp+2)%3; if(n1>n2){int t=n1;n1=n2;n2=t;}
    return cs2_L211_6(es,xi)*eta_pos_off(offs[n1],coords[n1])*eta_pos_off(offs[n2],coords[n2])
         + cs2_R(es,xi)*cs2_S6(eta_pos_off(offs[n1],coords[n1]),xi)*cs2_S6(eta_pos_off(offs[n2],coords[n2]),xi);
  } else if(nd.cls==2){
    int sp=nd.sp_axis; double es=nd.sp_is_neg?(1.0-coords[sp]):coords[sp];
    int n1=(sp+1)%3,n2=(sp+2)%3; if(n1>n2){int t=n1;n1=n2;n2=t;}
    return cs2_L311_6(es,xi)*eta_pos_off(offs[n1],coords[n1])*eta_pos_off(offs[n2],coords[n2]);
  } else if(nd.cls==3){
    int nm=nd.sp_axis; int s1=(nm+1)%3,s2=(nm+2)%3; if(s1>s2){int t=s1;s1=s2;s2=t;}
    double en=eta_pos_off(offs[nm],coords[nm]);
    double es1=(nd.sp_is_neg&1)?(1.0-coords[s1]):coords[s1];
    double es2=(nd.sp_is_neg&2)?(1.0-coords[s2]):coords[s2];
    return cs2_S6(en,xi)*cs2_R(es1,xi)*cs2_R(es2,xi);
  } else {
    double ex=(nd.sp_is_neg&1)?(1.0-tx):tx, ey=(nd.sp_is_neg&2)?(1.0-ty):ty, ez=(nd.sp_is_neg&4)?(1.0-tz):tz;
    return cs2_R(ex,xi)*cs2_R(ey,xi)*cs2_R(ez,xi);
  }
}

struct GpuWeightGrad { double v, dx, dy, dz; };
__device__ __forceinline__ GpuWeightGrad gg(double v,double dx=0,double dy=0,double dz=0){return {v,dx,dy,dz};}
__device__ __forceinline__ GpuWeightGrad ga(GpuWeightGrad a,GpuWeightGrad b){return gg(a.v+b.v,a.dx+b.dx,a.dy+b.dy,a.dz+b.dz);}
__device__ __forceinline__ GpuWeightGrad gs(GpuWeightGrad a,GpuWeightGrad b){return gg(a.v-b.v,a.dx-b.dx,a.dy-b.dy,a.dz-b.dz);}
__device__ __forceinline__ GpuWeightGrad gm(GpuWeightGrad a,GpuWeightGrad b){return gg(a.v*b.v,a.dx*b.v+a.v*b.dx,a.dy*b.v+a.v*b.dy,a.dz*b.v+a.v*b.dz);}
__device__ __forceinline__ GpuWeightGrad gc(double a,GpuWeightGrad b){return gg(a*b.v,a*b.dx,a*b.dy,a*b.dz);}
__device__ __forceinline__ GpuWeightGrad gL(GpuWeightGrad t,double xi){auto t2=gm(t,t),t3=gm(t2,t);return ga(ga(gc(-.5,t3),gc(.5,t2)),ga(gc(-(9*xi*xi-2)/6,t),gg(.5*xi*xi)));}
__device__ __forceinline__ GpuWeightGrad gR(GpuWeightGrad t,double xi){return ga(gc(1.0/6.0,gm(gm(t,t),t)),gc((3*xi*xi-1)/6,t));}
__device__ __forceinline__ GpuWeightGrad gL111(GpuWeightGrad t,double xi){double x2=xi*xi,x4=x2*x2;auto t2=gm(t,t),t3=gm(t2,t),t4=gm(t3,t),t5=gm(t4,t);return ga(ga(ga(gc(1.0/12,t5),gc(-1.0/6,t4)),ga(gc((10*x2-1)/12,t3),gc(-(6*x2-1)/6,t2))),ga(gc((5*x4-x2)/4,t),gg(-(3*x4-x2)/6)));}
__device__ __forceinline__ GpuWeightGrad gL311(GpuWeightGrad t,double xi){double x2=xi*xi,x4=x2*x2;auto t3=gm(gm(t,t),t),t5=gm(gm(t3,t),t);return ga(ga(gc(1.0/120,t5),gc((2*x2-1)/24,t3)),gc((15*x4-15*x2+4)/120,t));}
__device__ __forceinline__ GpuWeightGrad gL211(GpuWeightGrad t,double xi){return gs(gc(-.25,gL111(t,xi)),gc(2.5,gL311(t,xi)));}
__device__ __forceinline__ GpuWeightGrad gS6(GpuWeightGrad t,double xi){auto t2=gm(t,t),t3=gm(t2,t);return ga(ga(gc(-.5,t3),gc(.5,t2)),ga(gc(-(3*xi*xi-2)/2,t),gg(.5*xi*xi)));}
__device__ __forceinline__ GpuWeightGrad gcoord(double v,int a,bool refl=false){return gg(v,a==0?(refl?-1:1):0,a==1?(refl?-1:1):0,a==2?(refl?-1:1):0);}
__device__ __forceinline__ GpuWeightGrad geta(const GpuWeightGrad*c,const int*o,int a){return o[a]==1?c[a]:gcoord(1-c[a].v,a,true);}
__device__ __forceinline__ GpuWeightGrad cs2_wg4(double tx,double ty,double tz,const GpuNode&nd,double xi){
  GpuWeightGrad c[3]={gcoord(tx,0),gcoord(ty,1),gcoord(tz,2)};int o[3]={nd.dx,nd.dy,nd.dz};
  if(nd.cls==0){auto ex=geta(c,o,0),ey=geta(c,o,1),ez=geta(c,o,2);return ga(ga(gm(gm(gL(ex,xi),ey),ez),gm(gm(gL(ey,xi),ex),ez)),gm(gm(gL(ez,xi),ex),ey));}
  int sp=nd.sp_axis,n1=(sp+1)%3,n2=(sp+2)%3;if(n1>n2){int z=n1;n1=n2;n2=z;}auto es=nd.sp_is_neg?gcoord(1-c[sp].v,sp,true):c[sp];return gm(gm(gR(es,xi),geta(c,o,n1)),geta(c,o,n2));
}
__device__ __forceinline__ GpuWeightGrad cs2_wg6(double tx,double ty,double tz,const GpuNode&nd,double xi){
  GpuWeightGrad c[3]={gcoord(tx,0),gcoord(ty,1),gcoord(tz,2)};int o[3]={nd.dx,nd.dy,nd.dz};
  if(nd.cls==0){auto ex=geta(c,o,0),ey=geta(c,o,1),ez=geta(c,o,2);return ga(ga(ga(gm(gm(gL111(ex,xi),ey),ez),gm(gm(gL111(ey,xi),ez),ex)),gm(gm(gL111(ez,xi),ex),ey)),gm(gm(gS6(ex,xi),gS6(ey,xi)),gS6(ez,xi)));}
  if(nd.cls==1||nd.cls==2){int sp=nd.sp_axis,n1=(sp+1)%3,n2=(sp+2)%3;if(n1>n2){int z=n1;n1=n2;n2=z;}auto es=nd.sp_is_neg?gcoord(1-c[sp].v,sp,true):c[sp];if(nd.cls==1)return ga(gm(gm(gL211(es,xi),geta(c,o,n1)),geta(c,o,n2)),gm(gm(gR(es,xi),gS6(geta(c,o,n1),xi)),gS6(geta(c,o,n2),xi)));return gm(gm(gL311(es,xi),geta(c,o,n1)),geta(c,o,n2));}
  if(nd.cls==3){int nm=nd.sp_axis,s1=(nm+1)%3,s2=(nm+2)%3;if(s1>s2){int z=s1;s1=s2;s2=z;}auto e1=(nd.sp_is_neg&1)?gcoord(1-c[s1].v,s1,true):c[s1];auto e2=(nd.sp_is_neg&2)?gcoord(1-c[s2].v,s2,true):c[s2];return gm(gm(gS6(geta(c,o,nm),xi),gR(e1,xi)),gR(e2,xi));}
  auto ex=(nd.sp_is_neg&1)?gcoord(1-tx,0,true):c[0];auto ey=(nd.sp_is_neg&2)?gcoord(1-ty,1,true):c[1];auto ez=(nd.sp_is_neg&4)?gcoord(1-tz,2,true):c[2];return gm(gm(gR(ex,xi),gR(ey,xi)),gR(ez,xi));
}

// CubeS2 spread: 1 thread/atom, loop 32/88 nodes, atomicAdd into d_rho.real.
template<int ORD>
__global__ void k_spread_cubes2(int nlocal,const double*x,const double*boxlo,
    double lx,double ly,double lz,const double*q,double rho_scale,
    int nx,int ny,int nz,cufftDoubleComplex*rho){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=nlocal) return;
  double fx=periodic_frac(x[3*i+0],boxlo[0],lx)*nx;
  double fy=periodic_frac(x[3*i+1],boxlo[1],ly)*ny;
  double fz=periodic_frac(x[3*i+2],boxlo[2],lz)*nz;
  int ix0=(int)floor(fx),iy0=(int)floor(fy),iz0=(int)floor(fz);
  double tx=fx-ix0,ty=fy-iy0,tz=fz-iz0, qs=rho_scale*q[i];
  const int NN=(ORD==4)?32:88; const double xi=(ORD==4)?kXi4:kXi6;
  for(int k=0;k<NN;++k){
    GpuNode nd=(ORD==4)?c_nodes4[k]:c_nodes6[k];
    double w=(ORD==4)?cs2_w4(tx,ty,tz,nd,xi):cs2_w6(tx,ty,tz,nd,xi);
    if(w==0.0) continue;
    int gx=wrap(ix0+nd.dx,nx),gy=wrap(iy0+nd.dy,ny),gz=wrap(iz0+nd.dz,nz);
    atomicAdd(&rho[midx(gx,gy,gz,nx,ny)].x, qs*w);
  }
}

template<int ORD>
__global__ void k_gather_cubes2(int nlocal,const double*x,const double*boxlo,
    double lx,double ly,double lz,const double*q,double qscale,
    int nx,int ny,int nz,const cufftDoubleComplex*gx,const cufftDoubleComplex*gy,
    const cufftDoubleComplex*gz,const cufftDoubleComplex*pot,int want_pot,
    double*force,double*vpot,double*red){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=nlocal) return;
  double fx=periodic_frac(x[3*i+0],boxlo[0],lx)*nx;
  double fy=periodic_frac(x[3*i+1],boxlo[1],ly)*ny;
  double fz=periodic_frac(x[3*i+2],boxlo[2],lz)*nz;
  int ix0=(int)floor(fx),iy0=(int)floor(fy),iz0=(int)floor(fz);
  double tx=fx-ix0,ty=fy-iy0,tz=fz-iz0;
  const int NN=(ORD==4)?32:88; const double xi=(ORD==4)?kXi4:kXi6;
  double dtx=0,dty=0,dtz=0,gp=0;
  for(int k=0;k<NN;++k){
    GpuNode nd=(ORD==4)?c_nodes4[k]:c_nodes6[k];
    GpuWeightGrad wg=(ORD==4)?cs2_wg4(tx,ty,tz,nd,xi):cs2_wg6(tx,ty,tz,nd,xi);
    double w=wg.v; if(w==0.0) continue;
    int gxi=wrap(ix0+nd.dx,nx),gyi=wrap(iy0+nd.dy,ny),gzi=wrap(iz0+nd.dz,nz);
    size_t id=midx(gxi,gyi,gzi,nx,ny);
    double u=pot[id].x;
    dtx+=wg.dx*u; dty+=wg.dy*u; dtz+=wg.dz*u; gp+=w*u;
  }
  double qi=q[i];
  double fxs=-qscale*qi*dtx*nx/lx, fys=-qscale*qi*dty*ny/ly, fzs=-qscale*qi*dtz*nz/lz;
  force[3*i+0]=fxs; force[3*i+1]=fys; force[3*i+2]=fzs;
  vpot[i]=qscale*gp;
  // r⊗F virial: exact, captures all position-dependent strain effects
  // that the analytic Fourier formula misses (matching CPU vv_rf).
  // red[14..19] holds Σ r_iα · F_iβ.
  atomicAdd(&red[14], x[3*i+0]*fxs);  // r_x * F_x (xx)
  atomicAdd(&red[15], x[3*i+1]*fys);  // r_y * F_y (yy)
  atomicAdd(&red[16], x[3*i+2]*fzs);  // r_z * F_z (zz)
  atomicAdd(&red[17], x[3*i+0]*fys);  // r_x * F_y (xy)
  atomicAdd(&red[18], x[3*i+0]*fzs);  // r_x * F_z (xz)
  atomicAdd(&red[19], x[3*i+1]*fzs);  // r_y * F_z (yz)
}

// ── kernels ──
// QuadS spread: 1 thread/atom, (norder)^3 tensor-product atomicAdds into the real part of d_rho.
template<int NORD>
__global__ void k_spread_quads(int nlocal,const double*x,const double*boxlo,
    double lx,double ly,double lz,const double*q,double rho_scale,
    int nx,int ny,int nz,cufftDoubleComplex*rho){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=nlocal) return;
  double fx=periodic_frac(x[3*i+0],boxlo[0],lx)*nx;
  double fy=periodic_frac(x[3*i+1],boxlo[1],ly)*ny;
  double fz=periodic_frac(x[3*i+2],boxlo[2],lz)*nz;
  int ix0=(int)floor(fx), iy0=(int)floor(fy), iz0=(int)floor(fz);
  double tx=fx-ix0, ty=fy-iy0, tz=fz-iz0;
  double wx[NORD],wy[NORD],wz[NORD],dwx[NORD],dwy[NORD],dwz[NORD];
  if(NORD==4){ quads4_wd(tx,wx,dwx); quads4_wd(ty,wy,dwy); quads4_wd(tz,wz,dwz); }
  else       { quads6_wd(tx,wx,dwx); quads6_wd(ty,wy,dwy); quads6_wd(tz,wz,dwz); }
  const int off=(NORD==4)?-1:-2;
  double qs=rho_scale*q[i];
  for(int a=0;a<NORD;++a){int gx=wrap(ix0+off+a,nx); double wa=wx[a]*qs;
    for(int b=0;b<NORD;++b){int gy=wrap(iy0+off+b,ny); double wab=wa*wy[b];
      for(int c=0;c<NORD;++c){int gz=wrap(iz0+off+c,nz);
        atomicAdd(&rho[midx(gx,gy,gz,nx,ny)].x, wab*wz[c]); }}}
}

// green-multiply: 1 thread/grid-point. Builds ik*gf*rho grad meshes + ge*rho potential mesh,
// and reduces energy + 6 virial (block-reduced into d_red via atomicAdd).
__global__ void k_kspace(int nx,int ny,int nz,double lx,double ly,double lz,
    const cufftDoubleComplex*rho,const double*ge,const double*gf,const double*gv,const double*gs,
    const double*gsv,cufftDoubleComplex*gx,cufftDoubleComplex*gy,cufftDoubleComplex*gz,
    cufftDoubleComplex*pot,int want_pot,double scaleinv,double*red){
  size_t idx=(size_t)blockIdx.x*blockDim.x+threadIdx.x; size_t ngrid=(size_t)nx*ny*nz;
  if(idx>=ngrid) return;
  int ix=idx%nx, iy=(idx/nx)%ny, iz=idx/((size_t)nx*ny);
  int kxm=ix-nx*((2*ix)/nx), kym=iy-ny*((2*iy)/ny), kzm=iz-nz*((2*iz)/nz);
  double kx=2*M_PI/lx*kxm, ky=2*M_PI/ly*kym, kz=2*M_PI/lz*kzm;
  double geff=gf[idx], gee=ge[idx], gvv=gv[idx];
  double rr=rho[idx].x, ri=rho[idx].y, rho2=rr*rr+ri*ri;
  double s2=scaleinv*scaleinv;
  // diag_sum for the mesh self-term: Σ green_self over modes where green is non-zero (skip k=0),
  // matching sog.cpp's `if (geff==0 && geff_energy==0) continue; diag_sum += mesh_green_self`.
  bool active = !(geff==0.0 && gee==0.0);
  if(active) atomicAdd(&red[7], gs[idx]);
  // energy + virial reductions
  double e = s2*gee*rho2;
  atomicAdd(&red[0], e);
  atomicAdd(&red[1], s2*rho2*(gee - gvv*kx*kx));
  atomicAdd(&red[2], s2*rho2*(gee - gvv*ky*ky));
  atomicAdd(&red[3], s2*rho2*(gee - gvv*kz*kz));
  atomicAdd(&red[4], s2*rho2*(-gvv*kx*ky));
  atomicAdd(&red[5], s2*rho2*(-gvv*kx*kz));
  atomicAdd(&red[6], s2*rho2*(-gvv*ky*kz));
  // self-energy stress accumulator: Σ bare K_v(k²)·k_α k_β (config-independent, red[8..13])
  if(active){
    double sv=gsv[idx];
    atomicAdd(&red[8],  sv*kx*kx); atomicAdd(&red[9],  sv*ky*ky); atomicAdd(&red[10], sv*kz*kz);
    atomicAdd(&red[11], sv*kx*ky); atomicAdd(&red[12], sv*kx*kz); atomicAdd(&red[13], sv*ky*kz);
  }
  // gradient meshes: grad = i*k * (scaleinv*geff*rho)
  double vkr=scaleinv*geff*rr, vki=scaleinv*geff*ri;
  gx[idx].x=-kx*vki; gx[idx].y=kx*vkr;
  gy[idx].x=-ky*vki; gy[idx].y=ky*vkr;
  gz[idx].x=-kz*vki; gz[idx].y=kz*vkr;
  if(want_pot){ pot[idx].x=scaleinv*gee*rr; pot[idx].y=scaleinv*gee*ri; }
}

template<int NORD>
__global__ void k_gather_quads(int nlocal,const double*x,const double*boxlo,
    double lx,double ly,double lz,const double*q,double qscale,
    int nx,int ny,int nz,const cufftDoubleComplex*gx,const cufftDoubleComplex*gy,
    const cufftDoubleComplex*gz,const cufftDoubleComplex*pot,int want_pot,
    double*force,double*vpot,double*red){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=nlocal) return;
  double fx=periodic_frac(x[3*i+0],boxlo[0],lx)*nx;
  double fy=periodic_frac(x[3*i+1],boxlo[1],ly)*ny;
  double fz=periodic_frac(x[3*i+2],boxlo[2],lz)*nz;
  int ix0=(int)floor(fx),iy0=(int)floor(fy),iz0=(int)floor(fz);
  double tx=fx-ix0,ty=fy-iy0,tz=fz-iz0;
  double wx[NORD],wy[NORD],wz[NORD],dwx[NORD],dwy[NORD],dwz[NORD];
  if(NORD==4){ quads4_wd(tx,wx,dwx); quads4_wd(ty,wy,dwy); quads4_wd(tz,wz,dwz); }
  else       { quads6_wd(tx,wx,dwx); quads6_wd(ty,wy,dwy); quads6_wd(tz,wz,dwz); }
  const int off=(NORD==4)?-1:-2;
  double dtx=0,dty=0,dtz=0,gp=0;
  for(int a=0;a<NORD;++a){int gxi=wrap(ix0+off+a,nx);
    for(int b=0;b<NORD;++b){int gyi=wrap(iy0+off+b,ny); double wab=wx[a]*wy[b];
      for(int c=0;c<NORD;++c){int gzi=wrap(iz0+off+c,nz); double w=wab*wz[c];
        size_t id=midx(gxi,gyi,gzi,nx,ny);
        double u=pot[id].x;
        dtx+=dwx[a]*wy[b]*wz[c]*u; dty+=wx[a]*dwy[b]*wz[c]*u; dtz+=wx[a]*wy[b]*dwz[c]*u;
        gp+=w*u; }}}
  double qi=q[i];
  double fxs=-qscale*qi*dtx*nx/lx, fys=-qscale*qi*dty*ny/ly, fzs=-qscale*qi*dtz*nz/lz;
  force[3*i+0]=fxs; force[3*i+1]=fys; force[3*i+2]=fzs;
  vpot[i]=qscale*gp;
  // r⊗F virial: exact, captures all position-dependent strain effects
  // that the analytic Fourier formula misses (matching CPU vv_rf).
  // red[14..19] holds Σ r_iα · F_iβ.
  atomicAdd(&red[14], x[3*i+0]*fxs);  // r_x * F_x (xx)
  atomicAdd(&red[15], x[3*i+1]*fys);  // r_y * F_y (yy)
  atomicAdd(&red[16], x[3*i+2]*fzs);  // r_z * F_z (zz)
  atomicAdd(&red[17], x[3*i+0]*fys);  // r_x * F_y (xy)
  atomicAdd(&red[18], x[3*i+0]*fzs);  // r_x * F_z (xz)
  atomicAdd(&red[19], x[3*i+1]*fzs);  // r_y * F_z (yz)
}

__global__ void k_zero_complex(cufftDoubleComplex*a,size_t n){
  size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i<n){a[i].x=0;a[i].y=0;}
}

// ── host API ──
extern "C" SogGpuState *sog_gpu_create(){
  // upload CubeS2 node tables to constant memory (layout GpuNode == CubeS2Node4)
  cudaMemcpyToSymbol(c_nodes4, LAMMPS_NS::kCubes2Nodes4, 32*sizeof(GpuNode));
  cudaMemcpyToSymbol(c_nodes6, LAMMPS_NS::kCubes2Nodes6, 88*sizeof(GpuNode));
  return new SogGpuState();
}
extern "C" void sog_gpu_destroy(SogGpuState*s){
  if(!s) return;
  if(s->plan) cufftDestroy(s->plan);
  cudaFree(s->d_rho);cudaFree(s->d_gx);cudaFree(s->d_gy);cudaFree(s->d_gz);cudaFree(s->d_pot);
  cudaFree(s->d_ge);cudaFree(s->d_gf);cudaFree(s->d_gv);cudaFree(s->d_gs);cudaFree(s->d_gsv);
  cudaFree(s->d_x);cudaFree(s->d_q);cudaFree(s->d_force);cudaFree(s->d_vpot);cudaFree(s->d_red);
  delete s;
}

extern "C" void sog_gpu_setup(SogGpuState*s,int nx,int ny,int nz,
    const double*ge,const double*gf,const double*gs,const double*gvv,const double*gsv){
  size_t ng=(size_t)nx*ny*nz;
  if(nx!=s->nx||ny!=s->ny||nz!=s->nz){
    if(s->plan) cufftDestroy(s->plan);
    cudaFree(s->d_rho);cudaFree(s->d_gx);cudaFree(s->d_gy);cudaFree(s->d_gz);cudaFree(s->d_pot);
    cudaFree(s->d_ge);cudaFree(s->d_gf);cudaFree(s->d_gv);cudaFree(s->d_gs);cudaFree(s->d_gsv);
    CK(cudaMalloc(&s->d_rho,ng*sizeof(cufftDoubleComplex)));
    CK(cudaMalloc(&s->d_gx,ng*sizeof(cufftDoubleComplex)));
    CK(cudaMalloc(&s->d_gy,ng*sizeof(cufftDoubleComplex)));
    CK(cudaMalloc(&s->d_gz,ng*sizeof(cufftDoubleComplex)));
    CK(cudaMalloc(&s->d_pot,ng*sizeof(cufftDoubleComplex)));
    CK(cudaMalloc(&s->d_ge,ng*sizeof(double)));CK(cudaMalloc(&s->d_gf,ng*sizeof(double)));
    CK(cudaMalloc(&s->d_gv,ng*sizeof(double)));CK(cudaMalloc(&s->d_gs,ng*sizeof(double)));
    CK(cudaMalloc(&s->d_gsv,ng*sizeof(double)));
    // cuFFT 3D expects (nz,ny,nx) with x fastest -> pass dims as (nz,ny,nx)
    cufftPlan3d(&s->plan,nz,ny,nx,CUFFT_Z2Z);
    s->nx=nx;s->ny=ny;s->nz=nz;s->ngrid=ng;
  }
  CK(cudaMemcpy(s->d_ge,ge,ng*sizeof(double),cudaMemcpyHostToDevice));
  CK(cudaMemcpy(s->d_gf,gf,ng*sizeof(double),cudaMemcpyHostToDevice));
  CK(cudaMemcpy(s->d_gv,gvv,ng*sizeof(double),cudaMemcpyHostToDevice));
  CK(cudaMemcpy(s->d_gs,gs,ng*sizeof(double),cudaMemcpyHostToDevice));
  CK(cudaMemcpy(s->d_gsv,gsv,ng*sizeof(double),cudaMemcpyHostToDevice));
}

extern "C" double sog_gpu_compute(SogGpuState*s,int nlocal,const double*x,const double*boxlo,
    double lx,double ly,double lz,const double*q,double qscale,int spline,int want_pot,
    double*force,double*virial6,double*vpot,double self_coeff,double qsqsum,int remove_self){
  size_t ng=s->ngrid;
  if(nlocal>s->cap_atoms){
    cudaFree(s->d_x);cudaFree(s->d_q);cudaFree(s->d_force);cudaFree(s->d_vpot);
    CK(cudaMalloc(&s->d_x,3*nlocal*sizeof(double)));CK(cudaMalloc(&s->d_q,nlocal*sizeof(double)));
    CK(cudaMalloc(&s->d_force,3*nlocal*sizeof(double)));CK(cudaMalloc(&s->d_vpot,nlocal*sizeof(double)));
    if(!s->d_red) CK(cudaMalloc(&s->d_red,20*sizeof(double)));
    s->cap_atoms=nlocal;
  }
  double boxlo3[3]={boxlo[0],boxlo[1],boxlo[2]};
  double *d_boxlo; CK(cudaMalloc(&d_boxlo,3*sizeof(double)));
  CK(cudaMemcpy(d_boxlo,boxlo3,3*sizeof(double),cudaMemcpyHostToDevice));
  CK(cudaMemcpy(s->d_x,x,3*nlocal*sizeof(double),cudaMemcpyHostToDevice));
  CK(cudaMemcpy(s->d_q,q,nlocal*sizeof(double),cudaMemcpyHostToDevice));
  CK(cudaMemset(s->d_red,0,20*sizeof(double)));
  double volume=lx*ly*lz, rho_scale=(double)ng/volume, scaleinv=1.0/(double)ng;

  int tb=256, gg=(ng+tb-1)/tb, ga=(nlocal+tb-1)/tb;
  // ── Create timing events lazily ──
  if (!s->evt_spread_start) {
    cudaEventCreate(&s->evt_spread_start); cudaEventCreate(&s->evt_spread_stop);
    cudaEventCreate(&s->evt_fftfwd_start); cudaEventCreate(&s->evt_fftfwd_stop);
    cudaEventCreate(&s->evt_kspace_start); cudaEventCreate(&s->evt_kspace_stop);
    cudaEventCreate(&s->evt_fftinv_start); cudaEventCreate(&s->evt_fftinv_stop);
    cudaEventCreate(&s->evt_gather_start); cudaEventCreate(&s->evt_gather_stop);
  }
  k_zero_complex<<<gg,tb>>>(s->d_rho,ng);
  bool quads = (spline==SOG_QUADS_4||spline==SOG_QUADS_6);
  int nord = (spline==SOG_CUBES2_6||spline==SOG_QUADS_6)?6:4;
  // ── Phase 1: Spread ──
  cudaEventRecord(s->evt_spread_start, 0);
  if(quads){
    if(nord==4) k_spread_quads<4><<<ga,tb>>>(nlocal,s->d_x,d_boxlo,lx,ly,lz,s->d_q,rho_scale,s->nx,s->ny,s->nz,s->d_rho);
    else        k_spread_quads<6><<<ga,tb>>>(nlocal,s->d_x,d_boxlo,lx,ly,lz,s->d_q,rho_scale,s->nx,s->ny,s->nz,s->d_rho);
  } else {
    if(nord==4) k_spread_cubes2<4><<<ga,tb>>>(nlocal,s->d_x,d_boxlo,lx,ly,lz,s->d_q,rho_scale,s->nx,s->ny,s->nz,s->d_rho);
    else        k_spread_cubes2<6><<<ga,tb>>>(nlocal,s->d_x,d_boxlo,lx,ly,lz,s->d_q,rho_scale,s->nx,s->ny,s->nz,s->d_rho);
  }
  cudaEventRecord(s->evt_spread_stop, 0);
  // ── Phase 2: FFT forward ──
  cudaEventRecord(s->evt_fftfwd_start, 0);
  cufftExecZ2Z(s->plan,s->d_rho,s->d_rho,CUFFT_FORWARD);
  cudaEventRecord(s->evt_fftfwd_stop, 0);
  // ── Phase 3: Kspace (green multiply + energy/virial reductions) ──
  cudaEventRecord(s->evt_kspace_start, 0);
  k_kspace<<<gg,tb>>>(s->nx,s->ny,s->nz,lx,ly,lz,s->d_rho,s->d_ge,s->d_gf,s->d_gv,s->d_gs,
                      s->d_gsv,s->d_gx,s->d_gy,s->d_gz,s->d_pot,want_pot,scaleinv,s->d_red);
  cudaEventRecord(s->evt_kspace_stop, 0);
  // ── Phase 4: one inverse FFT of the mesh potential ──
  cudaEventRecord(s->evt_fftinv_start, 0);
  cufftExecZ2Z(s->plan,s->d_pot,s->d_pot,CUFFT_INVERSE);
  cudaEventRecord(s->evt_fftinv_stop, 0);
  // ── Phase 5: Gather ──
  cudaEventRecord(s->evt_gather_start, 0);
  if(quads){
    if(nord==4) k_gather_quads<4><<<ga,tb>>>(nlocal,s->d_x,d_boxlo,lx,ly,lz,s->d_q,qscale,s->nx,s->ny,s->nz,s->d_gx,s->d_gy,s->d_gz,s->d_pot,want_pot,s->d_force,s->d_vpot,s->d_red);
    else        k_gather_quads<6><<<ga,tb>>>(nlocal,s->d_x,d_boxlo,lx,ly,lz,s->d_q,qscale,s->nx,s->ny,s->nz,s->d_gx,s->d_gy,s->d_gz,s->d_pot,want_pot,s->d_force,s->d_vpot,s->d_red);
  } else {
    if(nord==4) k_gather_cubes2<4><<<ga,tb>>>(nlocal,s->d_x,d_boxlo,lx,ly,lz,s->d_q,qscale,s->nx,s->ny,s->nz,s->d_gx,s->d_gy,s->d_gz,s->d_pot,want_pot,s->d_force,s->d_vpot,s->d_red);
    else        k_gather_cubes2<6><<<ga,tb>>>(nlocal,s->d_x,d_boxlo,lx,ly,lz,s->d_q,qscale,s->nx,s->ny,s->nz,s->d_gx,s->d_gy,s->d_gz,s->d_pot,want_pot,s->d_force,s->d_vpot,s->d_red);
  }
  cudaEventRecord(s->evt_gather_stop, 0);
  // ── Accumulate timing ──
  { float ms;
    cudaEventSynchronize(s->evt_spread_stop); cudaEventElapsedTime(&ms,s->evt_spread_start,s->evt_spread_stop); s->acc_spread += ms;
    cudaEventSynchronize(s->evt_fftfwd_stop); cudaEventElapsedTime(&ms,s->evt_fftfwd_start,s->evt_fftfwd_stop); s->acc_fftfwd += ms;
    cudaEventSynchronize(s->evt_kspace_stop); cudaEventElapsedTime(&ms,s->evt_kspace_start,s->evt_kspace_stop); s->acc_kspace += ms;
    cudaEventSynchronize(s->evt_fftinv_stop); cudaEventElapsedTime(&ms,s->evt_fftinv_start,s->evt_fftinv_stop); s->acc_fftinv += ms;
    cudaEventSynchronize(s->evt_gather_stop); cudaEventElapsedTime(&ms,s->evt_gather_start,s->evt_gather_stop); s->acc_gather += ms;
  }
  ++s->timing_count;
  if (s->timing_count >= s->timing_interval && s->timing_count > 0) {
    if (getenv("SOG_GPU_TIMING")) {
      int n = s->timing_count;
      printf("SOG_GPU_TIMING (n=%d): spread=%.3f fft_fwd=%.3f kspace=%.3f fft_inv=%.3f gather=%.3f ms/step  grid=%dx%dx%d=%zu spline=%s\n",
             n, s->acc_spread/n, s->acc_fftfwd/n, s->acc_kspace/n, s->acc_fftinv/n, s->acc_gather/n,
             s->nx, s->ny, s->nz, ng, quads ? (nord==6?"QuadS-6":"QuadS-4") : (nord==6?"CubeS2-6":"CubeS2-4"));
    }
    s->acc_spread=s->acc_fftfwd=s->acc_kspace=s->acc_fftinv=s->acc_gather=0;
    s->timing_count=0;
  }

  double red[20]; CK(cudaMemcpy(red,s->d_red,20*sizeof(double),cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(force,s->d_force,3*nlocal*sizeof(double),cudaMemcpyDeviceToHost));
  if(want_pot){
    CK(cudaMemcpy(vpot,s->d_vpot,nlocal*sizeof(double),cudaMemcpyDeviceToHost));
    // v_i self-energy part (matches sog.cpp:2083): vpot[i] -= qscale·q_i·(rsi·diag_sum/V + 2·self_coeff)
    double rsi = remove_self ? red[7]/volume : 0.0;
    double sc = 2.0*self_coeff;
    for(int i=0;i<nlocal;++i) vpot[i] -= qscale*q[i]*(rsi + sc);
  }
  cudaFree(d_boxlo);
  // Energy = qscale * [ 0.5*V*Σ(s2·ge·|ρ|²)  −  (rsi) qsqsum·diag_sum/(2V)  −  self_coeff·qsqsum ]
  // (matches sog.cpp: k≠0 mesh sum + mesh self-term + real-space self-energy). red[7]=diag_sum.
  double e_kneq0 = 0.5*volume*red[0];
  double e_self  = self_coeff*qsqsum + (remove_self ? qsqsum*red[7]/(2.0*volume) : 0.0);
  double energy = qscale*(e_kneq0 - e_self);
  // ── Analytic Fourier virial: the COMPLETE strain-derivative of E_k ──
  // W_αβ = ½·V·qscale · Σ s²·|ρ̂(k)|²·(G_E·δ_αβ − K_v·k_α·k_β)
  // where G_E = Σ a_m e^{−½β_m k²}, K_v = Σ a_m·β_m e^{−½β_m k²}.
  // The strain-derivative of |ρ̂(k)|² is identically ZERO under fractional-
  // coordinate charge spreading + Form-B deconvolution. The r⊗F approach
  // (red[14..19]) is INCOMPLETE — it omits the box-explicit strain term.
  // This matches fastsog.cpp, PPPM vg, and the CPU sog.cpp analytic path.
  // red[1..6] = fv_local; red[14..19] = r⊗F (retained for diagnostics).
  for(int j=0;j<6;++j) virial6[j]=0.5*volume*qscale*red[1+j];
  // Self-energy strain-derivative virial (matches sog.cpp + fastsog.cpp).
  // W_self_αβ = qscale·qsqsum/(2V)·(Σ K_v·k_α k_β − δ·Σ K).  No /3 factor.
  if(remove_self){
    double sv_pref = qscale*qsqsum/(2.0*volume);
    virial6[0] += sv_pref*(red[8]  - red[7]);
    virial6[1] += sv_pref*(red[9]  - red[7]);
    virial6[2] += sv_pref*(red[10] - red[7]);
    virial6[3] += sv_pref* red[11];
    virial6[4] += sv_pref* red[12];
    virial6[5] += sv_pref* red[13];
  }
  (void)quads;
  return energy;
}
