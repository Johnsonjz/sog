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
  double *d_brick=nullptr;             // local owned+ghost scalar brick
  size_t cap_brick=0;
  double *d_boxlo=nullptr;           // 3 doubles, box origin (persistent device buffer)
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
// Local reduction slot -> legacy ``red[]`` output slot.  Slots 0..6 are
// diag_sum, energy and the six virial components; 7..13 the self-energy stress
// accumulators.
__device__ __constant__ int SELF_RED_MAP[14] =
    {7, 0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13};

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
  // ── Reductions ──
  // 14 global accumulators per grid point used to be one atomicAdd each onto the
  // same 20 doubles: at a 125³ mesh that is 27 M contending global atomics per
  // step and it dominated the whole k-space solve (~20 ms).  Reduce inside the
  // warp (shuffles), then per block in shared memory, then one atomicAdd per
  // block -- 8 warps x 7629 blocks instead of 1.95 M x 14.
  double acc[14];
  acc[0] = active ? gs[idx] : 0.0;
  const double e = s2*gee*rho2;
  acc[1]  = e;
  acc[2]  = s2*rho2*(gee - gvv*kx*kx);
  acc[3]  = s2*rho2*(gee - gvv*ky*ky);
  acc[4]  = s2*rho2*(gee - gvv*kz*kz);
  acc[5]  = s2*rho2*(-gvv*kx*ky);
  acc[6]  = s2*rho2*(-gvv*kx*kz);
  acc[7]  = s2*rho2*(-gvv*ky*kz);
  if(active){
    const double sv=gsv[idx];
    acc[8]  = sv*kx*kx; acc[9]  = sv*ky*ky; acc[10] = sv*kz*kz;
    acc[11] = sv*kx*ky; acc[12] = sv*kx*kz; acc[13] = sv*ky*kz;
  } else {
    for(int t=8;t<14;++t) acc[t]=0.0;
  }
  #pragma unroll
  for(int t=0;t<14;++t){
    #pragma unroll
    for(int o=16;o>0;o>>=1) acc[t]+=__shfl_down_sync(0xffffffffu,acc[t],o);
  }
  __shared__ double sred[14][8];
  const int lane=(int)(threadIdx.x&31u), warp=(int)(threadIdx.x>>5);
  if(lane==0){
    #pragma unroll
    for(int t=0;t<14;++t) sred[t][warp]=acc[t];
  }
  __syncthreads();
  const int nwarps=(int)(blockDim.x>>5);
  if((int)threadIdx.x<14){
    double v=sred[threadIdx.x][0];
    for(int w=1;w<nwarps;++w) v+=sred[threadIdx.x][w];
    atomicAdd(&red[SELF_RED_MAP[threadIdx.x]], v);
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

// ── Distributed Grid3d local-brick kernels ──
// Convert a global periodic grid index to the equivalent unwrapped index
// stored by this rank's owned+ghost brick.
__device__ __forceinline__ int brick_coord(int g,int n,int lo,int hi){
  if(g<lo) g += ((lo-g+n-1)/n)*n;
  if(g>hi) g -= ((g-hi+n-1)/n)*n;
  return (g>=lo && g<=hi) ? g-lo : -1;
}
__device__ __forceinline__ size_t bidx(int ix,int iy,int iz,int nx,int ny){
  return (size_t)iz*ny*nx+(size_t)iy*nx+ix;
}

template<int ORD>
__global__ void k_spread_cubes2_brick(int nlocal,const double*x,
    double xlo,double ylo,double zlo,double lx,double ly,double lz,
    const double*q,double rho_scale,int nx,int ny,int nz,
    int bxlo,int bxhi,int bylo,int byhi,int bzlo,int bzhi,double*rho){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=nlocal) return;
  double fx=periodic_frac(x[3*i+0],xlo,lx)*nx;
  double fy=periodic_frac(x[3*i+1],ylo,ly)*ny;
  double fz=periodic_frac(x[3*i+2],zlo,lz)*nz;
  int ix0=(int)floor(fx),iy0=(int)floor(fy),iz0=(int)floor(fz);
  double tx=fx-ix0,ty=fy-iy0,tz=fz-iz0,qs=rho_scale*q[i];
  int bnx=bxhi-bxlo+1,bny=byhi-bylo+1;
  const int NN=(ORD==4)?32:88; const double xi=(ORD==4)?kXi4:kXi6;
  for(int k=0;k<NN;++k){
    GpuNode nd=(ORD==4)?c_nodes4[k]:c_nodes6[k];
    double w=(ORD==4)?cs2_w4(tx,ty,tz,nd,xi):cs2_w6(tx,ty,tz,nd,xi);
    if(w==0.0) continue;
    int gx=brick_coord(ix0+nd.dx,nx,bxlo,bxhi);
    int gy=brick_coord(iy0+nd.dy,ny,bylo,byhi);
    int gz=brick_coord(iz0+nd.dz,nz,bzlo,bzhi);
    if(gx>=0&&gy>=0&&gz>=0) atomicAdd(&rho[bidx(gx,gy,gz,bnx,bny)],qs*w);
  }
}

template<int NORD>
__global__ void k_spread_quads_brick(int nlocal,const double*x,
    double xlo,double ylo,double zlo,double lx,double ly,double lz,
    const double*q,double rho_scale,int nx,int ny,int nz,
    int bxlo,int bxhi,int bylo,int byhi,int bzlo,int bzhi,double*rho){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=nlocal) return;
  double fx=periodic_frac(x[3*i+0],xlo,lx)*nx;
  double fy=periodic_frac(x[3*i+1],ylo,ly)*ny;
  double fz=periodic_frac(x[3*i+2],zlo,lz)*nz;
  int ix0=(int)floor(fx),iy0=(int)floor(fy),iz0=(int)floor(fz);
  double tx=fx-ix0,ty=fy-iy0,tz=fz-iz0,qs=rho_scale*q[i];
  double wx[NORD],wy[NORD],wz[NORD];
  if(NORD==4){quads4_w(tx,wx);quads4_w(ty,wy);quads4_w(tz,wz);}
  else {quads6_w(tx,wx);quads6_w(ty,wy);quads6_w(tz,wz);}
  int off=(NORD==4)?-1:-2,bnx=bxhi-bxlo+1,bny=byhi-bylo+1;
  for(int a=0;a<NORD;++a){int gx=brick_coord(ix0+off+a,nx,bxlo,bxhi);if(gx<0)continue;
    for(int b=0;b<NORD;++b){int gy=brick_coord(iy0+off+b,ny,bylo,byhi);if(gy<0)continue;
      for(int c=0;c<NORD;++c){int gz=brick_coord(iz0+off+c,nz,bzlo,bzhi);if(gz<0)continue;
        atomicAdd(&rho[bidx(gx,gy,gz,bnx,bny)],qs*wx[a]*wy[b]*wz[c]);}}}
}

template<int ORD>
__global__ void k_gather_cubes2_brick(int nlocal,const double*x,
    double xlo,double ylo,double zlo,double lx,double ly,double lz,
    const double*q,double qscale,int nx,int ny,int nz,
    int bxlo,int bxhi,int bylo,int byhi,int bzlo,int bzhi,const double*u,
    double*force,double*vpot){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=nlocal) return;
  double fx=periodic_frac(x[3*i+0],xlo,lx)*nx;
  double fy=periodic_frac(x[3*i+1],ylo,ly)*ny;
  double fz=periodic_frac(x[3*i+2],zlo,lz)*nz;
  int ix0=(int)floor(fx),iy0=(int)floor(fy),iz0=(int)floor(fz);
  double tx=fx-ix0,ty=fy-iy0,tz=fz-iz0,dtx=0,dty=0,dtz=0,gp=0;
  int bnx=bxhi-bxlo+1,bny=byhi-bylo+1;
  const int NN=(ORD==4)?32:88; const double xi=(ORD==4)?kXi4:kXi6;
  for(int k=0;k<NN;++k){
    GpuNode nd=(ORD==4)?c_nodes4[k]:c_nodes6[k];
    GpuWeightGrad wg=(ORD==4)?cs2_wg4(tx,ty,tz,nd,xi):cs2_wg6(tx,ty,tz,nd,xi);
    if(wg.v==0.0) continue;
    int gx=brick_coord(ix0+nd.dx,nx,bxlo,bxhi);
    int gy=brick_coord(iy0+nd.dy,ny,bylo,byhi);
    int gz=brick_coord(iz0+nd.dz,nz,bzlo,bzhi);
    if(gx<0||gy<0||gz<0) continue;
    double up=u[bidx(gx,gy,gz,bnx,bny)];
    dtx+=wg.dx*up;dty+=wg.dy*up;dtz+=wg.dz*up;gp+=wg.v*up;
  }
  double qi=q[i];
  force[3*i+0]=-qscale*qi*dtx*nx/lx;
  force[3*i+1]=-qscale*qi*dty*ny/ly;
  force[3*i+2]=-qscale*qi*dtz*nz/lz;
  vpot[i]=qscale*gp;
}

template<int NORD>
__global__ void k_gather_quads_brick(int nlocal,const double*x,
    double xlo,double ylo,double zlo,double lx,double ly,double lz,
    const double*q,double qscale,int nx,int ny,int nz,
    int bxlo,int bxhi,int bylo,int byhi,int bzlo,int bzhi,const double*u,
    double*force,double*vpot){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=nlocal) return;
  double fx=periodic_frac(x[3*i+0],xlo,lx)*nx;
  double fy=periodic_frac(x[3*i+1],ylo,ly)*ny;
  double fz=periodic_frac(x[3*i+2],zlo,lz)*nz;
  int ix0=(int)floor(fx),iy0=(int)floor(fy),iz0=(int)floor(fz);
  double tx=fx-ix0,ty=fy-iy0,tz=fz-iz0,dtx=0,dty=0,dtz=0,gp=0;
  double wx[NORD],wy[NORD],wz[NORD],dwx[NORD],dwy[NORD],dwz[NORD];
  if(NORD==4){quads4_wd(tx,wx,dwx);quads4_wd(ty,wy,dwy);quads4_wd(tz,wz,dwz);}
  else {quads6_wd(tx,wx,dwx);quads6_wd(ty,wy,dwy);quads6_wd(tz,wz,dwz);}
  int off=(NORD==4)?-1:-2,bnx=bxhi-bxlo+1,bny=byhi-bylo+1;
  for(int a=0;a<NORD;++a){int gx=brick_coord(ix0+off+a,nx,bxlo,bxhi);if(gx<0)continue;
    for(int b=0;b<NORD;++b){int gy=brick_coord(iy0+off+b,ny,bylo,byhi);if(gy<0)continue;
      for(int c=0;c<NORD;++c){int gz=brick_coord(iz0+off+c,nz,bzlo,bzhi);if(gz<0)continue;
        double up=u[bidx(gx,gy,gz,bnx,bny)];
        dtx+=dwx[a]*wy[b]*wz[c]*up;dty+=wx[a]*dwy[b]*wz[c]*up;
        dtz+=wx[a]*wy[b]*dwz[c]*up;gp+=wx[a]*wy[b]*wz[c]*up;}}}
  double qi=q[i];
  force[3*i+0]=-qscale*qi*dtx*nx/lx;
  force[3*i+1]=-qscale*qi*dty*ny/ly;
  force[3*i+2]=-qscale*qi*dtz*nz/lz;
  vpot[i]=qscale*gp;
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
  cudaFree(s->d_boxlo);
  cudaFree(s->d_brick);
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

// v_i self-energy correction, on the device:
//   vpot[i] -= qscale * q[i] * (rsi + 2*self_coeff)
__global__ void k_vpot_self(int n,double*vpot,const double*q,double qscale,double rsi,double sc){
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n) vpot[i]-=qscale*q[i]*(rsi+sc);
}

// Device-resident core: every per-atom array (x, q in; force, vpot out) already
// lives on the GPU, so a runner that keeps coordinates/charges/forces in device
// arrays (LAMMPS Kokkos views) pays no host round trip.  ``sog_gpu_compute``
// below is the host-buffer wrapper around it.
static double sog_gpu_compute_core(SogGpuState*s,int nlocal,const double*d_x,const double*d_q,
    const double*boxlo,double lx,double ly,double lz,double qscale,int spline,int want_pot,
    double*d_force,double*d_vpot,double self_coeff,double qsqsum,int remove_self,
    double*virial6){
  size_t ng=s->ngrid;
  (void)nlocal;
  double boxlo3[3]={boxlo[0],boxlo[1],boxlo[2]};
  if(s->d_boxlo==nullptr) CK(cudaMalloc(&s->d_boxlo,3*sizeof(double)));
  double *d_boxlo=s->d_boxlo;
  if(s->d_red==nullptr) CK(cudaMalloc(&s->d_red,20*sizeof(double)));
  CK(cudaMemcpy(d_boxlo,boxlo3,3*sizeof(double),cudaMemcpyHostToDevice));
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
    if(nord==4) k_spread_quads<4><<<ga,tb>>>(nlocal,d_x,d_boxlo,lx,ly,lz,d_q,rho_scale,s->nx,s->ny,s->nz,s->d_rho);
    else        k_spread_quads<6><<<ga,tb>>>(nlocal,d_x,d_boxlo,lx,ly,lz,d_q,rho_scale,s->nx,s->ny,s->nz,s->d_rho);
  } else {
    if(nord==4) k_spread_cubes2<4><<<ga,tb>>>(nlocal,d_x,d_boxlo,lx,ly,lz,d_q,rho_scale,s->nx,s->ny,s->nz,s->d_rho);
    else        k_spread_cubes2<6><<<ga,tb>>>(nlocal,d_x,d_boxlo,lx,ly,lz,d_q,rho_scale,s->nx,s->ny,s->nz,s->d_rho);
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
    if(nord==4) k_gather_quads<4><<<ga,tb>>>(nlocal,d_x,d_boxlo,lx,ly,lz,d_q,qscale,s->nx,s->ny,s->nz,s->d_gx,s->d_gy,s->d_gz,s->d_pot,want_pot,d_force,d_vpot,s->d_red);
    else        k_gather_quads<6><<<ga,tb>>>(nlocal,d_x,d_boxlo,lx,ly,lz,d_q,qscale,s->nx,s->ny,s->nz,s->d_gx,s->d_gy,s->d_gz,s->d_pot,want_pot,d_force,d_vpot,s->d_red);
  } else {
    if(nord==4) k_gather_cubes2<4><<<ga,tb>>>(nlocal,d_x,d_boxlo,lx,ly,lz,d_q,qscale,s->nx,s->ny,s->nz,s->d_gx,s->d_gy,s->d_gz,s->d_pot,want_pot,d_force,d_vpot,s->d_red);
    else        k_gather_cubes2<6><<<ga,tb>>>(nlocal,d_x,d_boxlo,lx,ly,lz,d_q,qscale,s->nx,s->ny,s->nz,s->d_gx,s->d_gy,s->d_gz,s->d_pot,want_pot,d_force,d_vpot,s->d_red);
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
  if(want_pot){
    // v_i self-energy part (matches sog.cpp:2083), applied on the device so the
    // potential never leaves the GPU between the gather and the consumer:
    //   vpot[i] -= qscale*q_i*(rsi*diag_sum/V + 2*self_coeff)
    const double rsi = remove_self ? red[7]/volume : 0.0;
    const double sc = 2.0*self_coeff;
    if(nlocal>0) k_vpot_self<<<ga,tb>>>(nlocal,d_vpot,d_q,qscale,rsi,sc);
  }
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

static void ensure_local_atoms(SogGpuState*s,int nlocal){
  if(nlocal<=s->cap_atoms) return;
  cudaFree(s->d_x);cudaFree(s->d_q);cudaFree(s->d_force);cudaFree(s->d_vpot);
  CK(cudaMalloc(&s->d_x,3*(size_t)nlocal*sizeof(double)));
  CK(cudaMalloc(&s->d_q,(size_t)nlocal*sizeof(double)));
  CK(cudaMalloc(&s->d_force,3*(size_t)nlocal*sizeof(double)));
  CK(cudaMalloc(&s->d_vpot,(size_t)nlocal*sizeof(double)));
  s->cap_atoms=nlocal;
}

static void ensure_local_brick(SogGpuState*s,size_t nbrick){
  if(nbrick<=s->cap_brick) return;
  cudaFree(s->d_brick);
  CK(cudaMalloc(&s->d_brick,nbrick*sizeof(double)));
  s->cap_brick=nbrick;
}

extern "C" void sog_gpu_spread_brick(SogGpuState*s,int nlocal,
    const double*x,const double*boxlo,double lx,double ly,double lz,
    const double*q,double rho_scale,int spline,int nx,int ny,int nz,
    int nxlo,int nxhi,int nylo,int nyhi,int nzlo,int nzhi,double*rho_brick){
  size_t nb=(size_t)(nxhi-nxlo+1)*(nyhi-nylo+1)*(nzhi-nzlo+1);
  ensure_local_brick(s,nb);
  CK(cudaMemset(s->d_brick,0,nb*sizeof(double)));
  if(nlocal>0){
    ensure_local_atoms(s,nlocal);
    CK(cudaMemcpy(s->d_x,x,3*(size_t)nlocal*sizeof(double),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(s->d_q,q,(size_t)nlocal*sizeof(double),cudaMemcpyHostToDevice));
    int tb=256,ga=(nlocal+tb-1)/tb;
    bool quads=(spline==SOG_QUADS_4||spline==SOG_QUADS_6);
    int nord=(spline==SOG_CUBES2_6||spline==SOG_QUADS_6)?6:4;
    if(quads){
      if(nord==4) k_spread_quads_brick<4><<<ga,tb>>>(nlocal,s->d_x,boxlo[0],boxlo[1],boxlo[2],lx,ly,lz,s->d_q,rho_scale,nx,ny,nz,nxlo,nxhi,nylo,nyhi,nzlo,nzhi,s->d_brick);
      else k_spread_quads_brick<6><<<ga,tb>>>(nlocal,s->d_x,boxlo[0],boxlo[1],boxlo[2],lx,ly,lz,s->d_q,rho_scale,nx,ny,nz,nxlo,nxhi,nylo,nyhi,nzlo,nzhi,s->d_brick);
    }else{
      if(nord==4) k_spread_cubes2_brick<4><<<ga,tb>>>(nlocal,s->d_x,boxlo[0],boxlo[1],boxlo[2],lx,ly,lz,s->d_q,rho_scale,nx,ny,nz,nxlo,nxhi,nylo,nyhi,nzlo,nzhi,s->d_brick);
      else k_spread_cubes2_brick<6><<<ga,tb>>>(nlocal,s->d_x,boxlo[0],boxlo[1],boxlo[2],lx,ly,lz,s->d_q,rho_scale,nx,ny,nz,nxlo,nxhi,nylo,nyhi,nzlo,nzhi,s->d_brick);
    }
    CK(cudaGetLastError());
  }
  CK(cudaMemcpy(rho_brick,s->d_brick,nb*sizeof(double),cudaMemcpyDeviceToHost));
}

extern "C" void sog_gpu_gather_brick(SogGpuState*s,int nlocal,
    const double*x,const double*boxlo,double lx,double ly,double lz,
    const double*q,double qscale,int spline,int nx,int ny,int nz,
    int nxlo,int nxhi,int nylo,int nyhi,int nzlo,int nzhi,const double*u_brick,
    double*force,double*vpot){
  if(nlocal<=0) return;
  size_t nb=(size_t)(nxhi-nxlo+1)*(nyhi-nylo+1)*(nzhi-nzlo+1);
  ensure_local_brick(s,nb);ensure_local_atoms(s,nlocal);
  CK(cudaMemcpy(s->d_brick,u_brick,nb*sizeof(double),cudaMemcpyHostToDevice));
  CK(cudaMemcpy(s->d_x,x,3*(size_t)nlocal*sizeof(double),cudaMemcpyHostToDevice));
  CK(cudaMemcpy(s->d_q,q,(size_t)nlocal*sizeof(double),cudaMemcpyHostToDevice));
  int tb=256,ga=(nlocal+tb-1)/tb;
  bool quads=(spline==SOG_QUADS_4||spline==SOG_QUADS_6);
  int nord=(spline==SOG_CUBES2_6||spline==SOG_QUADS_6)?6:4;
  if(quads){
    if(nord==4) k_gather_quads_brick<4><<<ga,tb>>>(nlocal,s->d_x,boxlo[0],boxlo[1],boxlo[2],lx,ly,lz,s->d_q,qscale,nx,ny,nz,nxlo,nxhi,nylo,nyhi,nzlo,nzhi,s->d_brick,s->d_force,s->d_vpot);
    else k_gather_quads_brick<6><<<ga,tb>>>(nlocal,s->d_x,boxlo[0],boxlo[1],boxlo[2],lx,ly,lz,s->d_q,qscale,nx,ny,nz,nxlo,nxhi,nylo,nyhi,nzlo,nzhi,s->d_brick,s->d_force,s->d_vpot);
  }else{
    if(nord==4) k_gather_cubes2_brick<4><<<ga,tb>>>(nlocal,s->d_x,boxlo[0],boxlo[1],boxlo[2],lx,ly,lz,s->d_q,qscale,nx,ny,nz,nxlo,nxhi,nylo,nyhi,nzlo,nzhi,s->d_brick,s->d_force,s->d_vpot);
    else k_gather_cubes2_brick<6><<<ga,tb>>>(nlocal,s->d_x,boxlo[0],boxlo[1],boxlo[2],lx,ly,lz,s->d_q,qscale,nx,ny,nz,nxlo,nxhi,nylo,nyhi,nzlo,nzhi,s->d_brick,s->d_force,s->d_vpot);
  }
  CK(cudaGetLastError());
  CK(cudaMemcpy(force,s->d_force,3*(size_t)nlocal*sizeof(double),cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(vpot,s->d_vpot,(size_t)nlocal*sizeof(double),cudaMemcpyDeviceToHost));
}

// ── Host-buffer wrapper (kept for the non-Kokkos call sites in sog.cpp) ──
extern "C" double sog_gpu_compute(SogGpuState*s,int nlocal,const double*x,const double*boxlo,
    double lx,double ly,double lz,const double*q,double qscale,int spline,int want_pot,
    double*force,double*virial6,double*vpot,double self_coeff,double qsqsum,int remove_self){
  if(nlocal>s->cap_atoms){
    cudaFree(s->d_x);cudaFree(s->d_q);cudaFree(s->d_force);cudaFree(s->d_vpot);
    CK(cudaMalloc(&s->d_x,3*nlocal*sizeof(double)));CK(cudaMalloc(&s->d_q,nlocal*sizeof(double)));
    CK(cudaMalloc(&s->d_force,3*nlocal*sizeof(double)));CK(cudaMalloc(&s->d_vpot,nlocal*sizeof(double)));
    s->cap_atoms=nlocal;
  }
  CK(cudaMemcpy(s->d_x,x,3*(size_t)nlocal*sizeof(double),cudaMemcpyHostToDevice));
  CK(cudaMemcpy(s->d_q,q,(size_t)nlocal*sizeof(double),cudaMemcpyHostToDevice));
  double e = sog_gpu_compute_core(s,nlocal,s->d_x,s->d_q,boxlo,lx,ly,lz,qscale,spline,
                                  want_pot,s->d_force,s->d_vpot,self_coeff,qsqsum,
                                  remove_self,virial6);
  CK(cudaMemcpy(force,s->d_force,3*(size_t)nlocal*sizeof(double),cudaMemcpyDeviceToHost));
  if(want_pot) CK(cudaMemcpy(vpot,s->d_vpot,(size_t)nlocal*sizeof(double),cudaMemcpyDeviceToHost));
  return e;
}

// ── Device-resident entry point (LAMMPS Kokkos) ──
// x, q, force and vpot are device pointers (e.g. Kokkos views).  Only the
// energy/virial reductions (160 bytes) cross back to the host.
extern "C" double sog_gpu_compute_dev(SogGpuState*s,int nlocal,const double*d_x,const double*boxlo,
    double lx,double ly,double lz,const double*d_q,double qscale,int spline,int want_pot,
    double*d_force,double*virial6,double*d_vpot,double self_coeff,double qsqsum,int remove_self){
  return sog_gpu_compute_core(s,nlocal,d_x,d_q,boxlo,lx,ly,lz,qscale,spline,want_pot,
                              d_force,d_vpot,self_coeff,qsqsum,remove_self,virial6);
}

// ── Multi-rank replicated-grid path ─────────────────────────────────────────
// The distributed (Grid3d/Remap) mesh keeps the FFT on the host, and at np=2
// the KISS pencils alone cost ~86 ms/step for a 125^3 grid -- 25x the whole
// single-GPU solve.  Instead every rank spreads its own atoms into a FULL grid,
// the grids are summed with one MPI_Allreduce, and each rank then runs the
// same device cuFFT/green/gather pipeline it would run alone.  Correct by
// construction: the summed density is the global one, so the potential is
// identical to the single-rank solve, and each rank gathers only its local
// atoms.  The price is one grid-sized reduction per step; the benefit is a
// device speed FFT at any rank count.

// Zero the device mesh and spread ``nlocal`` atoms into it.
extern "C" void sog_gpu_spread_full_dev(SogGpuState*s,int nlocal,const double*d_x,
    const double*boxlo,double lx,double ly,double lz,const double*d_q,
    double rho_scale,int spline){
  size_t ng=s->ngrid;
  if(s->d_rho==nullptr) return;
  int tb=256, gg=(int)((ng+tb-1)/tb), ga=(nlocal+tb-1)/tb;
  if(s->d_boxlo==nullptr) CK(cudaMalloc(&s->d_boxlo,3*sizeof(double)));
  CK(cudaMemcpy(s->d_boxlo,boxlo,3*sizeof(double),cudaMemcpyHostToDevice));
  k_zero_complex<<<gg,tb>>>(s->d_rho,ng);
  bool quads=(spline==SOG_QUADS_4||spline==SOG_QUADS_6);
  int nord=(spline==SOG_CUBES2_6||spline==SOG_QUADS_6)?6:4;
  if(nlocal<=0) return;
  if(quads){
    if(nord==4) k_spread_quads<4><<<ga,tb>>>(nlocal,d_x,s->d_boxlo,lx,ly,lz,d_q,rho_scale,s->nx,s->ny,s->nz,s->d_rho);
    else        k_spread_quads<6><<<ga,tb>>>(nlocal,d_x,s->d_boxlo,lx,ly,lz,d_q,rho_scale,s->nx,s->ny,s->nz,s->d_rho);
  } else {
    if(nord==4) k_spread_cubes2<4><<<ga,tb>>>(nlocal,d_x,s->d_boxlo,lx,ly,lz,d_q,rho_scale,s->nx,s->ny,s->nz,s->d_rho);
    else        k_spread_cubes2<6><<<ga,tb>>>(nlocal,d_x,s->d_boxlo,lx,ly,lz,d_q,rho_scale,s->nx,s->ny,s->nz,s->d_rho);
  }
}

// Device mesh <-> host buffer (the reduction runs on the host; OpenMPI here is
// not GPU-aware, so the grid is staged through host memory).
// Raw device mesh pointer, for a CUDA-aware MPI reduction (no host staging).
extern "C" double* sog_gpu_rho_device_ptr(SogGpuState*s){ return s->d_rho ? (double*)s->d_rho : nullptr; }

extern "C" void sog_gpu_rho_download(SogGpuState*s,double*host){
  CK(cudaMemcpy(host,s->d_rho,2*s->ngrid*sizeof(double),cudaMemcpyDeviceToHost));
}
extern "C" void sog_gpu_rho_upload(SogGpuState*s,const double*host){
  CK(cudaMemcpy(s->d_rho,host,2*s->ngrid*sizeof(double),cudaMemcpyHostToDevice));
}

// FFT -> green multiply -> inverse, then gather force/potential for ``nlocal``
// atoms.  The density must already be the summed global mesh.
extern "C" double sog_gpu_solve_dev(SogGpuState*s,int nlocal,const double*d_x,
    const double*boxlo,double lx,double ly,double lz,const double*d_q,
    double qscale,int spline,int want_pot,double*d_force,double*virial6,
    double*d_vpot,double self_coeff,double qsqsum,int remove_self){
  size_t ng=s->ngrid;
  if(s->d_boxlo==nullptr) CK(cudaMalloc(&s->d_boxlo,3*sizeof(double)));
  CK(cudaMemcpy(s->d_boxlo,boxlo,3*sizeof(double),cudaMemcpyHostToDevice));
  if(s->d_red==nullptr) CK(cudaMalloc(&s->d_red,20*sizeof(double)));
  CK(cudaMemset(s->d_red,0,20*sizeof(double)));
  double volume=lx*ly*lz, scaleinv=1.0/(double)ng;
  int tb=256, gg=(int)((ng+tb-1)/tb), ga=(nlocal+tb-1)/tb;
  bool quads=(spline==SOG_QUADS_4||spline==SOG_QUADS_6);
  int nord=(spline==SOG_CUBES2_6||spline==SOG_QUADS_6)?6:4;
  cufftExecZ2Z(s->plan,s->d_rho,s->d_rho,CUFFT_FORWARD);
  k_kspace<<<gg,tb>>>(s->nx,s->ny,s->nz,lx,ly,lz,s->d_rho,s->d_ge,s->d_gf,s->d_gv,
                      s->d_gs,s->d_gsv,s->d_gx,s->d_gy,s->d_gz,s->d_pot,want_pot,
                      scaleinv,s->d_red);
  cufftExecZ2Z(s->plan,s->d_pot,s->d_pot,CUFFT_INVERSE);
  if(nlocal>0){
    if(quads){
      if(nord==4) k_gather_quads<4><<<ga,tb>>>(nlocal,d_x,s->d_boxlo,lx,ly,lz,d_q,qscale,s->nx,s->ny,s->nz,s->d_gx,s->d_gy,s->d_gz,s->d_pot,want_pot,d_force,d_vpot,s->d_red);
      else        k_gather_quads<6><<<ga,tb>>>(nlocal,d_x,s->d_boxlo,lx,ly,lz,d_q,qscale,s->nx,s->ny,s->nz,s->d_gx,s->d_gy,s->d_gz,s->d_pot,want_pot,d_force,d_vpot,s->d_red);
    } else {
      if(nord==4) k_gather_cubes2<4><<<ga,tb>>>(nlocal,d_x,s->d_boxlo,lx,ly,lz,d_q,qscale,s->nx,s->ny,s->nz,s->d_gx,s->d_gy,s->d_gz,s->d_pot,want_pot,d_force,d_vpot,s->d_red);
      else        k_gather_cubes2<6><<<ga,tb>>>(nlocal,d_x,s->d_boxlo,lx,ly,lz,d_q,qscale,s->nx,s->ny,s->nz,s->d_gx,s->d_gy,s->d_gz,s->d_pot,want_pot,d_force,d_vpot,s->d_red);
    }
  }
  double red[20]; CK(cudaMemcpy(red,s->d_red,20*sizeof(double),cudaMemcpyDeviceToHost));
  if(want_pot){
    const double rsi=remove_self?red[7]/volume:0.0;
    if(nlocal>0) k_vpot_self<<<ga,tb>>>(nlocal,d_vpot,d_q,qscale,rsi,2.0*self_coeff);
  }
  double e_kneq0=0.5*volume*red[0];
  double e_self=self_coeff*qsqsum+(remove_self?qsqsum*red[7]/(2.0*volume):0.0);
  for(int j=0;j<6;++j) virial6[j]=0.5*volume*qscale*red[1+j];
  if(remove_self){
    double sv_pref=qscale*qsqsum/(2.0*volume);
    virial6[0]+=sv_pref*(red[8]-red[7]);
    virial6[1]+=sv_pref*(red[9]-red[7]);
    virial6[2]+=sv_pref*(red[10]-red[7]);
    virial6[3]+=sv_pref*red[11];
    virial6[4]+=sv_pref*red[12];
    virial6[5]+=sv_pref*red[13];
  }
  return qscale*(e_kneq0-e_self);
}
