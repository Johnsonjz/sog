// sog_gpu.cu — plugin-internal GPU SOG kspace (raw CUDA + cuFFT). Tier-2 full-GPU pipeline:
//   spread(q@x) -> cuFFT fwd -> green-multiply (grad meshes + potential + energy/6-virial reductions)
//   -> one cuFFT inverse of the mesh potential -> adjoint weight-derivative gather.
// Two splines: separable QuadS (PRIMARY — 3D tensor-product 1D weights, GPU-friendly scatter once
// cuFFT makes the FFT ~free and spreading dominates) and CubeS2-4 (BASELINE — bit-matches CPU sog.cpp).
// Math ported from sog.cpp::compute_single. STATUS: nvcc-syntax-compiled; NOT yet GPU-bit-validated
// against the CPU path (cuFFT is unnormalized vs LAMMPS FFT3d 1/ngrid — the normalization constants
// below must be confirmed by the run-0 bit-match once the GPU frees). See sog-progress.md §8.
#include "sog_gpu.cuh"
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
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
  cufftHandle plan=0;                 // Z2Z, 3D (replicated, full grid)
  // Distributed (slab) k-space plans.  The (x,y) pair is real-to-complex: the
  // density in the brick is real, so only the compact half-spectrum (Nh =
  // nx/2+1) is stored/transported along x.  The z transform stays full-length
  // complex (the +/-kz partners are needed after the x compression).
  cufftHandle plan_xy_rc=0, plan_xy_cr=0, plan_z=0;
  int slab_ZL=-1, slab_YL=-1, slab_Nh=-1;
  cufftDoubleComplex *d_rho=nullptr;  // charge mesh / rho_hat (in place)
  cufftDoubleComplex *d_gx=nullptr, *d_gy=nullptr, *d_gz=nullptr, *d_pot=nullptr;
  double *d_ge=nullptr, *d_gf=nullptr, *d_gv=nullptr;  // green energy/force/virial tables
  // (The bare K and K_v "self" tables are gone: the self term is closed-form,
  //  see self_coeff in sog_gpu_solve_dev, so no per-mode table is uploaded.)
  double *d_x=nullptr, *d_q=nullptr, *d_force=nullptr, *d_vpot=nullptr;  // per-atom
  double *d_brick=nullptr;             // local owned+ghost scalar brick
  size_t cap_brick=0;
  double *d_boxlo=nullptr;           // 3 doubles, box origin (persistent device buffer)
  double *d_red=nullptr;             // energy + 6 fv_local + 6 r⊗F (20 doubles; slots 7..13 unused)
  int cap_atoms=0;

  // GPU timing (cudaEvent-based, accumulated per N steps)
  cudaEvent_t evt_spread_start=0, evt_spread_stop=0;
  cudaEvent_t evt_fftfwd_start=0, evt_fftfwd_stop=0;
  cudaEvent_t evt_kspace_start=0, evt_kspace_stop=0;
  cudaEvent_t evt_fftinv_start=0, evt_fftinv_stop=0;
  cudaEvent_t evt_gather_start=0, evt_gather_stop=0;
  double acc_spread=0, acc_fftfwd=0, acc_kspace=0, acc_fftinv=0, acc_gather=0;
  int timing_count=0, timing_interval=100;  // report every N steps

  // ── slab (distributed) mode ──
  // In this mode the grid buffers are sized for ONE rank's slab (n_g/np) and
  // the green tables hold only this rank's y-slice, instead of the full grid.
  bool slab_mode=false;
  int slab_Y=0;                 // this rank's y length (pencil slab)
  size_t slab_cap=0;            // complex elements in d_rho / d_pot
  size_t slab_gn=0;             // doubles in each of d_ge / d_gf / d_gv
  // scratch for uploading a received halo plane before adding it on the device
  double *d_halo=nullptr;
  size_t cap_halo=0;
  // aligned real scratch for D2Z/Z2D when the brick offset is not 16-aligned
  double *d_real=nullptr;
  size_t cap_real=0;

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
//
// The k-space SELF term used to be accumulated here as a Riemann sum over mesh
// modes (`diag_sum`, legacy red[7]) plus six self-stress accumulators
// (legacy red[8..13]).  That sum is NOT volume-extensive: K(k^2) is dominated
// by the few lowest-|k| modes, whose sampling changes with the box, so for a
// 264-atom cell it grew 9.03x (2x replication) and 76.5x (4x) instead of 8x and
// 64x.  It also made sum(K_v k_a k_b) anisotropic -- (0.827, 1.085, 1.339) x
// the diagonal instead of isotropic -- injecting a spurious shear stress.
// Both are replaced by the exact continuum value, which is closed-form and
// needs no mesh at all; see the `self_coeff` handling in the solve functions.
__global__ void k_kspace(int nx,int ny,int nz,double lx,double ly,double lz,
    const cufftDoubleComplex*rho,const double*ge,const double*gf,const double*gv,
    cufftDoubleComplex*gx,cufftDoubleComplex*gy,cufftDoubleComplex*gz,
    cufftDoubleComplex*pot,int want_pot,double scaleinv,double*red){
  size_t idx=(size_t)blockIdx.x*blockDim.x+threadIdx.x; size_t ngrid=(size_t)nx*ny*nz;
  // Every thread of the block MUST reach the warp shuffles and __syncthreads()
  // below.  An early `return` here made the last (partially filled) block
  // undefined behaviour on three counts: __shfl_down_sync(0xffffffff) asserts
  // that all 32 lanes are active, warps that exited never wrote sred[t][warp]
  // (leaving uninitialised shared memory in the reduction), and
  // __syncthreads() was not reached by every thread.  The visible symptom was
  // a random, system-size-INDEPENDENT additive error in red[0] (energy) and
  // red[1..6] (virial), plus garbage pot/gx/gy/gz for the grid points of that
  // block: 11.4 eV (43 meV/atom) spread over identical calls at 264 atoms,
  // 0.6 eV (35 ueV/atom) at 16896, and a 1e7 atm pressure at 264 atoms.
  // Out-of-range threads now stay in the kernel and contribute exact zeros.
  const bool in_range = (idx < ngrid);
  if(!in_range) idx = 0;
  int ix=idx%nx, iy=(idx/nx)%ny, iz=idx/((size_t)nx*ny);
  int kxm=ix-nx*((2*ix)/nx), kym=iy-ny*((2*iy)/ny), kzm=iz-nz*((2*iz)/nz);
  double kx=2*M_PI/lx*kxm, ky=2*M_PI/ly*kym, kz=2*M_PI/lz*kzm;
  double geff=gf[idx], gee=ge[idx], gvv=gv[idx];
  double rr=rho[idx].x, ri=rho[idx].y, rho2=rr*rr+ri*ri;
  double s2=scaleinv*scaleinv;
  // ── Reductions ──
  // 7 global accumulators per grid point used to be one atomicAdd each onto the
  // same 20 doubles: at a 125³ mesh that is 27 M contending global atomics per
  // step and it dominated the whole k-space solve (~20 ms).  Reduce inside the
  // warp (shuffles), then per block in shared memory, then one atomicAdd per
  // block -- 8 warps x 7629 blocks instead of 1.95 M x 7.
  double acc[7];
  acc[0]  = s2*gee*rho2;                 // energy
  acc[1]  = s2*rho2*(gee - gvv*kx*kx);   // xx
  acc[2]  = s2*rho2*(gee - gvv*ky*ky);   // yy
  acc[3]  = s2*rho2*(gee - gvv*kz*kz);   // zz
  acc[4]  = s2*rho2*(-gvv*kx*ky);        // xy
  acc[5]  = s2*rho2*(-gvv*kx*kz);        // xz
  acc[6]  = s2*rho2*(-gvv*ky*kz);        // yz
  // Out-of-range lanes took the idx=0 loads above; zero them so they add
  // nothing to the block reduction while still participating in the shuffles.
  if(!in_range){
    for(int t=0;t<7;++t) acc[t]=0.0;
  }
  #pragma unroll
  for(int t=0;t<7;++t){
    #pragma unroll
    for(int o=16;o>0;o>>=1) acc[t]+=__shfl_down_sync(0xffffffffu,acc[t],o);
  }
  __shared__ double sred[7][8];
  const int lane=(int)(threadIdx.x&31u), warp=(int)(threadIdx.x>>5);
  if(lane==0){
    #pragma unroll
    for(int t=0;t<7;++t) sred[t][warp]=acc[t];
  }
  __syncthreads();
  const int nwarps=(int)(blockDim.x>>5);
  if((int)threadIdx.x<7){
    double v=sred[threadIdx.x][0];
    for(int w=1;w<nwarps;++w) v+=sred[threadIdx.x][w];
    atomicAdd(&red[threadIdx.x], v);
  }
  // gradient meshes: grad = i*k * (scaleinv*geff*rho)
  // Only real grid points may be written; out-of-range lanes operate on the
  // idx=0 values and must not store them.
  if(in_range){
    double vkr=scaleinv*geff*rr, vki=scaleinv*geff*ri;
    gx[idx].x=-kx*vki; gx[idx].y=kx*vkr;
    gy[idx].x=-ky*vki; gy[idx].y=ky*vkr;
    gz[idx].x=-kz*vki; gz[idx].y=kz*vkr;
    if(want_pot){ pot[idx].x=scaleinv*gee*rr; pot[idx].y=scaleinv*gee*ri; }
  }
}


// ── Distributed (slab) k-space: green multiply on a local PENCIL ─────────────
// The pencil holds the full z range, all x, and this rank's y-slab, with z
// FASTEST: index = iz + nz*(ix + nx*iy_local).  That layout makes both the
// z-transform a contiguous batch and the global-index arithmetic below cheap.
//
// Every quantity must be GLOBAL (theory doc I1/I2/I3): the green table is
// indexed by the global grid index gi = ix + nx*(iy + ny*iz) with iy = ylo +
// iy_local, and k comes from the global integer mode.  Using a local offset
// instead produces a plausible-looking but wrong energy with no fingerprint
// (measured 1.78x / 13.27x), which is why this is spelled out here.
//
// Reduces 7 accumulators (energy + 6 virial) exactly like k_kspace, so the
// caller's global Allreduce is the same one the replicated path already does.
// R2C pencil: only kh = 0..Nh-1 is stored.  Every internal kh stands for
// itself AND its -kx partner (Hermitian weight 2); kh=0 and, for even nx, the
// Nyquist column kh=Nh-1 have no partner (weight 1).  The green tables remain
// FULL-size in x, so the table index uses the original `nx` stride at column
// kh (the table only depends on kx^2).
__global__ void k_green_pencil(int Nh,int nx,int ny,int nz,int ylo,int YL,
    double lx,double ly,double lz,
    const cufftDoubleComplex*rho,const double*ge,const double*gf,const double*gv,
    cufftDoubleComplex*pot,double scaleinv,double*red){
  size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
  const size_t nloc=(size_t)nz*(size_t)Nh*(size_t)YL;
  const bool in_range = (i < nloc);
  if(!in_range) i = 0;
  int iz=(int)(i % nz);
  int kh=(int)((i/nz) % Nh);
  int iyl=(int)(i/((size_t)nz*Nh));
  int iy=ylo+iyl;
  if(!in_range){ iz=0; kh=0; iyl=0; iy=ylo; }
  // The green tables are this rank's COMPACT x half-spectrum y-slice:
  // kh + Nh*(iyl + YL*iz) (the -kx half is redundant for a real kernel).
  const size_t gi=(size_t)kh+(size_t)Nh*((size_t)iyl+(size_t)YL*(size_t)iz);
  const int even_nx = (2*Nh-1 != nx) ? 1 : 0;
  const double w = (kh==0 || (even_nx && kh==Nh-1)) ? 1.0 : 2.0;
  int kym=iy-ny*((2*iy)/ny), kzm=iz-nz*((2*iz)/nz);
  // Physical k = 2 pi / L * mode  (NOT 2 pi / n: the virial terms need it).
  // kx uses the +kh representative; the green table and every acc term depend
  // on kx^2 (or the pair-symmetric kx*ky / kx*kz), so the sign is immaterial.
  double kx=2*M_PI/lx*kh, ky=2*M_PI/ly*kym, kz=2*M_PI/lz*kzm;
  double geff=gf[gi], gee=ge[gi], gvv=gv[gi];
  double rr=rho[i].x, ri=rho[i].y, rho2=rr*rr+ri*ri;
  double s2=scaleinv*scaleinv;
  double acc[7];
  acc[0] = s2*gee*rho2;
  acc[1] = s2*rho2*(gee - gvv*kx*kx);
  acc[2] = s2*rho2*(gee - gvv*ky*ky);
  acc[3] = s2*rho2*(gee - gvv*kz*kz);
  acc[4] = s2*rho2*(-gvv*kx*ky);
  acc[5] = s2*rho2*(-gvv*kx*kz);
  acc[6] = s2*rho2*(-gvv*ky*kz);
  if(!in_range){ for(int t=0;t<7;++t) acc[t]=0.0; }
  else { for(int t=0;t<7;++t) acc[t]*=w; }   // half-spectrum Hermitian weight
  #pragma unroll
  for(int t=0;t<7;++t){
    #pragma unroll
    for(int o=16;o>0;o>>=1) acc[t]+=__shfl_down_sync(0xffffffffu,acc[t],o);
  }
  __shared__ double sred[7][8];
  const int lane=(int)(threadIdx.x&31u), warp=(int)(threadIdx.x>>5);
  if(lane==0){
    #pragma unroll
    for(int t=0;t<7;++t) sred[t][warp]=acc[t];
  }
  __syncthreads();
  const int nwarps=(int)(blockDim.x>>5);
  if((int)threadIdx.x<7){
    double v=sred[threadIdx.x][0];
    for(int w=1;w<nwarps;++w) v+=sred[threadIdx.x][w];
    atomicAdd(&red[threadIdx.x], v);
  }
  if(in_range){ pot[i].x=scaleinv*gee*rr; pot[i].y=scaleinv*gee*ri; }
}

// ── Transpose pack/unpack on the device ────────────────────────────────────
// The host used to repack the mesh with three nested loops per direction; at a
// 150^3 mesh that is ~0.5 GB of strided host traffic per step.  These kernels
// do the same permutation on the GPU so only the MPI message itself crosses
// the bus.
//
// Block decomposition (must match the host's scnt/rdisp exactly): destination
// d owns y in [d*ny/np, (d+1)*ny/np); sender r owns z in [r*nz/np, (r+1)*nz/np).
// A block is laid out [yl][ix][z_local] with z fastest.
__global__ void k_pack_xy_to_blocks(int Nh,int ny,int nz,int ZL,int np,
    const cufftDoubleComplex*cur,cufftDoubleComplex*snd){
  size_t t=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
  const size_t ntot=(size_t)Nh*ny*ZL;
  if(t>=ntot) return;
  const int yb=(int)(t/((size_t)ZL*Nh));          // global y
  int d=0; while(d+1<np && yb >= ((d+1)*ny)/np) ++d;
  const int dl=(d*ny)/np;
  const size_t b=t-(size_t)Nh*ZL*dl;              // index inside the block
  const int z=(int)(b%ZL), kh=(int)((b/ZL)%Nh);
  const size_t src=(size_t)kh+(size_t)Nh*((size_t)yb+(size_t)ny*z);
  snd[t]=cur[src];
}

__global__ void k_unpack_blocks_to_pen(int Nh,int ny,int nz,int np,int YL,
    const cufftDoubleComplex*rcv,cufftDoubleComplex*pen){
  size_t t=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
  const size_t ntot=(size_t)nz*Nh*YL;
  if(t>=ntot) return;
  const int zl=(int)(t%nz), kh=(int)((t/nz)%Nh),
            yl=(int)(t/((size_t)nz*Nh));
  int r=0; while(r+1<np && zl >= ((r+1)*nz)/np) ++r;
  const int rl=(r*nz)/np, rz=((r+1)*nz)/np-rl;
  const size_t start=(size_t)Nh*YL*rl;
  const size_t src=start+(size_t)(zl-rl)+
                   (size_t)rz*((size_t)kh+(size_t)Nh*yl);
  pen[t]=rcv[src];
}

__global__ void k_pack_pen_to_blocks(int Nh,int ny,int nz,int np,int YL,
    const cufftDoubleComplex*pen,cufftDoubleComplex*snd){
  size_t t=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
  const size_t ntot=(size_t)nz*Nh*YL;
  if(t>=ntot) return;
  const int zl=(int)(t%nz), kh=(int)((t/nz)%Nh),
            yl=(int)(t/((size_t)nz*Nh));
  int r=0; while(r+1<np && zl >= ((r+1)*nz)/np) ++r;
  const int rl=(r*nz)/np, rz=((r+1)*nz)/np-rl;
  const size_t start=(size_t)Nh*YL*rl;
  const size_t dst=start+(size_t)(zl-rl)+
                   (size_t)rz*((size_t)kh+(size_t)Nh*yl);
  snd[dst]=pen[t];
}

__global__ void k_unpack_blocks_to_xy(int Nh,int ny,int ZL,int np,
    const cufftDoubleComplex*back,cufftDoubleComplex*cur2){
  size_t t=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
  const size_t ntot=(size_t)Nh*ny*ZL;
  if(t>=ntot) return;
  // The compact slab is (kh fastest, y, z_local): index = kh + Nh*(y + ny*z).
  // Decoding it as the block layout instead permutes y and z (energy and forces
  // wrong, virial right -- the same trap as the full-spectrum version).
  const int z=(int)(t/((size_t)Nh*ny));
  const int y=(int)((t/(size_t)Nh)%ny);
  const int kh=(int)(t%Nh);
  int d=0; while(d+1<np && y >= ((d+1)*ny)/np) ++d;
  const int dl=(d*ny)/np;
  const size_t src=(size_t)Nh*ZL*dl+(size_t)z+
                   (size_t)ZL*((size_t)kh+(size_t)Nh*(y-dl));
  cur2[t]=back[src];
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
  if(s->plan_xy_rc) cufftDestroy(s->plan_xy_rc);
  if(s->plan_xy_cr) cufftDestroy(s->plan_xy_cr);
  if(s->plan_z) cufftDestroy(s->plan_z);
  cudaFree(s->d_rho);cudaFree(s->d_gx);cudaFree(s->d_gy);cudaFree(s->d_gz);cudaFree(s->d_pot);
  cudaFree(s->d_ge);cudaFree(s->d_gf);cudaFree(s->d_gv);
  cudaFree(s->d_x);cudaFree(s->d_q);cudaFree(s->d_force);cudaFree(s->d_vpot);cudaFree(s->d_red);
  cudaFree(s->d_boxlo);
  cudaFree(s->d_brick);
  cudaFree(s->d_halo);
  cudaFree(s->d_real);
  delete s;
}


// ── Distributed (slab) k-space: device side ────────────────────────────────
// The host orchestrator (sog_kspace_op.cc) owns the decomposition, the halo
// exchange and the transpose; these entry points only move one buffer to the
// device, transform it, and bring it back.  Correctness first: no device-side
// MPI (this machine has neither CUDA-aware MPI nor a working NCCL inside the
// op), so every exchange is host-staged by the caller.
//
// Layouts (both keep x in the middle, which is what cuFFT wants):
//   "xy": (nx, ny, ZL)         z slowest, contiguous (nx,ny) planes, one per
//                              local z.  rank-2 plan over (x,y), batch ZL.
//   "z" : (nz, nx, YL)         z FASTEST, so the z transform is a plain
//                              contiguous batch of nx*YL vectors of length nz.
extern "C" int sog_gpu_slab_setup(SogGpuState*s,int nx,int ny,int nz,
    int ZL,int YL){
  const int Nh=nx/2+1;
  // The plans depend on the GRID (nx,ny,nz), not just the slab shapes: a stale
  // reuse after a grid change was a latent bug even before R2C.
  if(s->slab_ZL==ZL && s->slab_YL==YL && s->slab_Nh==Nh && s->plan_xy_rc &&
     s->plan_xy_cr && s->plan_z) return 0;
  if(s->plan_xy_rc) cufftDestroy(s->plan_xy_rc);
  if(s->plan_xy_cr) cufftDestroy(s->plan_xy_cr);
  if(s->plan_z) cufftDestroy(s->plan_z);
  s->plan_xy_rc=0; s->plan_xy_cr=0; s->plan_z=0;
  // cuFFT takes sizes SLOWEST -> FASTEST.  Getting this backwards still looks
  // right on cubic grids and aliases once nx != ny; see
  // sog_work/dist_mesh/README.md.
  int n2[2]={ny,nx}, rin[2]={ny,nx}, cout[2]={ny,Nh};
  if(cufftPlanMany(&s->plan_xy_rc,2,n2,rin,1,(size_t)nx*ny,cout,1,
                   (size_t)ny*Nh,CUFFT_D2Z,ZL)!=CUFFT_SUCCESS) return -1;
  if(cufftPlanMany(&s->plan_xy_cr,2,n2,cout,1,(size_t)ny*Nh,rin,1,
                   (size_t)nx*ny,CUFFT_Z2D,ZL)!=CUFFT_SUCCESS){
    cufftDestroy(s->plan_xy_rc); s->plan_xy_rc=0; return -1;
  }
  int n1[1]={nz}, e1[1]={nz};
  if(cufftPlanMany(&s->plan_z,1,n1,e1,1,nz,e1,1,nz,CUFFT_Z2Z,Nh*YL)
     !=CUFFT_SUCCESS){
    cufftDestroy(s->plan_xy_rc); cufftDestroy(s->plan_xy_cr);
    s->plan_xy_rc=0; s->plan_xy_cr=0; return -1;
  }
  s->slab_ZL=ZL; s->slab_YL=YL; s->slab_Nh=Nh;
  return 0;
}

// green multiply on the local pencil; returns the 7 accumulators in red[0..6]
// ── Device-resident transpose stages ───────────────────────────────────────
// Everything between the MPI messages stays on the GPU: the host only sees the
// packed block matrices.  This removes ~0.9 GB/step of strided host traffic at
// a 150^3 mesh (four pack/unpack traversals plus four extra H<->D copies).
//
//   forward : cur (xy slab) -> FFT(x,y) -> pack -> [MPI] -> unpack -> FFT(z)
//             -> green -> FFT^-1(z) -> pack -> [MPI] -> unpack -> FFT^-1(x,y)

// Device-resident variant: the received blocks are ALREADY in d_rho; the
// packed inverse blocks are left in d_rho and the local accumulators in d_red.
extern "C" int sog_gpu_slab_unpack_green_izpack_dev(SogGpuState*s,int nx,int ny,
    int nz,int ylo,int YL,double lx,double ly,double lz,const double*ge,
    const double*gf,const double*gv,double scaleinv,int np){
  if(s->d_rho==nullptr||s->d_pot==nullptr||s->plan_z==0) return -1;
  const int Nh=nx/2+1;
  const size_t nc=(size_t)nz*Nh*YL;              // complex elements each way
  int tb=256; size_t gg=(nc+tb-1)/tb;
  k_unpack_blocks_to_pen<<<(int)gg,tb>>>(Nh,ny,nz,np,YL,s->d_rho,s->d_pot);
  CK(cudaGetLastError());
  if(cufftExecZ2Z(s->plan_z,s->d_pot,s->d_pot,CUFFT_FORWARD)!=CUFFT_SUCCESS)
    return -2;
  if(s->d_red==nullptr) CK(cudaMalloc(&s->d_red,20*sizeof(double)));
  CK(cudaMemset(s->d_red,0,20*sizeof(double)));
  k_green_pencil<<<(int)gg,tb>>>(Nh,nx,ny,nz,ylo,YL,lx,ly,lz,s->d_pot,
                                s->d_ge,s->d_gf,s->d_gv,s->d_pot,scaleinv,s->d_red);
  CK(cudaGetLastError());
  if(cufftExecZ2Z(s->plan_z,s->d_pot,s->d_pot,CUFFT_INVERSE)!=CUFFT_SUCCESS)
    return -3;
  k_pack_pen_to_blocks<<<(int)gg,tb>>>(Nh,ny,nz,np,YL,s->d_pot,s->d_rho);
  CK(cudaGetLastError());
  return 0;
}
// Host-buffer wrapper (MPI fallback path).
extern "C" int sog_gpu_slab_unpack_green_izpack(SogGpuState*s,int nx,int ny,
    int nz,int ylo,int YL,double lx,double ly,double lz,const double*ge,
    const double*gf,const double*gv,double scaleinv,int np,
    const double*host_rcv,double*host_snd_back,double*red7){
  const size_t nc=(size_t)nz*(size_t)(nx/2+1)*YL;
  CK(cudaMemcpy(s->d_rho,host_rcv,2*nc*sizeof(double),cudaMemcpyHostToDevice));
  const int rc=sog_gpu_slab_unpack_green_izpack_dev(s,nx,ny,nz,ylo,YL,lx,ly,lz,
                                                    ge,gf,gv,scaleinv,np);
  if(rc!=0) return rc;
  double red[20]; CK(cudaMemcpy(red,s->d_red,20*sizeof(double),cudaMemcpyDeviceToHost));
  for(int t=0;t<7;++t) red7[t]=red[t];
  CK(cudaMemcpy(host_snd_back,s->d_rho,2*nc*sizeof(double),cudaMemcpyDeviceToHost));
  return 0;
}

// unpack [MPI] -> FFT^-1(x,y) -> cur2 (xy slab)


// ── NCCL device collectives (dlopen, no build/link dependency) ─────────────
// The transpose all-to-all and the halo exchanges are device-to-device with
// NCCL, so the packed messages never cross the bus.  The op library is also
// compiled into the LAMMPS plugin (torch-free), so NCCL is resolved at run
// time instead of being a link dependency: if it cannot be loaded or the
// communicator cannot be created, every entry returns non-zero and the caller
// keeps using the host-staged MPI path.
namespace {
// ncclUniqueId is a 128-byte struct passed BY VALUE to ncclCommInitRank; using
// a pointer in the function-pointer cast changes the ABI and NCCL answers
// "invalid argument".
struct NcclUniqueId{ char internal[128]; };
struct NcclApi {
  bool loaded=false, ok=false;
  int (*get_id)(NcclUniqueId*)=nullptr;
  int (*init_rank)(void**,int,NcclUniqueId,int)=nullptr;
  int (*destroy)(void*)=nullptr;
  int (*group_start)()=nullptr;
  int (*group_end)()=nullptr;
  int (*send)(const void*,size_t,int,int,void*,cudaStream_t)=nullptr;
  int (*recv)(void*,size_t,int,int,void*,cudaStream_t)=nullptr;
  int (*all_reduce)(const void*,void*,size_t,int,int,void*,cudaStream_t)=nullptr;
  const char* (*err_str)(int)=nullptr;
};
NcclApi &nccl_api(){
  static NcclApi a;
  if(a.loaded) return a;
  a.loaded=true;
  const char *cands[5];
  int nc=0;
  const char *e=getenv("SOG_NCCL_LIB");
  if(e) cands[nc++]=e;
  cands[nc++]="libnccl.so.2";
  cands[nc++]="libnccl.so";
  // Last resort for this deployment: the NCCL that ships inside the torch
  // wheel's nvidia/ tree, which is not on LD_LIBRARY_PATH by default.
  cands[nc++]="/data/home/jiangzhen/.conda/envs/lr/lib/python3.11/"
              "site-packages/nvidia/nccl/lib/libnccl.so.2";
  for(int i=0;i<nc && !a.ok;++i){
    void *lib=dlopen(cands[i],RTLD_NOW|RTLD_LOCAL);
    if(!lib) continue;
    a.get_id=(int(*)(NcclUniqueId*))dlsym(lib,"ncclGetUniqueId");
    a.init_rank=(int(*)(void**,int,NcclUniqueId,int))dlsym(lib,"ncclCommInitRank");
    a.destroy=(int(*)(void*))dlsym(lib,"ncclCommDestroy");
    a.group_start=(int(*)())dlsym(lib,"ncclGroupStart");
    a.group_end=(int(*)())dlsym(lib,"ncclGroupEnd");
    a.send=(int(*)(const void*,size_t,int,int,void*,cudaStream_t))dlsym(lib,"ncclSend");
    a.recv=(int(*)(void*,size_t,int,int,void*,cudaStream_t))dlsym(lib,"ncclRecv");
    a.all_reduce=(int(*)(const void*,void*,size_t,int,int,void*,cudaStream_t))dlsym(lib,"ncclAllReduce");
    a.err_str=(const char*(*)(int))dlsym(lib,"ncclGetErrorString");
    a.ok=a.get_id&&a.init_rank&&a.destroy&&a.group_start&&a.group_end&&a.send&&
         a.recv&&a.all_reduce;
  }
  return a;
}
constexpr int kNcclDouble=8;    // ncclDataType_t: ncclFloat64/ncclDouble
constexpr int kNcclSum=0;       // ncclRedOp_t

// One communicator per process: neutralize/qmean (which have no handle) and
// solve_auto share it, so the first collective can be the neutralization.
void *g_comm=nullptr;
int g_rank=-1, g_np=-1, g_state=0;   // 0 = not tried, 1 = usable, -1 = unavailable
}  // namespace

extern "C" int sog_gpu_nccl_get_id(void *id){
  NcclApi &a=nccl_api();
  return (a.ok && a.get_id(static_cast<NcclUniqueId*>(id))==0) ? 0 : -1;
}

extern "C" int sog_gpu_nccl_init(int np,int me,const void *id){
  if(g_state==1 && g_np==np && g_rank==me) return 0;
  if(g_state==-1) return -1;
  if(getenv("SOG_NCCL")!=nullptr && atoi(getenv("SOG_NCCL"))==0){
    g_state=-1; return -1;
  }
  NcclApi &a=nccl_api();
  if(!a.ok){ g_state=-1; return -1; }
  void *comm=nullptr;
  const NcclUniqueId uid=*static_cast<const NcclUniqueId*>(id);
  const int rc=a.init_rank(&comm,np,uid,me);
  if(rc!=0){
    if(getenv("SOG_DIST_DEBUG"))
      fprintf(stderr,"[sog-nccl] init failed (%s); falling back to MPI\n",
              a.err_str?a.err_str(rc):"error");
    g_state=-1; return -1;
  }
  g_comm=comm; g_np=np; g_rank=me; g_state=1;
  return 0;
}
extern "C" int sog_gpu_nccl_ok(){ return g_state==1 ? 1 : 0; }
extern "C" int sog_gpu_nccl_available(){ return nccl_api().ok ? 1 : 0; }

extern "C" int sog_gpu_nccl_alltoall(SogGpuState*s,const double*send,
    const int*scnt,const int*sdisp,double*recv,const int*rcnt,
    const int*rdisp,int np){
  NcclApi &a=nccl_api();
  if(!a.ok||g_state!=1) return -1;
  if(a.group_start()!=0) return -2;
  for(int d=0;d<np;++d)
    a.send(send+sdisp[d],(size_t)scnt[d],kNcclDouble,d,g_comm,(cudaStream_t)0);
  for(int r=0;r<np;++r)
    a.recv(recv+rdisp[r],(size_t)rcnt[r],kNcclDouble,r,g_comm,(cudaStream_t)0);
  return a.group_end()==0 ? 0 : -3;
}

// Two-direction halo without tags.  NCCL pairs messages per peer by posting
// order, so the order below is chosen to be correct when prev == next (np=2):
//   send->prev, recv<-next, send->next, recv<-prev
// At np=2 the 1st send pairs with the peer's 1st recv and the 2nd with the 2nd;
// at np>2 each direction has a distinct peer, so the order is irrelevant.
extern "C" int sog_gpu_nccl_halo2(SogGpuState*s,int prev,int next,
    const double*send_prev,int cnt_prev,const double*send_next,int cnt_next,
    double*recv_prev,double*recv_next){
  NcclApi &a=nccl_api();
  if(!a.ok||g_state!=1) return -1;
  if(a.group_start()!=0) return -2;
  a.send(send_prev,(size_t)cnt_prev,kNcclDouble,prev,g_comm,(cudaStream_t)0);
  a.recv(recv_next,(size_t)cnt_next,kNcclDouble,next,g_comm,(cudaStream_t)0);
  a.send(send_next,(size_t)cnt_next,kNcclDouble,next,g_comm,(cudaStream_t)0);
  a.recv(recv_prev,(size_t)cnt_prev,kNcclDouble,prev,g_comm,(cudaStream_t)0);
  return a.group_end()==0 ? 0 : -3;
}

__global__ void k_add_into(double*dst,const double*src,size_t n);  // defined below

// Forward all-to-all: packed forward blocks in d_pot -> received blocks in d_rho.
extern "C" int sog_gpu_nccl_alltoall_fwd(SogGpuState*s,const int*scnt,
    const int*sdisp,const int*rcnt,const int*rdisp,int np){
  return sog_gpu_nccl_alltoall(s,(const double*)s->d_pot,scnt,sdisp,
                               (double*)s->d_rho,rcnt,rdisp,np);
}
// Inverse: packed inverse blocks in d_rho -> received blocks in d_pot.
extern "C" int sog_gpu_nccl_alltoall_bwd(SogGpuState*s,const int*scnt,
    const int*sdisp,const int*rcnt,const int*rdisp,int np){
  return sog_gpu_nccl_alltoall(s,(const double*)s->d_rho,scnt,sdisp,
                               (double*)s->d_pot,rcnt,rdisp,np);
}

// Reverse halo (density overlap-add) on the device brick.  Requires
// gh_lo == gh_hi (true by construction: both are the interpolation order).
extern "C" int sog_gpu_nccl_halo_rev(SogGpuState*s,int prev,int next,
    size_t plane,int zoff,int ZL,int gh_lo,int gh_hi){
  if(gh_lo!=gh_hi) return -4;
  const size_t n=(size_t)plane*(size_t)gh_lo;
  if(2*n>s->cap_halo){
    if(s->d_halo) cudaFree(s->d_halo);
    CK(cudaMalloc(&s->d_halo,2*n*sizeof(double)));
    s->cap_halo=2*n;
  }
  double *recv_prev=s->d_halo, *recv_next=s->d_halo+n;
  const double *send_lo=s->d_brick;                          // lower ghost
  const double *send_hi=s->d_brick+(size_t)(zoff+ZL)*plane;  // upper ghost
  if(sog_gpu_nccl_halo2(s,prev,next,send_lo,(int)n,send_hi,(int)n,recv_prev,
                        recv_next)!=0) return -1;
  int tb=256; size_t gg=(n+tb-1)/tb;
  k_add_into<<<(int)gg,tb>>>(s->d_brick+(size_t)zoff*plane,recv_prev,n);
  k_add_into<<<(int)gg,tb>>>(s->d_brick+(size_t)(zoff+ZL-gh_lo)*plane,recv_next,n);
  CK(cudaGetLastError());
  return 0;
}
// Forward halo (potential ghosts) on the device brick.
extern "C" int sog_gpu_nccl_halo_fwd(SogGpuState*s,int prev,int next,
    size_t plane,int zoff,int ZL,int gh_lo,int gh_hi){
  if(gh_lo!=gh_hi) return -4;
  const size_t n=(size_t)plane*(size_t)gh_lo;
  if(2*n>s->cap_halo){
    if(s->d_halo) cudaFree(s->d_halo);
    CK(cudaMalloc(&s->d_halo,2*n*sizeof(double)));
    s->cap_halo=2*n;
  }
  double *recv_prev=s->d_halo, *recv_next=s->d_halo+n;
  const double *send_first=s->d_brick+(size_t)zoff*plane;           // first owned
  const double *send_last=s->d_brick+(size_t)(zoff+ZL-gh_lo)*plane; // last owned
  if(sog_gpu_nccl_halo2(s,prev,next,send_first,(int)n,send_last,(int)n,
                        recv_prev,recv_next)!=0) return -1;
  CK(cudaMemcpy(s->d_brick,recv_prev,n*sizeof(double),cudaMemcpyDeviceToDevice));
  CK(cudaMemcpy(s->d_brick+(size_t)(zoff+ZL)*plane,recv_next,n*sizeof(double),
                cudaMemcpyDeviceToDevice));
  return 0;
}

// Generic in-place device all-reduce (double).  Used by neutralize/qmean.
extern "C" int sog_gpu_nccl_allreduce_dev(void *buf,size_t n){
  NcclApi &a=nccl_api();
  if(!a.ok||g_state!=1) return -1;
  return a.all_reduce(buf,buf,n,kNcclDouble,kNcclSum,g_comm,(cudaStream_t)0)==0
             ? 0 : -2;
}

extern "C" int sog_gpu_nccl_allreduce_red(SogGpuState*s){
  NcclApi &a=nccl_api();
  if(!a.ok||g_state!=1||s->d_red==nullptr) return -1;
  return a.all_reduce(s->d_red,s->d_red,(size_t)7,kNcclDouble,kNcclSum,
                      g_comm,(cudaStream_t)0)==0 ? 0 : -2;
}

extern "C" void sog_gpu_red_download(SogGpuState*s,double*host){
  if(s->d_red==nullptr) return;
  CK(cudaMemcpy(host,s->d_red,7*sizeof(double),cudaMemcpyDeviceToHost));
}

// cuFFT's D2Z/Z2D wants a 16-byte-aligned pointer.  The brick's slab offset
// zoff*plane*8 is 16-aligned only when the double offset is even; otherwise
// stage the contiguous plane run through this scratch.
static double *ensure_real(SogGpuState*s,size_t n){
  if(n>s->cap_real){
    if(s->d_real) cudaFree(s->d_real);
    CK(cudaMalloc(&s->d_real,n*sizeof(double)));
    s->cap_real=n;
  }
  return s->d_real;
}

// ── Device-resident brick staging ──────────────────────────────────────────
// The distributed solve keeps the owned+ghost brick on the GPU: the slab
// conversions and the halo overlap-adds run as kernels, and only the few ghost
// planes cross the bus.  This removes the full-brick D2H/H2D per step and the
// host rho_ext/u_ext/cur/cur2 buffers.
__global__ void k_brick_to_slab(int nx,int ny,int ZL,int zoff,
    const double*brick,cufftDoubleComplex*slab){
  size_t t=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
  const size_t n=(size_t)nx*ny*ZL;
  if(t>=n) return;
  const int ix=(int)(t%nx);
  const int y=(int)((t/nx)%ny);
  const int z=(int)(t/((size_t)nx*ny));
  slab[t].x=brick[(size_t)(zoff+z)*nx*ny+(size_t)y*nx+ix];
  slab[t].y=0.0;
}
__global__ void k_slab_to_brick(int nx,int ny,int ZL,int zoff,
    const cufftDoubleComplex*slab,double*brick){
  size_t t=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
  const size_t n=(size_t)nx*ny*ZL;
  if(t>=n) return;
  const int ix=(int)(t%nx);
  const int y=(int)((t/nx)%ny);
  const int z=(int)(t/((size_t)nx*ny));
  brick[(size_t)(zoff+z)*nx*ny+(size_t)y*nx+ix]=slab[t].x;
}
__global__ void k_add_into(double*dst,const double*src,size_t n){
  size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n) dst[i]+=src[i];
}

// plane ranges are in doubles: off = plane_index * nx*ny
extern "C" void sog_gpu_brick_get(SogGpuState*s,double*host,size_t off,size_t n){
  CK(cudaMemcpy(host,s->d_brick+off,n*sizeof(double),cudaMemcpyDeviceToHost));
}
extern "C" void sog_gpu_brick_set(SogGpuState*s,const double*host,size_t off,
    size_t n){
  CK(cudaMemcpy(s->d_brick+off,host,n*sizeof(double),cudaMemcpyHostToDevice));
}
extern "C" void sog_gpu_brick_add(SogGpuState*s,const double*host,size_t off,
    size_t n){
  if(n>s->cap_halo){
    if(s->d_halo) cudaFree(s->d_halo);
    CK(cudaMalloc(&s->d_halo,n*sizeof(double)));
    s->cap_halo=n;
  }
  CK(cudaMemcpy(s->d_halo,host,n*sizeof(double),cudaMemcpyHostToDevice));
  int tb=256; size_t gg=(n+tb-1)/tb;
  k_add_into<<<(int)gg,tb>>>(s->d_brick+off,s->d_halo,n);
  CK(cudaGetLastError());
}
extern "C" void sog_gpu_brick_zero(SogGpuState*s,size_t off,size_t n){
  CK(cudaMemset(s->d_brick+off,0,n*sizeof(double)));
}

// brick (owned planes) -> xy slab complex, then FFT(x,y), pack.  The packed
// blocks are LEFT IN d_pot so a device all-to-all can send them directly.
extern "C" int sog_gpu_slab_xyfft_pack_brick_dev(SogGpuState*s,int nx,int ny,
    int nz,int ZL,int np,int zoff){
  if(s->d_rho==nullptr||s->d_pot==nullptr||s->plan_xy_rc==0) return -1;
  const int Nh=nx/2+1;
  const size_t nc=(size_t)Nh*ny*ZL;
  // R2C out of the owned brick planes (x fastest, ny rows, ZL batches).  zoff
  // is even in the 1-D slab path, but the full-brick path can land the slab
  // at an odd double offset; then stage it through the aligned scratch.
  const size_t plane=(size_t)nx*ny;
  const size_t doff=(size_t)zoff*plane;
  const cufftDoubleReal *rin;
  if((doff%2)==0){
    rin=reinterpret_cast<const cufftDoubleReal*>(s->d_brick+doff);
  }else{
    double *tmp=ensure_real(s,(size_t)ZL*plane);
    CK(cudaMemcpy(tmp,s->d_brick+doff,(size_t)ZL*plane*sizeof(double),
                  cudaMemcpyDeviceToDevice));
    rin=reinterpret_cast<const cufftDoubleReal*>(tmp);
  }
  if(cufftExecD2Z(s->plan_xy_rc,const_cast<cufftDoubleReal*>(rin),
                  s->d_rho)!=CUFFT_SUCCESS) return -2;
  int tb=256; size_t gg=(nc+tb-1)/tb;
  k_pack_xy_to_blocks<<<(int)gg,tb>>>(Nh,ny,nz,ZL,np,s->d_rho,s->d_pot);
  CK(cudaGetLastError());
  return 0;
}
extern "C" int sog_gpu_slab_xyfft_pack_brick(SogGpuState*s,int nx,int ny,
    int nz,int ZL,int np,int zoff,double*host_snd){
  const int rc=sog_gpu_slab_xyfft_pack_brick_dev(s,nx,ny,nz,ZL,np,zoff);
  if(rc!=0) return rc;
  const size_t nc=(size_t)(nx/2+1)*ny*ZL;
  CK(cudaMemcpy(host_snd,s->d_pot,2*nc*sizeof(double),cudaMemcpyDeviceToHost));
  return 0;
}

// blocks -> xy slab, inverse FFT(x,y), then slab -> brick owned planes; the
// ghost planes are zeroed so the forward halo starts from a clean brick.
// Device-resident variant: the received blocks are ALREADY in d_pot.
extern "C" int sog_gpu_slab_unpack_ixyfft_to_brick_dev(SogGpuState*s,int nx,
    int ny,int nz,int ZL,int np,int zoff,int ZB){
  if(s->d_rho==nullptr||s->d_pot==nullptr||s->plan_xy_cr==0) return -1;
  const int Nh=nx/2+1;
  const size_t nc=(size_t)Nh*ny*ZL;
  int tb=256; size_t gg=(nc+tb-1)/tb;
  k_unpack_blocks_to_xy<<<(int)gg,tb>>>(Nh,ny,ZL,np,s->d_pot,s->d_rho);
  CK(cudaGetLastError());
  const size_t plane=(size_t)nx*ny;
  const size_t doff=(size_t)zoff*plane;
  cufftDoubleReal *rout;
  if((doff%2)==0){
    rout=reinterpret_cast<cufftDoubleReal*>(s->d_brick+doff);
  }else{
    rout=reinterpret_cast<cufftDoubleReal*>(ensure_real(s,(size_t)ZL*plane));
  }
  if(cufftExecZ2D(s->plan_xy_cr,s->d_rho,rout)!=CUFFT_SUCCESS) return -2;
  if((doff%2)!=0)
    CK(cudaMemcpy(s->d_brick+doff,s->d_real,(size_t)ZL*plane*sizeof(double),
                  cudaMemcpyDeviceToDevice));
  if(zoff>0) sog_gpu_brick_zero(s,0,(size_t)zoff*plane);
  const size_t upper=(size_t)(ZB-zoff-ZL);
  if(upper>0) sog_gpu_brick_zero(s,(size_t)(zoff+ZL)*plane,upper*plane);
  return 0;
}
// Host-buffer wrapper (MPI fallback path).
extern "C" int sog_gpu_slab_unpack_ixyfft_to_brick(SogGpuState*s,int nx,int ny,
    int nz,int ZL,int np,int zoff,int ZB,const double*host_back){
  const size_t nc=(size_t)(nx/2+1)*ny*ZL;
  CK(cudaMemcpy(s->d_pot,host_back,2*nc*sizeof(double),cudaMemcpyHostToDevice));
  return sog_gpu_slab_unpack_ixyfft_to_brick_dev(s,nx,ny,nz,ZL,np,zoff,ZB);
}

// Free the grid-sized buffers (full or slab) and the 3D plan.
static void free_grid_buffers(SogGpuState*s){
  if(s->plan){cufftDestroy(s->plan);s->plan=0;}
  if(s->d_rho){cudaFree(s->d_rho);s->d_rho=nullptr;}
  if(s->d_gx){cudaFree(s->d_gx);s->d_gx=nullptr;}
  if(s->d_gy){cudaFree(s->d_gy);s->d_gy=nullptr;}
  if(s->d_gz){cudaFree(s->d_gz);s->d_gz=nullptr;}
  if(s->d_pot){cudaFree(s->d_pot);s->d_pot=nullptr;}
  if(s->d_ge){cudaFree(s->d_ge);s->d_ge=nullptr;}
  if(s->d_gf){cudaFree(s->d_gf);s->d_gf=nullptr;}
  if(s->d_gv){cudaFree(s->d_gv);s->d_gv=nullptr;}
  s->slab_cap=0;s->slab_gn=0;s->slab_mode=false;s->slab_Y=0;
}

// Slab-sized setup for the distributed path.  d_rho / d_pot hold the largest
// slab this rank ever transforms, d_ge/d_gf/d_gv hold only this rank's y-slice
// of the green tables (indexed ix + nx*(iyl + YL*iz)), and the gradient meshes
// d_gx/d_gy/d_gz are NOT allocated -- the distributed solve never uses them.
// Nothing here creates the 3D plan; the slab plans are built by
// sog_gpu_slab_setup, which needs the shapes.
extern "C" void sog_gpu_setup_slab(SogGpuState*s,int nx,int ny,int nz,int YL,
    int ZL,const double*ge,const double*gf,const double*gvv,size_t gn){
  const int Nh=nx/2+1;
  const size_t cap=(size_t)((size_t)Nh*ny*ZL > (size_t)nz*Nh*YL
                                ? (size_t)Nh*ny*ZL : (size_t)nz*Nh*YL);
  const bool realloc = !s->slab_mode || s->nx!=nx || s->ny!=ny || s->nz!=nz ||
                       s->slab_cap!=cap || s->slab_gn!=gn || s->slab_Y!=YL;
  if(realloc){
    free_grid_buffers(s);
    CK(cudaMalloc(&s->d_rho,cap*sizeof(cufftDoubleComplex)));
    CK(cudaMalloc(&s->d_pot,cap*sizeof(cufftDoubleComplex)));
    CK(cudaMalloc(&s->d_ge,gn*sizeof(double)));
    CK(cudaMalloc(&s->d_gf,gn*sizeof(double)));
    CK(cudaMalloc(&s->d_gv,gn*sizeof(double)));
    s->nx=nx;s->ny=ny;s->nz=nz;s->ngrid=(size_t)nx*ny*nz;
    s->slab_cap=cap;s->slab_gn=gn;s->slab_Y=YL;s->slab_mode=true;
  }
  CK(cudaMemcpy(s->d_ge,ge,gn*sizeof(double),cudaMemcpyHostToDevice));
  CK(cudaMemcpy(s->d_gf,gf,gn*sizeof(double),cudaMemcpyHostToDevice));
  CK(cudaMemcpy(s->d_gv,gvv,gn*sizeof(double),cudaMemcpyHostToDevice));
  if(getenv("SOG_DIST_DEBUG")){
    size_t fb=0,tb=0; cudaMemGetInfo(&fb,&tb);
    fprintf(stderr,"[sog-mem] slab grid buffers: scratch %.1f MB + green %.1f MB "
            "(full grid would be %.1f MB); device free %.1f MB\n",
            2.0*cap*16/1048576.0, 3.0*gn*8/1048576.0,
            (5.0*nx*ny*nz*16 + 3.0*nx*ny*nz*8)/1048576.0, fb/1048576.0);
    fflush(stderr);
  }
}

extern "C" void sog_gpu_setup(SogGpuState*s,int nx,int ny,int nz,
    const double*ge,const double*gf,const double*gvv){
  size_t ng=(size_t)nx*ny*nz;
  if(s->slab_mode || nx!=s->nx||ny!=s->ny||nz!=s->nz){
    free_grid_buffers(s);
    CK(cudaMalloc(&s->d_rho,ng*sizeof(cufftDoubleComplex)));
    CK(cudaMalloc(&s->d_gx,ng*sizeof(cufftDoubleComplex)));
    CK(cudaMalloc(&s->d_gy,ng*sizeof(cufftDoubleComplex)));
    CK(cudaMalloc(&s->d_gz,ng*sizeof(cufftDoubleComplex)));
    CK(cudaMalloc(&s->d_pot,ng*sizeof(cufftDoubleComplex)));
    CK(cudaMalloc(&s->d_ge,ng*sizeof(double)));CK(cudaMalloc(&s->d_gf,ng*sizeof(double)));
    CK(cudaMalloc(&s->d_gv,ng*sizeof(double)));
    // cuFFT 3D expects (nz,ny,nx) with x fastest -> pass dims as (nz,ny,nx)
    cufftPlan3d(&s->plan,nz,ny,nx,CUFFT_Z2Z);
    s->nx=nx;s->ny=ny;s->nz=nz;s->ngrid=ng;
  }
  CK(cudaMemcpy(s->d_ge,ge,ng*sizeof(double),cudaMemcpyHostToDevice));
  CK(cudaMemcpy(s->d_gf,gf,ng*sizeof(double),cudaMemcpyHostToDevice));
  CK(cudaMemcpy(s->d_gv,gvv,ng*sizeof(double),cudaMemcpyHostToDevice));
}

// v_i self-energy correction, on the device:
//   vpot[i] -= qscale * q[i] * (rsi + sc)
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
  k_kspace<<<gg,tb>>>(s->nx,s->ny,s->nz,lx,ly,lz,s->d_rho,s->d_ge,s->d_gf,s->d_gv,
                      s->d_gx,s->d_gy,s->d_gz,s->d_pot,want_pot,scaleinv,s->d_red);
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
  // ── Self-energy: closed-form in real space, NO mesh sum ──────────────────
  // The k-space energy is the Fourier form of a PAIR potential.  Summing over
  // the complete reciprocal lattice reproduces the pair sum exactly, and its
  // (i == j, R == 0) term is ½·qsum_sq·g(0), with g(0) the multi-Gaussian
  // real-space kernel at zero separation:
  //     g(0) = sum_m A_m,   A_m = amp_m (2 pi bw_m)^(-3/2).
  // So "remove self interaction" is exactly ½·qsum_sq·g(0) -- a real-space
  // quantity, independent of any k-truncation, and exactly proportional to q^2.
  // Algebraically this equals the RBSOG closed form that sog.cpp already builds
  // for internally-generated kernels (`log(b)/(sqrt(2 pi) sigma)(w0 + sum b^-m)`),
  // verified to all 15 digits.  Callers therefore pass it in `self_coeff`; it
  // must NOT be zeroed for externally-supplied amp/bandwidth (the DeepMD path).
  //
  // The old mesh sum was NOT extensive (9.03x / 76.5x instead of 8x / 64x for
  // 2x / 4x replication): several bands are extremely narrow in k (bw up to
  // 2.2e5) and need a k-spacing far below 2 pi / L to resolve, so a finite
  // mesh under-counts the i == j term by a box-dependent amount.
  if(want_pot){
    // v_i = dE/dq_i, applied on the device so the potential never leaves the
    // GPU between the gather and the consumer:
    //   vpot[i] -= qscale*q_i*(rsi + sc),  rsi = g(0),  sc = 0
    // (rsi is exactly 2*self_coeff because self_coeff = ½ g(0)).
    const double rsi = remove_self ? 2.0*self_coeff : 0.0;
    if(nlocal>0) k_vpot_self<<<ga,tb>>>(nlocal,d_vpot,d_q,qscale,rsi,0.0);
  }
  // Energy = qscale * [ 0.5*V*Σ(s2·ge·|ρ|²)  −  remove_self·self_coeff·qsqsum ]
  double e_kneq0 = 0.5*volume*red[0];
  double e_self  = remove_self ? self_coeff*qsqsum : 0.0;
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
  // No self-energy virial.  The exact self term is qsum_sq * sum_m A_m / 2,
  // independent of V and of the cell, so its strain derivative is identically
  // zero.  (Analytically sum_mesh K_v k_a k_b = diag_sum * delta_ab; the old
  // mesh version was anisotropic -- 0.827/1.085/1.339 x the diagonal at a
  // 264-atom cell -- and injected a spurious shear that this removal deletes.)
  (void)remove_self;
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

// Spread the local atoms into the device brick.  No download: the distributed
// solve keeps working on d_brick and only fetches the ghost planes.
extern "C" void sog_gpu_spread_brick_dev(SogGpuState*s,int nlocal,
    const double*x,const double*boxlo,double lx,double ly,double lz,
    const double*q,double rho_scale,int spline,int nx,int ny,int nz,
    int nxlo,int nxhi,int nylo,int nyhi,int nzlo,int nzhi){
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
}

extern "C" void sog_gpu_spread_brick(SogGpuState*s,int nlocal,
    const double*x,const double*boxlo,double lx,double ly,double lz,
    const double*q,double rho_scale,int spline,int nx,int ny,int nz,
    int nxlo,int nxhi,int nylo,int nyhi,int nzlo,int nzhi,double*rho_brick){
  size_t nb=(size_t)(nxhi-nxlo+1)*(nyhi-nylo+1)*(nzhi-nzlo+1);
  sog_gpu_spread_brick_dev(s,nlocal,x,boxlo,lx,ly,lz,q,rho_scale,spline,nx,ny,
                           nz,nxlo,nxhi,nylo,nyhi,nzlo,nzhi);
  CK(cudaMemcpy(rho_brick,s->d_brick,nb*sizeof(double),cudaMemcpyDeviceToHost));
}

// Gather from the brick ALREADY on the device (the distributed solve fills it
// with the potential and the forward halo); only x/q go up and F/v come down.
extern "C" void sog_gpu_gather_brick_dev(SogGpuState*s,int nlocal,
    const double*x,const double*boxlo,double lx,double ly,double lz,
    const double*q,double qscale,int spline,int nx,int ny,int nz,
    int nxlo,int nxhi,int nylo,int nyhi,int nzlo,int nzhi,
    double*force,double*vpot){
  if(nlocal<=0) return;
  size_t nb=(size_t)(nxhi-nxlo+1)*(nyhi-nylo+1)*(nzhi-nzlo+1);
  ensure_local_brick(s,nb);ensure_local_atoms(s,nlocal);
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

extern "C" void sog_gpu_gather_brick(SogGpuState*s,int nlocal,
    const double*x,const double*boxlo,double lx,double ly,double lz,
    const double*q,double qscale,int spline,int nx,int ny,int nz,
    int nxlo,int nxhi,int nylo,int nyhi,int nzlo,int nzhi,const double*u_brick,
    double*force,double*vpot){
  size_t nb=(size_t)(nxhi-nxlo+1)*(nyhi-nylo+1)*(nzhi-nzlo+1);
  ensure_local_brick(s,nb);
  CK(cudaMemcpy(s->d_brick,u_brick,nb*sizeof(double),cudaMemcpyHostToDevice));
  sog_gpu_gather_brick_dev(s,nlocal,x,boxlo,lx,ly,lz,q,qscale,spline,nx,ny,nz,
                           nxlo,nxhi,nylo,nyhi,nzlo,nzhi,force,vpot);
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
                      s->d_gx,s->d_gy,s->d_gz,s->d_pot,want_pot,
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
  // Exact closed-form self term; see sog_gpu_compute_dev for the derivation.
  // rsi = g(0) = 2*self_coeff, and the self term contributes no virial.
  if(want_pot){
    const double rsi=remove_self?2.0*self_coeff:0.0;
    if(nlocal>0) k_vpot_self<<<ga,tb>>>(nlocal,d_vpot,d_q,qscale,rsi,0.0);
  }
  double e_kneq0=0.5*volume*red[0];
  double e_self=remove_self?self_coeff*qsqsum:0.0;
  for(int j=0;j<6;++j) virial6[j]=0.5*volume*qscale*red[1+j];
  return qscale*(e_kneq0-e_self);
}
