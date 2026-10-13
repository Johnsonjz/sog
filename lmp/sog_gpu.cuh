// sog_gpu.cuh — plugin-internal GPU SOG kspace (raw CUDA + cuFFT).
// Called from SOGKSpace::compute_single when `use_gpu` is set. Keeping the GPU path inside
// SOGKSpace leaves the fix sog/response interface (dynamic_cast<SOGKSpace*>, get_potential())
// untouched — no in-tree LAMMPS build, no lib/gpu. See papers/sog/sog-progress.md §8.
#ifndef SOG_GPU_CUH
#define SOG_GPU_CUH

struct SogGpuState;  // opaque device state (buffers + cuFFT plans + green tables)

// Spline selector for the GPU spread/gather.
enum SogSpline { SOG_CUBES2_4 = 4, SOG_CUBES2_6 = 6, SOG_QUADS_4 = 104, SOG_QUADS_6 = 106 };

extern "C" {
SogGpuState *sog_gpu_create();
void sog_gpu_destroy(SogGpuState *s);

// (Re)allocate device mesh buffers, upload the host-computed green tables, and (re)plan cuFFT
// for grid nx*ny*nz. Called from ensure_fft_plan whenever the grid changes.
void sog_gpu_setup(SogGpuState *s, int nx, int ny, int nz,
                   const double *green_energy, const double *green_force,
                   const double *green_self, const double *green_virial,
                   const double *green_self_virial);

// One SOG kspace step, entirely on device:
//   spread(q@x) -> cuFFT fwd -> green-multiply (grad meshes + potential + energy/virial reductions)
//   -> cuFFT inv -> gather force (+ potential v_i if want_potential).
// x is nlocal*3 wrapped coords (box origin already handled by caller via boxlo/lx..lz);
// qscale = force->qqrd2e*scale. Fills force[nlocal*3], virial6[6]; returns energy.
// If want_potential, fills vpot[nlocal].
double sog_gpu_compute(SogGpuState *s, int nlocal,
                       const double *x, const double *boxlo,
                       double lx, double ly, double lz,
                       const double *q, double qscale,
                       int spline, int want_potential,
                       double *force, double *virial6, double *vpot,
                       double self_coeff, double qsqsum, int remove_self);
}
#endif  // SOG_GPU_CUH
