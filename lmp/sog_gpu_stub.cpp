/* ----------------------------------------------------------------------
   sog_gpu_stub.cpp — CPU-only stand-ins for the sog_gpu.cu entry points.

   sog.cpp references sog_gpu_create/setup/compute/destroy unconditionally,
   so a CPU-only shared library would otherwise carry undefined symbols that
   only lazy binding hides (and that break on strict linkers).  Compile this
   stub instead of sog_gpu.cu when CUDA is off.

   If `use_gpu yes` is ever reached, the stub fails loudly instead of
   silently computing wrong physics.
------------------------------------------------------------------------- */

#include "sog_gpu.cuh"

#include <cstdio>
#include <cstdlib>

extern "C" {

SogGpuState *sog_gpu_create() {
  fprintf(stderr,
          "SOG ERROR: `use_gpu yes` requested but this libsog_lmp.so was "
          "built without CUDA support (sog_gpu_stub.cpp). Rebuild with "
          "CUDA enabled or remove `use_gpu yes` from the kspace_style sog "
          "command.\n");
  abort();
}

void sog_gpu_destroy(SogGpuState *) {}

void sog_gpu_setup(SogGpuState *, int, int, int, const double *, const double *,
                   const double *, const double *, const double *) {}

double sog_gpu_compute(SogGpuState *, int, const double *, const double *,
                       double, double, double, const double *, double, int, int,
                       double *, double *, double *, double, double, int) {
  fprintf(stderr,
          "SOG ERROR: `use_gpu yes` requested but this libsog_lmp.so was "
          "built without CUDA support (sog_gpu_stub.cpp).\n");
  abort();
  return 0.0;
}

}  // extern "C"
