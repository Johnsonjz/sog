/* ----------------------------------------------------------------------
   SOG adapter: null charge response for fixed-charge MLIPs.

   For a framework with no differentiable charge graph (∂q/∂r = ∂q/∂h = 0),
   the charge-response correction is identically zero.  This adapter makes
   that decision explicit — fix sog/response registers, runs, and adds a
   documented zero — instead of the correction being silently absent.

   Selected automatically by the build when SOG_WITH_DEEPMD is off; a
   framework with a real graph implements SogChargeResponse itself.
------------------------------------------------------------------------- */

#ifndef LMP_ADAPTERS_NATIVE_CHARGE_RESPONSE_H
#define LMP_ADAPTERS_NATIVE_CHARGE_RESPONSE_H

#include "error.h"
#include "sog_charge_response.h"

namespace LAMMPS_NS {

class NativeChargeResponse : public SogChargeResponse {
 public:
  explicit NativeChargeResponse(LAMMPS *lmp) {
    // Once per process: the fix was requested but this build has no
    // differentiable charge graph to respond with.
    lmp->error->warning(FLERR,
                        "fix sog/response: built without SOG_WITH_DEEPMD and "
                        "no SogChargeResponse was registered — charge response "
                        "is a no-op (fixed-charge adapter).");
  }

  double energy_unit_factor() const override { return 1.0; }
  double force_unit_factor() const override { return 1.0; }
  int nchannels() const override { return 1; }
  bool fused_short_range() const override { return false; }

  void response(std::vector<double> &f_ext, std::vector<double> &virial9,
                const double *v_per_atom, int nlocal, int nall,
                int nchannels) override {
    (void)v_per_atom; (void)nlocal; (void)nchannels;
    f_ext.assign(static_cast<size_t>(nall) * 3, 0.0);
    virial9.assign(9, 0.0);
  }
};

}  // namespace LAMMPS_NS

#endif  // LMP_ADAPTERS_NATIVE_CHARGE_RESPONSE_H
