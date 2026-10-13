/* ----------------------------------------------------------------------
   SOG adapter: DeepMD pair style as a SogChargeResponse.

   Compiled only with -DSOG_WITH_DEEPMD.  Holds every deepmd-specific piece
   of the charge-response fix: locating the pair, asking it to retain its
   forward graph, the cached VJP (pair's retained graph), and the standalone
   forward+backward fallback (its own DeepPot).  FixSOGResponse sees only
   SogChargeResponse.

   The C++-API extensions used here (set_retain_charge_graph,
   compute_charge_response_cached, set_charge_only_forward,
   compute_combined_response) exist only in this fork's api_cc
   (DeepPot.h); hence the SOG_WITH_DEEPMD gate — a stock deepmd-kit
   C-API build cannot satisfy them.
------------------------------------------------------------------------- */

#ifndef LMP_ADAPTERS_DEEPMD_CHARGE_RESPONSE_H
#define LMP_ADAPTERS_DEEPMD_CHARGE_RESPONSE_H

#ifdef SOG_WITH_DEEPMD

#include <string>
#include <vector>

#include "pair_deepmd.h"
#include "sog_charge_response.h"

#ifdef DP_USE_CXX_API
#ifdef LMPPLUGIN
#include "DeepPot.h"
#else
#include "deepmd/DeepPot.h"
#endif
namespace deepmd_compat = deepmd;
#else
#ifdef LMPPLUGIN
#include "deepmd.hpp"
#else
#include "deepmd/deepmd.hpp"
#endif
namespace deepmd_compat = deepmd::hpp;
#endif

namespace LAMMPS_NS {

class DeepMDChargeResponse : public SogChargeResponse {
 public:
  // `fix` must be the owning FixSOGResponse: the fallback's InputNlist is
  // assembled by FixSOGResponse::build_input_nlist, the only context that is a
  // friend of CommBrickDeepMD (see fix_sog_response.h).
  DeepMDChargeResponse(LAMMPS *lmp, FixSOGResponse *fix, const std::string &model,
                       bool fused, int pair_index);
  ~DeepMDChargeResponse() override = default;

  double energy_unit_factor() const override { return ener_unit_cvt_factor_; }
  double force_unit_factor() const override { return force_unit_cvt_factor_; }
  int nchannels() const override { return nchannels_; }
  bool fused_short_range() const override { return fused_mode_; }

  void response(std::vector<double> &f_ext, std::vector<double> &virial9,
                const double *v_per_atom, int nlocal, int nall,
                int nchannels) override;

 private:
  LAMMPS *lmp_;
  FixSOGResponse *fix_ = nullptr;  // back-pointer (friend-legal nlist factory)
  PairDeepMD *pair_ = nullptr;
  // The standalone DeepPot is ONLY used by the non-fused fallback (the
  // post_force catch branch).  In fused mode that branch is never reached,
  // so this redundant, GPU-resident model load is skipped.
  deepmd_compat::DeepPot dp_;
  bool fused_mode_ = false;
  bool warned_fallback_ = false;
  double ener_unit_cvt_factor_ = 1.0;
  double dist_unit_cvt_factor_ = 1.0;
  double force_unit_cvt_factor_ = 1.0;
  int nchannels_ = 1;
};

}  // namespace LAMMPS_NS

#endif  // SOG_WITH_DEEPMD
#endif  // LMP_ADAPTERS_DEEPMD_CHARGE_RESPONSE_H
