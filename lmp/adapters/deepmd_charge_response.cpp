// SPDX-License-Identifier: LGPL-3.0-or-later
#include "adapters/deepmd_charge_response.h"

#ifdef SOG_WITH_DEEPMD

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "fix_sog_response.h"
#include "force.h"
#include "lammps.h"
#include "neigh_list.h"
#include "neighbor.h"

using namespace LAMMPS_NS;

DeepMDChargeResponse::DeepMDChargeResponse(LAMMPS *lmp, FixSOGResponse *fix,
                                           const std::string &model, bool fused,
                                           int pair_index)
    : lmp_(lmp), fix_(fix), fused_mode_(fused) {
  if (!fix_)
    lmp_->error->all(FLERR,
                     "fix sog/response: deepmd adapter needs its owning fix "
                     "(friend-legal InputNlist factory)");
  // The standalone `dp_` is ONLY used by the non-fused fallback (the
  // post_force catch branch).  In fused mode that branch is never reached,
  // so skip this redundant, GPU-resident model load.
  if (!fused_mode_) dp_.init(model, 0);

  // locate the deepmd pair
  pair_ = dynamic_cast<PairDeepMD *>(
      lmp_->force->pair_match("deepmd", 1, pair_index > 0 ? pair_index : -1));
  if (!pair_)
    lmp_->error->all(FLERR, "fix sog/response requires pair_style deepmd");

  // Fuse: have the pair retain its forward autograd graph so the charge-response
  // correction is a VJP-only backward here, not a redundant second model forward.
  // fused_mode goes further: the pair runs an energy-only+charge forward (no
  // extended_force), and the fix does ONE combined backward for the total
  // short-range+response force.
  if (fused_mode_)
    pair_->enable_charge_only_forward();
  else
    pair_->enable_charge_response_graph();

  // deepmd unit conversions from the pair
  ener_unit_cvt_factor_ = pair_->ener_unit_cvt_factor;
  dist_unit_cvt_factor_ = pair_->dist_unit_cvt_factor;
  force_unit_cvt_factor_ = pair_->force_unit_cvt_factor;

  nchannels_ = pair_->ncharge_channels;
  if (nchannels_ < 1) nchannels_ = 1;
}

void DeepMDChargeResponse::response(std::vector<double> &f_ext,
                                    std::vector<double> &virial9,
                                    const double *v_per_atom, int nlocal,
                                    int nall, int nchannels) {
  (void)nall; (void)nchannels;  // the pair knows the model's layout
  std::vector<double> vp(v_per_atom,
                         v_per_atom + static_cast<size_t>(nlocal) * nchannels_);

  // ── Compute the charge-response correction ──
  // Fast path: VJP on the pair's retained forward graph (no second forward, no
  // model inputs).  Fallback: standalone forward+backward — its dcoord/dtype/
  // dbox packing and the message-passing InputNlist + mapping are built ONLY
  // here (deferred off the hot path; the cached path never reads them, so on
  // the steady-state fast path this O(nall) per-step work is skipped entirely).
  bool used_cached = false;
  if (fused_mode_) {
    // Tier-2: the pair ran the energy-only+charge forward (added zero force).
    // This ONE combined backward yields the TOTAL short-range+response force
    // (+ total virial).  No fallback: a missing retained graph would silently
    // drop the short-range force, so the pair's exception propagates.
    pair_->compute_combined_response_cached(f_ext, virial9, vp);
    used_cached = true;
  } else {
    try {
      pair_->compute_charge_response_cached(f_ext, virial9, vp);
      used_cached = true;
    } catch (deepmd_compat::deepmd_exception &) {
      double **x = lmp_->atom->x;
      int *type = lmp_->atom->type;
      std::vector<double> dcoord(static_cast<size_t>(nall) * 3, 0.0);
      std::vector<int> dtype(nall);
      for (int ii = 0; ii < nall; ++ii) {
        dtype[ii] = type[ii] - 1;
        dcoord[ii * 3 + 0] = (x[ii][0] * dist_unit_cvt_factor_ + lmp_->domain->boxlo[0]);
        dcoord[ii * 3 + 1] = (x[ii][1] * dist_unit_cvt_factor_ + lmp_->domain->boxlo[1]);
        dcoord[ii * 3 + 2] = (x[ii][2] * dist_unit_cvt_factor_ + lmp_->domain->boxlo[2]);
      }
      std::vector<double> dbox(9, 0.0);
      dbox[0] = (lmp_->domain->boxhi[0] - lmp_->domain->boxlo[0]) * dist_unit_cvt_factor_;
      dbox[4] = (lmp_->domain->boxhi[1] - lmp_->domain->boxlo[1]) * dist_unit_cvt_factor_;
      dbox[8] = (lmp_->domain->boxhi[2] - lmp_->domain->boxlo[2]) * dist_unit_cvt_factor_;
      dbox[3] = lmp_->domain->xy * dist_unit_cvt_factor_;
      dbox[6] = lmp_->domain->xz * dist_unit_cvt_factor_;
      dbox[7] = lmp_->domain->yz * dist_unit_cvt_factor_;
      NeighList *list = pair_->list;
      int nghost_val = nall - nlocal;
      // The comm-brick layout behind deepmd::InputNlist is protected; only
      // FixSOGResponse (the declared friend of CommBrickDeepMD) can read it.
      deepmd::InputNlist lmp_list = fix_->build_input_nlist(list);
      lmp_list.set_mask(NEIGHMASK);
      std::vector<int> mapping_vec(nall, -1);
      if (lmp_->comm->nprocs == 1 && lmp_->atom->map_style != Atom::MAP_NONE) {
        for (int ii = 0; ii < nall; ++ii)
          mapping_vec[ii] = lmp_->atom->map(lmp_->atom->tag[ii]);
        lmp_list.set_mapping(mapping_vec.data());
      }
      dp_.compute_charge_response(f_ext, virial9, vp, dcoord, dtype, dbox,
                                  nghost_val, lmp_list, 0);
    }
  }
  if (!used_cached && !warned_fallback_) {
    warned_fallback_ = true;
    lmp_->error->warning(FLERR,
                         "fix sog/response: charge-response graph unavailable; "
                         "using slower standalone forward (fusion inactive).");
  }
}

#endif  // SOG_WITH_DEEPMD
