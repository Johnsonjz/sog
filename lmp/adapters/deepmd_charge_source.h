/* ----------------------------------------------------------------------
   SOG adapter: DeepMD pair style as a SogChargeSource.

   Compiled only with -DSOG_WITH_DEEPMD (the build gate that makes
   pair_deepmd.h available).  This is the entire deepmd-specific code the
   kspace solver needs: dynamic_cast to the pair and a bounds guard.

   Guard rationale: pair_deepmd.cpp initializes ncharge_channels = 0 and
   only fills dcharge_multi when the model runs with latent_charge_to_q.
   Without the guard, a latent_charge_to_q-less model used to read an empty
   buffer out of bounds (dcharge_multi[i*nchannels] with nchannels garbage);
   the adapter degrades to the native atom->q path instead.
------------------------------------------------------------------------- */

#ifndef LMP_ADAPTERS_DEEPMD_CHARGE_SOURCE_H
#define LMP_ADAPTERS_DEEPMD_CHARGE_SOURCE_H

#ifdef SOG_WITH_DEEPMD

#include "pair_deepmd.h"
#include "sog_charge_source.h"

namespace LAMMPS_NS {

class DeepMDChargeSource : public SogChargeSource {
 public:
  explicit DeepMDChargeSource(PairDeepMD *pair) : pair_(pair) {}

  int nchannels() const override {
    if (pair_ == nullptr) return 1;
    if (pair_->dcharge_multi.empty()) return 1;  // latent_charge_to_q off
    return pair_->ncharge_channels;
  }

  const double *charges() const override {
    if (pair_ == nullptr || pair_->dcharge_multi.empty()) return nullptr;
    return pair_->dcharge_multi.data();
  }

 private:
  PairDeepMD *pair_;
};

}  // namespace LAMMPS_NS

#endif  // SOG_WITH_DEEPMD
#endif  // LMP_ADAPTERS_DEEPMD_CHARGE_SOURCE_H
