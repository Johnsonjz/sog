/* ----------------------------------------------------------------------
   SOG: abstract charge source — what the host MLIP hands the kspace solver.

   Coupling point #2 of the SOG integration contract: per-atom latent
   charges produced by the MLIP pair style, consumed by SOGKSpace::compute().
   Any framework implements this one interface; the solver never looks at
   the pair style's class name.

   Contract:
     * charges() returns a flat, channel-major buffer in LAMMPS units:
         charges()[i * nchannels() + ch] = q_{i,ch}   for i in [0, nlocal)
     * the buffer must be valid for the whole compute() call
     * nchannels() <= 1 selects the native atom->q path (any pair style
       that writes atom->q satisfies it — no SogChargeSource needed)

   Resolution order inside SOGKSpace::compute():
     1. a source registered explicitly via SOGKSpace::set_charge_source()
     2. the deepmd adapter (only when compiled with -DSOG_WITH_DEEPMD)
     3. atom->q (native single-channel path)
------------------------------------------------------------------------- */

#ifndef LMP_SOG_CHARGE_SOURCE_H
#define LMP_SOG_CHARGE_SOURCE_H

namespace LAMMPS_NS {

class SogChargeSource {
 public:
  virtual ~SogChargeSource() = default;

  // Number of latent charge channels; <=1 means "use atom->q".
  virtual int nchannels() const = 0;

  // Flat channel-major charge buffer, [i*nchannels + ch], i in [0, nlocal).
  // May return nullptr only when nchannels() <= 1.
  virtual const double *charges() const = 0;
};

}  // namespace LAMMPS_NS

#endif  // LMP_SOG_CHARGE_SOURCE_H
