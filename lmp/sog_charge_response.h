/* ----------------------------------------------------------------------
   SOG: abstract charge response — the derivatives SOG asks back from the MLIP.

   Coupling point #3 of the SOG integration contract.  The kspace solver
   produces a per-atom potential v_i = ∂E_k/∂q_i; the conservative force
   needs the MLIP's charge derivative:

       F_j = -Σ_i v_i · ∂q_i/∂r_j            (charge-response force)
       W   = the complete box-strain derivative of S = Σ_i v_i·q_i at fixed
             scaled coordinates -- the pressure tensor needs BOTH terms,
             Σ F⊗r plus the energy's explicit box derivative (∂S/∂h)|_r·hᵀ.
             For a box-free charge graph (q sees the box only through the
             extended coordinates, which scale affinely) W reduces to the
             extended-atom Σ_a F_a ⊗ r_a einsum -- the convention
             fix_sog_response tallies.  The k-space energy's own box term is
             separate: computed analytically inside sog.cpp ("Analytic
             Fourier virial"), never from r⊗F.

   Implement this for any framework whose charges depend differentiably on
   positions and the box (i.e. the framework keeps a torch graph alive).
   For fixed-charge models use NativeChargeResponse (adapters/) — it makes
   "no response exists" explicit instead of silently dropping the term.

   Unit contract: v_per_atom is passed in MLIP energy units and the outputs
   are returned in MLIP units; FixSOGResponse converts both sides using
   energy_unit_factor()/force_unit_factor().
------------------------------------------------------------------------- */

#ifndef LMP_SOG_CHARGE_RESPONSE_H
#define LMP_SOG_CHARGE_RESPONSE_H

#include <vector>

namespace LAMMPS_NS {

class SogChargeResponse {
 public:
  virtual ~SogChargeResponse() = default;

  // MLIP energy unit -> LAMMPS energy unit.
  virtual double energy_unit_factor() const = 0;
  // MLIP force unit -> LAMMPS force unit.
  virtual double force_unit_factor() const = 0;
  // Number of latent charge channels the MLIP model emits.
  virtual int nchannels() const = 0;
  // True when the MLIP's forward already dropped the short-range force and
  // response() returns the TOTAL short-range+response force/virial (fused
  // mode).  False when the pair has already added the short-range force and
  // response() returns only the charge-response correction.
  virtual bool fused_short_range() const = 0;

  // Compute the response.  v_per_atom has nlocal*nchannels entries
  // (v_i replicated per channel, MLIP energy units).  f_ext receives the
  // ghost-inclusive [nall*3] force; virial9 receives 9 components.
  virtual void response(std::vector<double> &f_ext,     // out: [nall*3]
                        std::vector<double> &virial9,   // out: [9]
                        const double *v_per_atom,       // in: [nlocal*nchannels]
                        int nlocal, int nall, int nchannels) = 0;
};

}  // namespace LAMMPS_NS

#endif  // LMP_SOG_CHARGE_RESPONSE_H
