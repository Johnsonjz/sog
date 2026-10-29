// SPDX-License-Identifier: LGPL-3.0-or-later
#include "fix_sog_response.h"

#include "atom.h"
#include "comm.h"
#include <sstream>
#include "domain.h"
#include "error.h"
#include "force.h"
#include "input.h"
#include "memory.h"
#include "modify.h"
#include "sog.h"
#include "update.h"
#include "variable.h"

#ifdef SOG_WITH_DEEPMD
#include "adapters/deepmd_charge_response.h"
#include "neigh_list.h"
#endif
#include "adapters/native_charge_response.h"

using namespace LAMMPS_NS;
using namespace FixConst;

namespace {

// Factory for coupling #3.  The `model`/`fused`/`pair_index` keywords are
// fix-level input syntax, forwarded to the adapter that understands them.
// In a native build (no differentiable charge graph) the null adapter keeps
// the fix functional and explicit about what it is (not) doing.
// `fix` is the owning FixSOGResponse: the deepmd adapter needs it back for the
// friend-legal build_input_nlist (see fix_sog_response.h).
SogChargeResponse *create_charge_response(LAMMPS *lmp, FixSOGResponse *fix,
                                          const std::string &model, bool fused,
                                          int pair_index) {
#ifdef SOG_WITH_DEEPMD
  return new DeepMDChargeResponse(lmp, fix, model, fused, pair_index);
#else
  (void)fix; (void)model; (void)fused; (void)pair_index;
  return new NativeChargeResponse(lmp);
#endif
}

}  // namespace

FixSOGResponse::FixSOGResponse(LAMMPS *lmp, int narg, char **arg)
    : Fix(lmp, narg, arg), sog_kspace(nullptr) {
  // LAMMPS passes [ID, group, style, keywords...] (narg counts from the ID).
  // All keywords are optional here: `model` is required only by the deepmd
  // adapter (checked below when SOG_WITH_DEEPMD), and a native build accepts
  // the bare `fix ID group sog/response` form.
  std::string model;
  bool fused_mode = false;
  int pair_deepmd_index = 0;

  int iarg = 3;
  while (iarg < narg) {
    if (strcmp(arg[iarg], "model") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "fix sog/response missing model path");
      model = arg[iarg + 1];
      iarg += 2;
    } else if (strcmp(arg[iarg], "pair_deepmd_index") == 0) {
      if (iarg + 1 >= narg)
        error->all(FLERR, "fix sog/response missing pair_deepmd_index value");
      pair_deepmd_index = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg], "fused") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "fix sog/response missing fused value");
      if (strcmp(arg[iarg + 1], "yes") == 0)
        fused_mode = true;
      else if (strcmp(arg[iarg + 1], "no") == 0)
        fused_mode = false;
      else
        error->all(FLERR, "fix sog/response fused expects yes or no");
      iarg += 2;
    } else {
      error->all(FLERR, fmt::format("Unknown fix sog/response keyword: {}", arg[iarg]));
    }
  }
#ifdef SOG_WITH_DEEPMD
  if (model.empty()) error->all(FLERR, "fix sog/response: must specify 'model'");
#endif

  charge_response = create_charge_response(lmp, this, model, fused_mode,
                                           pair_deepmd_index);

  comm_reverse = 3;  // dfcorr_buff: 3 doubles per atom
  virial_global_flag = 1;
  energy_global_flag = 1;
  // Without ``thermo_virial`` LAMMPS does not fold this fix's virial into the
  // thermo pressure (the deployed deepmd fix sets it too).
  thermo_virial = 1;
}

FixSOGResponse::~FixSOGResponse() { delete charge_response; }

#ifdef SOG_WITH_DEEPMD
#ifdef DP_USE_CXX_API
// Defined here (never in the adapter) because CommBrickDeepMD's friendship is
// granted to this exact class: the protected comm-brick layout cannot be read
// from a free function or from an adapter class, and the deepmd-kit repo that
// declares the friendship must stay untouched.
deepmd::InputNlist FixSOGResponse::build_input_nlist(NeighList *list) const {
  CommBrickDeepMD *commdata = (CommBrickDeepMD *)comm;
  return deepmd::InputNlist(
      list->inum, list->ilist, list->numneigh, list->firstneigh,
      commdata->nswap, commdata->sendnum, commdata->recvnum,
      commdata->firstrecv, commdata->sendlist, commdata->sendproc,
      commdata->recvproc, &world);
}
#endif
#endif

int FixSOGResponse::setmask() {
  int mask = 0;
  mask |= FixConst::PRE_FORCE;
  mask |= FixConst::POST_FORCE;
  mask |= FixConst::MIN_PRE_FORCE;
  mask |= FixConst::MIN_POST_FORCE;
  // (no THERMO_ENERGY in the fix mask — energy tally is handled by the pair+kspace)
  return mask;
}

void FixSOGResponse::init() {
  // locate the SOG kspace solver
  sog_kspace = dynamic_cast<SOGKSpace*>(force->kspace_match("sog", 1));
  if (!sog_kspace)
    error->all(FLERR,
               "fix sog/response requires kspace_style sog (not fastsog/pppm/...): "
               "the SOG kspace solver must be active to provide v_i via get_potential().");
}

void FixSOGResponse::setup(int vflag) { post_force(vflag); }
void FixSOGResponse::min_setup(int vflag) { setup(vflag); }
void FixSOGResponse::setup_pre_force(int vflag) { pre_force(vflag); }
void FixSOGResponse::min_pre_force(int vflag) { pre_force(vflag); }
void FixSOGResponse::min_post_force(int vflag) { post_force(vflag); }

void FixSOGResponse::pre_force(int /*vflag*/) {
  // Enable the SOG kspace solver to compute v_i = ∂E_k/∂q_i.
  // The Verlet force loop calls kspace->compute() between pre_force and
  // post_force, so vpot will be populated when post_force reads it.
  if (sog_kspace) sog_kspace->set_want_potential(true);
}

void FixSOGResponse::post_force(int vflag) {
  // Modified template from FixDPLR::post_force
  if (vflag) {
    v_setup(vflag);
  } else {
    evflag = 0;
  }
  // reject per-atom virial: global only
  if (vflag_atom) {
    error->all(FLERR, "fix sog/response does not support per-atom virial");
  }

  int nlocal = atom->nlocal;
  if (nlocal <= 0) return;
  int nall = nlocal + atom->nghost;

  // ── Read kspace potential v_i (needed by the response VJP) ──
  const std::vector<double> &vpot = sog_kspace->get_potential();
  if (static_cast<int>(vpot.size()) != nlocal)
    error->all(FLERR, "sog/response: vpot size != nlocal (forgot set_want_potential?)");

  // Build v_per_atom [nlocal * nchannels] (v_i replicated per channel) in
  // MLIP energy units.
  const int nchannels = charge_response->nchannels();
  std::vector<double> v_per_atom(static_cast<size_t>(nlocal) * nchannels, 0.0);
  const double inv_ener = 1.0 / charge_response->energy_unit_factor();
  for (int i = 0; i < nlocal; ++i) {
    const double vi = vpot[i] * inv_ener;
    for (int c = 0; c < nchannels; ++c)
      v_per_atom[i * nchannels + c] = vi;
  }

  // ── Compute the charge-response correction (framework-specific) ──
  std::vector<double> dfcorr, dvcorr;
  charge_response->response(dfcorr, dvcorr, v_per_atom.data(), nlocal, nall,
                            nchannels);

  // Convert correction back to LAMMPS units
  const double force_cvt = charge_response->force_unit_factor();
  const double ener_cvt = charge_response->energy_unit_factor();
  for (size_t ii = 0; ii < dfcorr.size(); ++ii)
    dfcorr[ii] *= force_cvt;
  for (int ii = 0; ii < 9; ++ii)
    dvcorr[ii] *= ener_cvt;

  // ── Reverse-comm the ghost-atom correction forces ──
  dfcorr_buff = dfcorr;
  comm->reverse_comm(this, 3);
  std::copy(dfcorr_buff.begin(), dfcorr_buff.end(), dfcorr.begin());

  // ── Accumulate into atom->f ──
  double **f = atom->f;
  for (int ii = 0; ii < nlocal; ++ii) {
    f[ii][0] += dfcorr[ii * 3 + 0];
    f[ii][1] += dfcorr[ii * 3 + 1];
    f[ii][2] += dfcorr[ii * 3 + 2];
  }

  // ── Diagnostic dump of dvcorr (SR+CR virial; 9 components, row-major
  //    Σ F⊗r for the box-free charge-graph case — see sog_charge_response.h) ──
  if (getenv("SOG_DUMP_VIRIAL") && comm->me == 0) {
    double dfnorm = 0.0;
    for (size_t ii = 0; ii < dfcorr.size(); ++ii) dfnorm += dfcorr[ii] * dfcorr[ii];
    utils::logmesg(lmp, fmt::format("SOG_VIRIAL_FIX nlocal={} dfnorm={:.14e}\n", nlocal, dfnorm));
    utils::logmesg(lmp, fmt::format("SOG_VIRIAL_FIX dvcorr: {:.14e} {:.14e} {:.14e} {:.14e} {:.14e} {:.14e} {:.14e} {:.14e} {:.14e}\n",
        dvcorr[0], dvcorr[1], dvcorr[2], dvcorr[3], dvcorr[4],
        dvcorr[5], dvcorr[6], dvcorr[7], dvcorr[8]));
  }

  // ── Global virial ──
  // For a box-free charge graph the adapter's virial is already the complete
  // strain derivative of S = Σ v_i·q_i (extended-atom F⊗r), so no extra
  // outer-product term is added here (unlike DPLR, whose modifier omits it).
  // PBC pressure = this + the k-space's own box-deformation term, which
  // sog.cpp computes analytically (never as r⊗F for the reciprocal sum).
  if (evflag) {
    double vv[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    vv[0] += dvcorr[0];
    vv[1] += dvcorr[4];
    vv[2] += dvcorr[8];
    vv[3] += dvcorr[3];
    vv[4] += dvcorr[6];
    vv[5] += dvcorr[7];
    v_tally(0, vv);
  }
}

int FixSOGResponse::pack_reverse_comm(int n, int first, double *buf) {
  int offset = first * 3;
  int last = offset + n * 3;
  for (int i = offset; i < last; ++i) { buf[i - offset] = dfcorr_buff[i]; }
  return last - offset;  // n * 3 doubles packed
}

void FixSOGResponse::unpack_reverse_comm(int n, int *list, double *buf) {
  for (int ii = 0; ii < n; ++ii) {
    int idx = list[ii] * 3;
    dfcorr_buff[idx + 0] += buf[ii * 3 + 0];
    dfcorr_buff[idx + 1] += buf[ii * 3 + 1];
    dfcorr_buff[idx + 2] += buf[ii * 3 + 2];
  }
}
