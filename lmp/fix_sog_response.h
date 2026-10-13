// SPDX-License-Identifier: LGPL-3.0-or-later
#ifdef FIX_CLASS

FixStyle(sog/response, FixSOGResponse)

#else

#ifndef LMP_FIX_SOG_RESPONSE_H
#define LMP_FIX_SOG_RESPONSE_H

#include <string>
#include <vector>

#include "fix.h"
#include "sog_charge_response.h"

#if defined(SOG_WITH_DEEPMD) && defined(DP_USE_CXX_API)
namespace deepmd {
class InputNlist;  // complete type lives in this fork's api_cc DeepPot.h
}
#endif

namespace LAMMPS_NS {

class NeighList;

class FixSOGResponse : public Fix {
 public:
  FixSOGResponse(class LAMMPS*, int, char**);
  ~FixSOGResponse() override;
  int setmask() override;
  void init() override;
  void setup(int) override;
  void setup_pre_force(int) override;
  void min_setup(int) override;
  void pre_force(int) override;
  void post_force(int) override;
  void min_pre_force(int) override;
  void min_post_force(int) override;
  int pack_reverse_comm(int, int, double*) override;
  void unpack_reverse_comm(int, int*, double*) override;

#if defined(SOG_WITH_DEEPMD) && defined(DP_USE_CXX_API)
  // ── Friend-legal deepmd InputNlist factory ──
  // CommBrickDeepMD (deepmd-kit source/lmp/pair_deepmd.h) exposes its protected
  // comm-brick layout (nswap/sendnum/sendlist/...) to `friend class PairDeepMD`
  // and `friend class FixSOGResponse` only.  Every deepmd-specific helper in
  // this tree lives in an adapter class that is NOT a friend, so the neighbor
  // list the standalone charge-response fallback needs must be assembled inside
  // a member of THIS class (deepmd-kit itself must not be modified).
  deepmd::InputNlist build_input_nlist(NeighList* list) const;
#endif

 private:
  // Framework-specific machinery behind coupling #3 (see
  // sog_charge_response.h): a DeepMDChargeResponse when built with
  // -DSOG_WITH_DEEPMD, NativeChargeResponse (documented no-op) otherwise.
  SogChargeResponse* charge_response = nullptr;  // owned
  class SOGKSpace* sog_kspace = nullptr;
  std::vector<double> dfcorr_buff;
};

}  // namespace LAMMPS_NS

#endif  // LMP_FIX_SOG_RESPONSE_H
#endif  // FIX_CLASS
