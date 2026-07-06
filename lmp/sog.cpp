/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://lammps.sandia.gov/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing authors: Zhen Jiang (SJTU)

   SOG: FFT (from fastsog.cpp)-grid based Sum-of-Gaussians kspace solver.
   Uses B-spline charge spreading (order 5) + FFT + precomputed Green
   functions with alias fast-path.  Based on the optimized sog.cpp
   approach from deepmd-kit, adapted to the native LAMMPS KSpace API
   and the RBSOG SOG kernel convention.
------------------------------------------------------------------------- */

#include "sog.h"
#include "sog_spline.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "fft3d_wrap.h"
#include "force.h"
#include "math_const.h"
#include "pair.h"
#include "pair_deepmd.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <limits>
#include <string>

using namespace LAMMPS_NS;
using namespace MathConst;

namespace {

// ── Grid helpers ──

bool factorable_235(int n) {
  while (n > 1) {
    if ((n % 2) == 0)
      n /= 2;
    else if ((n % 3) == 0)
      n /= 3;
    else if ((n % 5) == 0)
      n /= 5;
    else
      return false;
  }
  return true;
}

// ── Math helpers ──

double sinc(const double x) {
  if (std::fabs(x) < 1e-14) return 1.0;
  return std::sin(x) / x;
}

double sinc_pow(const double x, const int p) {
  const double s = sinc(x);
  return std::pow(s, static_cast<double>(p));
}

int wrap_index(int i, const int n) {
  i %= n;
  if (i < 0) i += n;
  return i;
}

// ── B-spline weights (order 5) ──
constexpr int kSog_BSplineOrder = 5;

void sog_bspline_weights_1d(const double frac,
                        std::array<double, kSog_BSplineOrder> &w) {
  w.fill(0.0);
  w[0] = 1.0 - frac;
  w[1] = frac;
  for (int k = 3; k <= kSog_BSplineOrder; ++k) {
    const double inv = 1.0 / static_cast<double>(k - 1);
    w[static_cast<size_t>(k - 1)] =
        frac * w[static_cast<size_t>(k - 2)] * inv;
    for (int j = 1; j <= k - 2; ++j) {
      w[static_cast<size_t>(k - 1 - j)] =
          ((frac + static_cast<double>(j)) *
               w[static_cast<size_t>(k - 2 - j)] +
           (static_cast<double>(k - j) - frac) *
               w[static_cast<size_t>(k - 1 - j)]) *
          inv;
    }
    w[0] = (1.0 - frac) * w[0] * inv;
  }
}

// ── 1D analytic Fourier integrals for CubeS₂ influence function ──
// I_p(α) = ∫₀¹ t^p · exp(i·α·t) dt
// Recurrence (p≥1): I_p = (exp(iα) - p·I_{p-1}) / (iα)
// For |α| < 1e-8, use Taylor expansion to avoid division by near-zero.

inline std::complex<double> I_int_0(const double alpha) {
  if (std::fabs(alpha) < 1e-8) {
    // Taylor: I_0 ≈ 1 + iα/2 - α²/6 - iα³/24 + α⁴/120
    return std::complex<double>(1.0 - alpha * alpha / 6.0,
                                alpha / 2.0 - alpha * alpha * alpha / 24.0);
  }
  const double cos_a = std::cos(alpha);
  const double sin_a = std::sin(alpha);
  return std::complex<double>(sin_a / alpha, (1.0 - cos_a) / alpha);
}

inline std::complex<double> I_int_1(const double alpha) {
  if (std::fabs(alpha) < 1e-8) {
    // Taylor: I_1 ≈ 1/2 + iα/3 - α²/8 - iα³/30
    return std::complex<double>(0.5 - alpha * alpha / 8.0,
                                alpha / 3.0 - alpha * alpha * alpha / 30.0);
  }
  const double cos_a = std::cos(alpha);
  const double sin_a = std::sin(alpha);
  const double a2 = alpha * alpha;
  return std::complex<double>(
      (alpha * sin_a + cos_a - 1.0) / a2,
      (sin_a - alpha * cos_a) / a2);
}

inline std::complex<double> I_int_2(const double alpha) {
  if (std::fabs(alpha) < 1e-8) {
    // Taylor: I_2 ≈ 1/3 + iα/4 - α²/10
    return std::complex<double>(1.0 / 3.0 - alpha * alpha / 10.0, alpha / 4.0);
  }
  const double cos_a = std::cos(alpha);
  const double sin_a = std::sin(alpha);
  const double a2 = alpha * alpha;
  const double a3 = a2 * alpha;
  return std::complex<double>(
      (2.0 * alpha * sin_a + (a2 - 2.0) * cos_a + 2.0) / a3,
      ((a2 - 2.0) * sin_a + 2.0 * alpha * cos_a) / a3);
}

inline std::complex<double> I_int_3(const double alpha) {
  if (std::fabs(alpha) < 1e-8) {
    // Taylor: I_3 ≈ 1/4 + iα/5
    return std::complex<double>(0.25, alpha / 5.0);
  }
  const double cos_a = std::cos(alpha);
  const double sin_a = std::sin(alpha);
  const double a2 = alpha * alpha;
  const double a3 = a2 * alpha;
  const double a4 = a3 * alpha;
  return std::complex<double>(
      ((3.0 * a2 - 6.0) * alpha * sin_a + (a3 - 6.0 * alpha) * cos_a + 6.0 * alpha) / a4,
      ((a3 - 6.0 * alpha) * sin_a + (6.0 - 3.0 * a2) * cos_a + 3.0 * a2 - 6.0) / a4);
}

// ── Monomial expansion for CubeS₂ 4th-order node weights ──
// Each entry: (pow_x, pow_y, pow_z, real_coeff)
struct MonomialTerm {
  int px, py, pz;
  double coeff;
};

// Maximum 64 monomials per node (covers all cubic cross-terms)
constexpr int kMaxMonomialsPerNode = 64;

struct CubeS2NodeMonomial {
  int num_terms;
  MonomialTerm terms[kMaxMonomialsPerNode];
};

// Precompute monomial expansion for all 32 nodes.
// Class-0 nodes (offsets in {0,1}³): c_d = L(ηx)·ηy·ηz + cyclic.
// Class-1 nodes (one offset -1 or 2): c_d = R(η_special) · η_n1 · η_n2.
// L(t) = -½t³ + ½t² - ξ²_adj·t + ξ²/2
// R(t) =  ⅙t³ + (3ξ²-1)/6·t      (paper Eq. 15-16)
inline void build_monomials_for_node(const CubeS2Node4 &node, const double xi,
                                      CubeS2NodeMonomial &result) {
  result.num_terms = 0;
  const int dx = node.dx, dy = node.dy, dz = node.dz;
  const double a[3] = {static_cast<double>(dx),
                        static_cast<double>(dy),
                        static_cast<double>(dz)};
  const double b[3] = {1.0 - 2.0 * a[0],
                        1.0 - 2.0 * a[1],
                        1.0 - 2.0 * a[2]};

  const double xi2 = xi * xi;

  // Binomial coefficients C(n,k)
  auto binom = [](int n, int k) -> double {
    if (k < 0 || k > n) return 0.0;
    // n ≤ 3
    constexpr double C[4][4] = {
      {1, 0, 0, 0},
      {1, 1, 0, 0},
      {1, 2, 1, 0},
      {1, 3, 3, 1},
    };
    return C[n][k];
  };

  if (node.cls == 0) {
    // Class 0: c_d = L(ηx)·ηy·ηz + L(ηy)·ηz·ηx + L(ηz)·ηx·ηy
    const double xi2_adj = (9.0 * xi2 - 2.0) / 6.0;
    const double L_coeffs[4] = {0.5 * xi2, -xi2_adj, 0.5, -0.5};

    for (int term_idx = 0; term_idx < 3; ++term_idx) {
      int axis_L = term_idx;
      int axis_n1 = (term_idx + 1) % 3;
      int axis_n2 = (term_idx + 2) % 3;

      for (int pL = 0; pL <= 3; ++pL) {
        const double c_L = L_coeffs[pL];
        if (c_L == 0.0) continue;
        for (int jL = 0; jL <= pL; ++jL) {
          const double cf_L = c_L * binom(pL, jL) *
            std::pow(a[axis_L], static_cast<double>(pL - jL)) *
            std::pow(b[axis_L], static_cast<double>(jL));
          for (int jn1 = 0; jn1 <= 1; ++jn1) {
            const double cf_n1 = binom(1, jn1) *
              std::pow(a[axis_n1], static_cast<double>(1 - jn1)) *
              std::pow(b[axis_n1], static_cast<double>(jn1));
            for (int jn2 = 0; jn2 <= 1; ++jn2) {
              const double cf_n2 = binom(1, jn2) *
                std::pow(a[axis_n2], static_cast<double>(1 - jn2)) *
                std::pow(b[axis_n2], static_cast<double>(jn2));
              const double coeff = cf_L * cf_n1 * cf_n2;
              if (coeff == 0.0) continue;
              int pows[3] = {0, 0, 0};
              pows[axis_L] = jL;
              pows[axis_n1] = jn1;
              pows[axis_n2] = jn2;
              bool merged = false;
              for (int m = 0; m < result.num_terms; ++m) {
                if (result.terms[m].px == pows[0] &&
                    result.terms[m].py == pows[1] &&
                    result.terms[m].pz == pows[2]) {
                  result.terms[m].coeff += coeff;
                  merged = true;
                  break;
                }
              }
              if (!merged && result.num_terms < kMaxMonomialsPerNode) {
                result.terms[result.num_terms] = {pows[0], pows[1], pows[2], coeff};
                result.num_terms++;
              }
            }
          }
        }
      }
    }
  } else {
    // Class 1: c_d = R(η_special) · η_n1 · η_n2  (single term, paper Eq. 16)
    const double R_coeffs[4] = {0.0, (3.0 * xi2 - 1.0) / 6.0, 0.0, 1.0 / 6.0};
    int axis_L = node.sp_axis;
    int axis_n1 = (axis_L + 1) % 3;
    int axis_n2 = (axis_L + 2) % 3;

    for (int pL = 0; pL <= 3; ++pL) {
      const double c_R = R_coeffs[pL];
      if (c_R == 0.0) continue;
      for (int jL = 0; jL <= pL; ++jL) {
        const double cf_L = c_R * binom(pL, jL) *
          std::pow(a[axis_L], static_cast<double>(pL - jL)) *
          std::pow(b[axis_L], static_cast<double>(jL));
        for (int jn1 = 0; jn1 <= 1; ++jn1) {
          const double cf_n1 = binom(1, jn1) *
            std::pow(a[axis_n1], static_cast<double>(1 - jn1)) *
            std::pow(b[axis_n1], static_cast<double>(jn1));
          for (int jn2 = 0; jn2 <= 1; ++jn2) {
            const double cf_n2 = binom(1, jn2) *
              std::pow(a[axis_n2], static_cast<double>(1 - jn2)) *
              std::pow(b[axis_n2], static_cast<double>(jn2));
            const double coeff = cf_L * cf_n1 * cf_n2;
            if (coeff == 0.0) continue;
            int pows[3] = {0, 0, 0};
            pows[axis_L] = jL;
            pows[axis_n1] = jn1;
            pows[axis_n2] = jn2;
            bool merged = false;
            for (int m = 0; m < result.num_terms; ++m) {
              if (result.terms[m].px == pows[0] &&
                  result.terms[m].py == pows[1] &&
                  result.terms[m].pz == pows[2]) {
                result.terms[m].coeff += coeff;
                merged = true;
                break;
              }
            }
            if (!merged && result.num_terms < kMaxMonomialsPerNode) {
              result.terms[result.num_terms] = {pows[0], pows[1], pows[2], coeff};
              result.num_terms++;
            }
          }
        }
      }
    }
  }
}

// ── SOG real-space helpers (matching pair_lj_cut_coul_user convention) ──

double G_sigma(const double sigma, const double r) {
  return std::exp(-r * r / (2.0 * sigma * sigma)) /
         std::sqrt(2.0 * MY_PI * sigma * sigma);
}

double compute_w0(const double r0, const double b) {
  // r0 = rcut / sigma
  double sum = 0.0;
  for (int i = 1; i < 200; ++i) {
    const double bi = std::pow(b, static_cast<double>(-i));
    sum += bi * G_sigma(1.0, bi * r0);
  }
  const double w0 =
      (1.0 / G_sigma(1.0, r0)) * ((1.0 / (2.0 * std::log(b) * r0)) - sum);
  return w0;
}

// ── PPPM-style grid estimation ──

constexpr int kGridMin = 8;
constexpr int kGridMaxIter = 500;
constexpr int kSog_AssignOrder = kSog_BSplineOrder;
constexpr int kAliasExtent = 8;

double pppm_ik_error_estimate_order5(const double h, const double prd,
                                     const bigint natoms, const double q2,
                                     const double g_eff) {
  if (!(natoms > 0) || !(h > 0.0) || !(prd > 0.0) || !(q2 > 0.0) ||
      !(g_eff > 0.0))
    return std::numeric_limits<double>::infinity();

  static constexpr double acons_order5[] = {
      1.0 / 23232.0,
      7601.0 / 13628160.0,
      143.0 / 69120.0,
      517231.0 / 106536960.0,
      106640677.0 / 11737571328.0,
  };

  double series = 0.0;
  for (int m = 0; m < 5; ++m) {
    series +=
        acons_order5[m] * std::pow(h * g_eff, 2.0 * static_cast<double>(m));
  }

  const double prefactor = q2 * std::pow(h * g_eff, 5.0);
  const double root = std::sqrt(g_eff * prd * std::sqrt(MY_2PI) * series /
                                static_cast<double>(natoms));
  return prefactor * root / (prd * prd);
}

}  // namespace

// ──────────────────────────────────────────────────────────────────────
// SOG constructor / destructor
// ──────────────────────────────────────────────────────────────────────

SOGKSpace::SOGKSpace(LAMMPS *lmp)
    : KSpace(lmp),
      b_param(0.0),
      sigma_param(0.0),
      M_param(6),
      accuracy_in(1e-6),
      n_dl(-1.0),             // -1 = auto‑compute from accuracy + sigma
      remove_self_interaction(false),
      mesh_oversample(1.5),
      mesh_alias_extent(kAliasExtent),
      spline_type(4),         // default: CubeS₂ 4th order
      grid_method(0),         // default: SOG‑bandwidth grid estimation
      phi_max_user(-1.0),    // −1 = auto-compute from paper Table III
      w0(0.0),
      self_coeff(0.0),
      mesh_nx(0),
      mesh_ny(0),
      mesh_nz(0),
      mesh_lx(0.0),
      mesh_ly(0.0),
      mesh_lz(0.0),
      mesh_fft(nullptr),
      mesh_ready(false),
      amp_from_user(false) {
  triclinic_support = 0;
}

SOGKSpace::~SOGKSpace() { destroy_fft_plan(); }

// ──────────────────────────────────────────────────────────────────────
// settings / parameters
// ──────────────────────────────────────────────────────────────────────

void SOGKSpace::settings(int narg, char **arg) {
  // Required: accuracy b sigma M  [n_dl]  [options...]
  // n_dl is optional; auto‑computed from accuracy + sigma when omitted.
  if (narg < 4)
    error->all(FLERR,
               "Illegal kspace_style sog command: expected "
               "accuracy b sigma M [n_dl] [options]");

  accuracy_in = std::fabs(atof(arg[0]));
  if (!(std::isfinite(accuracy_in) && accuracy_in > 0.0))
    error->all(FLERR, "sog requires a positive accuracy argument");

  b_param = atof(arg[1]);
  sigma_param = atof(arg[2]);
  M_param = atoi(arg[3]);

  if (!(b_param > 0.0)) error->all(FLERR, "sog requires b > 0");
  if (!(sigma_param > 0.0)) error->all(FLERR, "sog requires sigma > 0");
  if (M_param < 1) error->all(FLERR, "sog requires M >= 1");

  // Reset optionals to defaults before parsing
  n_dl = -1.0;  // auto‑compute
  remove_self_interaction = false;
  mesh_oversample = 1.5;
  mesh_alias_extent = kAliasExtent;
  spline_type = 4;   // CubeS₂ 4th
  grid_method = 0;   // SOG bandwidth
  phi_max_user = -1.0;  // auto-compute

  // Parse optional 5th argument: n_dl or first option keyword
  int iarg = 4;
  if (iarg < narg) {
    // Try to parse as a number (n_dl); if it fails or starts with a letter,
    // treat it as the first option keyword.
    char *endptr = nullptr;
    double maybe_n_dl = strtod(arg[iarg], &endptr);
    if (endptr != arg[iarg] && *endptr == '\0' && std::isfinite(maybe_n_dl) &&
        maybe_n_dl > 0.0) {
      n_dl = maybe_n_dl;
      ++iarg;
    }
    // else: not a number → leave n_dl at -1 (auto), treat this arg as keyword
  }

  while (iarg < narg) {
    const std::string key(arg[iarg]);
    if (key == "remove_self_interaction") {
      if (iarg + 1 >= narg)
        error->all(FLERR, "sog missing remove_self_interaction value");
      const std::string val(arg[iarg + 1]);
      if (val == "1" || val == "yes" || val == "on" || val == "true")
        remove_self_interaction = true;
      else if (val == "0" || val == "no" || val == "off" || val == "false")
        remove_self_interaction = false;
      else
        error->all(FLERR, "sog remove_self_interaction expects yes/no");
      iarg += 2;
    } else if (key == "mesh_oversample") {
      if (iarg + 1 >= narg)
        error->all(FLERR, "sog missing mesh_oversample value");
      mesh_oversample = atof(arg[iarg + 1]);
      if (!(mesh_oversample >= 1.0))
        error->all(FLERR, "sog mesh_oversample must be >= 1.0");
      iarg += 2;
    } else if (key == "mesh_alias_extent") {
      if (iarg + 1 >= narg)
        error->all(FLERR, "sog missing mesh_alias_extent value");
      mesh_alias_extent = atoi(arg[iarg + 1]);
      if (mesh_alias_extent < 1)
        error->all(FLERR, "sog mesh_alias_extent must be >= 1");
      iarg += 2;
    } else if (key == "spline") {
      if (iarg + 1 >= narg)
        error->all(FLERR, "sog missing spline value");
      const std::string val(arg[iarg + 1]);
      if (val == "bspline")
        spline_type = 0;
      else if (val == "cubes2_4")
        spline_type = 4;
      else if (val == "cubes2_6")
        spline_type = 6;
      else
        error->all(FLERR, "sog spline expects bspline, cubes2_4, or cubes2_6");
      iarg += 2;
    } else if (key == "grid_method") {
      if (iarg + 1 >= narg)
        error->all(FLERR, "sog missing grid_method value");
      const std::string val(arg[iarg + 1]);
      if (val == "sog_bandwidth")
        grid_method = 0;
      else if (val == "pppm_legacy")
        grid_method = 1;
      else
        error->all(FLERR, "sog grid_method expects sog_bandwidth or pppm_legacy");
      iarg += 2;
    } else if (key == "phi_max") {
      if (iarg + 1 >= narg)
        error->all(FLERR, "sog missing phi_max value");
      phi_max_user = atof(arg[iarg + 1]);
      if (!(phi_max_user > 0.0))
        error->all(FLERR, "sog phi_max must be > 0");
      iarg += 2;
    } else if (key == "b") {
      if (iarg + 1 >= narg) error->all(FLERR, "sog missing b value");
      b_param = atof(arg[iarg + 1]);
      if (!(b_param > 0.0)) error->all(FLERR, "sog requires b > 0");
      iarg += 2;
    } else if (key == "sigma") {
      if (iarg + 1 >= narg) error->all(FLERR, "sog missing sigma value");
      sigma_param = atof(arg[iarg + 1]);
      if (!(sigma_param > 0.0)) error->all(FLERR, "sog requires sigma > 0");
      iarg += 2;
    } else if (key == "m") {
      if (iarg + 1 >= narg) error->all(FLERR, "sog missing M value");
      M_param = atoi(arg[iarg + 1]);
      if (M_param < 1) error->all(FLERR, "sog requires M >= 1");
      iarg += 2;
    } else if (key == "n_dl") {
      if (iarg + 1 >= narg) error->all(FLERR, "sog missing n_dl value");
      n_dl = atof(arg[iarg + 1]);
      if (!(n_dl > 0.0)) error->all(FLERR, "sog requires n_dl > 0");
      iarg += 2;
    } else if (key == "use_finufft") {
      iarg += 2;  // accepted for compatibility (ignored)
    } else if (key == "amp") {
      amp.clear(); amp_from_user = true; ++iarg;
      while (iarg < narg) {
        const std::string tok(arg[iarg]);
        if (tok == "b" || tok == "sigma" || tok == "m" || tok == "n_dl" ||
            tok == "amp" || tok == "bandwidth" || tok == "remove_self_interaction" ||
            tok == "use_finufft" || tok == "spline" || tok == "mesh_oversample" ||
            tok == "mesh_alias_extent" || tok == "grid_method" || tok == "phi_max")
          break;
        char *ep = nullptr; double v = strtod(arg[iarg], &ep);
        if (ep == arg[iarg] || !std::isfinite(v)) break;
        amp.push_back(v); ++iarg;
      }
      if (amp.empty()) error->all(FLERR, "sog amp keyword requires at least one value");
    } else if (key == "bandwidth") {
      bandwidth.clear(); amp_from_user = true; ++iarg;
      while (iarg < narg) {
        const std::string tok(arg[iarg]);
        if (tok == "b" || tok == "sigma" || tok == "m" || tok == "n_dl" ||
            tok == "amp" || tok == "bandwidth" || tok == "remove_self_interaction" ||
            tok == "use_finufft" || tok == "spline" || tok == "mesh_oversample" ||
            tok == "mesh_alias_extent" || tok == "grid_method" || tok == "phi_max")
          break;
        char *ep = nullptr; double v = strtod(arg[iarg], &ep);
        if (ep == arg[iarg] || !std::isfinite(v)) break;
        bandwidth.push_back(v); ++iarg;
      }
      if (bandwidth.empty()) error->all(FLERR, "sog bandwidth keyword requires at least one value");
    } else {
      error->all(FLERR, "Unknown sog option: {}", key);
    }
  }
}

// ──────────────────────────────────────────────────────────────────────
// kernel parameter finalization
// ──────────────────────────────────────────────────────────────────────

void SOGKSpace::finalize_kernel_parameters() {
  // When amp/bandwidth are externally provided (e.g. from a frozen DeepMD
  // model), skip internal generation and set self_coeff = 0.
  if (amp_from_user) {
    if (amp.empty() || bandwidth.empty())
      error->all(FLERR, "sog external amp/bandwidth must both be provided");
    if (amp.size() != bandwidth.size())
      error->all(FLERR, "sog amp and bandwidth must have same length");
    M_param = static_cast<int>(amp.size());
    self_coeff = 0.0;
    finalize_virial_parameters();
    // Total SOG amplitude at k=0: A = Σ_m amp_m
    amp_sum = 0.0;
    for (size_t m = 0; m < amp.size(); ++m) amp_sum += amp[m];
    return;
  }

  // Compute amp / bandwidth arrays using the RBSOG convention:
  //   coef[0] = 4*pi*log(b) * w0 * sigma^2
  //   coef[1] = 4*pi*log(b) * sigma^2 * b^2
  //   coef[m] = coef[m-1] * b^2  (m >= 2)
  //   band_m  = b^(2m) * sigma^2
  //
  //   K(k^2) = sum_{m=0}^{M-1} coef[m] * exp(-band_m * k^2 / 2)

  const double sigma2 = sigma_param * sigma_param;
  const double b2 = b_param * b_param;
  const double logb = std::log(b_param);

  amp.resize(static_cast<size_t>(M_param));
  bandwidth.resize(static_cast<size_t>(M_param));

  amp[0] = 4.0 * MY_PI * logb * w0 * sigma2;
  bandwidth[0] = sigma2;

  for (int m = 1; m < M_param; ++m) {
    bandwidth[static_cast<size_t>(m)] =
        bandwidth[static_cast<size_t>(m - 1)] * b2;
    if (m == 1) {
      amp[1] = 4.0 * MY_PI * logb * sigma2 * b2;
    } else {
      amp[static_cast<size_t>(m)] =
          amp[static_cast<size_t>(m - 1)] * b2;
    }
  }

  // Self-energy coefficient (matching RBSOG convention):
  // coeff = log(b) / (sqrt(2*pi) * sigma) * (w0 + (1 - b^{-M}) / (b - 1))
  double sum_b = 0.0;
  for (int m = 1; m < M_param; ++m) {
    sum_b += std::pow(b_param, static_cast<double>(-m));
  }
  if (!amp_from_user)
    self_coeff = (logb / (std::sqrt(2.0 * MY_PI) * sigma_param)) *
               (w0 + sum_b);

  // Total SOG amplitude at k=0: A = Σ_m amp_m
  amp_sum = 0.0;
  for (size_t m = 0; m < amp.size(); ++m) amp_sum += amp[m];
}

// ──────────────────────────────────────────────────────────────────────
// spectral kernel: K(k^2)
// ──────────────────────────────────────────────────────────────────────

double SOGKSpace::spectral_kernel(const double ksq) const {
  if (!(ksq > 0.0)) return 0.0;

  double sum = 0.0;
  for (size_t m = 0; m < amp.size(); ++m) {
    sum += amp[m] * std::exp(-0.5 * bandwidth[m] * ksq);
  }
  return sum;
}

// ──────────────────────────────────────────────────────────────────────
// virial spectral kernel: K_v(k²) = Σ amp·band · exp(-band·k²/2)
// Used for the anisotropic (k⊗k) contribution to the virial tensor.
// ──────────────────────────────────────────────────────────────────────

void SOGKSpace::finalize_virial_parameters() {
  amp_virial.resize(amp.size());
  for (size_t m = 0; m < amp.size(); ++m) {
    // amp_virial[m] = amp[m] * bandwidth[m]
    // This captures the b⁴ vs b² scaling in the NPT virial kernel
    // (one extra factor of σ²·b^{2m} = bandwidth[m])
    amp_virial[m] = amp[m] * bandwidth[m];
  }
}

double SOGKSpace::spectral_kernel_virial(const double ksq) const {
  if (!(ksq > 0.0)) return 0.0;

  double sum = 0.0;
  for (size_t m = 0; m < amp_virial.size(); ++m) {
    sum += amp_virial[m] * std::exp(-0.5 * bandwidth[m] * ksq);
  }
  return sum;
}

// ──────────────────────────────────────────────────────────────────────
// init — called once before run
// ──────────────────────────────────────────────────────────────────────

void SOGKSpace::init() {
  triclinic_check();

  if (domain->dimension == 2)
    error->all(FLERR, "Cannot use sog with 2d simulation");
  if (domain->triclinic != 0)
    error->all(FLERR, "sog currently supports only orthorhombic boxes");
  if (!atom->q_flag)
    error->all(FLERR, "sog requires atom attribute q");
  if (domain->nonperiodic > 0 || domain->xperiodic != 1 ||
      domain->yperiodic != 1 || domain->zperiodic != 1)
    error->all(FLERR, "sog requires fully periodic boundaries");

  two_charge();
  pair_check();

  // Extract cutoff from pair style (needed for w0 computation)
  int itmp = 0;
  auto *p_cutoff = (double *)force->pair->extract("cut_coul", itmp);
  const double rcut = (p_cutoff != nullptr && *p_cutoff > 0.0)
                          ? *p_cutoff
                          : n_dl;
  if (!(rcut > 0.0))
    error->all(FLERR, "sog requires positive rcut (from cut_coul or n_dl)");

  // Compute w0 (real-space correction factor)
  const double r0 = rcut / sigma_param;
  w0 = compute_w0(r0, b_param);

  // Compute amp / bandwidth / self_coeff from w0
  finalize_kernel_parameters();
  finalize_virial_parameters();

  // Auto‑compute n_dl if not user‑specified
  // n_dl sets the k‑space cutoff: k_max = 2π/n_dl.
  // Require exp(-σ²·k_max²/2) ≤ accuracy, i.e. k_max² ≥ 2·ln(1/acc)/σ².
  if (!(n_dl > 0.0)) {
    n_dl = MY_PI * std::sqrt(2.0) * sigma_param /
           std::sqrt(std::log(1.0 / accuracy_in));
  }

  // Set g_ewald for pair style compatibility (pair_lj_cut_coul_user reads it)
  // SOG doesn't use Ewald splitting; set to a safe non-zero value
  g_ewald = 1.0 / sigma_param;

  scale = 1.0;
  qqrd2e = force->qqrd2e;
  qsum_qsq();
  natoms_original = atom->natoms;

  // Print info
  if (comm->me == 0) {
    std::string mesg = "SOG initialization ...\n";
    mesg += fmt::format("  b = {:.6g}, sigma = {:.6g}, M = {}\n", b_param,
                        sigma_param, M_param);
    mesg += fmt::format("  n_dl = {:.6g}, accuracy = {:.6g}\n", n_dl,
                        accuracy_in);
    mesg += fmt::format("  w0 = {:.8g}, rcut = {:.6g}\n", w0, rcut);
    utils::logmesg(lmp, mesg);
  }

  setup();
}

// ──────────────────────────────────────────────────────────────────────
// setup — called whenever volume changes
// ──────────────────────────────────────────────────────────────────────

void SOGKSpace::setup() { ensure_fft_plan(); }

// ──────────────────────────────────────────────────────────────────────
// FFT plan management
// ──────────────────────────────────────────────────────────────────────

void SOGKSpace::destroy_fft_plan() {
  if (mesh_fft != nullptr) {
    delete mesh_fft;
    mesh_fft = nullptr;
  }
  mesh_ready = false;
  mesh_nx = mesh_ny = mesh_nz = 0;
  mesh_lx = mesh_ly = mesh_lz = 0.0;
  mesh_rho.clear();
  mesh_fft_work.clear();
  mesh_gradx.clear();
  mesh_grady.clear();
  mesh_gradz.clear();
  mesh_green_energy.clear();
  mesh_green_force.clear();
  mesh_green_self.clear();
  mesh_green_virial.clear();
  sinc_table_x.clear();
  sinc_table_y.clear();
  sinc_table_z.clear();
  sinc_sum_x.clear();
  sinc_sum_y.clear();
  sinc_sum_z.clear();
  cubes2_influence_re.clear();
  cubes2_influence_im.clear();
  cubes2_influence_sq.clear();
}

size_t SOGKSpace::mesh_index(int ix, int iy, int iz) const {
  return static_cast<size_t>(ix + mesh_nx * (iy + mesh_ny * iz));
}

double SOGKSpace::periodic_fraction(double x, double xlo, double prd) const {
  double frac = (x - xlo) / prd;
  frac -= std::floor(frac);
  return frac;
}

int SOGKSpace::wrap_index(int i, const int n) const { return ::wrap_index(i, n); }

void SOGKSpace::ensure_fft_plan() {
  if (!(domain->xprd > 0.0 && domain->yprd > 0.0 && domain->zprd > 0.0)) {
    error->all(FLERR, "sog encountered non-positive box length");
  }

  const double lx = domain->xprd;
  const double ly = domain->yprd;
  const double lz = domain->zprd;

  // Retrieve rcut once for grid sizing
  int itmp = 0;
  auto *p_cutoff = (double *)force->pair->extract("cut_coul", itmp);
  const double rcut = (p_cutoff != nullptr && *p_cutoff > 0.0)
                          ? *p_cutoff
                          : n_dl;

  int nx, ny, nz;

  if (grid_method == 0 && spline_type > 0) {
    // ── SOG-bandwidth grid estimation ──
    // φ_max from midtown-sog.md Table III (paper JCP 153, 224117):
    //   CubeS₂ 4th, b=2:      φ_max = 0.23
    //   CubeS₂ 4th, b≈1.630:  φ_max = 0.065
    //   CubeS₂ 6th, b=2:      φ_max = 0.35
    //   CubeS₂ 6th, b≈1.630:  φ_max = 0.160
    // Linear interpolation between tabulated b values; floor at b=1.63,
    // cap at b=2.0 to stay within paper's validated range.
    double phi_max;
    if (phi_max_user > 0.0) {
      phi_max = phi_max_user;  // explicit user override
    } else {
      const double b_ref_lo = 1.6297670882677647;  // paper's b≈1.63
      const double phi_lo = (spline_type == 6) ? 0.160 : 0.065;
      const double b_ref_hi = 2.0;
      const double phi_hi = (spline_type == 6) ? 0.350 : 0.230;
      if (b_param <= b_ref_lo) {
        phi_max = phi_lo;
      } else if (b_param >= b_ref_hi) {
        phi_max = phi_hi;
      } else {
        phi_max = phi_lo + (b_param - b_ref_lo) / (b_ref_hi - b_ref_lo) *
                                (phi_hi - phi_lo);
      }
    }

    const double delta = phi_max * rcut;
    nx = std::max(kGridMin, static_cast<int>(std::ceil(lx / delta)));
    ny = std::max(kGridMin, static_cast<int>(std::ceil(ly / delta)));
    nz = std::max(kGridMin, static_cast<int>(std::ceil(lz / delta)));

    if (comm->me == 0 && !mesh_ready)
      utils::logmesg(lmp, fmt::format("  SOG-bandwidth grid: "
                     "phi_max={:.4f} delta={:.6g} -> {}x{}x{}\n",
                     phi_max, delta, nx, ny, nz));
  } else {
    // ── Legacy PPPM-style grid estimation ──
    const double mesh_scale = std::max(1.0, mesh_oversample);
    nx = std::max(kGridMin,
                  static_cast<int>(std::ceil(mesh_scale * 2.0 * lx / n_dl)));
    ny = std::max(kGridMin,
                  static_cast<int>(std::ceil(mesh_scale * 2.0 * ly / n_dl)));
    nz = std::max(kGridMin,
                  static_cast<int>(std::ceil(mesh_scale * 2.0 * lz / n_dl)));

    // PPPM-style accuracy-based refinement (only during initial build)
    if (!mesh_ready && accuracy_in > 0.0 && q2 > 0.0 && atom->natoms > 0) {
      const double volume = lx * ly * lz;
      const double natoms = static_cast<double>(atom->natoms);

      double g_eff =
          accuracy_in * std::sqrt(natoms * rcut * volume) / (2.0 * q2);
      if (!(g_eff > 0.0) || !std::isfinite(g_eff))
        g_eff = MY_2PI / n_dl;
      else if (g_eff >= 1.0)
        g_eff = (1.35 - 0.15 * std::log(accuracy_in)) / rcut;
      else
        g_eff = std::sqrt(-std::log(g_eff)) / rcut;

      if (g_eff > 0.0 && std::isfinite(g_eff)) {
        double hx = 4.0 / g_eff, hy = 4.0 / g_eff, hz = 4.0 / g_eff;
        int nx_pppm = std::max(2, static_cast<int>(lx / hx));
        int ny_pppm = std::max(2, static_cast<int>(ly / hy));
        int nz_pppm = std::max(2, static_cast<int>(lz / hz));
        int count = 0;
        while (true) {
          double err =
              std::max({pppm_ik_error_estimate_order5(hx, lx, atom->natoms, q2,
                                                       g_eff),
                         pppm_ik_error_estimate_order5(hy, ly, atom->natoms, q2,
                                                       g_eff),
                         pppm_ik_error_estimate_order5(hz, lz, atom->natoms, q2,
                                                       g_eff)});
          if (err <= accuracy_in) break;
          if (++count > kGridMaxIter) break;
          hx *= 0.95;
          hy *= 0.95;
          hz *= 0.95;
          nx_pppm = std::max(2, static_cast<int>(lx / hx));
          ny_pppm = std::max(2, static_cast<int>(ly / hy));
          nz_pppm = std::max(2, static_cast<int>(lz / hz));
        }
        nx = std::max(nx, nx_pppm);
        ny = std::max(ny, ny_pppm);
        nz = std::max(nz, nz_pppm);
      }
    }
  }

  while (!factorable_235(nx)) ++nx;
  while (!factorable_235(ny)) ++ny;
  while (!factorable_235(nz)) ++nz;

  // ── Case 1: Nothing changed ──
  if (mesh_ready && mesh_nx == nx && mesh_ny == ny && mesh_nz == nz &&
      std::fabs(mesh_lx - lx) < 1e-12 && std::fabs(mesh_ly - ly) < 1e-12 &&
      std::fabs(mesh_lz - lz) < 1e-12)
    return;

  // ── Case 2: Box changed, grid count unchanged ──
  if (mesh_ready && mesh_nx == nx && mesh_ny == ny && mesh_nz == nz) {
    mesh_lx = lx;
    mesh_ly = ly;
    mesh_lz = lz;
    precompute_green_functions();
    return;
  }

  // ── Case 3: First build or grid count changed ──
  destroy_fft_plan();

  const int64_t ngrid64 =
      static_cast<int64_t>(nx) * static_cast<int64_t>(ny) *
      static_cast<int64_t>(nz);
  if (ngrid64 <= 0 ||
      ngrid64 > static_cast<int64_t>(std::numeric_limits<size_t>::max() / 2))
    error->all(FLERR, "sog mesh grid size overflow");

  mesh_nx = nx;
  mesh_ny = ny;
  mesh_nz = nz;
  mesh_lx = lx;
  mesh_ly = ly;
  mesh_lz = lz;

  const size_t ngrid = static_cast<size_t>(ngrid64);
  mesh_rho.assign(ngrid, 0.0);
  mesh_fft_work.assign(2 * ngrid, 0.0);
  mesh_gradx.assign(2 * ngrid, 0.0);
  mesh_grady.assign(2 * ngrid, 0.0);
  mesh_gradz.assign(2 * ngrid, 0.0);

  int tmp = 0;
  mesh_fft = new FFT3d(lmp, world, mesh_nx, mesh_ny, mesh_nz, 0, mesh_nx - 1,
                        0, mesh_ny - 1, 0, mesh_nz - 1, 0, mesh_nx - 1, 0,
                        mesh_ny - 1, 0, mesh_nz - 1, 0, 0, &tmp,
                        collective_flag);

  if (spline_type >= 4) {
    precompute_cubes2_influence();
  } else {
    precompute_sinc_tables();
  }
  precompute_green_functions();
  mesh_ready = true;
}

// ──────────────────────────────────────────────────────────────────────
// sinc table precomputation (box-independent)
// ──────────────────────────────────────────────────────────────────────

void SOGKSpace::precompute_sinc_tables() {
  const int assign_pow = 2 * kSog_AssignOrder;  // = 10
  const int alias_cnt = 2 * mesh_alias_extent + 1;

  // X
  sinc_table_x.assign(
      static_cast<size_t>(mesh_nx) * static_cast<size_t>(alias_cnt), 0.0);
  sinc_sum_x.assign(static_cast<size_t>(mesh_nx), 0.0);
  for (int ix = 0; ix < mesh_nx; ++ix) {
    const int kx_mode = ix - mesh_nx * (2 * ix / mesh_nx);
    const double arg_base =
        MY_PI * static_cast<double>(kx_mode) / static_cast<double>(mesh_nx);
    const size_t base =
        static_cast<size_t>(ix) * static_cast<size_t>(alias_cnt);
    double sum = 0.0;
    for (int jx = -mesh_alias_extent; jx <= mesh_alias_extent; ++jx) {
      const double arg = arg_base + MY_PI * static_cast<double>(jx);
      const double val = sinc_pow(arg, assign_pow);
      sinc_table_x[base + static_cast<size_t>(jx + mesh_alias_extent)] = val;
      sum += val;
    }
    sinc_sum_x[static_cast<size_t>(ix)] = sum;
  }

  // Y
  sinc_table_y.assign(
      static_cast<size_t>(mesh_ny) * static_cast<size_t>(alias_cnt), 0.0);
  sinc_sum_y.assign(static_cast<size_t>(mesh_ny), 0.0);
  for (int iy = 0; iy < mesh_ny; ++iy) {
    const int ky_mode = iy - mesh_ny * (2 * iy / mesh_ny);
    const double arg_base =
        MY_PI * static_cast<double>(ky_mode) / static_cast<double>(mesh_ny);
    const size_t base =
        static_cast<size_t>(iy) * static_cast<size_t>(alias_cnt);
    double sum = 0.0;
    for (int jy = -mesh_alias_extent; jy <= mesh_alias_extent; ++jy) {
      const double arg = arg_base + MY_PI * static_cast<double>(jy);
      const double val = sinc_pow(arg, assign_pow);
      sinc_table_y[base + static_cast<size_t>(jy + mesh_alias_extent)] = val;
      sum += val;
    }
    sinc_sum_y[static_cast<size_t>(iy)] = sum;
  }

  // Z
  sinc_table_z.assign(
      static_cast<size_t>(mesh_nz) * static_cast<size_t>(alias_cnt), 0.0);
  sinc_sum_z.assign(static_cast<size_t>(mesh_nz), 0.0);
  for (int iz = 0; iz < mesh_nz; ++iz) {
    const int kz_mode = iz - mesh_nz * (2 * iz / mesh_nz);
    const double arg_base =
        MY_PI * static_cast<double>(kz_mode) / static_cast<double>(mesh_nz);
    const size_t base =
        static_cast<size_t>(iz) * static_cast<size_t>(alias_cnt);
    double sum = 0.0;
    for (int jz = -mesh_alias_extent; jz <= mesh_alias_extent; ++jz) {
      const double arg = arg_base + MY_PI * static_cast<double>(jz);
      const double val = sinc_pow(arg, assign_pow);
      sinc_table_z[base + static_cast<size_t>(jz + mesh_alias_extent)] = val;
      sum += val;
    }
    sinc_sum_z[static_cast<size_t>(iz)] = sum;
  }
}

// ──────────────────────────────────────────────────────────────────────
// CubeS₂ influence function precomputation
// ──────────────────────────────────────────────────────────────────────

void SOGKSpace::precompute_cubes2_influence() {
  const size_t ngrid = static_cast<size_t>(mesh_nx) *
                       static_cast<size_t>(mesh_ny) *
                       static_cast<size_t>(mesh_nz);

  if (comm->me == 0)
    utils::logmesg(lmp, fmt::format("  CubeS2 influence: grid {}x{}x{} ngrid={}\n",
                   mesh_nx, mesh_ny, mesh_nz, ngrid));

  cubes2_influence_re.assign(ngrid, 0.0);
  cubes2_influence_im.assign(ngrid, 0.0);
  cubes2_influence_sq.assign(ngrid, 0.0);


  // Precompute monomial expansions for all 32 nodes (heap-allocated to avoid stack overflow)
  auto *node_mono = new CubeS2NodeMonomial[kCubes2NumNodes4];
  const double xi = (spline_type == 4) ? kCubes2Xi4 : kCubes2Xi6;
  for (int k = 0; k < kCubes2NumNodes4; ++k) {
    build_monomials_for_node(kCubes2Nodes4[k], xi, node_mono[k]);
  }
  if (comm->me == 0)
    utils::logmesg(lmp, "  CubeS2 monomials built\n");

  const double twopi_over_x = MY_2PI / mesh_lx;
  const double twopi_over_y = MY_2PI / mesh_ly;
  const double twopi_over_z = MY_2PI / mesh_lz;
  const double dx_grid = mesh_lx / static_cast<double>(mesh_nx);
  const double dy_grid = mesh_ly / static_cast<double>(mesh_ny);
  const double dz_grid = mesh_lz / static_cast<double>(mesh_nz);

  // Precompute 1D integrals I_p for each k-mode per axis
  // I_p(alpha) where alpha = k·Δ (dimensionless)
  std::vector<std::complex<double>> Ipx[4]; // Ipx[p][ix], p=0..3
  for (int p = 0; p < 4; ++p) {
    Ipx[p].resize(static_cast<size_t>(mesh_nx));
  }
  for (int ix = 0; ix < mesh_nx; ++ix) {
    const int kx_mode = ix - mesh_nx * (2 * ix / mesh_nx);
    const double alpha_x = twopi_over_x * static_cast<double>(kx_mode) * dx_grid;
    Ipx[0][static_cast<size_t>(ix)] = I_int_0(alpha_x);
    Ipx[1][static_cast<size_t>(ix)] = I_int_1(alpha_x);
    Ipx[2][static_cast<size_t>(ix)] = I_int_2(alpha_x);
    Ipx[3][static_cast<size_t>(ix)] = I_int_3(alpha_x);
  }
  std::vector<std::complex<double>> Ipy[4];
  for (int p = 0; p < 4; ++p) {
    Ipy[p].resize(static_cast<size_t>(mesh_ny));
  }
  for (int iy = 0; iy < mesh_ny; ++iy) {
    const int ky_mode = iy - mesh_ny * (2 * iy / mesh_ny);
    const double alpha_y = twopi_over_y * static_cast<double>(ky_mode) * dy_grid;
    Ipy[0][static_cast<size_t>(iy)] = I_int_0(alpha_y);
    Ipy[1][static_cast<size_t>(iy)] = I_int_1(alpha_y);
    Ipy[2][static_cast<size_t>(iy)] = I_int_2(alpha_y);
    Ipy[3][static_cast<size_t>(iy)] = I_int_3(alpha_y);
  }
  std::vector<std::complex<double>> Ipz[4];
  for (int p = 0; p < 4; ++p) {
    Ipz[p].resize(static_cast<size_t>(mesh_nz));
  }
  for (int iz = 0; iz < mesh_nz; ++iz) {
    const int kz_mode = iz - mesh_nz * (2 * iz / mesh_nz);
    const double alpha_z = twopi_over_z * static_cast<double>(kz_mode) * dz_grid;
    Ipz[0][static_cast<size_t>(iz)] = I_int_0(alpha_z);
    Ipz[1][static_cast<size_t>(iz)] = I_int_1(alpha_z);
    Ipz[2][static_cast<size_t>(iz)] = I_int_2(alpha_z);
    Ipz[3][static_cast<size_t>(iz)] = I_int_3(alpha_z);
  }

  if (comm->me == 0)
    utils::logmesg(lmp, "  CubeS2 1D integrals done, starting 3D loop\n");

  // For each k-mode, compute Φ(k) = Σ_d exp(i·k·d·Δ) · Σ_{a,b,c} C_d · I_a · I_b · I_c
  int loop_count = 0;
  for (int iz = 0; iz < mesh_nz; ++iz) {
    const int kz_mode = iz - mesh_nz * (2 * iz / mesh_nz);
    const double kz = twopi_over_z * static_cast<double>(kz_mode);

    for (int iy = 0; iy < mesh_ny; ++iy) {
      const int ky_mode = iy - mesh_ny * (2 * iy / mesh_ny);
      const double ky = twopi_over_y * static_cast<double>(ky_mode);

      for (int ix = 0; ix < mesh_nx; ++ix) {
        const int kx_mode = ix - mesh_nx * (2 * ix / mesh_nx);
        const double kx = twopi_over_x * static_cast<double>(kx_mode);

        // Skip k=0 mode (DC component)
        const double sqk = kx * kx + ky * ky + kz * kz;
        if (sqk == 0.0) continue;

        const size_t idx = mesh_index(ix, iy, iz);
        std::complex<double> phi_k(0.0, 0.0);

        for (int d = 0; d < kCubes2NumNodes4; ++d) {
          const auto &node = kCubes2Nodes4[d];
          const auto &mono = node_mono[d];

          // exp(i·k·d·Δ)
          const double phase = kx * static_cast<double>(node.dx) * dx_grid +
                               ky * static_cast<double>(node.dy) * dy_grid +
                               kz * static_cast<double>(node.dz) * dz_grid;
          const std::complex<double> eikd(std::cos(phase), std::sin(phase));

          // Σ_{a,b,c} C_d(a,b,c) · I_a(α_x) · I_b(α_y) · I_c(α_z)
          std::complex<double> integral(0.0, 0.0);
          for (int m = 0; m < mono.num_terms; ++m) {
            const auto &term = mono.terms[m];
            const std::complex<double> prod =
              Ipx[term.px][static_cast<size_t>(ix)] *
              Ipy[term.py][static_cast<size_t>(iy)] *
              Ipz[term.pz][static_cast<size_t>(iz)];
            integral += term.coeff * prod;
          }
          phi_k += eikd * integral;
        }

        cubes2_influence_re[idx] = phi_k.real();
        cubes2_influence_im[idx] = phi_k.imag();
        const double abs_sq = phi_k.real() * phi_k.real() +
                              phi_k.imag() * phi_k.imag();
        cubes2_influence_sq[idx] = abs_sq;
      }
    }
  }

  delete[] node_mono;
  if (comm->me == 0)
    utils::logmesg(lmp, "  CubeS2 influence done\n");
}

// ──────────────────────────────────────────────────────────────────────
// Green function precomputation
// ──────────────────────────────────────────────────────────────────────

void SOGKSpace::precompute_green_functions() {
  const size_t ngrid = static_cast<size_t>(mesh_nx) *
                       static_cast<size_t>(mesh_ny) *
                       static_cast<size_t>(mesh_nz);

  mesh_green_energy.assign(ngrid, 0.0);
  mesh_green_force.assign(ngrid, 0.0);
  mesh_green_self.assign(ngrid, 0.0);
  mesh_green_virial.assign(ngrid, 0.0);

  // k_sq_max: cutoff for which k‑modes contribute.
  // CubeS₂ (spline_type ≥ 4): use grid Nyquist — all principal modes.
  // B‑spline (spline_type == 0): use n_dl to bound alias loop cost.
  double k_sq_max;
  if (spline_type >= 4) {
    const double dx = mesh_lx / static_cast<double>(mesh_nx);
    const double dy = mesh_ly / static_cast<double>(mesh_ny);
    const double dz = mesh_lz / static_cast<double>(mesh_nz);
    k_sq_max = MY_PI * MY_PI * (1.0 / (dx * dx) + 1.0 / (dy * dy) +
                                 1.0 / (dz * dz));
  } else {
    k_sq_max = (MY_2PI / n_dl) * (MY_2PI / n_dl);
  }

  const double twopi_over_x = MY_2PI / mesh_lx;
  const double twopi_over_y = MY_2PI / mesh_ly;
  const double twopi_over_z = MY_2PI / mesh_lz;
  const int alias_extent = mesh_alias_extent;
  const int alias_cnt = 2 * alias_extent + 1;

  // Alias fast-path check
  const double k_max = std::sqrt(k_sq_max);
  const bool alias_fast_path =
      (twopi_over_x * static_cast<double>(mesh_nx) > 2.0 * k_max) &&
      (twopi_over_y * static_cast<double>(mesh_ny) > 2.0 * k_max) &&
      (twopi_over_z * static_cast<double>(mesh_nz) > 2.0 * k_max);

  for (int iz = 0; iz < mesh_nz; ++iz) {
    const int kz_mode = iz - mesh_nz * (2 * iz / mesh_nz);
    const double kz = twopi_over_z * static_cast<double>(kz_mode);

    for (int iy = 0; iy < mesh_ny; ++iy) {
      const int ky_mode = iy - mesh_ny * (2 * iy / mesh_ny);
      const double ky = twopi_over_y * static_cast<double>(ky_mode);

      for (int ix = 0; ix < mesh_nx; ++ix) {
        const int kx_mode = ix - mesh_nx * (2 * ix / mesh_nx);
        const double kx = twopi_over_x * static_cast<double>(kx_mode);

        const double sqk = kx * kx + ky * ky + kz * kz;
        if (!(sqk > 0.0 && sqk <= k_sq_max)) continue;

        const size_t idx = mesh_index(ix, iy, iz);

        if (spline_type >= 4) {
          // ── CubeS₂ Green function (alias fast path) ──
          const double kfac = spectral_kernel(sqk);
          const double inf_sq = cubes2_influence_sq[idx];
          if (!(inf_sq > 1e-20) || !std::isfinite(inf_sq)) continue;

          mesh_green_energy[idx] = kfac / inf_sq;
          mesh_green_force[idx] = kfac / inf_sq;  // alias fast path
          mesh_green_self[idx] = kfac;
          mesh_green_virial[idx] = spectral_kernel_virial(sqk) / inf_sq;
          continue;
        }

        // ── Legacy B-spline Green function ──
        const double sz_sum = sinc_sum_z[static_cast<size_t>(iz)];
        const double sy_sum = sinc_sum_y[static_cast<size_t>(iy)];
        const double sx_sum = sinc_sum_x[static_cast<size_t>(ix)];

        const double denom_lin = sx_sum * sy_sum * sz_sum;
        const double denominator = denom_lin * denom_lin;
        if (!(denominator > 1e-20) || !std::isfinite(denominator)) continue;

        const size_t tbl_base_x =
            static_cast<size_t>(ix) * static_cast<size_t>(alias_cnt);
        const size_t tbl_base_y =
            static_cast<size_t>(iy) * static_cast<size_t>(alias_cnt);
        const size_t tbl_base_z =
            static_cast<size_t>(iz) * static_cast<size_t>(alias_cnt);

        if (alias_fast_path) {
          // Only principal mode (j=0,0,0) contributes
          const double w2x =
              sinc_table_x[tbl_base_x + static_cast<size_t>(alias_extent)];
          const double w2y =
              sinc_table_y[tbl_base_y + static_cast<size_t>(alias_extent)];
          const double w2z =
              sinc_table_z[tbl_base_z + static_cast<size_t>(alias_extent)];
          const double w2_principal = w2x * w2y * w2z;

          const double kfac = spectral_kernel(sqk);
          const double sum0 = kfac * w2_principal;

          const double geff_energy = sum0 / denominator;
          mesh_green_energy[idx] = geff_energy;
          mesh_green_force[idx] = geff_energy;  // geff = geff_energy in fast path
          mesh_green_virial[idx] =
              spectral_kernel_virial(sqk) * w2_principal / denominator;
          if (w2_principal > 1e-20) {
            mesh_green_self[idx] = sum0 / w2_principal;
          }
        } else {
          // Full alias loop
          double sum0 = 0.0, sum1 = 0.0, sum_virial = 0.0;
          for (int jx = -alias_extent; jx <= alias_extent; ++jx) {
            const double qx =
                twopi_over_x * static_cast<double>(kx_mode + mesh_nx * jx);
            const double wx_alias =
                sinc_table_x[tbl_base_x +
                             static_cast<size_t>(jx + alias_extent)];

            for (int jy = -alias_extent; jy <= alias_extent; ++jy) {
              const double qy =
                  twopi_over_y * static_cast<double>(ky_mode + mesh_ny * jy);
              const double wy_alias =
                  sinc_table_y[tbl_base_y +
                               static_cast<size_t>(jy + alias_extent)];

              for (int jz = -alias_extent; jz <= alias_extent; ++jz) {
                const double qz =
                    twopi_over_z * static_cast<double>(kz_mode + mesh_nz * jz);
                const double wz_alias =
                    sinc_table_z[tbl_base_z +
                                 static_cast<size_t>(jz + alias_extent)];

                const double qsq = qx * qx + qy * qy + qz * qz;
                if (!(qsq > 0.0 && qsq <= k_sq_max)) continue;

                const double kfac_alias = spectral_kernel(qsq);
                if (!std::isfinite(kfac_alias) || kfac_alias == 0.0) continue;

                const double wprod = wx_alias * wy_alias * wz_alias;
                sum0 += kfac_alias * wprod;
                const double dot1 = kx * qx + ky * qy + kz * qz;
                sum1 += dot1 * kfac_alias * wprod;

                // Virial: sum over aliases of K_v(q²) * wprod
                const double kfac_virial = spectral_kernel_virial(qsq);
                if (std::isfinite(kfac_virial) && kfac_virial != 0.0) {
                  sum_virial += kfac_virial * wprod;
                }
              }
            }
          }

          const double geff_energy = sum0 / denominator;
          const double geff = sum1 / (sqk * denominator);
          if (!std::isfinite(geff_energy) || !std::isfinite(geff)) continue;

          mesh_green_energy[idx] = geff_energy;
          mesh_green_force[idx] = geff;
          mesh_green_virial[idx] = sum_virial / denominator;

          const double w2x =
              sinc_table_x[tbl_base_x + static_cast<size_t>(alias_extent)];
          const double w2y =
              sinc_table_y[tbl_base_y + static_cast<size_t>(alias_extent)];
          const double w2z =
              sinc_table_z[tbl_base_z + static_cast<size_t>(alias_extent)];
          const double w2_principal = w2x * w2y * w2z;
          if (w2_principal > 1e-20) {
            mesh_green_self[idx] = sum0 / w2_principal;
          }
        }
      }
    }
  }
}

// ── Multi-channel wrapper ──
void SOGKSpace::compute(int eflag, int vflag) {
  auto *pair_dp = dynamic_cast<PairDeepMD *>(force->pair);
  const int nchannels =
      (pair_dp != nullptr) ? pair_dp->ncharge_channels : 1;

  if (nchannels <= 1) {
    compute_single(eflag, vflag);
    return;
  }

  const int nlocal = atom->nlocal;
  std::vector<double> q_orig(nlocal);
  std::vector<std::array<double, 3>> f_orig(nlocal);
  for (int i = 0; i < nlocal; ++i) {
    q_orig[i] = atom->q[i];
    f_orig[i][0] = atom->f[i][0];
    f_orig[i][1] = atom->f[i][1];
    f_orig[i][2] = atom->f[i][2];
  }

  energy = 0.0;
  for (int j = 0; j < 6; ++j) virial[j] = 0.0;

  for (int ch = 0; ch < nchannels; ++ch) {
    for (int i = 0; i < nlocal; ++i)
      atom->q[i] = pair_dp->dcharge_multi[i * nchannels + ch];
    for (int i = 0; i < nlocal; ++i) {
      atom->f[i][0] = 0.0; atom->f[i][1] = 0.0; atom->f[i][2] = 0.0;
    }
    compute_single(eflag, vflag);
    for (int i = 0; i < nlocal; ++i) {
      f_orig[i][0] += atom->f[i][0];
      f_orig[i][1] += atom->f[i][1];
      f_orig[i][2] += atom->f[i][2];
    }
  }

  for (int i = 0; i < nlocal; ++i) {
    atom->q[i] = q_orig[i];
    atom->f[i][0] = f_orig[i][0];
    atom->f[i][1] = f_orig[i][1];
    atom->f[i][2] = f_orig[i][2];
  }
}

// ──────────────────────────────────────────────────────────────────────
// compute — per-step force/energy/virial
// ──────────────────────────────────────────────────────────────────────

void SOGKSpace::compute_single(int eflag, int vflag) {
  ev_init(eflag, vflag, 0);

  if (atom->natoms != natoms_original) {
    qsum_qsq();
    natoms_original = atom->natoms;
  }

  if (qsqsum == 0.0) {
    energy = 0.0;
    for (int j = 0; j < 6; ++j) virial[j] = 0.0;
    return;
  }

  ensure_fft_plan();

  const bool want_energy = (eflag & ENERGY_GLOBAL);
  const bool want_virial = (vflag & (VIRIAL_PAIR | VIRIAL_FDOTR));

  energy = 0.0;
  for (int j = 0; j < 6; ++j) virial[j] = 0.0;

  const int nlocal = atom->nlocal;
  if (nlocal <= 0) return;

  // ── Clear mesh arrays ──
  std::fill(mesh_rho.begin(), mesh_rho.end(), 0.0);
  std::fill(mesh_fft_work.begin(), mesh_fft_work.end(), 0.0);
  std::fill(mesh_gradx.begin(), mesh_gradx.end(), 0.0);
  std::fill(mesh_grady.begin(), mesh_grady.end(), 0.0);
  std::fill(mesh_gradz.begin(), mesh_gradz.end(), 0.0);

  const double volume = mesh_lx * mesh_ly * mesh_lz;
  const double rho_scale =
      static_cast<double>(mesh_nx * mesh_ny * mesh_nz) / volume;

  constexpr int assign_order = kSog_AssignOrder;
  constexpr int assign_half = (kSog_AssignOrder - 1) / 2;  // = 2

  double **x = atom->x;
  double *q = atom->q;

  // ── Charge spreading ──

  if (spline_type >= 4) {
    // CubeS₂ charge spreading
    const int num_nodes = (spline_type == 4) ? kCubes2NumNodes4 : kCubes2NumNodes6;
    const double xi = (spline_type == 4) ? kCubes2Xi4 : kCubes2Xi6;

    for (int i = 0; i < nlocal; ++i) {
      const double fx = periodic_fraction(x[i][0], domain->boxlo[0], mesh_lx) *
                        static_cast<double>(mesh_nx);
      const double fy = periodic_fraction(x[i][1], domain->boxlo[1], mesh_ly) *
                        static_cast<double>(mesh_ny);
      const double fz = periodic_fraction(x[i][2], domain->boxlo[2], mesh_lz) *
                        static_cast<double>(mesh_nz);

      const int ix0 = static_cast<int>(std::floor(fx));
      const int iy0 = static_cast<int>(std::floor(fy));
      const int iz0 = static_cast<int>(std::floor(fz));
      const double tx = fx - static_cast<double>(ix0);
      const double ty = fy - static_cast<double>(iy0);
      const double tz = fz - static_cast<double>(iz0);

      const double q_scaled = rho_scale * q[i];

      if (spline_type == 4) {
        for (int k = 0; k < kCubes2NumNodes4; ++k) {
          const auto &node = kCubes2Nodes4[k];
          const double w = cubes2_weight_4(tx, ty, tz, node, xi);
          if (w == 0.0) continue;
          const int igx = wrap_index(ix0 + node.dx, mesh_nx);
          const int igy = wrap_index(iy0 + node.dy, mesh_ny);
          const int igz = wrap_index(iz0 + node.dz, mesh_nz);
          const size_t idx = mesh_index(igx, igy, igz);
          mesh_rho[idx] += static_cast<FFT_SCALAR>(q_scaled * w);
        }
      }
      // 6th order placeholder — will be filled when 88-node set is defined
    }
  } else {
    // Legacy B-spline charge spreading (order 5)
    for (int i = 0; i < nlocal; ++i) {
      const double fx = periodic_fraction(x[i][0], domain->boxlo[0], mesh_lx) *
                        static_cast<double>(mesh_nx);
      const double fy = periodic_fraction(x[i][1], domain->boxlo[1], mesh_ly) *
                        static_cast<double>(mesh_ny);
      const double fz = periodic_fraction(x[i][2], domain->boxlo[2], mesh_lz) *
                        static_cast<double>(mesh_nz);

      const int ix0 = static_cast<int>(std::floor(fx));
      const int iy0 = static_cast<int>(std::floor(fy));
      const int iz0 = static_cast<int>(std::floor(fz));
      const double tx = fx - static_cast<double>(ix0);
      const double ty = fy - static_cast<double>(iy0);
      const double tz = fz - static_cast<double>(iz0);

      std::array<double, assign_order> wx, wy, wz;
      sog_bspline_weights_1d(tx, wx);
      sog_bspline_weights_1d(ty, wy);
      sog_bspline_weights_1d(tz, wz);

      std::array<int, assign_order> ix, iy, iz;
      for (int a = 0; a < assign_order; ++a) {
        ix[static_cast<size_t>(a)] =
            wrap_index(ix0 - assign_half + a, mesh_nx);
        iy[static_cast<size_t>(a)] =
            wrap_index(iy0 - assign_half + a, mesh_ny);
        iz[static_cast<size_t>(a)] =
            wrap_index(iz0 - assign_half + a, mesh_nz);
      }

      for (int a = 0; a < assign_order; ++a) {
        for (int b = 0; b < assign_order; ++b) {
          for (int c = 0; c < assign_order; ++c) {
            const size_t idx = mesh_index(ix[a], iy[b], iz[c]);
            mesh_rho[idx] += static_cast<FFT_SCALAR>(rho_scale * q[i] * wx[a] *
                                                      wy[b] * wz[c]);
          }
        }
      }
    }
  }

  // ── FFT forward ──
  const size_t ngrid = mesh_rho.size();
  for (size_t idx = 0; idx < ngrid; ++idx) {
    mesh_fft_work[2 * idx] = mesh_rho[idx];
    mesh_fft_work[2 * idx + 1] = 0.0;
  }
  mesh_fft->compute(mesh_fft_work.data(), mesh_fft_work.data(),
                    FFT3d::FORWARD);

  // ── k-space loop: multiply by Green functions, compute gradients ──
  const double scaleinv = 1.0 / static_cast<double>(ngrid);
  const double s2 = scaleinv * scaleinv;
  const double twopi_over_x = MY_2PI / mesh_lx;
  const double twopi_over_y = MY_2PI / mesh_ly;
  const double twopi_over_z = MY_2PI / mesh_lz;

  double energy_local = 0.0;
  double diag_sum_local = 0.0;
  std::array<double, 6> fv_local = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

  for (int iz = 0; iz < mesh_nz; ++iz) {
    const int kz_mode = iz - mesh_nz * (2 * iz / mesh_nz);
    const double kz = twopi_over_z * static_cast<double>(kz_mode);

    for (int iy = 0; iy < mesh_ny; ++iy) {
      const int ky_mode = iy - mesh_ny * (2 * iy / mesh_ny);
      const double ky = twopi_over_y * static_cast<double>(ky_mode);

      for (int ix = 0; ix < mesh_nx; ++ix) {
        const int kx_mode = ix - mesh_nx * (2 * ix / mesh_nx);
        const double kx = twopi_over_x * static_cast<double>(kx_mode);

        const size_t idx = mesh_index(ix, iy, iz);

        const double geff_energy = mesh_green_energy[idx];
        const double geff = mesh_green_force[idx];

        if (geff == 0.0 && geff_energy == 0.0) {
          mesh_gradx[2 * idx] = mesh_gradx[2 * idx + 1] = 0.0;
          mesh_grady[2 * idx] = mesh_grady[2 * idx + 1] = 0.0;
          mesh_gradz[2 * idx] = mesh_gradz[2 * idx + 1] = 0.0;
          continue;
        }

        diag_sum_local += mesh_green_self[idx];

        const double rho_re =
            static_cast<double>(mesh_fft_work[2 * idx]);
        const double rho_im =
            static_cast<double>(mesh_fft_work[2 * idx + 1]);

        if (want_energy) {
          energy_local +=
              s2 * geff_energy * (rho_re * rho_re + rho_im * rho_im);
        }

        // ── Fourier-space virial (SOG-correct formula) ──
        // W_{αβ}(k) = s2 * |ρ|² * (ge * δ_{αβ} - gv * k_α * k_β)
        //   ge = mesh_green_energy = K(k²)/|Φ|²
        //   gv = mesh_green_virial = K_v(k²)/|Φ|²
        //   K(k²)  = Σ amp[m] · exp(-band[m]·k²/2)
        //   K_v(k²)= Σ amp[m]·band[m]·exp(-band[m]·k²/2)
        // Ref: rbsog-npt Eq. 3.8, rbsog_intel.cpp virial loop
        if (want_virial) {
          const double rho2 = rho_re * rho_re + rho_im * rho_im;
          const double gv = mesh_green_virial[idx];
          // geff_energy already loaded as mesh_green_energy[idx]
          fv_local[0] += s2 * rho2 * (geff_energy - gv * kx * kx);
          fv_local[1] += s2 * rho2 * (geff_energy - gv * ky * ky);
          fv_local[2] += s2 * rho2 * (geff_energy - gv * kz * kz);
          fv_local[3] += s2 * rho2 * (-gv * kx * ky);
          fv_local[4] += s2 * rho2 * (-gv * kx * kz);
          fv_local[5] += s2 * rho2 * (-gv * ky * kz);
        }

        const double vk_re = scaleinv * geff * rho_re;
        const double vk_im = scaleinv * geff * rho_im;

        mesh_gradx[2 * idx] = static_cast<FFT_SCALAR>(-kx * vk_im);
        mesh_gradx[2 * idx + 1] = static_cast<FFT_SCALAR>(kx * vk_re);
        mesh_grady[2 * idx] = static_cast<FFT_SCALAR>(-ky * vk_im);
        mesh_grady[2 * idx + 1] = static_cast<FFT_SCALAR>(ky * vk_re);
        mesh_gradz[2 * idx] = static_cast<FFT_SCALAR>(-kz * vk_im);
        mesh_gradz[2 * idx + 1] = static_cast<FFT_SCALAR>(kz * vk_re);
      }
    }
  }

  // ── FFT backward (3 gradients) ──
  mesh_fft->compute(mesh_gradx.data(), mesh_gradx.data(), FFT3d::BACKWARD);
  mesh_fft->compute(mesh_grady.data(), mesh_grady.data(), FFT3d::BACKWARD);
  mesh_fft->compute(mesh_gradz.data(), mesh_gradz.data(), FFT3d::BACKWARD);

  // ── Force interpolation ──
  const double qscale = force->qqrd2e * scale;
  std::array<double, 6> virial_local = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

  if (spline_type >= 4) {
    // CubeS₂ force interpolation
    const int num_nodes = (spline_type == 4) ? kCubes2NumNodes4 : kCubes2NumNodes6;
    const double xi = (spline_type == 4) ? kCubes2Xi4 : kCubes2Xi6;

    for (int i = 0; i < nlocal; ++i) {
      const double fx = periodic_fraction(x[i][0], domain->boxlo[0], mesh_lx) *
                        static_cast<double>(mesh_nx);
      const double fy = periodic_fraction(x[i][1], domain->boxlo[1], mesh_ly) *
                        static_cast<double>(mesh_ny);
      const double fz = periodic_fraction(x[i][2], domain->boxlo[2], mesh_lz) *
                        static_cast<double>(mesh_nz);

      const int ix0 = static_cast<int>(std::floor(fx));
      const int iy0 = static_cast<int>(std::floor(fy));
      const int iz0 = static_cast<int>(std::floor(fz));
      const double tx = fx - static_cast<double>(ix0);
      const double ty = fy - static_cast<double>(iy0);
      const double tz = fz - static_cast<double>(iz0);

      double gx = 0.0, gy = 0.0, gz = 0.0;

      if (spline_type == 4) {
        for (int k = 0; k < kCubes2NumNodes4; ++k) {
          const auto &node = kCubes2Nodes4[k];
          const double w = cubes2_weight_4(tx, ty, tz, node, xi);
          if (w == 0.0) continue;
          const int igx = wrap_index(ix0 + node.dx, mesh_nx);
          const int igy = wrap_index(iy0 + node.dy, mesh_ny);
          const int igz = wrap_index(iz0 + node.dz, mesh_nz);
          const size_t idx = mesh_index(igx, igy, igz);
          gx += w * static_cast<double>(mesh_gradx[2 * idx]);
          gy += w * static_cast<double>(mesh_grady[2 * idx]);
          gz += w * static_cast<double>(mesh_gradz[2 * idx]);
        }
      }
      // 6th order placeholder

      const double qi = q[i];
      const double fxs = -qscale * qi * gx;
      const double fys = -qscale * qi * gy;
      const double fzs = -qscale * qi * gz;

      atom->f[i][0] += fxs;
      atom->f[i][1] += fys;
      atom->f[i][2] += fzs;

      if (want_virial) {
        virial_local[0] += x[i][0] * fxs;
        virial_local[1] += x[i][1] * fys;
        virial_local[2] += x[i][2] * fzs;
        virial_local[3] += x[i][0] * fys;
        virial_local[4] += x[i][0] * fzs;
        virial_local[5] += x[i][1] * fzs;
      }
    }
  } else {
    // Legacy B-spline force interpolation
    for (int i = 0; i < nlocal; ++i) {
      const double fx = periodic_fraction(x[i][0], domain->boxlo[0], mesh_lx) *
                        static_cast<double>(mesh_nx);
      const double fy = periodic_fraction(x[i][1], domain->boxlo[1], mesh_ly) *
                        static_cast<double>(mesh_ny);
      const double fz = periodic_fraction(x[i][2], domain->boxlo[2], mesh_lz) *
                        static_cast<double>(mesh_nz);

      const int ix0 = static_cast<int>(std::floor(fx));
      const int iy0 = static_cast<int>(std::floor(fy));
      const int iz0 = static_cast<int>(std::floor(fz));
      const double tx = fx - static_cast<double>(ix0);
      const double ty = fy - static_cast<double>(iy0);
      const double tz = fz - static_cast<double>(iz0);

      std::array<double, assign_order> wx, wy, wz;
      sog_bspline_weights_1d(tx, wx);
      sog_bspline_weights_1d(ty, wy);
      sog_bspline_weights_1d(tz, wz);

      std::array<int, assign_order> ix, iy, iz;
      for (int a = 0; a < assign_order; ++a) {
        ix[static_cast<size_t>(a)] =
            wrap_index(ix0 - assign_half + a, mesh_nx);
        iy[static_cast<size_t>(a)] =
            wrap_index(iy0 - assign_half + a, mesh_ny);
        iz[static_cast<size_t>(a)] =
            wrap_index(iz0 - assign_half + a, mesh_nz);
      }

      double gx = 0.0, gy = 0.0, gz = 0.0;
      for (int a = 0; a < assign_order; ++a) {
        for (int b = 0; b < assign_order; ++b) {
          for (int c = 0; c < assign_order; ++c) {
            const size_t idx = mesh_index(ix[a], iy[b], iz[c]);
            const double w = wx[a] * wy[b] * wz[c];
            gx += w * static_cast<double>(mesh_gradx[2 * idx]);
            gy += w * static_cast<double>(mesh_grady[2 * idx]);
            gz += w * static_cast<double>(mesh_gradz[2 * idx]);
          }
        }
      }

      const double qi = q[i];
      const double fxs = -qscale * qi * gx;
      const double fys = -qscale * qi * gy;
      const double fzs = -qscale * qi * gz;

      atom->f[i][0] += fxs;
      atom->f[i][1] += fys;
      atom->f[i][2] += fzs;

      if (want_virial) {
        virial_local[0] += x[i][0] * fxs;
        virial_local[1] += x[i][1] * fys;
        virial_local[2] += x[i][2] * fzs;
        virial_local[3] += x[i][0] * fys;
        virial_local[4] += x[i][0] * fzs;
        virial_local[5] += x[i][1] * fzs;
      }
    }
  }

  // ── Energy ──
  if (want_energy) {
    double energy_all = 0.0;
    MPI_Allreduce(&energy_local, &energy_all, 1, MPI_DOUBLE, MPI_SUM, world);

    double diag_sum_all = 0.0;
    MPI_Allreduce(&diag_sum_local, &diag_sum_all, 1, MPI_DOUBLE, MPI_SUM,
                  world);

    double qsqsum_local = 0.0;
    for (int i = 0; i < nlocal; ++i) qsqsum_local += q[i] * q[i];
    double qsqsum_all = 0.0;
    MPI_Allreduce(&qsqsum_local, &qsqsum_all, 1, MPI_DOUBLE, MPI_SUM, world);

    energy = 0.5 * volume * energy_all;
    if (remove_self_interaction) {
      const double self_term = diag_sum_all / (2.0 * volume);
      energy -= qsqsum_all * self_term;
    }
    // Also subtract the real-space self-energy (matching RBSOG convention)
    energy -= self_coeff * qsqsum_all;

    // ── k=0 corrections (k=0 excluded by spectral_kernel(0)=0) ──
    // Do NOT use raw amp_sum at k=0: amp[m] grows as b^(2m) (geometric growth),
    // and at k=0, exp(0)=1 for ALL m → dominated by high-m "inactive" terms.
    // At any finite k, exp decay suppresses high-m terms. Use the regularized
    // kernel value at the smallest physical |k| for consistency with k≠0 energy.
    // k_min is determined by the reciprocal lattice vectors:
    //   b_i = 2π · (a_j × a_k) / V,  k_min = min(|b1|, |b2|, |b3|)
    // For orthorhombic: k_min = 2π / max(Lx, Ly, Lz).
    {
      double Lx = boxhi[0] - boxlo[0];
      double Ly = boxhi[1] - boxlo[1];
      double Lz = boxhi[2] - boxlo[2];
      double k_min_sq;

      if (domain->triclinic) {
        // Lattice vectors (LAMMPS convention):
        //   a1 = [Lx, 0, 0], a2 = [xy, Ly, 0], a3 = [xz, yz, Lz]
        double xy = domain->xy, xz_d = domain->xz, yz = domain->yz;
        double V = Lx * Ly * Lz;
        double twopi_over_V = 2.0 * M_PI / V;

        // |b1|² from a2×a3 = [Ly·Lz, −xy·Lz, xy·yz − Ly·xz]
        double b1_sq = (Ly*Lz)*(Ly*Lz) + (xy*Lz)*(xy*Lz)
                     + (xy*yz - Ly*xz_d)*(xy*yz - Ly*xz_d);
        b1_sq *= twopi_over_V * twopi_over_V;

        // |b2|² from a3×a1 = [0, Lz·Lx, −yz·Lx]
        double b2_sq = Lx*Lx * (Lz*Lz + yz*yz);
        b2_sq *= twopi_over_V * twopi_over_V;

        // |b3|² from a1×a2 = [0, 0, Lx·Ly]
        double b3_sq = Lx*Lx * Ly*Ly * twopi_over_V * twopi_over_V;

        k_min_sq = std::min({b1_sq, b2_sq, b3_sq});
      } else {
        double L_max = std::max({Lx, Ly, Lz});
        k_min_sq = (2.0 * M_PI / L_max) * (2.0 * M_PI / L_max);
      }

      double kfac_eff = 0.0;
      for (size_t m = 0; m < amp.size(); ++m) {
        kfac_eff += amp[m] * std::exp(-0.5 * bandwidth[m] * k_min_sq);
      }
      // Charge cross-term (always on, physical): +A_eff·Q²/(2V)
      energy += kfac_eff * qsum * qsum / (2.0 * volume);
      // k=0 self term (tied to remove_self_interaction): −A_eff·Σq_i²/(2V)
      if (remove_self_interaction) {
        energy -= kfac_eff * qsqsum_all / (2.0 * volume);
      }
    }

    energy *= qscale;
  }

  // ── Virial ──
  if (want_virial) {
    double vr_all[6] = {0.0}, vf_all[6] = {0.0};

    // Force·r virial (diagnostic only)
    MPI_Allreduce(virial_local.data(), vr_all, 6, MPI_DOUBLE, MPI_SUM, world);

    // Fourier-space virial (primary, matches rbsog_intel & PPPM convention)
    MPI_Allreduce(fv_local.data(), vf_all, 6, MPI_DOUBLE, MPI_SUM, world);
    // Scale Fourier virial: W = 0.5 * V * qscale * Σ s2 * |ρ|² * (ge·I - gv·k⊗k)
    const double virial_scale = 0.5 * volume * qscale;
    for (int j = 0; j < 6; ++j) virial[j] = virial_scale * vf_all[j];

    // Diagnostic: compare force·r vs Fourier virial
    double max_rel_diff = 0.0;
    for (int j = 0; j < 6; ++j) {
      double denom = std::max(std::fabs(vr_all[j]), std::fabs(virial[j]));
      if (denom > 0.0) {
        double rd = std::fabs(vr_all[j] - virial[j]) / denom;
        if (rd > max_rel_diff) max_rel_diff = rd;
      }
    }
    if (comm->me == 0 && max_rel_diff > 0.0001) {
      std::string msg = fmt::format(
          "  SOG virial: max |force·r - Fourier|/max = {:.6f}\n"
          "    force·r: {:12.4f} {:12.4f} {:12.4f} {:12.4f} {:12.4f} {:12.4f}\n"
          "    Fourier: {:12.4f} {:12.4f} {:12.4f} {:12.4f} {:12.4f} {:12.4f}\n",
          max_rel_diff,
          vr_all[0], vr_all[1], vr_all[2], vr_all[3], vr_all[4], vr_all[5],
          virial[0], virial[1], virial[2], virial[3], virial[4], virial[5]);
      utils::logmesg(lmp, msg);
    }
  }
}

// ──────────────────────────────────────────────────────────────────────
// memory_usage
// ──────────────────────────────────────────────────────────────────────

double SOGKSpace::memory_usage() {
  double bytes = 0.0;
  bytes += mesh_rho.capacity() * sizeof(FFT_SCALAR);
  bytes += mesh_fft_work.capacity() * sizeof(FFT_SCALAR);
  bytes += mesh_gradx.capacity() * sizeof(FFT_SCALAR);
  bytes += mesh_grady.capacity() * sizeof(FFT_SCALAR);
  bytes += mesh_gradz.capacity() * sizeof(FFT_SCALAR);
  bytes += mesh_green_energy.capacity() * sizeof(double);
  bytes += mesh_green_force.capacity() * sizeof(double);
  bytes += mesh_green_self.capacity() * sizeof(double);
  bytes += sinc_table_x.capacity() * sizeof(double);
  bytes += sinc_table_y.capacity() * sizeof(double);
  bytes += sinc_table_z.capacity() * sizeof(double);
  bytes += sinc_sum_x.capacity() * sizeof(double);
  bytes += sinc_sum_y.capacity() * sizeof(double);
  bytes += sinc_sum_z.capacity() * sizeof(double);
  bytes += amp.capacity() * sizeof(double);
  bytes += bandwidth.capacity() * sizeof(double);
  return bytes;
}
