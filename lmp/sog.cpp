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
// GPU path (raw CUDA + cuFFT) is a compile-time option. The deepmd-kit plugin build defines
// SOG_ENABLE_GPU and ships sog_gpu.cu/.cuh; this standalone sog/lmp build is CPU-only (CXX only),
// so the GPU include and call sites are compiled out. Algorithm otherwise identical to deepmd-kit.
#ifdef SOG_ENABLE_GPU
#include "sog_gpu.cuh"
#endif
#include "sog_spline.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "fft3d_wrap.h"
#include "force.h"
#include "math_const.h"
#include "pair.h"
// Multi-channel latent charges couple to the deepmd pair (ncharge_channels). The deepmd-kit plugin
// build defines SOG_WITH_DEEPMD; this standalone build is single-channel and omits the coupling.
#ifdef SOG_WITH_DEEPMD
#include "pair_deepmd.h"
#endif

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
  if (narg < 1)
    error->all(FLERR,
               "Illegal kspace_style sog command: need arguments (legacy "
               "'accuracy b sigma M [n_dl] [options]' or the clean form "
               "'amp ... bandwidth ... [options]')");

  // Reset optionals to defaults before parsing
  n_dl = -1.0;  // auto‑compute
  remove_self_interaction = false;
  mesh_oversample = 1.5;
  mesh_alias_extent = kAliasExtent;
  spline_type = 4;   // CubeS₂ 4th
  grid_method = 0;   // SOG bandwidth
  phi_max_user = -1.0;  // auto-compute
  phi_accuracy_user = -1.0;  // default per-order ε

  // The leading positional header 'accuracy b sigma M [n_dl]' is OPTIONAL and fully
  // backward-compatible. It is only needed to AUTO-GENERATE the kernel; when amp+bandwidth are
  // supplied explicitly (the frozen-model / production path) these scalars are vestigial:
  // accuracy feeds only auto-n_dl / legacy-PPPM refinement; b,sigma feed only the unused w0 +
  // pair-compat g_ewald; M is overridden by amp.size(). Detect the form by whether arg[0] parses
  // as a number: numeric → legacy header; keyword (e.g. 'amp') → clean form with the defaults below.
  accuracy_in = 1e-6;   // default; only used for auto-n_dl / legacy-PPPM refinement
  b_param = 1.0;        // default; feeds only the unused w0 when amp/bandwidth are supplied
  sigma_param = 1.0;    // default; feeds unused w0 + pair-compat g_ewald
  M_param = 1;          // default; overridden by amp.size() when amp is supplied
  bool have_positional_header = false;
  {
    char *ep0 = nullptr;
    double a0 = strtod(arg[0], &ep0);
    have_positional_header = (ep0 != arg[0] && *ep0 == '\0' && std::isfinite(a0));
  }

  int iarg = 0;
  if (have_positional_header) {
    if (narg < 4)
      error->all(FLERR,
                 "Illegal kspace_style sog command: a numeric positional header must be "
                 "'accuracy b sigma M [n_dl]'");
    accuracy_in = std::fabs(atof(arg[0]));
    if (!(std::isfinite(accuracy_in) && accuracy_in > 0.0))
      error->all(FLERR, "sog requires a positive accuracy argument");
    b_param = atof(arg[1]);
    sigma_param = atof(arg[2]);
    M_param = atoi(arg[3]);
    if (!(b_param > 0.0)) error->all(FLERR, "sog requires b > 0");
    if (!(sigma_param > 0.0)) error->all(FLERR, "sog requires sigma > 0");
    if (M_param < 1) error->all(FLERR, "sog requires M >= 1");
    iarg = 4;
    // Parse optional 5th argument: n_dl or first option keyword
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
  }
  // else: clean form — iarg stays 0; the keyword loop below parses amp/bandwidth/options.

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
      else if (val == "quads_4") {
        spline_type = 4;
        is_quads = true;
      } else if (val == "quads_6") {
        spline_type = 6;
        is_quads = true;
      } else
        error->all(FLERR,
                   "sog spline expects bspline, cubes2_4, cubes2_6, quads_4, or quads_6");
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
    } else if (key == "use_gpu") {
      // Enable plugin-internal GPU kspace (raw CUDA + cuFFT in sog_gpu.cu).
      // Defaults off (CPU); setting a value of "yes" enables the device path.
      if (iarg + 1 >= narg)
        error->all(FLERR, "sog missing use_gpu value");
      if (strcmp(arg[iarg + 1], "yes") == 0)
        enable_gpu = true;
      else if (strcmp(arg[iarg + 1], "no") == 0)
        enable_gpu = false;
      else
        error->all(FLERR, "sog use_gpu expects yes or no");
      iarg += 2;
    } else if (key == "phi_accuracy") {
      // Target relative energy accuracy ε for the φ_max general method (grid sizing).
      if (iarg + 1 >= narg)
        error->all(FLERR, "sog missing phi_accuracy value");
      phi_accuracy_user = atof(arg[iarg + 1]);
      if (!(phi_accuracy_user > 0.0))
        error->all(FLERR, "sog phi_accuracy must be > 0");
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
            tok == "mesh_alias_extent" || tok == "grid_method" || tok == "phi_max" ||
            tok == "use_gpu" || tok == "phi_accuracy")
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
            tok == "mesh_alias_extent" || tok == "grid_method" || tok == "phi_max" ||
            tok == "use_gpu" || tok == "phi_accuracy")
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

  // Clean form (no numeric header) is only valid with an explicit kernel: without amp/bandwidth
  // there are no b/sigma/M to auto-generate from, so fail loudly rather than silently build a
  // bogus kernel from the placeholder defaults above.
  if (!have_positional_header && !amp_from_user)
    error->all(FLERR,
               "kspace_style sog: clean form (no numeric header) requires explicit "
               "'amp ... bandwidth ...'; otherwise supply the legacy 'accuracy b sigma M' "
               "header so the kernel can be auto-generated.");
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
    // ── diagnostic: print amp/bandwidth arrays
    if (comm->me == 0) {
      std::string amp_str, bw_str;
      for (size_t m = 0; m < amp.size(); ++m) {
        char buf[64];
        snprintf(buf, sizeof(buf), " %.10f", amp[m]);
        amp_str += buf;
        snprintf(buf, sizeof(buf), " %.10f", bandwidth[m]);
        bw_str += buf;
      }
      utils::logmesg(lmp, fmt::format("SOG external amp:{}\n", amp_str));
      utils::logmesg(lmp, fmt::format("SOG external bandwidth:{}\n", bw_str));
      utils::logmesg(lmp, fmt::format("SOG amp_sum={:.6f} self_coeff={:.6f}\n",
                                 amp_sum, self_coeff));
    }
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
      // ── φ_max general method (accuracy inversion) ──
      // The paper has NO φ_max formula (Table III is empirical). The general rule inverts the
      // MEASURED grid-error law  rel ≈ C_ν·(Δ/σ_min)^{p_ν}  for a target relative accuracy ε:
      //     φ_max = (ε/C_ν)^{1/p_ν} · σ_min/r_c ,   clamped below the validity ceiling
      //     Δ_max = σ_min/(√2·ξ₀).
      // (C_ν, p_ν) are calibrated OFFLINE by the Python direct-k-sum tools
      // (sog/phi_max_rule.py, verify_anchors.py, refine_phi_max_constants.py) on a representative
      // water kernel — C_ν is kernel/system-dependent (~100× spread), so the precise per-kernel
      // value comes from those tools; the constants below are the cons/water FORCE-rel calibration,
      // with each φ_max point pinned by BISECTION on the true FFT-vs-direct curve (no fit scatter).
      // ε is a target FORCE-rel accuracy; default 1e-4 gives a genuine 1e-4-force-rel grid
      // (order-6 φ=0.068, order-4 φ=0.032 on cons); override with the `phi_accuracy` keyword. bandwidth is
      // always populated by finalize_kernel_parameters (init), so σ_min = √β_min is available.
      double bw_min = bandwidth.empty() ? sigma_param * sigma_param : bandwidth[0];
      for (double bw : bandwidth)
        if (bw < bw_min) bw_min = bw;
      const double sigma_min = std::sqrt(bw_min);
      // (C_ν, p_ν) = HONEST FORCE-rel error law from calibrate_phi_max_anchors.py (2026-07-10):
      // φ_max pinned by BISECTION of the true FFT-vs-direct force-rel curve on a PANEL of random
      // systems × kernels, then pooled log-log refit. The single-parameter (Δ/σ_min) law collapses
      // tightly for force-rel (CV ~5%); energy-rel does not (15-30%). These REPLACE the old back-fit
      // constants (2.10e-3/7.59, 1.90e-3/3.69) that were tuned so ε=1e-4 reproduced φ=0.10/0.0675 —
      // the true force-rel at φ=0.10 (order-6 cons) is ~2e-3, NOT 1e-4 (optimistic ~30×). With these
      // honest constants ε=1e-4 gives φ=0.068 (order-6) / 0.032 (order-4) on the cons kernel → a finer
      // mesh (75×150×150). PRODUCTION pins explicit phi_max=0.10 (force-rel ~2e-3, validated adequate:
      // RDF/density match DPA + experiment); auto-derive here targets genuine 1e-4 accuracy.
      const double C_nu = (spline_type == 6) ? 1.681e-2 : 4.465e-2; // honest force-rel prefactor (bisection panel)
      const double p_nu = (spline_type == 6) ? 6.533 : 3.956;       // honest force-rel convergence exponent
      const double eps_default = 1.0e-4;                            // canonical target force-rel accuracy
      const double eps = (phi_accuracy_user > 0.0) ? phi_accuracy_user : eps_default;
      const double ds = std::pow(eps / C_nu, 1.0 / p_nu);          // Δ/σ_min at target ε
      const double xi0 = (spline_type == 6) ? kCubes2Xi6 : kCubes2Xi4;
      const double delta_max = sigma_min / (std::sqrt(2.0) * xi0); // validity ceiling
      const double delta_want = std::min(ds * sigma_min, 0.95 * delta_max);
      phi_max = delta_want / rcut;
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
#ifdef SOG_ENABLE_GPU
    if (enable_gpu) {
      if (!gpu_) gpu_ = sog_gpu_create();
      sog_gpu_setup((SogGpuState*)gpu_, mesh_nx, mesh_ny, mesh_nz,
                    mesh_green_energy.data(), mesh_green_force.data(),
                    mesh_green_self.data(), mesh_green_virial.data(),
                    mesh_green_self_virial.data());
    }
#endif
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
  mesh_pot.assign(2 * ngrid, 0.0);

  int tmp = 0;
  mesh_fft = new FFT3d(lmp, world, mesh_nx, mesh_ny, mesh_nz, 0, mesh_nx - 1,
                        0, mesh_ny - 1, 0, mesh_nz - 1, 0, mesh_nx - 1, 0,
                        mesh_ny - 1, 0, mesh_nz - 1, 0, 0, &tmp,
                        collective_flag);

  // CubeS₂ (Form B) needs no spline-influence precompute: precompute_green_functions() builds the
  // separable variance-subtraction Green tables directly from amp/bandwidth. Only the legacy B-spline
  // Green path needs the sinc influence/alias tables. (The old non-separable Form-A influence build,
  // precompute_cubes2_influence(), was dead — its |Φ|² output is never read, and dividing by it is
  // unstable for the non-convolutional CubeS₂ window — so it is no longer called.)
  if (spline_type == 0) {
    precompute_sinc_tables();
  }
  precompute_green_functions();
#ifdef SOG_ENABLE_GPU
  if (enable_gpu) {
    if (!gpu_) gpu_ = sog_gpu_create();
    sog_gpu_setup((SogGpuState*)gpu_, mesh_nx, mesh_ny, mesh_nz,
                  mesh_green_energy.data(), mesh_green_force.data(),
                  mesh_green_self.data(), mesh_green_virial.data(),
                  mesh_green_self_virial.data());
  }
#endif
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
//
// DEAD CODE — retained for reference, NOT called. This builds the exact
// non-separable CubeS₂ spectral influence Φ(k) (Form A, SPME-style |Φ|²
// division). It is superseded by the separable variance-subtraction Green
// function (Form B) in precompute_green_functions(): dividing by |Φ|² is
// numerically unstable for the non-convolutional CubeS₂ window (near-zeros
// at high k), and the cubes2_influence_* outputs were never read. Kept only
// as an executable record of the non-separable form.
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
  mesh_green_self_virial.assign(ngrid, 0.0);
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

  // ── CubeS₂ variance-subtraction Green function (paper Eq. 70) ──
  // G(k) = K(k²)·exp(+Σ_α σ_{s,α}² k_α²), σ_{s,α} = ξ₀·Δ_α. The CubeS₂ spread
  // approximates an ideal Gaussian of variance σ_s²; this deconvolution is
  // analytic and stable. (Dividing by the spline influence |Φ|² is unstable
  // for the non-convolutional CubeS₂ window and is NOT used.)
  double sig_sx2 = 0.0, sig_sy2 = 0.0, sig_sz2 = 0.0;
  if (spline_type >= 4) {
    const double xi_cs2 = (spline_type == 4) ? kCubes2Xi4 : kCubes2Xi6;
    sig_sx2 = (xi_cs2 * mesh_lx / mesh_nx) * (xi_cs2 * mesh_lx / mesh_nx);
    sig_sy2 = (xi_cs2 * mesh_ly / mesh_ny) * (xi_cs2 * mesh_ly / mesh_ny);
    sig_sz2 = (xi_cs2 * mesh_lz / mesh_nz) * (xi_cs2 * mesh_lz / mesh_nz);
  }

  // Alias fast-path check
  const double k_max = std::sqrt(k_sq_max);
  const bool alias_fast_path =
      (twopi_over_x * static_cast<double>(mesh_nx) > 2.0 * k_max) &&
      (twopi_over_y * static_cast<double>(mesh_ny) > 2.0 * k_max) &&
      (twopi_over_z * static_cast<double>(mesh_nz) > 2.0 * k_max);

  // ── CubeS₂ separable Green build: per-axis 1D exp tables (PPPM-style factorization) ──
  // Form B is  Ĝ(k) = [Σ_m amp[m]·exp(-½·bandwidth[m]·k²)] · exp(Σ_α σ_{s,α}²·k_α²). Because k²=kx²+ky²+kz²
  // and the deconv argument enter ONLY as exponents, each Gaussian factors per axis. Precompute the
  // per-axis factors ONCE here — (n_gauss+1)·(nx+ny+nz) ≈ a few thousand exp — so the per-grid loop below
  // is pure multiply-add with ZERO exp. This is the whole point under NPT: `fix_nh` calls
  // `kspace->setup()` every step, so this table was rebuilt with ~12M exp/mesh; the factorization drops
  // that to O(nx+ny+nz) exp and makes NPT fast even single-threaded (no longer OpenMP-dependent).
  //   Bα[m][iα] = exp(-½·bandwidth[m]·kα²)   → folds into spectral_kernel / spectral_kernel_virial
  //   Dα[iα]    = exp(σ_{s,α}²·kα²)           → the separable variance-subtraction deconv
  const size_t n_gauss = amp.size();
  // QuadS (is_quads): the separable deconv D_α becomes the EXACT inverse window influence
  // 1/|Ŵ_α(k_α)|² (Form A, mode-based → strain-invariant / NPT-free), replacing the CubeS₂
  // Form-B Gaussian D_α = exp(σ_{s,α}²·k_α²). Everything else (B_α, kfac, kfacv) is identical.
  const double xi_quads = is_quads ? quads_xi(spline_type) : 0.0;
  std::vector<double> Bx, By, Bz, Dx, Dy, Dz;
  if (spline_type >= 4) {
    Bx.assign(n_gauss * static_cast<size_t>(mesh_nx), 0.0);
    By.assign(n_gauss * static_cast<size_t>(mesh_ny), 0.0);
    Bz.assign(n_gauss * static_cast<size_t>(mesh_nz), 0.0);
    Dx.assign(static_cast<size_t>(mesh_nx), 0.0);
    Dy.assign(static_cast<size_t>(mesh_ny), 0.0);
    Dz.assign(static_cast<size_t>(mesh_nz), 0.0);
    for (int ix = 0; ix < mesh_nx; ++ix) {
      const int kx_mode = ix - mesh_nx * (2 * ix / mesh_nx);
      const double kx = twopi_over_x * static_cast<double>(kx_mode);
      const double kx2 = kx * kx;
      Dx[static_cast<size_t>(ix)] =
          is_quads ? 1.0 / quads_influence_1d(kx_mode, mesh_nx, xi_quads, spline_type)
                   : std::exp(sig_sx2 * kx2);
      for (size_t m = 0; m < n_gauss; ++m)
        Bx[m * static_cast<size_t>(mesh_nx) + static_cast<size_t>(ix)] =
            std::exp(-0.5 * bandwidth[m] * kx2);
    }
    for (int iy = 0; iy < mesh_ny; ++iy) {
      const int ky_mode = iy - mesh_ny * (2 * iy / mesh_ny);
      const double ky = twopi_over_y * static_cast<double>(ky_mode);
      const double ky2 = ky * ky;
      Dy[static_cast<size_t>(iy)] =
          is_quads ? 1.0 / quads_influence_1d(ky_mode, mesh_ny, xi_quads, spline_type)
                   : std::exp(sig_sy2 * ky2);
      for (size_t m = 0; m < n_gauss; ++m)
        By[m * static_cast<size_t>(mesh_ny) + static_cast<size_t>(iy)] =
            std::exp(-0.5 * bandwidth[m] * ky2);
    }
    for (int iz = 0; iz < mesh_nz; ++iz) {
      const int kz_mode = iz - mesh_nz * (2 * iz / mesh_nz);
      const double kz = twopi_over_z * static_cast<double>(kz_mode);
      const double kz2 = kz * kz;
      Dz[static_cast<size_t>(iz)] =
          is_quads ? 1.0 / quads_influence_1d(kz_mode, mesh_nz, xi_quads, spline_type)
                   : std::exp(sig_sz2 * kz2);
      for (size_t m = 0; m < n_gauss; ++m)
        Bz[m * static_cast<size_t>(mesh_nz) + static_cast<size_t>(iz)] =
            std::exp(-0.5 * bandwidth[m] * kz2);
    }
  }

  // Parallelize the per-grid-point green-function build. Each iteration writes only its own
  // mesh_green_*[idx] (idx unique per ix,iy,iz) and reads shared const tables — embarrassingly
  // parallel. This matters because under NPT `fix_nh` calls `kspace->setup()` EVERY step
  // (fix_nh.cpp), so this loop (~12M exp over the mesh) reran single-threaded at ~114 ms/step
  // inside the barostat and doubled NPT wall time; threaded over the cores it is ~1-2 ms.
#pragma omp parallel for schedule(static)
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
          // ── CubeS₂ Green function (variance-subtraction, Form B) — separable synthesis ──
          // kfac ≡ spectral_kernel(sqk), kfacv ≡ spectral_kernel_virial(sqk), dc ≡ deconv, rebuilt
          // from the per-axis tables above: exact to fp rounding, with NO exp() in this hot loop.
          double kfac = 0.0, kfacv = 0.0;
          for (size_t m = 0; m < n_gauss; ++m) {
            const double p =
                Bx[m * static_cast<size_t>(mesh_nx) + static_cast<size_t>(ix)] *
                By[m * static_cast<size_t>(mesh_ny) + static_cast<size_t>(iy)] *
                Bz[m * static_cast<size_t>(mesh_nz) + static_cast<size_t>(iz)];
            kfac += amp[m] * p;
            kfacv += amp_virial[m] * p;
          }
          const double dc = Dx[static_cast<size_t>(ix)] *
                            Dy[static_cast<size_t>(iy)] *
                            Dz[static_cast<size_t>(iz)];
          mesh_green_energy[idx] = kfac * dc;
          mesh_green_force[idx] = kfac * dc;
          mesh_green_self[idx] = kfac;
          mesh_green_self_virial[idx] = kfacv;  // bare K_v(k²) for the self-energy stress term
          mesh_green_virial[idx] = kfacv * dc;
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
  // (Per-step Green-function statistics / GF-diag prints removed — validated.)
}

// ── k=0 Q² cross-term correction (applied ONCE with total charge) ──
// The k=0 energy has a term ∝ (Σq)² that is quadratic in the total charge.
// For multi-channel latent charges, individual channels are NOT charge-neutral,
// so applying the Q² term per-channel produces enormous unphysical energies/virials.
// These helpers apply the correction once with the total Q across all channels.

void SOGKSpace::apply_k0_correction_single_channel(int eflag, int vflag) {
  // For single-channel, compute total qsum/qsqsum/kfac_eff and apply Q² term.
  const int nlocal = atom->nlocal;
  double *q = atom->q;
  double qsum_local = 0.0, qsqsum_local = 0.0;
  for (int i = 0; i < nlocal; ++i) {
    qsum_local += q[i];
    qsqsum_local += q[i] * q[i];
  }
  double qsum_all = 0.0, qsqsum_all = 0.0;
  MPI_Allreduce(&qsum_local, &qsum_all, 1, MPI_DOUBLE, MPI_SUM, world);
  MPI_Allreduce(&qsqsum_local, &qsqsum_all, 1, MPI_DOUBLE, MPI_SUM, world);

  double kfac_eff = 0.0;
  {
    double Lx = domain->xprd, Ly = domain->yprd, Lz = domain->zprd;
    double k_min_sq;
    if (domain->triclinic) {
      double xy = domain->xy, xz_d = domain->xz, yz = domain->yz;
      double Vcell = Lx * Ly * Lz;
      double twopi_over_V = 2.0 * MY_PI / Vcell;
      double b1_sq = (Ly*Lz)*(Ly*Lz) + (xy*Lz)*(xy*Lz) + (xy*yz - Ly*xz_d)*(xy*yz - Ly*xz_d);
      b1_sq *= twopi_over_V * twopi_over_V;
      double b2_sq = Lx*Lx * (Lz*Lz + yz*yz);
      b2_sq *= twopi_over_V * twopi_over_V;
      double b3_sq = Lx*Lx * Ly*Ly * twopi_over_V * twopi_over_V;
      k_min_sq = std::min({b1_sq, b2_sq, b3_sq});
    } else {
      double L_max = std::max({Lx, Ly, Lz});
      k_min_sq = (2.0 * MY_PI / L_max) * (2.0 * MY_PI / L_max);
    }
    for (size_t m = 0; m < amp.size(); ++m)
      kfac_eff += amp[m] * std::exp(-0.5 * bandwidth[m] * k_min_sq);
  }

  const double volume = mesh_lx * mesh_ly * mesh_lz;
  const double qscale = force->qqrd2e * scale;

  // Energy correction
  if (eflag & ENERGY_GLOBAL) {
    energy += qscale * kfac_eff * qsum_all * qsum_all / (2.0 * volume);
    if (comm->me == 0) {
      std::string msg = fmt::format(
          "  SOG k0 Q² (1-ch): Q={:.6e} dE={:.6e}\n",
          qsum_all, qscale * kfac_eff * qsum_all * qsum_all / (2.0 * volume));
      utils::logmesg(lmp, msg);
    }
  }

  // Virial correction: W_diag = E_k0_cross for isotropic E∝1/V
  if (vflag & (VIRIAL_PAIR | VIRIAL_FDOTR)) {
    double k0_cross = qscale * kfac_eff * qsum_all * qsum_all / (2.0 * volume);
    // Also add self-term removal if remove_self_interaction is set
    if (remove_self_interaction) {
      k0_cross -= qscale * kfac_eff * qsqsum_all / (2.0 * volume);
    }
    virial[0] += k0_cross;
    virial[1] += k0_cross;
    virial[2] += k0_cross;
    if (comm->me == 0) {
      std::string msg = fmt::format(
          "  SOG k0 virial Q² (1-ch): dW_diag={:.6e}\n", k0_cross);
      utils::logmesg(lmp, msg);
    }
  }
}

void SOGKSpace::apply_k0_correction_multi_channel(
    double &energy_acc, double virial_acc[6],
    int eflag, int vflag,
    double qsum_total, double qsqsum_total) {
  // Compute kfac_eff (same as compute_single)
  double kfac_eff = 0.0;
  {
    double Lx = domain->xprd, Ly = domain->yprd, Lz = domain->zprd;
    double k_min_sq;
    if (domain->triclinic) {
      double xy = domain->xy, xz_d = domain->xz, yz = domain->yz;
      double Vcell = Lx * Ly * Lz;
      double twopi_over_V = 2.0 * MY_PI / Vcell;
      double b1_sq = (Ly*Lz)*(Ly*Lz) + (xy*Lz)*(xy*Lz) + (xy*yz - Ly*xz_d)*(xy*yz - Ly*xz_d);
      b1_sq *= twopi_over_V * twopi_over_V;
      double b2_sq = Lx*Lx * (Lz*Lz + yz*yz);
      b2_sq *= twopi_over_V * twopi_over_V;
      double b3_sq = Lx*Lx * Ly*Ly * twopi_over_V * twopi_over_V;
      k_min_sq = std::min({b1_sq, b2_sq, b3_sq});
    } else {
      double L_max = std::max({Lx, Ly, Lz});
      k_min_sq = (2.0 * MY_PI / L_max) * (2.0 * MY_PI / L_max);
    }
    for (size_t m = 0; m < amp.size(); ++m)
      kfac_eff += amp[m] * std::exp(-0.5 * bandwidth[m] * k_min_sq);
  }

  const double volume = mesh_lx * mesh_ly * mesh_lz;
  const double qscale = force->qqrd2e * scale;

  // Energy correction: Q² cross term using TOTAL charge
  if (eflag & ENERGY_GLOBAL) {
    double dE = qscale * kfac_eff * qsum_total * qsum_total / (2.0 * volume);
    energy_acc += dE;
    if (comm->me == 0) {
      std::string msg = fmt::format(
          "  SOG k0 Q² (multi-ch): Q_total={:.6e} dE={:.6e}\n",
          qsum_total, dE);
      utils::logmesg(lmp, msg);
    }
  }

  // Virial correction
  if (vflag & (VIRIAL_PAIR | VIRIAL_FDOTR)) {
    double k0_cross = qscale * kfac_eff * qsum_total * qsum_total / (2.0 * volume);
    // Self-term for multi-channel: qsqsum_total is the sum across channels
    if (remove_self_interaction) {
      k0_cross -= qscale * kfac_eff * qsqsum_total / (2.0 * volume);
    }
    virial_acc[0] += k0_cross;
    virial_acc[1] += k0_cross;
    virial_acc[2] += k0_cross;
    if (comm->me == 0) {
      std::string msg = fmt::format(
          "  SOG k0 virial Q² (multi-ch): Q_total={:.6e} dW_diag={:.6e}\n",
          qsum_total, k0_cross);
      utils::logmesg(lmp, msg);
    }
  }
}

// ── Multi-channel wrapper ──
// Multi-channel latent charges: channels are NOT individually neutral.
// Per-channel FFT would miss cross-terms ∝ ρ_ch(k)·ρ_{ch'}(-k).
// Instead, collapse all channels to a single effective charge per atom:
//   q_eff[i] = Σ_ch q_{i,ch}
// This gives the correct total charge density for the kspace solver.
void SOGKSpace::compute(int eflag, int vflag) {
#ifdef SOG_WITH_DEEPMD
  auto *pair_dp = dynamic_cast<PairDeepMD *>(force->pair);
  const int nchannels =
      (pair_dp != nullptr) ? pair_dp->ncharge_channels : 1;
#else
  const int nchannels = 1;  // standalone build: single-channel only (no PairDeepMD coupling)
#endif

  const int nlocal = atom->nlocal;

  // ── Single-channel: use atom->q directly ──
  if (nchannels <= 1) {
    compute_single(eflag, vflag);
    // NOTE: k=0 Q² cross-term is NOT applied. The training SOG kernel
    // skips k=0, so the model was trained without this contribution.
    return;
  }

#ifdef SOG_WITH_DEEPMD
  // ── Multi-channel: collapse to single effective charge ──
  // Save original charges and forces
  std::vector<double> q_orig(nlocal);
  std::vector<std::array<double, 3>> f_orig(nlocal);
  for (int i = 0; i < nlocal; ++i) {
    q_orig[i] = atom->q[i];
    f_orig[i][0] = atom->f[i][0];
    f_orig[i][1] = atom->f[i][1];
    f_orig[i][2] = atom->f[i][2];
  }

  // Set atom->q to sum of all channels (total effective charge)
  for (int i = 0; i < nlocal; ++i)
    atom->q[i] = pair_dp->dcharge_multi[i * nchannels];
  for (int ch = 1; ch < nchannels; ++ch) {
    for (int i = 0; i < nlocal; ++i)
      atom->q[i] += pair_dp->dcharge_multi[i * nchannels + ch];
  }

  // Zero forces for kspace accumulation
  for (int i = 0; i < nlocal; ++i) {
    atom->f[i][0] = 0.0; atom->f[i][1] = 0.0; atom->f[i][2] = 0.0;
  }

  // Run single-channel kspace with combined charges
  // NOTE: k=0 Q² cross-term is NOT applied — training SOG kernel skips k=0.
  compute_single(eflag, vflag);

  if (comm->me == 0) {
    std::string msg = fmt::format(
        "  SOG multi-ch collapsed ({}ch): energy={:.6e} virial=[{:.4e} {:.4e} {:.4e} {:.4e} {:.4e} {:.4e}]\n",
        nchannels, energy, virial[0], virial[1], virial[2], virial[3], virial[4], virial[5]);
    utils::logmesg(lmp, msg);
  }

  // Accumulate kspace forces (preserving original pair forces)
  for (int i = 0; i < nlocal; ++i) {
    f_orig[i][0] += atom->f[i][0];
    f_orig[i][1] += atom->f[i][1];
    f_orig[i][2] += atom->f[i][2];
  }

  // Restore original charges and write back combined forces
  for (int i = 0; i < nlocal; ++i) {
    atom->q[i] = q_orig[i];
    atom->f[i][0] = f_orig[i][0];
    atom->f[i][1] = f_orig[i][1];
    atom->f[i][2] = f_orig[i][2];
  }
#endif  // SOG_WITH_DEEPMD
}

// ──────────────────────────────────────────────────────────────────────
// compute — per-step force/energy/virial
// ──────────────────────────────────────────────────────────────────────

void SOGKSpace::compute_single(int eflag, int vflag) {
  ev_init(eflag, vflag, 0);

  // ── GPU path (plugin-internal raw CUDA + cuFFT) ──
  // ── local variables used by both CPU and GPU paths ──
  int nlocal = atom->nlocal;
  if (nlocal <= 0) return;
  double **x = atom->x, *q = atom->q;

  // ── GPU path (plugin-internal raw CUDA + cuFFT) ──
#ifdef SOG_ENABLE_GPU
  if (enable_gpu) {
    ensure_fft_plan();                 // sizes grid + (re)creates plan/green tables on box/grid change
    auto g = (SogGpuState *)gpu_;
    if (!g) { g = sog_gpu_create(); gpu_ = g; }
    int spl = is_quads ? (100 + spline_type) : spline_type;  // SOG_QUADS_4=104 / _6=106
    double qscale = force->qqrd2e * scale;
    double boxlo[3] = {domain->boxlo[0], domain->boxlo[1], domain->boxlo[2]};
    double lx = mesh_lx, ly = mesh_ly, lz = mesh_lz;
    double **fx = atom->f;
    std::vector<double> xx(nlocal * 3);
    for (int i = 0; i < nlocal; ++i) {
      xx[i * 3 + 0] = x[i][0];
      xx[i * 3 + 1] = x[i][1];
      xx[i * 3 + 2] = x[i][2];
    }
    if (want_potential) vpot.assign(nlocal, 0.0);
    double qsqsum_gpu = 0.0;
    for (int i = 0; i < nlocal; ++i) qsqsum_gpu += q[i] * q[i];
    // (single-rank isolated kspace; MPI-reduce qsqsum when this path goes multi-rank)
    std::vector<double> vv(6), fk(3 * nlocal);
    energy = sog_gpu_compute(g, nlocal, xx.data(), boxlo, lx, ly, lz, q, qscale,
                             spl, want_potential ? 1 : 0, fk.data(), vv.data(),
                             want_potential ? vpot.data() : nullptr,
                             self_coeff, qsqsum_gpu, remove_self_interaction ? 1 : 0);
    for (int i = 0; i < nlocal; ++i) {   // ADD kspace force to atom->f (pair already wrote its part)
      fx[i][0] += fk[i * 3 + 0];
      fx[i][1] += fk[i * 3 + 1];
      fx[i][2] += fk[i * 3 + 2];
    }
    for (int j = 0; j < 6; ++j) virial[j] = vv[j];
    return;
  }
#endif

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

  // TEMP validation hook: SOG_DUMP_POT=1 forces v_i computation + one-time dump,
  // so v_i = ∂E_k/∂q_i can be finite-difference validated before the fix exists.
  if (!want_potential && getenv("SOG_DUMP_POT")) want_potential = true;

  energy = 0.0;
  for (int j = 0; j < 6; ++j) virial[j] = 0.0;

  // nlocal/x/q/already declared above (also used by GPU path)

  // ── Pre-compute qsum/qsqsum (used for diagnostics and self-interaction) ──
  double qsum_all = 0.0, qsqsum_all = 0.0;
  {
    double qsqsum_local = 0.0, qsum_local = 0.0;
    for (int i = 0; i < nlocal; ++i) {
      qsqsum_local += q[i] * q[i];
      qsum_local += q[i];
    }
    MPI_Allreduce(&qsqsum_local, &qsqsum_all, 1, MPI_DOUBLE, MPI_SUM, world);
    MPI_Allreduce(&qsum_local, &qsum_all, 1, MPI_DOUBLE, MPI_SUM, world);
  }

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

  // ── Charge spreading ──

  if (spline_type >= 4 && is_quads) {
    // QuadS separable charge spreading: 3-D weight = wx[a]·wy[b]·wz[c] over the
    // (2ν)³ nodes at offsets quads_offsets(order).
    const double xi = quads_xi(spline_type);
    const int n1d = spline_type;              // 2ν
    const int *offs = quads_offsets(spline_type);
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
      double wx[6], wy[6], wz[6];
      quads_weights_1d(tx, xi, spline_type, wx);
      quads_weights_1d(ty, xi, spline_type, wy);
      quads_weights_1d(tz, xi, spline_type, wz);
      for (int a = 0; a < n1d; ++a) {
        const int igx = wrap_index(ix0 + offs[a], mesh_nx);
        for (int b = 0; b < n1d; ++b) {
          const int igy = wrap_index(iy0 + offs[b], mesh_ny);
          const double wxy = wx[a] * wy[b];
          for (int c = 0; c < n1d; ++c) {
            const int igz = wrap_index(iz0 + offs[c], mesh_nz);
            const size_t idx = mesh_index(igx, igy, igz);
            mesh_rho[idx] += static_cast<FFT_SCALAR>(q_scaled * wxy * wz[c]);
          }
        }
      }
    }
  } else if (spline_type >= 4) {
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
      } else {  // spline_type == 6 (88-node CubeS₂)
        for (int k = 0; k < kCubes2NumNodes6; ++k) {
          const auto &node = kCubes2Nodes6[k];
          const double w = cubes2_weight_6(tx, ty, tz, node, xi);
          if (w == 0.0) continue;
          const int igx = wrap_index(ix0 + node.dx, mesh_nx);
          const int igy = wrap_index(iy0 + node.dy, mesh_ny);
          const int igz = wrap_index(iz0 + node.dz, mesh_nz);
          const size_t idx = mesh_index(igx, igy, igz);
          mesh_rho[idx] += static_cast<FFT_SCALAR>(q_scaled * w);
        }
      }
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
  // Anisotropic self-energy stress accumulator: Σ_mesh K_v(k²)·k_α k_β. Combined with
  // diag_sum (Σ K) it gives the strain-derivative of the reciprocal self-removal energy
  // −qscale·qsqsum·Σ K/(2V), which the reciprocal virial omits (~2.5%→~0.5% fix).
  std::array<double, 6> sv_local = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

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
          if (want_potential) {
            mesh_pot[2 * idx] = mesh_pot[2 * idx + 1] = 0.0;
          }
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
          // Anisotropic self-energy stress: Σ bare K_v(k²)·k_α k_β (config-independent,
          // like diag_sum_local; combined with diag_sum in the virial finalize below).
          const double gsv = mesh_green_self_virial[idx];
          sv_local[0] += gsv * kx * kx;
          sv_local[1] += gsv * ky * ky;
          sv_local[2] += gsv * kz * kz;
          sv_local[3] += gsv * kx * ky;
          sv_local[4] += gsv * kx * kz;
          sv_local[5] += gsv * ky * kz;
        }

        const double vk_re = scaleinv * geff * rho_re;
        const double vk_im = scaleinv * geff * rho_im;

        mesh_gradx[2 * idx] = static_cast<FFT_SCALAR>(-kx * vk_im);
        mesh_gradx[2 * idx + 1] = static_cast<FFT_SCALAR>(kx * vk_re);
        mesh_grady[2 * idx] = static_cast<FFT_SCALAR>(-ky * vk_im);
        mesh_grady[2 * idx + 1] = static_cast<FFT_SCALAR>(ky * vk_re);
        mesh_gradz[2 * idx] = static_cast<FFT_SCALAR>(-kz * vk_im);
        mesh_gradz[2 * idx + 1] = static_cast<FFT_SCALAR>(kz * vk_re);

        // ── Potential mesh u(k) = scaleinv·geff_energy·ρ(k) (no ik factor) ──
        // v_i = ∂E_k/∂q_i is the potential at atom i: gather this mesh with the same
        // CubeS₂ weights as the force (uses the ENERGY Green function, not the force one).
        if (want_potential) {
          mesh_pot[2 * idx] = static_cast<FFT_SCALAR>(scaleinv * geff_energy * rho_re);
          mesh_pot[2 * idx + 1] = static_cast<FFT_SCALAR>(scaleinv * geff_energy * rho_im);
        }
      }
    }
  }

  // ── FFT backward (3 gradients) ──
  mesh_fft->compute(mesh_gradx.data(), mesh_gradx.data(), FFT3d::BACKWARD);
  mesh_fft->compute(mesh_grady.data(), mesh_grady.data(), FFT3d::BACKWARD);
  mesh_fft->compute(mesh_gradz.data(), mesh_gradz.data(), FFT3d::BACKWARD);
  // 4th inverse FFT: real-space mesh potential u_j (for v_i = ∂E_k/∂q_i)
  if (want_potential) {
    mesh_fft->compute(mesh_pot.data(), mesh_pot.data(), FFT3d::BACKWARD);
  }

  // ── Force interpolation ──
  const double qscale = force->qqrd2e * scale;
  // Diagnostic: print key scale factors once
  static bool _diag_printed = false;
  if (!_diag_printed && comm->me == 0) {
    _diag_printed = true;
    std::string msg = fmt::format(
        "  SOG diag: qscale={:.8e} volume={:.6e} virial_scale={:.8e} "
        "want_energy={} want_virial={} vflag={} spline_type={}\n",
        qscale, volume, 0.5 * volume * qscale,
        want_energy, want_virial, vflag, spline_type);
    utils::logmesg(lmp, msg);
  }

  if (spline_type >= 4 && is_quads) {
    // QuadS separable force/potential interpolation.
    const double xi = quads_xi(spline_type);
    const int n1d = spline_type;
    const int *offs = quads_offsets(spline_type);
    if (want_potential) vpot.assign(nlocal, 0.0);
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
      double gx = 0.0, gy = 0.0, gz = 0.0, gpot = 0.0;
      double wx[6], wy[6], wz[6];
      quads_weights_1d(tx, xi, spline_type, wx);
      quads_weights_1d(ty, xi, spline_type, wy);
      quads_weights_1d(tz, xi, spline_type, wz);
      for (int a = 0; a < n1d; ++a) {
        const int igx = wrap_index(ix0 + offs[a], mesh_nx);
        for (int b = 0; b < n1d; ++b) {
          const int igy = wrap_index(iy0 + offs[b], mesh_ny);
          const double wxy = wx[a] * wy[b];
          for (int c = 0; c < n1d; ++c) {
            const int igz = wrap_index(iz0 + offs[c], mesh_nz);
            const size_t idx = mesh_index(igx, igy, igz);
            const double w = wxy * wz[c];
            gx += w * static_cast<double>(mesh_gradx[2 * idx]);
            gy += w * static_cast<double>(mesh_grady[2 * idx]);
            gz += w * static_cast<double>(mesh_gradz[2 * idx]);
            if (want_potential) gpot += w * static_cast<double>(mesh_pot[2 * idx]);
          }
        }
      }
      const double qi = q[i];
      atom->f[i][0] += -qscale * qi * gx;
      atom->f[i][1] += -qscale * qi * gy;
      atom->f[i][2] += -qscale * qi * gz;
      if (want_potential) vpot[i] = qscale * gpot;
    }
  } else if (spline_type >= 4) {
    // CubeS₂ force interpolation
    const int num_nodes = (spline_type == 4) ? kCubes2NumNodes4 : kCubes2NumNodes6;
    const double xi = (spline_type == 4) ? kCubes2Xi4 : kCubes2Xi6;
    if (want_potential) vpot.assign(nlocal, 0.0);

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
      double gpot = 0.0;

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
          if (want_potential) gpot += w * static_cast<double>(mesh_pot[2 * idx]);
        }
      } else {  // spline_type == 6 (88-node CubeS₂)
        for (int k = 0; k < kCubes2NumNodes6; ++k) {
          const auto &node = kCubes2Nodes6[k];
          const double w = cubes2_weight_6(tx, ty, tz, node, xi);
          if (w == 0.0) continue;
          const int igx = wrap_index(ix0 + node.dx, mesh_nx);
          const int igy = wrap_index(iy0 + node.dy, mesh_ny);
          const int igz = wrap_index(iz0 + node.dz, mesh_nz);
          const size_t idx = mesh_index(igx, igy, igz);
          gx += w * static_cast<double>(mesh_gradx[2 * idx]);
          gy += w * static_cast<double>(mesh_grady[2 * idx]);
          gz += w * static_cast<double>(mesh_gradz[2 * idx]);
          if (want_potential) gpot += w * static_cast<double>(mesh_pot[2 * idx]);
        }
      }

      const double qi = q[i];
      const double fxs = -qscale * qi * gx;
      const double fys = -qscale * qi * gy;
      const double fzs = -qscale * qi * gz;

      atom->f[i][0] += fxs;
      atom->f[i][1] += fys;
      atom->f[i][2] += fzs;

      // Mesh part of the per-atom potential v_i = ∂E_k/∂q_i (same qscale as the
      // force). Self-energy contributions are added after diag_sum_all is reduced.
      if (want_potential) vpot[i] = qscale * gpot;
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
    }
  }

  // ── Energy ──
  if (want_energy) {
    double energy_all = 0.0;
    MPI_Allreduce(&energy_local, &energy_all, 1, MPI_DOUBLE, MPI_SUM, world);

    double diag_sum_all = 0.0;
    MPI_Allreduce(&diag_sum_local, &diag_sum_all, 1, MPI_DOUBLE, MPI_SUM,
                  world);

    energy = 0.5 * volume * energy_all;
    if (remove_self_interaction) {
      const double self_term = diag_sum_all / (2.0 * volume);
      energy -= qsqsum_all * self_term;
    }
    // Also subtract the real-space self-energy (matching RBSOG convention)
    energy -= self_coeff * qsqsum_all;

    // The k≠0 Fourier modes, real-space self-energy, and mesh self-term
    // fully define the SOG long-range energy. k=0 is excluded (Σq=0 by
    // model-layer charge neutrality), matching the native fastsog.cpp.

    energy *= qscale;
  }

  // ── Self-energy part of the per-atom potential v_i = ∂E_k/∂q_i ──
  // E_self = qscale·[ −(rsi)·qsqsum·diag_sum/(2V) − self_coeff·qsqsum ]
  //   ⇒ ∂E_self/∂q_i = −qscale·q_i·[ (rsi)·diag_sum/V + 2·self_coeff ]
  if (want_potential) {
    double diag_sum_all_p = 0.0;
    MPI_Allreduce(&diag_sum_local, &diag_sum_all_p, 1, MPI_DOUBLE, MPI_SUM, world);
    const double rsi_coef =
        remove_self_interaction ? (diag_sum_all_p / volume) : 0.0;
    const double self_c = 2.0 * self_coeff;
    for (int i = 0; i < nlocal; ++i) {
      vpot[i] -= qscale * q[i] * (rsi_coef + self_c);
    }
    if (getenv("SOG_DUMP_POT") && comm->me == 0) {
      utils::logmesg(lmp, fmt::format(
          "  SOG vpot: tag[0]={} v[0]={:.10e}  tag[1]={} v[1]={:.10e}  "
          "q[0]={:.6f} q[1]={:.6f}\n",
          atom->tag[0], vpot[0], (nlocal > 1 ? atom->tag[1] : 0),
          (nlocal > 1 ? vpot[1] : 0.0), q[0], (nlocal > 1 ? q[1] : 0.0)));
    }
  }

  // ── Virial ──
  if (want_virial) {
    double vf_all[6] = {0.0};

    // Fourier-space virial (primary, matches rbsog_intel & PPPM convention)
    MPI_Allreduce(fv_local.data(), vf_all, 6, MPI_DOUBLE, MPI_SUM, world);
    // Scale Fourier virial: W = 0.5 * V * qscale * Σ s2 * |ρ|² * (ge·I - gv·k⊗k)
    const double virial_scale = 0.5 * volume * qscale;
    for (int j = 0; j < 6; ++j) virial[j] = virial_scale * vf_all[j];

    // Self-energy strain-derivative (the term the reciprocal formula omits, ~2.5%→~0.5%):
    //   W_self_αβ = (qscale·qsqsum/2V)·(Σ K_v·k_α k_β − δ_αβ·Σ K)
    // Σ K_v·k_α k_β = sv_all, Σ K = diag_sum_all (both raw mesh sums, no s2). Isotropic
    // δ part matches the energy's −qsqsum·diag_sum/(2V); the k=0 correction is separate.
    if (remove_self_interaction) {
      double sv_all[6] = {0.0};
      MPI_Allreduce(sv_local.data(), sv_all, 6, MPI_DOUBLE, MPI_SUM, world);
      double diag_sum_all_v = 0.0;
      MPI_Allreduce(&diag_sum_local, &diag_sum_all_v, 1, MPI_DOUBLE, MPI_SUM, world);
      const double self_pref = qscale * qsqsum_all / (2.0 * volume);
      virial[0] += self_pref * (sv_all[0] - diag_sum_all_v);
      virial[1] += self_pref * (sv_all[1] - diag_sum_all_v);
      virial[2] += self_pref * (sv_all[2] - diag_sum_all_v);
      virial[3] += self_pref * sv_all[3];
      virial[4] += self_pref * sv_all[4];
      virial[5] += self_pref * sv_all[5];
    }

    // NOTE: k=0 Q² virial correction is NOT applied here per-channel.
    // It is applied once in the multi-channel wrapper using total Q.
  }

  // NOTE: k=0 terms are intentionally NOT applied here. Charge neutrality
  // is enforced at the model layer (lr_fitting._corr_head subtracts the
  // per-frame per-channel mean), so Σq = 0 and the k=0 Q² cross-term
  // vanishes. This keeps sog.cpp bit-for-bit consistent with the native
  // LAMMPS fastsog.cpp, which has no k=0 correction.
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
