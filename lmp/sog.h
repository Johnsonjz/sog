/* -*- c++ -*- ----------------------------------------------------------
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
   Contributing authors: Zhen Jiang (SJTU), based on sog.cpp (deepmd-kit)
   and rbsog_intel.cpp
------------------------------------------------------------------------- */

#ifdef KSPACE_CLASS
// clang-format off
KSpaceStyle(sog, SOGKSpace)
// clang-format on
#else

#ifndef LMP_SOG_H
#define LMP_SOG_H

#include "kspace.h"
#include "lmpfftsettings.h"

#include <array>
#include <vector>

namespace LAMMPS_NS {

class FFT3d;

class SOGKSpace : public KSpace {
 public:
  explicit SOGKSpace(class LAMMPS *lmp);
  ~SOGKSpace() override;

  void settings(int narg, char **arg) override;
  void init() override;
  void setup() override;
  void compute(int eflag, int vflag) override;
  void compute_single(int eflag, int vflag);
  void apply_k0_correction_single_channel(int eflag, int vflag);
  void apply_k0_correction_multi_channel(double &energy_acc, double virial_acc[6],
                                         int eflag, int vflag,
                                         double qsum_total, double qsqsum_total);
  double memory_usage() override;

  // Per-atom electrostatic potential v_i = ∂E_k/∂q_i (filled in compute_single when
  // want_potential is set). Consumed by the charge-response fix to feed the model's
  // ∂q/∂r backward (DPLR analog of pppm_dplr::get_fele()).
  const std::vector<double> &get_potential() const { return vpot; }
  void set_want_potential(bool v) { want_potential = v; }

 protected:
  // ── SOG kernel parameters ──
  double b_param;        // b: geometric scaling factor
  double sigma_param;    // sigma: base Gaussian width
  int M_param;           // M: number of Gaussians
  double accuracy_in;    // force accuracy target
  double n_dl;           // mesh resolution parameter (legacy PPPM grid)
  bool remove_self_interaction;
  double mesh_oversample;
  int mesh_alias_extent;

  // ── Spline / grid method selection ──
  int spline_type;       // 0 = B-spline (legacy; order in bspline_order), 4 = CubeS2 4th, 6 = CubeS2 6th
  int bspline_order = 5; // B-spline order when spline_type == 0 (4, 5, or 6)
  bool is_quads = false; // false = CubeS₂ (non-separable); true = QuadS (separable, Form-A)
  int grid_method;       // 0 = SOG bandwidth (new), 1 = PPPM iteration (legacy)
  double phi_max_user;   // user-specified φ_max override (>0 means active, −1 = auto)
  double phi_accuracy_user;  // target rel-accuracy ε for the φ_max general method (−1 = default)
  bool enable_gpu = false;   // route compute_single to sog_gpu.cu (raw CUDA + cuFFT)
  void *gpu_ = nullptr;      // opaque SogGpuState* (global-ns; sog_gpu.cuh is C)

  // ── Computed from SOG params + cutoff ──
  double w0;             // real-space correction factor (only on m=0 term)
  std::vector<double> amp;       // coef[m] = A_m (for energy)
  std::vector<double> bandwidth; // band_m = b^(2m) * sigma^2
  std::vector<double> amp_virial; // coef_virial[m] = A_m * band_m (for virial)
  double self_coeff;     // self-energy coefficient
  double amp_sum;       // Σ_m amp_m, total SOG amplitude at k=0
  bool amp_from_user;  // true when amp/bandwidth provided externally

  // ── Mesh + FFT ──
  int mesh_nx, mesh_ny, mesh_nz;
  double mesh_lx, mesh_ly, mesh_lz;
  FFT3d *mesh_fft;
  bool mesh_ready;

  std::vector<FFT_SCALAR> mesh_rho;
  std::vector<FFT_SCALAR> mesh_fft_work;
  std::vector<FFT_SCALAR> mesh_gradx;  std::vector<FFT_SCALAR> mesh_grady;
  std::vector<FFT_SCALAR> mesh_gradz;

  // ── Per-atom potential v_i = ∂E_k/∂q_i (charge-response feedback) ──
  bool want_potential = false;            // gate: fill vpot in compute_single
  std::vector<FFT_SCALAR> mesh_pot;       // k-space then real-space mesh potential u_j
  std::vector<double> vpot;               // gathered per-atom potential v_i (nlocal)

  // ── Precomputed Green functions ──
  std::vector<double> mesh_green_energy;
  std::vector<double> mesh_green_force;
  std::vector<double> mesh_green_self;
  std::vector<double> mesh_green_self_virial;  // bare K_v(k²): self-energy strain-derivative virial
  std::vector<double> mesh_green_virial;  // K_virial(k²)/|Φ(k)|²

  // ── Box-independent sinc tables ──
  std::vector<double> sinc_table_x;
  std::vector<double> sinc_table_y;
  std::vector<double> sinc_table_z;
  std::vector<double> sinc_sum_x;
  std::vector<double> sinc_sum_y;
  std::vector<double> sinc_sum_z;

  // ── CubeS₂ influence function (replaces sinc tables when spline_type > 0) ──
  std::vector<double> cubes2_influence_re;  // Re[Φ(k)]
  std::vector<double> cubes2_influence_im;  // Im[Φ(k)]
  std::vector<double> cubes2_influence_sq;  // |Φ(k)|²

  // ── Direct k-space mode (exact reference, no mesh/FFT) ──
  bool use_direct = false;                  // gate: route compute_single to compute_direct
  double direct_kmax = -1.0;                // k_max in Å⁻¹ for direct k-vector sphere
  int nk_direct = 0;                        // actual half-sphere k-vector count
  std::vector<int> kx_d, ky_d, kz_d;        // Miller indices (half-sphere)
  std::vector<double> ksq_d;                // k_cart² per k-vector
  std::vector<double> kfac_d;               // K(k²) per k-vector
  std::vector<double> kfac_virial_d;        // K_v(k²) = Σ a_m·β_m·exp(-½β_m·k²)
  std::vector<double> cs_x, cs_y, cs_z;     // per-atom cos(k·r_i) by axis, per k, per atom
  std::vector<double> sn_x, sn_y, sn_z;     // per-atom sin(k·r_i) by axis, per k, per atom
  std::vector<double> sf_re, sf_im;         // global structure factor per k (after MPI)
  std::vector<double> sf_re_loc, sf_im_loc; // local contribution (pre-reduce)

  // ── Methods ──
  void finalize_kernel_parameters();
  void finalize_virial_parameters();
  double spectral_kernel(double ksq) const;
  double spectral_kernel_virial(double ksq) const;
  void ensure_fft_plan();
  void destroy_fft_plan();
  void precompute_sinc_tables();
  void precompute_cubes2_influence();
  void precompute_green_functions();

  // Direct k-space mode
  void enumerate_direct_kvecs();
  void compute_direct(int eflag, int vflag);

  size_t mesh_index(int ix, int iy, int iz) const;
  double periodic_fraction(double x, double xlo, double prd) const;
  int wrap_index(int i, int n) const;
};

}  // namespace LAMMPS_NS

#endif
#endif
