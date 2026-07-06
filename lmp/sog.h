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
  double memory_usage() override;

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
  int spline_type;       // 0 = B-spline order 5 (legacy), 4 = CubeS2 4th, 6 = CubeS2 6th
  int grid_method;       // 0 = SOG bandwidth (new), 1 = PPPM iteration (legacy)
  double phi_max_user;   // user-specified φ_max override (>0 means active, −1 = auto)

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
  std::vector<FFT_SCALAR> mesh_gradx;
  std::vector<FFT_SCALAR> mesh_grady;
  std::vector<FFT_SCALAR> mesh_gradz;

  // ── Precomputed Green functions ──
  std::vector<double> mesh_green_energy;
  std::vector<double> mesh_green_force;
  std::vector<double> mesh_green_self;
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

  size_t mesh_index(int ix, int iy, int iz) const;
  double periodic_fraction(double x, double xlo, double prd) const;
  int wrap_index(int i, int n) const;
};

}  // namespace LAMMPS_NS

#endif
#endif
