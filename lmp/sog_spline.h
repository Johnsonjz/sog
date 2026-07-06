/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   FastSOG shared spline definitions (B-spline + CubeS₂ Midtown splines)
------------------------------------------------------------------------- */

#ifndef LMP_SOG_SPLINE_H
#define LMP_SOG_SPLINE_H

#include <array>
#include <cmath>

namespace LAMMPS_NS {

// ── B-spline order (legacy) ──
constexpr int kSog_BSplineOrder = 5;
constexpr int kSog_AssignOrder = kSog_BSplineOrder;  // used by both .cpp files

inline void fastsog_bspline_weights_1d(const double frac,
                                        std::array<double, 5> &w) {
  w.fill(0.0);
  w[0] = 1.0 - frac;
  w[1] = frac;
  for (int k = 3; k <= 5; ++k) {
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

// ── CubeS₂ Midtown splines ──
// Reference: Predescu, Bergdorf, Shaw, J. Chem. Phys. 153, 224117 (2020)

// Optimal ξ values (paper Table I)
constexpr double kCubes2Xi4 = 0.5773502691896258;     // 1/√3, C² continuous
constexpr double kCubes2Xi4_alt = 0.5163977794943222; // 2/√15, general purpose
constexpr double kCubes2Xi6 = 0.72879;                 // 6th order optimal

// Node counts
constexpr int kCubes2NumNodes4 = 32;
constexpr int kCubes2NumNodes6 = 88;

// ── CubeS₂ 4th-order node definition ──
struct CubeS2Node4 {
  int dx, dy, dz;  // offset from floor grid index
  int cls;         // 0 = (1,1,1) symmetry class, 1 = (2,1,1) class
  int sp_axis;     // for cls==1: which axis (0=x,1=y,2=z) has the special component
  int sp_is_neg;   // for cls==1: 1 if d=-1, 0 if d=2
};

constexpr CubeS2Node4 kCubes2Nodes4[32] = {
  // ── Class 0: offsets in {0,1}³ (8 nodes) ──
  { 0, 0, 0, 0, -1, 0}, { 0, 0, 1, 0, -1, 0}, { 0, 1, 0, 0, -1, 0}, { 0, 1, 1, 0, -1, 0},
  { 1, 0, 0, 0, -1, 0}, { 1, 0, 1, 0, -1, 0}, { 1, 1, 0, 0, -1, 0}, { 1, 1, 1, 0, -1, 0},
  // ── Class 1: one component is -1 or 2, others in {0,1} (24 nodes) ──
  // x-axis special
  {-1, 0, 0, 1, 0, 1}, {-1, 0, 1, 1, 0, 1}, {-1, 1, 0, 1, 0, 1}, {-1, 1, 1, 1, 0, 1},
  { 2, 0, 0, 1, 0, 0}, { 2, 0, 1, 1, 0, 0}, { 2, 1, 0, 1, 0, 0}, { 2, 1, 1, 1, 0, 0},
  // y-axis special
  { 0,-1, 0, 1, 1, 1}, { 0,-1, 1, 1, 1, 1}, { 1,-1, 0, 1, 1, 1}, { 1,-1, 1, 1, 1, 1},
  { 0, 2, 0, 1, 1, 0}, { 0, 2, 1, 1, 1, 0}, { 1, 2, 0, 1, 1, 0}, { 1, 2, 1, 1, 1, 0},
  // z-axis special
  { 0, 0,-1, 1, 2, 1}, { 0, 1,-1, 1, 2, 1}, { 1, 0,-1, 1, 2, 1}, { 1, 1,-1, 1, 2, 1},
  { 0, 0, 2, 1, 2, 0}, { 0, 1, 2, 1, 2, 0}, { 1, 0, 2, 1, 2, 0}, { 1, 1, 2, 1, 2, 0},
};

// Weight functions for CubeS₂ 4th order
// Eq. 15: L(θ,ξ) = -½θ³ + ½θ² - (9ξ²-2)/6·θ + ξ²/2
//          R(θ,ξ) =  ⅙θ³ + (3ξ²-1)/6·θ
inline double cubes2_L(const double theta, const double xi) {
  return -0.5 * theta * theta * theta + 0.5 * theta * theta
         - (9.0 * xi * xi - 2.0) / 6.0 * theta + 0.5 * xi * xi;
}

inline double cubes2_R(const double theta, const double xi) {
  return (1.0 / 6.0) * theta * theta * theta
         + (3.0 * xi * xi - 1.0) / 6.0 * theta;
}

// Compute CubeS₂ 4th-order weight for a single node at fractional (tx,ty,tz)
inline double cubes2_weight_4(const double tx, const double ty, const double tz,
                               const CubeS2Node4 &node, const double xi) {
  if (node.cls == 0) {
    const double eta_x = (node.dx == 0) ? tx : (1.0 - tx);
    const double eta_y = (node.dy == 0) ? ty : (1.0 - ty);
    const double eta_z = (node.dz == 0) ? tz : (1.0 - tz);
    return cubes2_L(eta_x, xi) * eta_y * eta_z +
           cubes2_L(eta_y, xi) * eta_x * eta_z +
           cubes2_L(eta_z, xi) * eta_x * eta_y;
  } else {
    double eta_special, eta_n1, eta_n2;
    if (node.sp_axis == 0) {
      eta_special = node.sp_is_neg ? tx : (1.0 - tx);
      eta_n1 = (node.dy == 0) ? ty : (1.0 - ty);
      eta_n2 = (node.dz == 0) ? tz : (1.0 - tz);
    } else if (node.sp_axis == 1) {
      eta_special = node.sp_is_neg ? ty : (1.0 - ty);
      eta_n1 = (node.dx == 0) ? tx : (1.0 - tx);
      eta_n2 = (node.dz == 0) ? tz : (1.0 - tz);
    } else {
      eta_special = node.sp_is_neg ? tz : (1.0 - tz);
      eta_n1 = (node.dx == 0) ? tx : (1.0 - tx);
      eta_n2 = (node.dy == 0) ? ty : (1.0 - ty);
    }
    return cubes2_R(eta_special, xi) * eta_n1 * eta_n2;
  }
}

}  // namespace LAMMPS_NS

#endif
