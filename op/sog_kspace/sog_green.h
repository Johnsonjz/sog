// sog_green.h — self-contained grid sizing + CubeS₂ green tables for the
// SOG kspace custom op.
//
// Ported line-for-line from source/lmp/sog.cpp so the op reproduces the
// DEPLOYED chain's discretization, not an approximation of it:
//   * grid sizing        : SOGKSpace::ensure_fft_plan (sog.cpp:965-1027, the
//                          grid_method==0 / phi_max path) + factorable_235
//   * green tables       : SOGKSpace::precompute_green_functions (sog.cpp:1413-1671,
//                          spline_type >= 4 / CubeS₂ Form-B branch)
// Keeping the two implementations separate is deliberate: sog_op.py's python
// green builder is an independent second implementation, and validate192.py
// checks the op against the LAMMPS run-0 energy, so a transcription error in
// either one shows up as a disagreement.
//
// Only the CubeS₂ family (spline 4/6) is supported: those are the production
// splines, and the legacy B-spline (spline_type == 0) branch needs the sinc
// alias tables that live in the SOGKSpace class.
#ifndef SOG_GREEN_H
#define SOG_GREEN_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace sog_green {

constexpr int kGridMin = 8;                    // sog.cpp:350
constexpr double kCubes2Xi4 = 0.5773502691896258;   // sog_spline.h:46
constexpr double kCubes2Xi6 = 0.6503998764035732;   // sog_spline.h:48

inline bool factorable_235(int n) {            // sog.cpp:66
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

// SOGKSpace::ensure_fft_plan, grid_method==0 branch (sog.cpp:974-1027).
//
// phi_max is derived by inverting the measured grid-error law
//   rel ~= C_nu * (delta/sigma_min)^p_nu
// for a target force-relative accuracy eps, clamped below the validity ceiling
// sigma_min/(sqrt(2)*xi0).  (C_nu, p_nu) are the CubeS₂ Form-B constants,
// using the paper's reference values (sog-lib-paper,
// sections/03-general-kernel.tex Eq. eq:phimax-rule; fit in
// sections/06-results.tex Table tab:spline-slopes): trained conservative
// kernel, FORCE-relative metric, C_4 = 1.0e-1, C_6 = 2.4e-2.  Must stay
// identical to source/lmp/sog.cpp so the op matches the deployed chain.
inline int grid_axis(double L, const std::vector<double> &bandwidth, double rcut,
                     int spline, double phi_accuracy) {
  double bw_min = bandwidth.empty() ? 1.0 : bandwidth[0];
  for (double bw : bandwidth)
    if (bw < bw_min) bw_min = bw;
  const double sigma_min = std::sqrt(bw_min);

  const double C_nu = (spline == 6) ? 2.4e-2 : 1.0e-1;
  const double p_nu = (spline == 6) ? 6.0 : 4.0;
  const double eps = (phi_accuracy > 0.0) ? phi_accuracy : 1.0e-4;

  const double ds = std::pow(eps / C_nu, 1.0 / p_nu);
  const double xi0 = (spline == 6) ? kCubes2Xi6 : kCubes2Xi4;
  const double delta_max = sigma_min / (std::sqrt(2.0) * xi0);
  const double delta_want = std::min(ds * sigma_min, 0.95 * delta_max);
  const double phi_max = delta_want / rcut;

  const double delta = phi_max * rcut;
  int n = std::max(kGridMin, static_cast<int>(std::ceil(L / delta)));
  while (!factorable_235(n)) ++n;
  return n;
}

// SOGKSpace::precompute_green_functions, CubeS₂ branch (sog.cpp:1413-1565).
// Fills the five tables flat in mesh_index(ix,iy,iz) = ix + nx*(iy + ny*iz).
// Only the three deconvolved tables are produced.  The bare K and K_v "self"
// tables that used to live alongside them are gone: the self-energy term is
// closed-form (see sog_kspace_op.cc / sog_gpu.cu), needs no per-mode table,
// and the mesh sum it was used for was not volume-extensive.
// Shared body for the full and slab (y-sliced) tables.  `ylo`/`YL` select the
// y-range to fill and `ystride` is the storage extent along y: the full table
// uses (0, ny, ny), the per-rank slab uses (ylo, YL, YL) and stores
// idx = ix + nx*(iyl + YL*iz).  Both variants run the SAME arithmetic, so a
// slab value at a given (ix,iy,iz) is bit-identical to the full table.
inline void build_green_range(int nx, int ny, int nz, double lx, double ly,
                              double lz, const std::vector<double> &amp,
                              const std::vector<double> &bandwidth, int spline,
                              int ylo, int YL, int ystride,
                              std::vector<double> &ge, std::vector<double> &gf,
                              std::vector<double> &gv) {
  const size_t ng = static_cast<size_t>(nx) * YL * nz;
  ge.assign(ng, 0.0);
  gf.assign(ng, 0.0);
  gv.assign(ng, 0.0);

  const double dx = lx / nx, dy = ly / ny, dz = lz / nz;
  // Grid Nyquist for the CubeS₂ family (sog.cpp:1428-1433).
  const double k_sq_max = M_PI * M_PI *
                          (1.0 / (dx * dx) + 1.0 / (dy * dy) + 1.0 / (dz * dz));
  const double tx = 2.0 * M_PI / lx, ty = 2.0 * M_PI / ly, tz = 2.0 * M_PI / lz;

  const double xi = (spline == 6) ? kCubes2Xi6 : kCubes2Xi4;
  const double sig_x2 = (xi * dx) * (xi * dx);
  const double sig_y2 = (xi * dy) * (xi * dy);
  const double sig_z2 = (xi * dz) * (xi * dz);

  const size_t M = amp.size();
  std::vector<double> Bx(M * nx), By(M * ny), Bz(M * nz);
  std::vector<double> Dx(nx), Dy(ny), Dz(nz);
  for (int ix = 0; ix < nx; ++ix) {
    const int mode = ix - nx * (2 * ix / nx);
    const double k = tx * mode, k2 = k * k;
    Dx[ix] = std::exp(sig_x2 * k2);
    for (size_t m = 0; m < M; ++m)
      Bx[m * nx + ix] = std::exp(-0.5 * bandwidth[m] * k2);
  }
  for (int iy = 0; iy < ny; ++iy) {
    const int mode = iy - ny * (2 * iy / ny);
    const double k = ty * mode, k2 = k * k;
    Dy[iy] = std::exp(sig_y2 * k2);
    for (size_t m = 0; m < M; ++m)
      By[m * ny + iy] = std::exp(-0.5 * bandwidth[m] * k2);
  }
  for (int iz = 0; iz < nz; ++iz) {
    const int mode = iz - nz * (2 * iz / nz);
    const double k = tz * mode, k2 = k * k;
    Dz[iz] = std::exp(sig_z2 * k2);
    for (size_t m = 0; m < M; ++m)
      Bz[m * nz + iz] = std::exp(-0.5 * bandwidth[m] * k2);
  }

#pragma omp parallel for schedule(static)
  for (int iz = 0; iz < nz; ++iz) {
    const int mz = iz - nz * (2 * iz / nz);
    const double kz = tz * mz;
    for (int iyl = 0; iyl < YL; ++iyl) {
      const int iy = ylo + iyl;
      const int my = iy - ny * (2 * iy / ny);
      const double ky = ty * my;
      for (int ix = 0; ix < nx; ++ix) {
        const int mx = ix - nx * (2 * ix / nx);
        const double kx = tx * mx;
        const double sqk = kx * kx + ky * ky + kz * kz;
        if (!(sqk > 0.0 && sqk <= k_sq_max)) continue;   // k == 0 dropped
        const size_t idx = static_cast<size_t>(ix) +
                           nx * (static_cast<size_t>(iyl) +
                                 static_cast<size_t>(ystride) * static_cast<size_t>(iz));
        double kfac = 0.0, kfacv = 0.0;
        for (size_t m = 0; m < M; ++m) {
          const double p = Bx[m * nx + ix] * By[m * ny + iy] * Bz[m * nz + iz];
          kfac += amp[m] * p;
          kfacv += amp[m] * bandwidth[m] * p;   // amp_virial[m] = amp[m]*beta[m]
        }
        const double dc = Dx[ix] * Dy[iy] * Dz[iz];
        ge[idx] = kfac * dc;
        gf[idx] = kfac * dc;
        gv[idx] = kfacv * dc;
      }
    }
  }
}

// Half-spectrum slab tables: only the compact x column (Nh = nx/2+1) is
// stored, because the (x,y) transform is real-to-complex and the green
// function depends on kx^2 only.  Layout: idx = kh + Nh*(iyl + YL*iz).
// Values are identical to the full table's column ix = kh.
inline void build_green_slab_half(int nx, int ny, int nz, double lx,
                                  double ly, double lz,
                                  const std::vector<double> &amp,
                                  const std::vector<double> &bandwidth,
                                  int spline, int ylo, int YL,
                                  std::vector<double> &ge,
                                  std::vector<double> &gf,
                                  std::vector<double> &gv) {
  const int Nh = nx / 2 + 1;
  const size_t ng = static_cast<size_t>(Nh) * YL * nz;
  ge.assign(ng, 0.0);
  gf.assign(ng, 0.0);
  gv.assign(ng, 0.0);

  const double dx = lx / nx, dy = ly / ny, dz = lz / nz;
  const double k_sq_max = M_PI * M_PI *
                          (1.0 / (dx * dx) + 1.0 / (dy * dy) + 1.0 / (dz * dz));
  const double tx = 2.0 * M_PI / lx, ty = 2.0 * M_PI / ly, tz = 2.0 * M_PI / lz;
  const double xi = (spline == 6) ? kCubes2Xi6 : kCubes2Xi4;
  const double sig_y2 = (xi * dy) * (xi * dy);
  const double sig_z2 = (xi * dz) * (xi * dz);
  const size_t M = amp.size();
  std::vector<double> Bx(M * Nh), By(M * ny), Bz(M * nz);
  std::vector<double> Dx(Nh), Dy(ny), Dz(nz);
  for (int kh = 0; kh < Nh; ++kh) {
    const double k = tx * kh, k2 = k * k;   // +kx representative
    Dx[kh] = std::exp((xi * dx) * (xi * dx) * k2);
    for (size_t m = 0; m < M; ++m) Bx[m * Nh + kh] = std::exp(-0.5 * bandwidth[m] * k2);
  }
  for (int iy = 0; iy < ny; ++iy) {
    const int mode = iy - ny * (2 * iy / ny);
    const double k = ty * mode, k2 = k * k;
    Dy[iy] = std::exp(sig_y2 * k2);
    for (size_t m = 0; m < M; ++m) By[m * ny + iy] = std::exp(-0.5 * bandwidth[m] * k2);
  }
  for (int iz = 0; iz < nz; ++iz) {
    const int mode = iz - nz * (2 * iz / nz);
    const double k = tz * mode, k2 = k * k;
    Dz[iz] = std::exp(sig_z2 * k2);
    for (size_t m = 0; m < M; ++m) Bz[m * nz + iz] = std::exp(-0.5 * bandwidth[m] * k2);
  }

#pragma omp parallel for schedule(static)
  for (int iz = 0; iz < nz; ++iz) {
    const int mz = iz - nz * (2 * iz / nz);
    const double kz = tz * mz;
    for (int iyl = 0; iyl < YL; ++iyl) {
      const int iy = ylo + iyl;
      const int my = iy - ny * (2 * iy / ny);
      const double ky = ty * my;
      for (int kh = 0; kh < Nh; ++kh) {
        const double kx = tx * kh;
        const double sqk = kx * kx + ky * ky + kz * kz;
        if (!(sqk > 0.0 && sqk <= k_sq_max)) continue;   // k == 0 dropped
        const size_t idx = static_cast<size_t>(kh) +
                           Nh * (static_cast<size_t>(iyl) +
                                 static_cast<size_t>(YL) * static_cast<size_t>(iz));
        double kfac = 0.0, kfacv = 0.0;
        for (size_t m = 0; m < M; ++m) {
          const double p = Bx[m * Nh + kh] * By[m * ny + iy] * Bz[m * nz + iz];
          kfac += amp[m] * p;
          kfacv += amp[m] * bandwidth[m] * p;
        }
        const double dc = Dx[kh] * Dy[iy] * Dz[iz];
        ge[idx] = kfac * dc;
        gf[idx] = kfac * dc;
        gv[idx] = kfacv * dc;
      }
    }
  }
}

// Full-grid tables (replicated path and LAMMPS kspace).
inline void build_green(int nx, int ny, int nz, double lx, double ly, double lz,
                        const std::vector<double> &amp,
                        const std::vector<double> &bandwidth, int spline,
                        std::vector<double> &ge, std::vector<double> &gf,
                        std::vector<double> &gv) {
  build_green_range(nx, ny, nz, lx, ly, lz, amp, bandwidth, spline, 0, ny, ny,
                    ge, gf, gv);
}

// This rank's y-slab of the tables (distributed path): YL*nx*nz entries,
// indexed ix + nx*(iyl + YL*iz).  The green pencil kernel consumes exactly
// this layout, so per-rank green memory is 1/np of the full tables.
inline void build_green_slab(int nx, int ny, int nz, double lx, double ly,
                             double lz, const std::vector<double> &amp,
                             const std::vector<double> &bandwidth, int spline,
                             int ylo, int YL, std::vector<double> &ge,
                             std::vector<double> &gf,
                             std::vector<double> &gv) {
  build_green_range(nx, ny, nz, lx, ly, lz, amp, bandwidth, spline, ylo, YL,
                    YL, ge, gf, gv);
}

}  // namespace sog_green

#endif  // SOG_GREEN_H
