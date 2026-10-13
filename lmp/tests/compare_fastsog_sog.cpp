/**
 * Standalone comparison: fastsog.cpp vs sog.cpp core algorithms.
 *
 * Compares:
 *   1. |Phi(k)|^2  (influence function for CubeS2)
 *   2. Green functions (geff_energy, geff_force, geff_virial)
 *   3. Charge spreading + FFT + energy / forces / virial
 *
 * Build:
 *   g++ -std=c++17 -O2 -o compare_fastsog_sog compare_fastsog_sog.cpp -lm
 *
 * Run:
 *   ./compare_fastsog_sog
 */

#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// ───────────────────────────────────────────────────────────────
// Shared constants
// ───────────────────────────────────────────────────────────────
constexpr double MY_PI  = 3.14159265358979323846;
constexpr double MY_2PI = 6.28318530717958647692;

// Test grid: small box, small grid for manual DFT
constexpr int    NX = 4, NY = 4, NZ = 4;
constexpr double LX = 10.0, LY = 10.0, LZ = 10.0;
constexpr int    NGRID = NX * NY * NZ;

// SOG kernel parameters
constexpr double B_VAL     = 2.0;
constexpr double SIGMA_VAL = 2.0;
constexpr int    M_VAL     = 4;
constexpr double RCUT      = 10.0;    // needed for fastsog w0

// ───────────────────────────────────────────────────────────────
// CubeS₂ 4th-order structures (shared between both)
// ───────────────────────────────────────────────────────────────
struct CubeS2Node4 {
  int dx, dy, dz, cls, sp_axis, sp_is_neg;
};

constexpr double XI_4 = 0.5773502691896258;  // 1/√3

constexpr CubeS2Node4 NODES_4[32] = {
  { 0, 0, 0, 0, -1, 0}, { 0, 0, 1, 0, -1, 0}, { 0, 1, 0, 0, -1, 0}, { 0, 1, 1, 0, -1, 0},
  { 1, 0, 0, 0, -1, 0}, { 1, 0, 1, 0, -1, 0}, { 1, 1, 0, 0, -1, 0}, { 1, 1, 1, 0, -1, 0},
  {-1, 0, 0, 1, 0, 1}, {-1, 0, 1, 1, 0, 1}, {-1, 1, 0, 1, 0, 1}, {-1, 1, 1, 1, 0, 1},
  { 2, 0, 0, 1, 0, 0}, { 2, 0, 1, 1, 0, 0}, { 2, 1, 0, 1, 0, 0}, { 2, 1, 1, 1, 0, 0},
  { 0,-1, 0, 1, 1, 1}, { 0,-1, 1, 1, 1, 1}, { 1,-1, 0, 1, 1, 1}, { 1,-1, 1, 1, 1, 1},
  { 0, 2, 0, 1, 1, 0}, { 0, 2, 1, 1, 1, 0}, { 1, 2, 0, 1, 1, 0}, { 1, 2, 1, 1, 1, 0},
  { 0, 0,-1, 1, 2, 1}, { 0, 1,-1, 1, 2, 1}, { 1, 0,-1, 1, 2, 1}, { 1, 1,-1, 1, 2, 1},
  { 0, 0, 2, 1, 2, 0}, { 0, 1, 2, 1, 2, 0}, { 1, 0, 2, 1, 2, 0}, { 1, 1, 2, 1, 2, 0},
};

// ───────────────────────────────────────────────────────────────
// Shared: CubeS₂ weight functions
// ───────────────────────────────────────────────────────────────
inline double cubes2_L(double theta, double xi) {
  return -0.5*theta*theta*theta + 0.5*theta*theta
         - (9.0*xi*xi - 2.0)/6.0*theta + 0.5*xi*xi;
}
inline double cubes2_R(double theta, double xi) {
  return (1.0/6.0)*theta*theta*theta + (3.0*xi*xi - 1.0)/6.0*theta;
}

inline double cubes2_weight_4(double tx, double ty, double tz,
                              const CubeS2Node4 &node, double xi) {
  if (node.cls == 0) {
    double ex = (node.dx == 0) ? tx : (1.0 - tx);
    double ey = (node.dy == 0) ? ty : (1.0 - ty);
    double ez = (node.dz == 0) ? tz : (1.0 - tz);
    return cubes2_L(ex,xi)*ey*ez + cubes2_L(ey,xi)*ex*ez + cubes2_L(ez,xi)*ex*ey;
  } else {
    double es, en1, en2;
    if (node.sp_axis == 0) {
      es = node.sp_is_neg ? tx : (1.0 - tx);
      en1 = (node.dy == 0) ? ty : (1.0 - ty);
      en2 = (node.dz == 0) ? tz : (1.0 - tz);
    } else if (node.sp_axis == 1) {
      es = node.sp_is_neg ? ty : (1.0 - ty);
      en1 = (node.dx == 0) ? tx : (1.0 - tx);
      en2 = (node.dz == 0) ? tz : (1.0 - tz);
    } else {
      es = node.sp_is_neg ? tz : (1.0 - tz);
      en1 = (node.dx == 0) ? tx : (1.0 - tx);
      en2 = (node.dy == 0) ? ty : (1.0 - ty);
    }
    return cubes2_R(es,xi) * en1 * en2;
  }
}

// ───────────────────────────────────────────────────────────────
// Shared: 1D Fourier integrals I_p(alpha) = ∫₀¹ t^p e^{iαt} dt
// ───────────────────────────────────────────────────────────────
inline std::complex<double> I_int(int p, double alpha) {
  if (std::abs(alpha) < 1e-12)
    return std::complex<double>(1.0/(p+1), 0.0);

  std::complex<double> eia(std::cos(alpha), std::sin(alpha));
  std::complex<double> ia(0.0, alpha);
  double a2 = alpha * alpha;

  if (p == 0) return (eia - 1.0) / ia;
  if (p == 1) return ((1.0 - ia) * eia - 1.0) / a2;
  if (p == 2) return ((2.0 - 2.0*ia - a2) * eia - 2.0) / (ia * a2);
  // p == 3
  double a4 = a2 * a2;
  std::complex<double> ia3(0.0, a2 * alpha);
  return ((6.0 - 6.0*ia - 3.0*a2 + ia3) * eia - 6.0) / a4;
}

// ───────────────────────────────────────────────────────────────
// Shared: SOG spectral kernel and band-limited virial kernel
// ───────────────────────────────────────────────────────────────
inline double spectral_kernel(double ksq,
                               const std::vector<double> &amp,
                               const std::vector<double> &bw) {
  double s = 0.0;
  for (size_t m = 0; m < amp.size(); ++m)
    s += amp[m] * std::exp(-0.5 * bw[m] * ksq);
  return s;
}
inline double virial_kernel(double ksq,
                            const std::vector<double> &amp,
                            const std::vector<double> &bw) {
  double s = 0.0;
  for (size_t m = 0; m < amp.size(); ++m)
    s += amp[m] * bw[m] * std::exp(-0.5 * bw[m] * ksq);
  return s;
}

// ───────────────────────────────────────────────────────────────
// fastsog: auto-generate amp/bandwidth (includes w0, self_coeff)
// ───────────────────────────────────────────────────────────────
struct FastSOGParams {
  std::vector<double> amp, bandwidth;
  double w0, self_coeff;
};

FastSOGParams fastsog_make_params(double b, double sigma, int M, double rcut) {
  FastSOGParams p;
  double sigma2 = sigma * sigma;
  double logb = std::log(b);

  // ── fastsog w0 (u-series continuity at rcut) ──
  double r0 = rcut / sigma;
  auto G = [](double s, double r) {
    return std::exp(-r*r/(2*s*s)) / std::sqrt(2*MY_PI*s*s);
  };
  double S = 0.0, absS = 0.0;
  for (int i = 1; i <= 500; ++i) {
    double bi = std::pow(b, -i);
    double term = bi * G(1.0, bi * r0);
    S += term; absS += std::abs(term);
    if (i >= 10 && std::abs(term) < 1e-14 * std::max(absS, 1e-30)) break;
  }
  p.w0 = (1.0 / G(1.0, r0)) * (1.0 / (2.0 * logb * r0) - S);

  // ── amp / bandwidth (fastsog convention: amp[0] includes w0) ──
  p.bandwidth.resize(M);
  p.amp.resize(M);
  p.bandwidth[0] = sigma2;
  p.amp[0] = 4.0 * MY_PI * logb * p.w0 * sigma2;
  double b2 = b * b;
  for (int m = 1; m < M; ++m) {
    p.bandwidth[m] = p.bandwidth[m-1] * b2;
    p.amp[m] = p.amp[m-1] * b2;
  }

  // ── self_coeff ──
  double sum_b = 0.0;
  for (int m = 1; m < M; ++m) sum_b += std::pow(b, -m);
  p.self_coeff = logb / (std::sqrt(2*MY_PI) * sigma) * (p.w0 + sum_b);

  return p;
}

// ───────────────────────────────────────────────────────────────
// sog.cpp-style: auto-generate amp/bandwidth (no w0)
// ───────────────────────────────────────────────────────────────
struct SOGParams {
  std::vector<double> amp, bandwidth;
};

SOGParams sog_make_params(double b, double sigma, int M) {
  SOGParams p;
  double sigma2 = sigma * sigma;
  double logb = std::log(b);
  double b2 = b * b;

  p.bandwidth.resize(M);
  p.amp.resize(M);

  // sog.cpp convention: amp[m] = 4π·ln(b) · bandwidth[m]  for all m
  double bw_m = sigma2;
  for (int m = 0; m < M; ++m) {
    p.bandwidth[m] = bw_m;
    p.amp[m] = 4.0 * MY_PI * logb * bw_m;
    bw_m *= b2;
  }

  return p;
}

// ───────────────────────────────────────────────────────────────
// fastsog: analytic |Phi(k)|^2 via monomial expansion
// ───────────────────────────────────────────────────────────────
// (Simplified monomial structure — just enough for correctness check)
struct MonomialTerm { int px, py, pz; double coeff; };
struct NodeMonomial { int n; MonomialTerm terms[64]; };

void build_monomials(const CubeS2Node4 &node, double xi, NodeMonomial &mono) {
  mono.n = 0;
  int dx = node.dx, dy = node.dy, dz = node.dz;
  double a[3] = {(double)dx, (double)dy, (double)dz};
  double b[3] = {1.0-2.0*a[0], 1.0-2.0*a[1], 1.0-2.0*a[2]};
  double xi2 = xi*xi;

  auto binom = [](int n, int k) -> double {
    constexpr double C[4][4] = {{1,0,0,0},{1,1,0,0},{1,2,1,0},{1,3,3,1}};
    return (k>=0 && k<=n) ? C[n][k] : 0.0;
  };

  // Accumulate: monomial -> coeff  (max 64 terms per node)
  auto add = [&](int px, int py, int pz, double c) {
    if (std::abs(c) < 1e-30) return;
    for (int i = 0; i < mono.n; ++i)
      if (mono.terms[i].px==px && mono.terms[i].py==py && mono.terms[i].pz==pz) {
        mono.terms[i].coeff += c; return;
      }
    if (mono.n < 64) mono.terms[mono.n++] = {px, py, pz, c};
  };

  if (node.cls == 0) {
    double xi2_adj = (9.0*xi2 - 2.0)/6.0;
    double Lc[4] = {0.5*xi2, -xi2_adj, 0.5, -0.5};
    for (int term_idx = 0; term_idx < 3; ++term_idx) {
      int axL = term_idx, ax1 = (term_idx+1)%3, ax2 = (term_idx+2)%3;
      for (int pL = 0; pL <= 3; ++pL) {
        double cL = Lc[pL]; if (cL == 0.0) continue;
        for (int jL = 0; jL <= pL; ++jL) {
          double cfL = cL * binom(pL,jL) * std::pow(a[axL],pL-jL) * std::pow(b[axL],jL);
          for (int j1 = 0; j1 <= 1; ++j1) {
            double cf1 = binom(1,j1)*std::pow(a[ax1],1-j1)*std::pow(b[ax1],j1);
            for (int j2 = 0; j2 <= 1; ++j2) {
              double cf2 = binom(1,j2)*std::pow(a[ax2],1-j2)*std::pow(b[ax2],j2);
              double coeff = cfL * cf1 * cf2;
              if (coeff == 0.0) continue;
              int pows[3] = {0,0,0};
              pows[axL]=jL; pows[ax1]=j1; pows[ax2]=j2;
              add(pows[0],pows[1],pows[2],coeff);
            }
          }
        }
      }
    }
  } else {
    double Rc[4] = {0.0, (3.0*xi2-1.0)/6.0, 0.0, 1.0/6.0};
    int axL = node.sp_axis, ax1 = (axL+1)%3, ax2 = (axL+2)%3;
    for (int pL = 0; pL <= 3; ++pL) {
      double cR = Rc[pL]; if (cR == 0.0) continue;
      for (int jL = 0; jL <= pL; ++jL) {
        double cfL = cR * binom(pL,jL) * std::pow(a[axL],pL-jL) * std::pow(b[axL],jL);
        for (int j1 = 0; j1 <= 1; ++j1) {
          double cf1 = binom(1,j1)*std::pow(a[ax1],1-j1)*std::pow(b[ax1],j1);
          for (int j2 = 0; j2 <= 1; ++j2) {
            double cf2 = binom(1,j2)*std::pow(a[ax2],1-j2)*std::pow(b[ax2],j2);
            double coeff = cfL * cf1 * cf2;
            if (coeff == 0.0) continue;
            int pows[3] = {0,0,0};
            pows[axL]=jL; pows[ax1]=j1; pows[ax2]=j2;
            add(pows[0],pows[1],pows[2],coeff);
          }
        }
      }
    }
  }
}

std::vector<double> fastsog_influence_sq() {
  // -- Exactly replicates fastsog.cpp precompute_cubes2_influence() --
  std::vector<double> inf_sq(NGRID, 0.0);

  NodeMonomial monos[32];
  for (int k = 0; k < 32; ++k) build_monomials(NODES_4[k], XI_4, monos[k]);

  double dx = LX/NX, dy = LY/NY, dz = LZ/NZ;
  double twx = MY_2PI/LX, twy = MY_2PI/LY, twz = MY_2PI/LZ;

  // Precompute 1D integrals I_p for each axis
  std::vector<std::complex<double>> Ipx[4], Ipy[4], Ipz[4];
  for (int p = 0; p < 4; ++p) {
    Ipx[p].resize(NX); Ipy[p].resize(NY); Ipz[p].resize(NZ);
  }
  for (int ix = 0; ix < NX; ++ix) {
    int km = ix - NX*(2*ix/NX);
    double ax = twx * km * dx;
    for (int p = 0; p < 4; ++p) Ipx[p][ix] = I_int(p, ax);
  }
  for (int iy = 0; iy < NY; ++iy) {
    int km = iy - NY*(2*iy/NY);
    double ay = twy * km * dy;
    for (int p = 0; p < 4; ++p) Ipy[p][iy] = I_int(p, ay);
  }
  for (int iz = 0; iz < NZ; ++iz) {
    int km = iz - NZ*(2*iz/NZ);
    double az = twz * km * dz;
    for (int p = 0; p < 4; ++p) Ipz[p][iz] = I_int(p, az);
  }

  for (int iz = 0; iz < NZ; ++iz) {
    int kzm = iz - NZ*(2*iz/NZ);
    double kz = twz * kzm;
    for (int iy = 0; iy < NY; ++iy) {
      int kym = iy - NY*(2*iy/NY);
      double ky = twy * kym;
      for (int ix = 0; ix < NX; ++ix) {
        int kxm = ix - NX*(2*ix/NX);
        double kx = twx * kxm;
        double sqk = kx*kx + ky*ky + kz*kz;
        if (sqk == 0.0) continue;

        int idx = (iz*NY + iy)*NX + ix;
        std::complex<double> phi(0,0);

        for (int d = 0; d < 32; ++d) {
          const auto &node = NODES_4[d];
          const auto &mono = monos[d];
          double phase = kx*node.dx*dx + ky*node.dy*dy + kz*node.dz*dz;
          std::complex<double> eikd(std::cos(phase), std::sin(phase));
          std::complex<double> integral(0,0);
          for (int m = 0; m < mono.n; ++m) {
            integral += mono.terms[m].coeff *
              Ipx[mono.terms[m].px][ix] *
              Ipy[mono.terms[m].py][iy] *
              Ipz[mono.terms[m].pz][iz];
          }
          phi += eikd * integral;
        }
        inf_sq[idx] = phi.real()*phi.real() + phi.imag()*phi.imag();
      }
    }
  }
  return inf_sq;
}

// ───────────────────────────────────────────────────────────────
// sog.cpp: ensemble-averaged |Phi(k)|^2 (simplified for small grid)
// ───────────────────────────────────────────────────────────────
std::vector<double> sog_influence_sq() {
  // Uses the same algorithm as sog.cpp precompute_cubes2_influence():
  // spread unit charge at random positions, FFT, accumulate |Phi|^2
  std::vector<double> inf_sq(NGRID, 0.0);
  const int N_SAMPLES = 64;

  double dx = LX/NX, dy = LY/NY, dz = LZ/NZ;

  // Simple LCG random generator
  uint64_t rng = 12345ULL;
  auto rand_u = [&]() -> double {
    rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
    return (double)((rng >> 11) & 0x1FFFFFFFFFFFFFULL) * 0x1.0p-53;
  };

  for (int sample = 0; sample < N_SAMPLES; ++sample) {
    double tx0 = rand_u(), ty0 = rand_u(), tz0 = rand_u();
    double fx = tx0 * NX, fy = ty0 * NY, fz = tz0 * NZ;
    int ix0 = (int)std::floor(fx), iy0 = (int)std::floor(fy), iz0 = (int)std::floor(fz);
    double tx = fx - ix0, ty = fy - iy0, tz = fz - iz0;

    // Spread unit charge
    std::vector<double> rho(NGRID, 0.0);
    double total_w = 0.0;
    for (int k = 0; k < 32; ++k) {
      double w = cubes2_weight_4(tx, ty, tz, NODES_4[k], XI_4);
      if (w == 0.0) continue;
      int gx = (ix0 + NODES_4[k].dx) % NX; if (gx < 0) gx += NX;
      int gy = (iy0 + NODES_4[k].dy) % NY; if (gy < 0) gy += NY;
      int gz = (iz0 + NODES_4[k].dz) % NZ; if (gz < 0) gz += NZ;
      rho[(gz*NY + gy)*NX + gx] += w;
      total_w += w;
    }
    if (total_w <= 0.0) continue;

    // Normalize & manual DFT (forward, no 1/N)
    std::vector<std::complex<double>> rho_k(NGRID, {0,0});
    double inv_tw = 1.0 / total_w;
    for (int iz = 0; iz < NZ; ++iz)
      for (int iy = 0; iy < NY; ++iy)
        for (int ix = 0; ix < NX; ++ix) {
          double val = rho[(iz*NY + iy)*NX + ix] * inv_tw;
          // DFT: sum_r f(r) * exp(-i k·r)
          // k = 2pi * (kxm/LX, kym/LY, kzm/LZ)
          // r = (ix*dx, iy*dy, iz*dz)
          for (int kz = 0; kz < NZ; ++kz) {
            int kzm = kz - NZ*(2*kz/NZ);
            double kz_val = MY_2PI * kzm / LZ;
            for (int ky = 0; ky < NY; ++ky) {
              int kym = ky - NY*(2*ky/NY);
              double ky_val = MY_2PI * kym / LY;
              for (int kx = 0; kx < NX; ++kx) {
                int kxm = kx - NX*(2*kx/NX);
                double kx_val = MY_2PI * kxm / LX;
                double phase = -(kx_val*ix*dx + ky_val*iy*dy + kz_val*iz*dz);
                rho_k[(kz*NY + ky)*NX + kx] +=
                  val * std::complex<double>(std::cos(phase), std::sin(phase));
              }
            }
          }
        }

    // Accumulate |Phi(k)|^2
    for (int kz = 0; kz < NZ; ++kz)
      for (int ky = 0; ky < NY; ++ky)
        for (int kx = 0; kx < NX; ++kx) {
          int idx = (kz*NY + ky)*NX + kx;
          double re = rho_k[idx].real(), im = rho_k[idx].imag();
          inf_sq[idx] += re*re + im*im;
        }
  }

  double inv_n = 1.0 / N_SAMPLES;
  for (auto &v : inf_sq) v *= inv_n;
  return inf_sq;
}

// ───────────────────────────────────────────────────────────────
// Manual DFT utilities for full computation test
// ───────────────────────────────────────────────────────────────
inline int mesh_idx(int ix, int iy, int iz) { return (iz*NY + iy)*NX + ix; }
inline int wrap(int i, int n) { int r = i % n; return r < 0 ? r + n : r; }

// Spread charges using CubeS2
void spread_charges(const std::vector<double> &q,
                    const std::vector<double> &x,
                    const std::vector<double> &y,
                    const std::vector<double> &z,
                    std::vector<double> &rho_grid,
                    double rho_scale) {
  std::fill(rho_grid.begin(), rho_grid.end(), 0.0);
  for (size_t i = 0; i < q.size(); ++i) {
    double fx = x[i] * NX / LX;
    double fy = y[i] * NY / LY;
    double fz = z[i] * NZ / LZ;
    int ix0 = (int)std::floor(fx), iy0 = (int)std::floor(fy), iz0 = (int)std::floor(fz);
    double tx = fx - ix0, ty = fy - iy0, tz = fz - iz0;

    for (int k = 0; k < 32; ++k) {
      double w = cubes2_weight_4(tx, ty, tz, NODES_4[k], XI_4);
      if (w == 0.0) continue;
      int gx = wrap(ix0 + NODES_4[k].dx, NX);
      int gy = wrap(iy0 + NODES_4[k].dy, NY);
      int gz = wrap(iz0 + NODES_4[k].dz, NZ);
      rho_grid[mesh_idx(gx, gy, gz)] += rho_scale * q[i] * w;
    }
  }
}

// Forward DFT (no 1/N)
void forward_dft(const std::vector<double> &grid_r,
                 std::vector<std::complex<double>> &grid_k) {
  double dx = LX/NX, dy = LY/NY, dz = LZ/NZ;
  for (int kz = 0; kz < NZ; ++kz) {
    int kzm = kz - NZ*(2*kz/NZ);
    double kz_v = MY_2PI * kzm / LZ;
    for (int ky = 0; ky < NY; ++ky) {
      int kym = ky - NY*(2*ky/NY);
      double ky_v = MY_2PI * kym / LY;
      for (int kx = 0; kx < NX; ++kx) {
        int kxm = kx - NX*(2*kx/NX);
        double kx_v = MY_2PI * kxm / LX;
        std::complex<double> sum(0,0);
        for (int iz = 0; iz < NZ; ++iz) {
          double z = iz * dz;
          for (int iy = 0; iy < NY; ++iy) {
            double y = iy * dy;
            for (int ix = 0; ix < NX; ++ix) {
              double x = ix * dx;
              double phase = -(kx_v*x + ky_v*y + kz_v*z);
              sum += grid_r[mesh_idx(ix,iy,iz)] *
                     std::complex<double>(std::cos(phase), std::sin(phase));
            }
          }
        }
        grid_k[mesh_idx(kx,ky,kz)] = sum;
      }
    }
  }
}

// Backward DFT (no 1/N)
void backward_dft(const std::vector<std::complex<double>> &grid_k,
                  std::vector<double> &grid_r) {
  double dx = LX/NX, dy = LY/NY, dz = LZ/NZ;
  for (int iz = 0; iz < NZ; ++iz) {
    double z = iz * dz;
    for (int iy = 0; iy < NY; ++iy) {
      double y = iy * dy;
      for (int ix = 0; ix < NX; ++ix) {
        double x = ix * dx;
        std::complex<double> sum(0,0);
        for (int kz = 0; kz < NZ; ++kz) {
          int kzm = kz - NZ*(2*kz/NZ);
          double kz_v = MY_2PI * kzm / LZ;
          for (int ky = 0; ky < NY; ++ky) {
            int kym = ky - NY*(2*ky/NY);
            double ky_v = MY_2PI * kym / LY;
            for (int kx = 0; kx < NX; ++kx) {
              int kxm = kx - NX*(2*kx/NX);
              double kx_v = MY_2PI * kxm / LX;
              double phase = kx_v*x + ky_v*y + kz_v*z;
              sum += grid_k[mesh_idx(kx,ky,kz)] *
                     std::complex<double>(std::cos(phase), std::sin(phase));
            }
          }
        }
        grid_r[mesh_idx(ix,iy,iz)] = sum.real();
      }
    }
  }
}

// ───────────────────────────────────────────────────────────────
// Main comparison routine
// ───────────────────────────────────────────────────────────────
int main() {
  printf("=== fastsog vs sog.cpp standalone comparison ===\n");
  printf("Grid: %d x %d x %d, Box: %.1f x %.1f x %.1f\n", NX, NY, NZ, LX, LY, LZ);
  printf("SOG: b=%.1f, sigma=%.1f, M=%d, rcut=%.1f\n\n", B_VAL, SIGMA_VAL, M_VAL, RCUT);

  // ── Generate kernel parameters ──
  auto fp = fastsog_make_params(B_VAL, SIGMA_VAL, M_VAL, RCUT);
  auto sp = sog_make_params(B_VAL, SIGMA_VAL, M_VAL);

  printf("--- Kernel parameters ---\n");
  printf("fastsog w0 = %.10f, self_coeff = %.10f\n", fp.w0, fp.self_coeff);
  printf("%4s %16s %16s %16s %16s\n", "m", "fs_amp", "sog_amp", "fs_bw", "sog_bw");
  for (int m = 0; m < M_VAL; ++m) {
    printf("%4d %16.6f %16.6f %16.6f %16.6f\n",
           m, fp.amp[m], sp.amp[m], fp.bandwidth[m], sp.bandwidth[m]);
  }
  printf("amp ratio (fs/sog):");
  for (int m = 0; m < M_VAL; ++m) printf(" %.4f", fp.amp[m]/sp.amp[m]);
  printf("\n\n");

  // ── Compare |Phi(k)|^2 ──
  auto fs_inf = fastsog_influence_sq();
  auto sog_inf = sog_influence_sq();

  printf("--- |Phi(k)|^2 comparison ---\n");
  printf("%4s %4s %4s %16s %16s %12s\n", "kx", "ky", "kz", "fastsog", "sog", "ratio");
  int count = 0;
  for (int iz = 0; iz < NZ && count < 25; ++iz) {
    int kzm = iz - NZ*(2*iz/NZ);
    for (int iy = 0; iy < NY && count < 25; ++iy) {
      int kym = iy - NY*(2*iy/NY);
      for (int ix = 0; ix < NX && count < 25; ++ix) {
        int kxm = ix - NX*(2*ix/NX);
        int idx = mesh_idx(ix, iy, iz);
        double fs = fs_inf[idx], so = sog_inf[idx];
        if (fs > 1e-10 || so > 1e-10) {
          double ratio = fs > 1e-20 ? so/fs : 0.0;
          printf("%4d %4d %4d %16.10f %16.10f %12.6f\n",
                 kxm, kym, kzm, fs, so, ratio);
          count++;
        }
      }
    }
  }
  printf("\n");

  // ── Compare Green functions ──
  printf("--- Green functions (using fastsog analytic |Phi|^2, NO ngrid scaling) ---\n");
  printf("Note: Green function comparison uses the SAME amp/bandwidth (fastsog) for both.\n");
  printf("%4s %4s %4s %16s %16s %16s\n", "kx", "ky", "kz", "g_energy", "g_force", "g_virial");
  count = 0;
  double twx = MY_2PI/LX, twy = MY_2PI/LY, twz = MY_2PI/LZ;
  for (int iz = 0; iz < NZ && count < 15; ++iz) {
    int kzm = iz - NZ*(2*iz/NZ);
    double kz = twz * kzm;
    for (int iy = 0; iy < NY && count < 15; ++iy) {
      int kym = iy - NY*(2*iy/NY);
      double ky = twy * kym;
      for (int ix = 0; ix < NX && count < 15; ++ix) {
        int kxm = ix - NX*(2*ix/NX);
        double kx = twx * kxm;
        double ksq = kx*kx + ky*ky + kz*kz;
        if (ksq <= 0.0) continue;

        int idx = mesh_idx(ix, iy, iz);
        double inf_sq_fs = fs_inf[idx];
        if (inf_sq_fs <= 1e-20) continue;

        double kfac = spectral_kernel(ksq, fp.amp, fp.bandwidth);
        double vfac = virial_kernel(ksq, fp.amp, fp.bandwidth);
        double ge = kfac / inf_sq_fs;
        double gf = kfac / inf_sq_fs;  // alias fast path
        double gv = vfac / inf_sq_fs;

        printf("%4d %4d %4d %16.8f %16.8f %16.8f\n",
               kxm, kym, kzm, ge, gf, gv);
        count++;
      }
    }
  }
  printf("\n");

  // ── Compare charge spreading + energy ──
  printf("--- Full energy / virial test ---\n");
  // Simple test system: 2 charges (±1) in a neutral pair
  std::vector<double> q = {1.0, -1.0};
  std::vector<double> x = {3.0, 7.0};
  std::vector<double> y = {5.0, 5.0};
  std::vector<double> z = {5.0, 5.0};

  double volume = LX * LY * LZ;
  double rho_scale = (double)NGRID / volume;
  double scaleinv = 1.0 / (double)NGRID;
  double s2 = scaleinv * scaleinv;

  std::vector<double> rho_grid(NGRID);
  spread_charges(q, x, y, z, rho_grid, rho_scale);

  // Forward DFT
  std::vector<std::complex<double>> rho_k(NGRID);
  forward_dft(rho_grid, rho_k);

  // Compute energy and virial using fastsog formulas
  double energy_fs = 0.0, virial_fs[6] = {0};
  for (int iz = 0; iz < NZ; ++iz) {
    int kzm = iz - NZ*(2*iz/NZ);
    double kz = twz * kzm;
    for (int iy = 0; iy < NY; ++iy) {
      int kym = iy - NY*(2*iy/NY);
      double ky = twy * kym;
      for (int ix = 0; ix < NX; ++ix) {
        int kxm = ix - NX*(2*ix/NX);
        double kx = twx * kxm;
        double ksq = kx*kx + ky*ky + kz*kz;
        if (ksq <= 0.0) continue;

        int idx = mesh_idx(ix, iy, iz);
        double inf_sq = fs_inf[idx];
        if (inf_sq <= 1e-20) continue;

        double kfac = spectral_kernel(ksq, fp.amp, fp.bandwidth);
        double vfac = virial_kernel(ksq, fp.amp, fp.bandwidth);
        double ge = kfac / inf_sq;
        double gv = vfac / inf_sq;

        double rho_sq = std::norm(rho_k[idx]);

        energy_fs += s2 * ge * rho_sq;
        virial_fs[0] += s2 * rho_sq * (ge - gv * kx * kx);
        virial_fs[1] += s2 * rho_sq * (ge - gv * ky * ky);
        virial_fs[2] += s2 * rho_sq * (ge - gv * kz * kz);
        virial_fs[3] += s2 * rho_sq * (-gv * kx * ky);
        virial_fs[4] += s2 * rho_sq * (-gv * kx * kz);
        virial_fs[5] += s2 * rho_sq * (-gv * ky * kz);
      }
    }
  }

  double qscale = 14.3996454784255;  // qqrd2e for 'real' units
  double E_fs = 0.5 * volume * qscale * energy_fs;
  printf("Energy (fastsog style): %.8f kcal/mol\n", E_fs);

  double vs = 0.5 * volume * qscale;
  printf("Virial (fastsog style, analytic):\n");
  printf("  xx=%.6f yy=%.6f zz=%.6f xy=%.6f xz=%.6f yz=%.6f\n",
         vs*virial_fs[0], vs*virial_fs[1], vs*virial_fs[2],
         vs*virial_fs[3], vs*virial_fs[4], vs*virial_fs[5]);

  // ── Compare influence function key stats ──
  printf("\n--- Influence function summary ---\n");
  double fs_sum = 0, sog_sum = 0;
  double fs_max_diff = 0;
  for (int i = 0; i < NGRID; ++i) {
    fs_sum += fs_inf[i];
    sog_sum += sog_inf[i];
    double diff = std::abs(fs_inf[i] - sog_inf[i]);
    if (diff > fs_max_diff) fs_max_diff = diff;
  }
  printf("fastsog sum = %.6f, sog sum = %.6f\n", fs_sum, sog_sum);
  printf("Max |diff| = %.6e\n", fs_max_diff);

  // DC mode check
  printf("DC mode (0,0,0): fastsog=%.10f, sog=%.10f\n",
         fs_inf[0], sog_inf[0]);

  return 0;
}
