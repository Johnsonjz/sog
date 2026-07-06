"""Analytic CubeS₂ influence function via 1D Fourier integrals I_p(α).

Mirrors sog.cpp precompute_cubes2_influence() — EXACT (no MC noise).

Φ(k) = Σ_d exp(i·k·d·Δ) · Σ_{pqr} C_d(p,q,r) · I_p(αx)·I_q(αy)·I_r(αz)

where I_p(α) = ∫₀¹ t^p · exp(i·α·t) dt  (closed-form + Taylor for |α| < 1e-8).
"""

import math
import torch
from typing import Dict, List, Tuple

from .cubes2_spline import get_nodes

# ── 1D analytic Fourier integrals ──

def _I_int_0(alpha: float) -> complex:
    """I₀(α) = ∫₀¹ exp(iαt) dt"""
    if alpha < 0:
        return _I_int_0(-alpha).conjugate()
    a = abs(alpha)
    if a < 1e-8:
        return complex(1.0 - a*a/6.0, alpha/2.0 - alpha*alpha*alpha/24.0)
    return complex(math.sin(alpha)/alpha, (1.0 - math.cos(alpha))/alpha)


def _I_int_1(alpha: float) -> complex:
    """I₁(α) = ∫₀¹ t·exp(iαt) dt"""
    if alpha < 0:
        return _I_int_1(-alpha).conjugate()
    a = abs(alpha)
    if a < 1e-8:
        return complex(0.5 - a*a/8.0, alpha/3.0 - alpha*alpha*alpha/30.0)
    cos_a = math.cos(alpha); sin_a = math.sin(alpha)
    a2 = alpha * alpha
    return complex((alpha*sin_a + cos_a - 1.0)/a2, (sin_a - alpha*cos_a)/a2)


def _I_int_2(alpha: float) -> complex:
    """I₂(α) = ∫₀¹ t²·exp(iαt) dt"""
    if alpha < 0:
        return _I_int_2(-alpha).conjugate()
    a = abs(alpha)
    if a < 1e-8:
        return complex(1.0/3.0 - a*a/10.0, alpha/4.0)
    cos_a = math.cos(alpha); sin_a = math.sin(alpha)
    a2 = alpha * alpha; a3 = a2 * alpha
    return complex((2.0*alpha*sin_a + (a2-2.0)*cos_a + 2.0)/a3,
                   ((a2-2.0)*sin_a + 2.0*alpha*cos_a)/a3)


def _I_int_3(alpha: float) -> complex:
    """I₃(α) = ∫₀¹ t³·exp(iαt) dt"""
    if alpha < 0:
        return _I_int_3(-alpha).conjugate()
    a = abs(alpha)
    if a < 1e-8:
        return complex(0.25, alpha/5.0)
    cos_a = math.cos(alpha); sin_a = math.sin(alpha)
    a2 = alpha * alpha; a3 = a2 * alpha; a4 = a3 * alpha
    return complex(((3.0*a2-6.0)*alpha*sin_a + (a3-6.0*alpha)*cos_a + 6.0*alpha)/a4,
                   ((a3-6.0*alpha)*sin_a + (6.0-3.0*a2)*cos_a + 3.0*a2 - 6.0)/a4)


_I_TABLE = [_I_int_0, _I_int_1, _I_int_2, _I_int_3]


# ── Monomial expansion for CubeS₂ 4th-order ──

def _binom(n: int, k: int) -> float:
    if k < 0 or k > n:
        return 0.0
    C = [[1, 0, 0, 0],
         [1, 1, 0, 0],
         [1, 2, 1, 0],
         [1, 3, 3, 1]]
    return C[n][k]


def _build_monomials_for_node(node, xi: float) -> List[Tuple[int, int, int, float]]:
    """Expand CubeS₂ node weight into monomials Σ (px,py,pz,coeff).

    Uses the same geometric convention as C++ fastsog.cpp build_monomials_for_node():
    η_i = d_i + (1-2d_i)·θ_i for ALL axes (including class-1 special axes).
    """
    dx, dy, dz = node.dx, node.dy, node.dz
    a = [float(dx), float(dy), float(dz)]
    b = [1.0 - 2.0 * ax for ax in a]

    xi2 = xi * xi

    terms: Dict[Tuple[int, int, int], float] = {}

    def _add(pows, coeff):
        if coeff == 0.0:
            return
        key = (pows[0], pows[1], pows[2])
        terms[key] = terms.get(key, 0.0) + coeff

    if hasattr(node, 'cls') and node.cls == 0:
        # Class 0: c_d = L(ηx)·ηy·ηz + L(ηy)·ηz·ηx + L(ηz)·ηx·ηy
        xi2_adj = (9.0 * xi2 - 2.0) / 6.0
        L_coeffs = [0.5 * xi2, -xi2_adj, 0.5, -0.5]

        for term_idx in range(3):
            axis_L = term_idx
            axis_n1 = (term_idx + 1) % 3
            axis_n2 = (term_idx + 2) % 3

            for pL in range(4):
                c_L = L_coeffs[pL]
                if c_L == 0.0:
                    continue
                for jL in range(pL + 1):
                    cf_L = c_L * _binom(pL, jL) * (a[axis_L] ** (pL - jL)) * (b[axis_L] ** jL)
                    for jn1 in range(2):
                        cf_n1 = _binom(1, jn1) * (a[axis_n1] ** (1 - jn1)) * (b[axis_n1] ** jn1)
                        for jn2 in range(2):
                            cf_n2 = _binom(1, jn2) * (a[axis_n2] ** (1 - jn2)) * (b[axis_n2] ** jn2)
                            coeff = cf_L * cf_n1 * cf_n2
                            pows = [0, 0, 0]
                            pows[axis_L] = jL
                            pows[axis_n1] = jn1
                            pows[axis_n2] = jn2
                            _add(pows, coeff)

    elif hasattr(node, 'cls') and node.cls == 1:
        # Class 1: c_d = R(η_sp)·η_n1·η_n2
        R_coeffs = [0.0, (3.0 * xi2 - 1.0) / 6.0, 0.0, 1.0 / 6.0]
        axis_L = getattr(node, 'sp_axis', 0)
        axis_n1 = (axis_L + 1) % 3
        axis_n2 = (axis_L + 2) % 3

        for pL in range(4):
            c_R = R_coeffs[pL]
            if c_R == 0.0:
                continue
            for jL in range(pL + 1):
                cf_L = c_R * _binom(pL, jL) * (a[axis_L] ** (pL - jL)) * (b[axis_L] ** jL)
                for jn1 in range(2):
                    cf_n1 = _binom(1, jn1) * (a[axis_n1] ** (1 - jn1)) * (b[axis_n1] ** jn1)
                    for jn2 in range(2):
                        cf_n2 = _binom(1, jn2) * (a[axis_n2] ** (1 - jn2)) * (b[axis_n2] ** jn2)
                        coeff = cf_L * cf_n1 * cf_n2
                        pows = [0, 0, 0]
                        pows[axis_L] = jL
                        pows[axis_n1] = jn1
                        pows[axis_n2] = jn2
                        _add(pows, coeff)

    return [(p, q, r, c) for (p, q, r), c in terms.items()]


# ── Precomputed monomial expansions (cached per xi) ──

_MONOMIAL_CACHE: Dict[float, List[List[Tuple[int, int, int, float]]]] = {}


def _get_monomials(xi: float) -> List[List[Tuple[int, int, int, float]]]:
    """Build or retrieve cached monomial expansions for all 32 CubeS₂ nodes."""
    key = round(xi, 12)
    if key in _MONOMIAL_CACHE:
        return _MONOMIAL_CACHE[key]
    nodes = get_nodes(4)
    mono = [_build_monomials_for_node(n, xi) for n in nodes]
    _MONOMIAL_CACHE[key] = mono
    return mono


# ── Analytic influence function ──

_ANALYTIC_CACHE: Dict = {}


def precompute_influence_analytic(
    nx: int, ny: int, nz: int,
    lx: float, ly: float, lz: float,
    xi: float,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Compute |Φ(k)|² analytically using 1D Fourier integrals.

    Identical algorithm to sog.cpp precompute_cubes2_influence().
    """
    cache_key = (nx, ny, nz, round(lx, 6), round(ly, 6), round(lz, 6),
                 round(xi, 6), str(device), str(dtype))
    cached = _ANALYTIC_CACHE.get(cache_key)
    if cached is not None:
        return cached

    mono = _get_monomials(xi)
    num_nodes = len(mono)
    dx_grid = lx / nx
    dy_grid = ly / ny
    dz_grid = lz / nz
    two_pi = 2.0 * math.pi
    dkx = two_pi / lx
    dky = two_pi / ly
    dkz = two_pi / lz

    # Precompute 1D integrals I_p(α) for each k-mode per axis
    Ipx = [[0j] * nx for _ in range(4)]
    for ix in range(nx):
        kx_mode = ix - nx * (2 * ix // nx)
        alpha_x = two_pi * kx_mode * dx_grid / lx
        for p in range(4):
            Ipx[p][ix] = _I_TABLE[p](alpha_x)

    Ipy = [[0j] * ny for _ in range(4)]
    for iy in range(ny):
        ky_mode = iy - ny * (2 * iy // ny)
        alpha_y = two_pi * ky_mode * dy_grid / ly
        for p in range(4):
            Ipy[p][iy] = _I_TABLE[p](alpha_y)

    Ipz = [[0j] * nz for _ in range(4)]
    for iz in range(nz):
        kz_mode = iz - nz * (2 * iz // nz)
        alpha_z = two_pi * kz_mode * dz_grid / lz
        for p in range(4):
            Ipz[p][iz] = _I_TABLE[p](alpha_z)

    # |Φ(k)|² on FULL grid (not rfftn — we store full and slice later)
    influence_sq = torch.zeros(nz, ny, nx // 2 + 1, dtype=dtype, device=device)

    for iz in range(nz):
        kz_mode = iz - nz * (2 * iz // nz)
        kz = dkz * kz_mode
        for iy in range(ny):
            ky_mode = iy - ny * (2 * iy // ny)
            ky = dky * ky_mode
            for ix in range(nx // 2 + 1):
                kx = dkx * ix  # rfftn: mx ≥ 0 only
                sqk = kx*kx + ky*ky + kz*kz
                if sqk == 0.0:
                    continue

                phi_k = complex(0.0, 0.0)
                for d in range(num_nodes):
                    node = get_nodes(4)[d]
                    node_mono = mono[d]
                    phase = (kx * node.dx * dx_grid +
                             ky * node.dy * dy_grid +
                             kz * node.dz * dz_grid)
                    eikd = complex(math.cos(phase), math.sin(phase))

                    integral = complex(0.0, 0.0)
                    for px, py, pz, coeff in node_mono:
                        prod = Ipx[px][ix] * Ipy[py][iy] * Ipz[pz][iz]
                        integral += coeff * prod

                    phi_k += eikd * integral

                abs_sq = phi_k.real * phi_k.real + phi_k.imag * phi_k.imag
                influence_sq[iz, iy, ix] = abs_sq

    influence_sq = influence_sq.clamp(min=1e-20)
    _ANALYTIC_CACHE[cache_key] = influence_sq
    return influence_sq
