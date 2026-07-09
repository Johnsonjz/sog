"""CubeS₂ Midtown Splines + PyTorch FFT — SOG long-range solver.

Vectorized charge spreading and force interpolation using torch.scatter_add_.
Fully autograd-compatible for MLIP training.

Reference: Predescu, Bergdorf, Shaw, J. Chem. Phys. 153, 224117 (2020)
           sog-error.md — grid sizing from Gaussian width
"""

from __future__ import annotations

import math
import warnings
from typing import Dict, List, Optional, Tuple

import torch

from .cubes2_spline import (
    CUBES2_NODES_4,
    XI_4,
    CubeS2Node,
    CubeS2Node4,
    _cubes2_L,
    _cubes2_R,
    _get_xi,
    _get_num_nodes,
    get_nodes,
    cubes2_weight,
)
from .influence_analytic import precompute_influence_analytic


# ── φ-based grid sizing (Predescu 2020 Table III) ──

# φ_max = Δ / r_c: maximum grid spacing that keeps mesh discretization
# error ≤ u-series intrinsic error.  Data from Predescu 2020 Table III.
PHI_MAX_TABLE: Dict[int, Dict[str, float]] = {
    4: {"b2": 0.23, "b163": 0.065},    # CubeS₂ 4th order, 32 nodes
    6: {"b2": 0.35, "b163": 0.160},    # CubeS₂ 6th order, 88 nodes
}

# b values used as interpolation anchors in Table III.
# The b≈1.63 anchor corresponds to b = u_{1/2} = 1.6297670882677647
# (Predescu et al., JCP 153, 224117, 2020).
_PHI_B_LO = 1.6297670882677647
_PHI_B_HI = 2.0


def _compute_phi_max(b: float, spline_order: int = 4) -> float:
    """Look up φ_max from Predescu 2020 Table III with linear b-interpolation.

    Clamps to nearest table value for b outside [b_lo, b_hi].
    """
    if spline_order not in PHI_MAX_TABLE:
        raise ValueError(
            f"No φ_max data for spline order {spline_order}. "
            f"Available: {list(PHI_MAX_TABLE.keys())}"
        )
    entry = PHI_MAX_TABLE[spline_order]
    phi_lo, phi_hi = entry["b163"], entry["b2"]

    if b <= 0:
        raise ValueError(f"b must be positive, got {b}")

    if b <= _PHI_B_LO:
        return phi_lo
    elif b >= _PHI_B_HI:
        return phi_hi
    else:
        # Linear interpolation between table anchors
        frac = (b - _PHI_B_LO) / (_PHI_B_HI - _PHI_B_LO)
        return phi_lo + (phi_hi - phi_lo) * frac


def _compute_grid_from_phi(
    lx: float, ly: float, lz: float,
    r_c: float,
    phi: float,
    min_grid: int = 8,
) -> Tuple[int, int, int]:
    """Compute FFT grid dimensions from φ = Δ / r_c.

    Returns (nx, ny, nz) guaranteed to be >= min_grid and FFT-friendly
    (2,3,5-smooth).
    """
    if r_c <= 0:
        raise ValueError(f"r_c must be positive for φ-based grid, got {r_c}")
    if phi <= 0:
        raise ValueError(f"φ must be positive, got {phi}")

    delta = phi * r_c
    nx = max(min_grid, int(math.ceil(lx / delta)))
    ny = max(min_grid, int(math.ceil(ly / delta)))
    nz = max(min_grid, int(math.ceil(lz / delta)))
    # FFT-friendly rounding (2,3,5-smooth)
    nx = _next_fft_friendly(nx)
    ny = _next_fft_friendly(ny)
    nz = _next_fft_friendly(nz)
    return nx, ny, nz


def _resolve_grid(
    lx: float, ly: float, lz: float,
    n_dl: Optional[float],
    cubes2_phi_max: Optional[float],
    r_c: Optional[float],
    b: float = 2.0,
    spline_order: int = 4,
    min_grid: int = 8,
) -> Tuple[int, int, int, float]:
    """Resolve grid dimensions from either φ-based or n_dl-based specs.

    Priority:
      1. cubes2_phi_max + r_c  → φ-based grid (recommended)
      2. n_dl                  → legacy n_dl-based grid (deprecated)
      3. neither               → auto from Table III φ_max

    Returns (nx, ny, nz, k_sq_max).
    """
    if cubes2_phi_max is not None and n_dl is not None:
        raise ValueError(
            "Cannot specify both `cubes2_phi_max` and `n_dl`. "
            "Use `cubes2_phi_max` (φ = Δ/r_c, recommended) or `n_dl` (legacy)."
        )

    if cubes2_phi_max is not None:
        # ── φ-based grid (recommended) ──
        if r_c is None or r_c <= 0:
            raise ValueError(
                "r_c (real-space cutoff) is required for φ-based grid sizing. "
                "Set rcut in Gaussian/Sog constructor."
            )
        phi = float(cubes2_phi_max)
        nx, ny, nz = _compute_grid_from_phi(lx, ly, lz, r_c, phi, min_grid=min_grid)
        # k_sq_max = grid Nyquist (all principal modes within the mesh are valid)
        dx = lx / nx
        dy = ly / ny
        dz = lz / nz
        k_sq_max = math.pi ** 2 * (1.0 / (dx * dx) + 1.0 / (dy * dy) + 1.0 / (dz * dz))
        return nx, ny, nz, k_sq_max

    if n_dl is not None:
        # ── Legacy n_dl-based grid ──
        warnings.warn(
            "`n_dl` is deprecated. Use `cubes2_phi_max` (φ = Δ/r_c) instead. "
            "See Predescu 2020 Table III: φ_max=0.23 for CubeS₂ 4th/b=2, "
            "φ_max=0.065 for b≈1.63.",
            DeprecationWarning,
            stacklevel=2,
        )
        nk_x = max(1, int(math.floor(lx / n_dl)))
        nk_y = max(1, int(math.floor(ly / n_dl)))
        nk_z = max(1, int(math.floor(lz / n_dl)))
        nx = max(min_grid, 2 * nk_x + 1)
        ny = max(min_grid, 2 * nk_y + 1)
        nz = max(min_grid, 2 * nk_z + 1)
        nx = _next_fft_friendly(nx)
        ny = _next_fft_friendly(ny)
        nz = _next_fft_friendly(nz)
        k_sq_max = (2.0 * math.pi / n_dl) ** 2
        return nx, ny, nz, k_sq_max

    # ── Auto from Table III φ_max ──
    phi_default = _compute_phi_max(b, spline_order=spline_order)
    if r_c is None or r_c <= 0:
        raise ValueError(
            "r_c is required for auto φ-based grid (neither cubes2_phi_max "
            "nor n_dl specified). Set rcut in Gaussian/Sog constructor."
        )
    nx, ny, nz = _compute_grid_from_phi(lx, ly, lz, r_c, phi_default, min_grid=min_grid)
    dx = lx / nx
    dy = ly / ny
    dz = lz / nz
    k_sq_max = math.pi ** 2 * (1.0 / (dx * dx) + 1.0 / (dy * dy) + 1.0 / (dz * dz))
    return nx, ny, nz, k_sq_max


# ── FFT-friendly grid sizing ──

def _next_fft_friendly(n: int) -> int:
    while True:
        m = n
        for p in (2, 3, 5):
            while m % p == 0:
                m //= p
        if m == 1:
            return n
        n += 1


# ── Batched CubeS₂ weight ──

def _cubes2_weight_batch(
    tx: torch.Tensor, ty: torch.Tensor, tz: torch.Tensor,
    node: CubeS2Node, xi: float, order: int = 4,
) -> torch.Tensor:
    """Batched CubeS₂ weight for N atoms.

    Works for both scalar tensors (shape []) and batched tensors (shape [N]).
    The implementation routes through the shared spline dispatcher so that
    order-6 uses the same primitive logic as the scalar path.
    """
    return cubes2_weight(tx, ty, tz, node, xi, order=order)


# ── Influence function: |Φ(k)|² via FFT of window on grid ──

# Backward-compat alias — analytic influence is the only path now.
# Tests should import _ANALYTIC_CACHE directly from influence_analytic.
from .influence_analytic import _ANALYTIC_CACHE as _INFLUENCE_CACHE  # noqa: F401


# ── 1D analytic Fourier integrals for the influence function ──
# I_p(α) = ∫₀¹ t^p · exp(i·α·t) dt
# Closed forms with small-α Taylor expansions.  Mirrors sog.cpp / fastsog.cpp.

def _I_int_0(alpha: torch.Tensor) -> torch.Tensor:
    """∫₀¹ exp(i·α·t) dt.  Returns complex tensor (real, imag)."""
    small = torch.abs(alpha) < 1e-8
    cos_a = torch.cos(alpha); sin_a = torch.sin(alpha)
    re = torch.where(small, 1.0 - alpha * alpha / 6.0, sin_a / alpha)
    im = torch.where(small, alpha / 2.0 - alpha**3 / 24.0, (1.0 - cos_a) / alpha)
    return torch.complex(re, im)

def _I_int_1(alpha: torch.Tensor) -> torch.Tensor:
    """∫₀¹ t · exp(i·α·t) dt."""
    small = torch.abs(alpha) < 1e-8
    cos_a = torch.cos(alpha); sin_a = torch.sin(alpha); a2 = alpha * alpha
    re = torch.where(small, 0.5 - a2 / 8.0, (alpha * sin_a + cos_a - 1.0) / a2)
    im = torch.where(small, alpha / 3.0 - alpha**3 / 30.0, (sin_a - alpha * cos_a) / a2)
    return torch.complex(re, im)

def _I_int_2(alpha: torch.Tensor) -> torch.Tensor:
    """∫₀¹ t² · exp(i·α·t) dt."""
    small = torch.abs(alpha) < 1e-8
    cos_a = torch.cos(alpha); sin_a = torch.sin(alpha)
    a2 = alpha * alpha; a3 = a2 * alpha
    re = torch.where(small, 1.0 / 3.0 - a2 / 10.0,
                     (2.0 * alpha * sin_a + (a2 - 2.0) * cos_a + 2.0) / a3)
    im = torch.where(small, alpha / 4.0,
                     ((a2 - 2.0) * sin_a + 2.0 * alpha * cos_a) / a3)
    return torch.complex(re, im)

def _I_int_3(alpha: torch.Tensor) -> torch.Tensor:
    """∫₀¹ t³ · exp(i·α·t) dt."""
    small = torch.abs(alpha) < 1e-8
    cos_a = torch.cos(alpha); sin_a = torch.sin(alpha)
    a2 = alpha * alpha; a3 = a2 * alpha; a4 = a3 * alpha
    re = torch.where(small, 0.25,
                     ((3.0 * a2 - 6.0) * alpha * sin_a
                      + (a3 - 6.0 * alpha) * cos_a + 6.0 * alpha) / a4)
    im = torch.where(small, alpha / 5.0,
                     ((a3 - 6.0 * alpha) * sin_a
                      + (6.0 - 3.0 * a2) * cos_a + 3.0 * a2 - 6.0) / a4)
    return torch.complex(re, im)

_I_INT_FNS = [_I_int_0, _I_int_1, _I_int_2, _I_int_3]

# ── Monomial expansion helpers (mirrors sog_build_monomials_for_node) ──

def _binom(n: int, k: int) -> int:
    """Binomial coefficient for small n (n≤5)."""
    _C = {
        (0, 0): 1,
        (1, 0): 1, (1, 1): 1,
        (2, 0): 1, (2, 1): 2, (2, 2): 1,
        (3, 0): 1, (3, 1): 3, (3, 2): 3, (3, 3): 1,
        (4, 0): 1, (4, 1): 4, (4, 2): 6, (4, 3): 4, (4, 4): 1,
        (5, 0): 1, (5, 1): 5, (5, 2): 10, (5, 3): 10, (5, 4): 5, (5, 5): 1,
    }
    return _C.get((n, k), 0)


def _expand_node_monomials_4(node: CubeS2Node4, xi: float) -> list:
    """Expand 4th-order node weight into monomials.

    Returns list of (px, py, pz, coeff) tuples where the weight is
    c(θ) = Σ coeff · θ₁^px · θ₂^py · θ₃^pz.
    """
    xi2 = xi * xi
    a = [float(node.dx), float(node.dy), float(node.dz)]
    b = [1.0 - 2.0 * a[0], 1.0 - 2.0 * a[1], 1.0 - 2.0 * a[2]]

    # Accumulate monomials: dict (px, py, pz) → coeff
    terms: dict = {}

    def add_term(px: int, py: int, pz: int, coeff: float):
        if abs(coeff) < 1e-30:
            return
        key = (px, py, pz)
        terms[key] = terms.get(key, 0.0) + coeff

    if node.cls == 0:
        # c(η) = L(η₁)η₂η₃ + L(η₂)η₃η₁ + L(η₃)η₁η₂
        # L(η) = -½η³ + ½η² - (9ξ²-2)/6·η + ξ²/2
        xi2_adj = (9.0 * xi2 - 2.0) / 6.0
        L_coeffs = [0.5 * xi2, -xi2_adj, 0.5, -0.5]  # [η⁰, η¹, η², η³]

        for term_idx in range(3):  # three cyclic terms
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
                            add_term(pows[0], pows[1], pows[2], coeff)
    else:
        # cls == 1: c(η) = R(η_sp) · η_n1 · η_n2
        # R(η) = ⅙η³ + (3ξ²-1)/6·η
        R_coeffs = [0.0, (3.0 * xi2 - 1.0) / 6.0, 0.0, 1.0 / 6.0]

        axis_L = node.sp_axis
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
                        add_term(pows[0], pows[1], pows[2], coeff)

    return [(px, py, pz, c) for (px, py, pz), c in terms.items()]


# ── Monomial cache ──
_MONOMIAL_CACHE: dict = {}  # (order, xi) → [(dx, dy, dz, sp_axis, sp_is_neg, [(px,py,pz,coeff),...]), ...]


def _get_monomials(order: int, xi: float) -> list:
    """Get (or build) monomial expansions for all nodes at given order and xi."""
    key = (order, round(xi, 10))
    if key in _MONOMIAL_CACHE:
        return _MONOMIAL_CACHE[key]

    nodes = get_nodes(order)
    mono_list = []
    for node in nodes:
        if order == 4:
            terms = _expand_node_monomials_4(node, xi)
        else:
            # 6th-order: fallback to 4th-order monomials (weights fall back to 4th)
            terms = _expand_node_monomials_4(node, XI_4)
        mono_list.append((node.dx, node.dy, node.dz, terms))
    _MONOMIAL_CACHE[key] = mono_list
    return mono_list


# ── Vectorized charge spreading ──

def _cubes2_all_node_indices_weights(
    tx: torch.Tensor, ty: torch.Tensor, tz: torch.Tensor,
    ix0: torch.Tensor, iy0: torch.Tensor, iz0: torch.Tensor,
    nx: int, ny: int, nz: int, xi: float, order: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute all nodes' flat indices and weights in a single batch.

    Returns:
        all_idx: [N_nodes*N] flat linear indices into the grid
        all_w:   [N_nodes*N] CubeS₂ weights
    """
    n_atoms = tx.shape[0]
    device = tx.device
    nodes = get_nodes(order)
    n_nodes = len(nodes)

    all_idx_list = []
    all_w_list = []
    ny_nx = ny * nx

    for node in nodes:
        w = _cubes2_weight_batch(tx, ty, tz, node, xi, order=order).to(device=device)
        gx = torch.remainder(ix0 + node.dx, nx)
        gy = torch.remainder(iy0 + node.dy, ny)
        gz = torch.remainder(iz0 + node.dz, nz)
        idx = gz * ny_nx + gy * nx + gx
        all_idx_list.append(idx)
        all_w_list.append(w)

    all_idx = torch.cat(all_idx_list)   # [n_nodes*N]
    all_w = torch.cat(all_w_list)       # [n_nodes*N]
    return all_idx, all_w


def cubes2_spread(
    q: torch.Tensor,
    r_frac: torch.Tensor,
    nx: int, ny: int, nz: int,
    xi: float = XI_4,
    order: int = 4,
    rho_scale: Optional[float] = None,
    volume: Optional[float] = None,
) -> torch.Tensor:
    """Vectorized spread: CubeS₂ assignment using scatter_add_.

    Optimized: all nodes batched into a single scatter_add_ call.

    Args:
        q: Charges [N].
        r_frac: Fractional coords [N, 3] in [0,1).
        order: CubeS₂ order (4 or 6).
        rho_scale: Charge density scaling (default: N/V, matching C++ FastSOG).
        volume: Box volume (required if rho_scale not explicitly given).
    Returns:
        rho_grid [nz, ny, nx].
    """
    if q.dim() > 1:
        q = q.reshape(-1)
    n_atoms = q.shape[0]
    device = q.device
    dtype = q.dtype
    ngrid = nz * ny * nx
    nodes = get_nodes(order)

    # Default rho_scale = N/V (C++ FastSOG convention)
    if rho_scale is None:
        if volume is not None:
            rho_scale = float(ngrid) / volume
        else:
            rho_scale = 1.0  # backward compatibility

    rho = torch.zeros(ngrid, dtype=dtype, device=device)

    fx = r_frac[:, 0] * float(nx)
    fy = r_frac[:, 1] * float(ny)
    fz = r_frac[:, 2] * float(nz)

    ix0 = torch.floor(fx).long()
    iy0 = torch.floor(fy).long()
    iz0 = torch.floor(fz).long()

    tx = fx - ix0.float()
    ty = fy - iy0.float()
    tz = fz - iz0.float()

    # Batch all nodes into single scatter_add_
    all_idx, all_w = _cubes2_all_node_indices_weights(
        tx, ty, tz, ix0, iy0, iz0, nx, ny, nz, xi, order=order,
    )
    # Repeat q for all nodes, apply rho_scale
    q_repeated = q.repeat(len(nodes))
    rho.scatter_add_(0, all_idx, rho_scale * q_repeated * all_w)

    return rho.reshape(nz, ny, nx)


# ── Vectorized force interpolation ──

def cubes2_interpolate(
    q: torch.Tensor,
    r_frac: torch.Tensor,
    grad_grid: torch.Tensor,
    nx: int, ny: int, nz: int,
    xi: float = XI_4,
    order: int = 4,
) -> torch.Tensor:
    """Vectorized interpolate: gather forces from grid.

    Optimized: all nodes batched into single gather + reshape+sum.

    Args:
        q: Charges [N].
        r_frac: Fractional coords [N, 3] in [0,1).
        grad_grid: Force grid [3, nz, ny, nx].
        order: CubeS₂ order (4 or 6).
    Returns:
        forces [N, 3].
    """
    if q.dim() > 1:
        q = q.reshape(-1)
    n_atoms = q.shape[0]
    device = q.device
    dtype = q.dtype
    n_nodes = _get_num_nodes(order)

    gx_f = grad_grid[0].reshape(-1)
    gy_f = grad_grid[1].reshape(-1)
    gz_f = grad_grid[2].reshape(-1)

    fx = r_frac[:, 0] * float(nx)
    fy = r_frac[:, 1] * float(ny)
    fz = r_frac[:, 2] * float(nz)

    ix0 = torch.floor(fx).long()
    iy0 = torch.floor(fy).long()
    iz0 = torch.floor(fz).long()

    tx = fx - ix0.float()
    ty = fy - iy0.float()
    tz = fz - iz0.float()

    # Batch all nodes into single gather + reshape+sum
    all_idx, all_w = _cubes2_all_node_indices_weights(
        tx, ty, tz, ix0, iy0, iz0, nx, ny, nz, xi, order=order,
    )

    # Gather grid values for all nodes at once
    gx_vals = gx_f[all_idx]  # [n_nodes*N]
    gy_vals = gy_f[all_idx]
    gz_vals = gz_f[all_idx]

    # Reshape to [n_nodes, N] and sum over nodes
    w_gx = (all_w * gx_vals).reshape(n_nodes, n_atoms).sum(dim=0)  # [N]
    w_gy = (all_w * gy_vals).reshape(n_nodes, n_atoms).sum(dim=0)
    w_gz = (all_w * gz_vals).reshape(n_nodes, n_atoms).sum(dim=0)

    forces = torch.stack([-w_gx * q, -w_gy * q, -w_gz * q], dim=1)
    return forces


# ── Vectorized potential interpolation (no charge weighting) ──

def cubes2_interpolate_potential(
    r_frac: torch.Tensor,
    grid: torch.Tensor,
    nx: int, ny: int, nz: int,
    xi: float = XI_4,
    order: int = 4,
) -> torch.Tensor:
    """Interpolate scalar potential grid to atom positions (no charge weighting).

    Returns ∂E/∂q_j per atom: the electrostatic potential at each atomic position.

    Optimized: all nodes batched into single gather + reshape+sum.

    Args:
        r_frac: Fractional coords [N, 3] in [0,1).
        grid: Scalar grid [nz, ny, nx] (e.g. potential from IFFT).
        order: CubeS₂ order (4 or 6).
    Returns:
        potential [N] — scalar potential value at each atom position.
    """
    n_atoms = r_frac.shape[0]
    device = r_frac.device
    dtype = r_frac.dtype
    n_nodes = _get_num_nodes(order)

    g_flat = grid.reshape(-1)

    fx = r_frac[:, 0] * float(nx)
    fy = r_frac[:, 1] * float(ny)
    fz = r_frac[:, 2] * float(nz)

    ix0 = torch.floor(fx).long()
    iy0 = torch.floor(fy).long()
    iz0 = torch.floor(fz).long()

    tx = fx - ix0.float()
    ty = fy - iy0.float()
    tz = fz - iz0.float()

    # Batch all nodes into single gather + reshape+sum
    all_idx, all_w = _cubes2_all_node_indices_weights(
        tx, ty, tz, ix0, iy0, iz0, nx, ny, nz, xi, order=order,
    )

    g_vals = g_flat[all_idx]  # [n_nodes*N]
    potential = (all_w * g_vals).reshape(n_nodes, n_atoms).sum(dim=0)  # [N]

    return potential


# ── Custom autograd Function: correct gradients through CubeS₂+FFT ──

class Cubes2FFTFunction(torch.autograd.Function):
    """Custom autograd Function for CubeS₂+FFT long-range solver.

    Forward: computes SOG energy via spread→FFT (same as compute_cubes2_fft).
    Backward: recomputes ∂E/∂r (explicit force) and ∂E/∂q (potential) from
    saved spectral intermediates using PyTorch-native irfftn, which has a
    proper grad_fn — enabling create_graph=True support.

    This fixes the gradient truncation caused by torch.floor() in the
    charge spreading step, which otherwise breaks autograd through the
    particle-mesh path.
    """

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,           # [N] charges
        r: torch.Tensor,           # [N, 3] positions
        cell: torch.Tensor,        # [3, 3] box matrix
        amp: torch.Tensor,         # [M] kernel amplitudes
        bw2: torch.Tensor,         # [M] kernel bandwidths squared
        volume_val: float,         # box volume (scalar)
        diag_sum_val: float,       # self-interaction sum (scalar)
        cubes2_phi_max: Optional[float],  # φ = Δ/r_c (None→auto)
        n_dl: Optional[float],     # legacy grid density (deprecated)
        r_c: Optional[float],      # real-space cutoff (required for φ)
        b: float,                  # geometric base (for φ_max table)
        order: int,                # CubeS₂ spline order (4 or 6)
        xi: float,                 # CubeS₂ xi parameter
        remove_self_interaction: bool,
        self_coeff: float,         # real-space self-energy coefficient
        norm_factor: float,        # energy unit conversion
    ) -> torch.Tensor:
        device = r.device
        dtype = r.dtype

        if q.dim() > 1:
            q = q.reshape(-1)

        # ── Grid sizing (φ-based or n_dl-based) ──
        with torch.no_grad():
            box_norms = torch.norm(cell, dim=1)
            lx = float(box_norms[0].item())
            ly = float(box_norms[1].item())
            lz = float(box_norms[2].item())
            nx, ny, nz, k_sq_max = _resolve_grid(
                lx, ly, lz,
                n_dl=n_dl,
                cubes2_phi_max=cubes2_phi_max,
                r_c=r_c,
                b=b,
                spline_order=order,
                min_grid=8,
            )

        # ── Fractional coords ──
        cell_inv = torch.linalg.inv(cell)
        r_frac = r @ cell_inv
        r_frac = torch.remainder(r_frac, 1.0)

        # ── k-space grid + spectral kernel ──
        # Reciprocal-lattice k² (correct for triclinic cells).
        # FFT mode (mx,my,mz) has physical k = 2π * [mx,my,mz] @ cell_inv
        # where mx ∈ [0..nx//2] (rfft), my ∈ [-ny//2+1..ny//2] (fft).
        mx = torch.arange(0, nx // 2 + 1, device=device, dtype=torch.float64)
        my = torch.cat([
            torch.arange(0, ny // 2 + 1, device=device, dtype=torch.float64),
            torch.arange(-ny // 2 + 1, 0, device=device, dtype=torch.float64),
        ]) if ny > 1 else torch.zeros(1, device=device, dtype=torch.float64)
        mz = torch.cat([
            torch.arange(0, nz // 2 + 1, device=device, dtype=torch.float64),
            torch.arange(-nz // 2 + 1, 0, device=device, dtype=torch.float64),
        ]) if nz > 1 else torch.zeros(1, device=device, dtype=torch.float64)
        MX, MY, MZ = torch.meshgrid(mx, my, mz, indexing="ij")
        two_pi = 2.0 * math.pi
        # k_phys = 2π * [MX, MY, MZ] @ cell_inv
        KX_phys = two_pi * (MX * cell_inv[0, 0] + MY * cell_inv[1, 0] + MZ * cell_inv[2, 0])
        KY_phys = two_pi * (MX * cell_inv[0, 1] + MY * cell_inv[1, 1] + MZ * cell_inv[2, 1])
        KZ_phys = two_pi * (MX * cell_inv[0, 2] + MY * cell_inv[1, 2] + MZ * cell_inv[2, 2])
        k_sq = (KX_phys ** 2 + KY_phys ** 2 + KZ_phys ** 2).permute(2, 1, 0)

        amp_dev = amp.to(device=device, dtype=dtype)
        bw2_dev = bw2.to(device=device, dtype=dtype)
        kfac_fft = (
            amp_dev.view(1, 1, 1, -1)
            * torch.exp(-0.5 * bw2_dev.view(1, 1, 1, -1) * k_sq.unsqueeze(-1))
        ).sum(dim=-1)

        kfac_fft[0, 0, 0] = 0.0
        kfac_fft = kfac_fft.masked_fill(k_sq > k_sq_max, 0.0)

        # ── Green function via analytic variance-subtraction (paper Eq. 70) ──
        # G(k) = K(k²)·exp(+Σ_α σ_{s,α}² k_α²), σ_{s,α}=ξ₀·Δ_α. No division by
        # the spline influence (unstable for non-convolutional CubeS₂).
        sig_sx2 = (xi * lx / nx) ** 2
        sig_sy2 = (xi * ly / ny) ** 2
        sig_sz2 = (xi * lz / nz) ** 2
        kx2_p = (KX_phys ** 2).permute(2, 1, 0)
        ky2_p = (KY_phys ** 2).permute(2, 1, 0)
        kz2_p = (KZ_phys ** 2).permute(2, 1, 0)
        deconv = torch.exp(sig_sx2 * kx2_p + sig_sy2 * ky2_p + sig_sz2 * kz2_p)
        green_k = kfac_fft * deconv
        green_k[0, 0, 0] = 0.0
        green_k = green_k.masked_fill(k_sq > k_sq_max, 0.0)
        volume = volume_val
        N3 = float(nx * ny * nz)
        s2 = 1.0 / (N3 * N3)
        # rfftn stores only mx ≥ 0; each mx>0 mode represents both +k_x and -k_x.
        _rfft_w = torch.ones(1, 1, nx // 2 + 1, dtype=dtype, device=device)
        if nx % 2 == 0:
            _rfft_w[:, :, 1:nx // 2] = 2.0
        else:
            _rfft_w[:, :, 1:] = 2.0
        diag_sum_fft = (_rfft_w * kfac_fft).sum() / (2.0 * volume)

        # ── Spread + FFT + Energy (rho_scale=N/V; σ_s² cancels) ──
        rho_grid = cubes2_spread(q, r_frac, nx, ny, nz, xi=xi, order=order,
                                  rho_scale=float(nx*ny*nz)/volume)
        rho_k = torch.fft.rfftn(rho_grid)
        rho_sq = rho_k.real**2 + rho_k.imag**2

        energy = 0.5 * volume * s2 * (_rfft_w * green_k * rho_sq).sum()
        if remove_self_interaction:
            energy = energy - (q * q).sum() * diag_sum_fft
        if self_coeff != 0.0:
            energy = energy - (q * q).sum() * self_coeff
        energy = energy * norm_factor

        # ── Save intermediates for backward ──
        # Save physical k-vector components (already computed above)
        ctx.KX3 = KX_phys.permute(2, 1, 0)  # [nz, ny, nx//2+1] physical k_x
        ctx.KY3 = KY_phys.permute(2, 1, 0)  # physical k_y
        ctx.KZ3 = KZ_phys.permute(2, 1, 0)  # physical k_z
        ctx.green_k = green_k.detach().requires_grad_(True)
        ctx.rho_k = rho_k.detach().requires_grad_(True)
        ctx.r_frac = r_frac          # retains grad_fn → r
        ctx.q_val = q                # retains grad_fn → upstream model
        ctx.nx, ctx.ny, ctx.nz = nx, ny, nz
        ctx.volume = volume
        ctx.norm_factor = norm_factor
        ctx.xi = xi
        ctx.order = order
        ctx.remove_self_interaction = remove_self_interaction
        ctx.self_coeff = self_coeff
        ctx.diag_sum_fft = diag_sum_fft
        # ── Save for amp/bw gradient computation ──
        ctx.deconv = deconv.detach()                  # exp(+Σ σ_{s,α}² k_α²)
        ctx.k_sq = k_sq                                # k² grid
        ctx.k_sq_max = k_sq_max                        # Nyquist cutoff
        ctx._rfft_w = _rfft_w                          # rfftn weight
        ctx.rho_sq = rho_sq.detach()                   # |ρ(k)|²
        ctx.amp_d = amp_dev.detach()                   # [M] amp values
        ctx.bw2_d = bw2_dev.detach()                   # [M] bw values
        ctx.s2_val = s2                                 # 1/N⁶
        ctx.q_sq_sum = (q * q).sum().detach()          # Q² for self-int correction

        return energy

    @staticmethod
    def backward(ctx, grad_output):
        green_k = ctx.green_k
        rho_k = ctx.rho_k
        r_frac = ctx.r_frac
        q = ctx.q_val
        KX3 = ctx.KX3
        KY3 = ctx.KY3
        KZ3 = ctx.KZ3
        nx, ny, nz = ctx.nx, ctx.ny, ctx.nz
        volume = ctx.volume
        norm_factor = ctx.norm_factor
        xi = ctx.xi
        order = ctx.order

        dtype = green_k.dtype
        device = green_k.device
        complex_dtype = torch.complex128 if dtype == torch.float64 else torch.complex64
        N3 = float(nx * ny * nz)

        # ── Use saved physical k-vector components (from forward) ──
        KX3c = KX3.to(dtype=complex_dtype)
        KY3c = KY3.to(dtype=complex_dtype)
        KZ3c = KZ3.to(dtype=complex_dtype)

        # ── conv_k = (1/N) · G(k) · ρ(k) (no rfft weight for forces) ──
        scaleinv_bw = 1.0 / N3
        conv_k = (scaleinv_bw * green_k).to(dtype=complex_dtype) * rho_k.to(dtype=complex_dtype)

        # ── grad_r: ∂E/∂r|_q via ik·(1/N)·G·ρ → irfftn → interpolate ──
        grad_kx = 1j * KX3c * conv_k
        grad_ky = 1j * KY3c * conv_k
        grad_kz = 1j * KZ3c * conv_k

        force_grid_x = torch.fft.irfftn(grad_kx, s=(nz, ny, nx))
        force_grid_y = torch.fft.irfftn(grad_ky, s=(nz, ny, nx))
        force_grid_z = torch.fft.irfftn(grad_kz, s=(nz, ny, nx))

        force_grid = torch.stack(
            [force_grid_x, force_grid_y, force_grid_z], dim=0,
        ).to(dtype=dtype)

        explicit_force = cubes2_interpolate(
            q, r_frac, force_grid, nx, ny, nz, xi=xi, order=order,
        )
        explicit_force = explicit_force * N3 * norm_factor

        # ── grad_q: ∂E/∂q via (1/N)·G·ρ → irfftn → interpolate ──
        phi_k = conv_k
        phi_grid = torch.fft.irfftn(phi_k, s=(nz, ny, nx)).to(dtype=dtype)
        potential = cubes2_interpolate_potential(
            r_frac, phi_grid, nx, ny, nz, xi=xi, order=order,
        )
        grad_q = potential * N3 * norm_factor

        # Self-interaction corrections: ∂/∂q of -(q²)·diag_sum and -(q²)·self_coeff
        if ctx.remove_self_interaction:
            grad_q = grad_q - 2.0 * q * ctx.diag_sum_fft * norm_factor
        if ctx.self_coeff != 0.0:
            grad_q = grad_q - 2.0 * q * ctx.self_coeff * norm_factor

        # ∂E/∂r = -force  (force = -∂E/∂r, so ∂E/∂r = -explicit_force)
        grad_r = grad_output * (-explicit_force)

        # ∂E/∂q
        grad_q_out = grad_output * grad_q if ctx.needs_input_grad[0] else None

        # ── amp / bandwidth gradients: kernel parameter optimization ──
        # ∂E/∂amp[m] = ½·V/N⁶·Σ w·exp(-½·bw[m]·k²)/|Φ|²·|ρ|²
        # ∂E/∂amp[m] = ½·V/N⁶·Σ w·exp(-½·bw[m]·k²)·deconv·|ρ|²
        #               - Q²·Σ w·exp(-½·bw[m]·k²)/(2V)  (if remove_self_interaction)
        # ∂E/∂bw[m]  = ½·V/N⁶·Σ w·[-½·amp[m]·k²·exp(-½·bw[m]·k²)]·deconv·|ρ|²
        #               - Q²·Σ w·[-½·amp[m]·k²·exp(-½·bw[m]·k²)]/(2V)  (if remove_self_interaction)
        grad_amp = None
        grad_bw2 = None

        if ctx.needs_input_grad[3] or ctx.needs_input_grad[4]:
            amp_bw = ctx.amp_d
            bw2_bw = ctx.bw2_d
            M = amp_bw.shape[0]
            k_sq_bw = ctx.k_sq
            deconv = ctx.deconv  # exp(+Σ σ_{s,α}² k_α²), replaces 1/|Φ|²

            prefactor_main = 0.5 * ctx.volume * ctx.s2_val  # ½·V/N⁶
            prefactor_diag = 1.0 / (2.0 * ctx.volume)         # 1/(2V)

            # k-space mask: match forward — exclude k=0 AND modes beyond Nyquist
            k_mask = (k_sq_bw <= ctx.k_sq_max) & (k_sq_bw > 0.0)  # [nz, ny, nx//2+1]

            # ── Batched over M: single fused kernel launch instead of M iterations ──
            # Expand to [M, nz, ny, nx//2+1]
            bw2_4d = bw2_bw.view(M, 1, 1, 1)               # [M, 1, 1, 1]
            k_sq_4d = k_sq_bw.unsqueeze(0)                  # [1, nz, ny, nx//2+1]
            exp_all = torch.exp(-0.5 * bw2_4d * k_sq_4d)    # [M, nz, ny, nx//2+1]
            exp_all = exp_all.masked_fill(~k_mask.unsqueeze(0), 0.0)

            # weighted = _rfft_w * exp * deconv * rho_sq    [M, nz, ny, nx//2+1]
            weighted = ctx._rfft_w * exp_all * deconv * ctx.rho_sq

            if ctx.needs_input_grad[3]:
                grad_amp = prefactor_main * weighted.sum(dim=(1, 2, 3))  # [M]
                if ctx.remove_self_interaction:
                    diag_all = (ctx._rfft_w * exp_all).sum(dim=(1, 2, 3)) * prefactor_diag
                    grad_amp = grad_amp - ctx.q_sq_sum * diag_all

            if ctx.needs_input_grad[4]:
                ksq_w = (-0.5 * k_sq_4d) * weighted        # [M, nz, ny, nx//2+1]
                amp_4d = amp_bw.view(M, 1, 1, 1)
                grad_bw2 = prefactor_main * (amp_4d * ksq_w).sum(dim=(1, 2, 3))
                if ctx.remove_self_interaction:
                    diag_all = (amp_4d * (-0.5 * k_sq_4d) * ctx._rfft_w * exp_all).sum(dim=(1, 2, 3)) * prefactor_diag
                    grad_bw2 = grad_bw2 - ctx.q_sq_sum * diag_all

            if ctx.needs_input_grad[3] and grad_amp is not None:
                grad_amp = grad_amp * norm_factor * grad_output
            if ctx.needs_input_grad[4] and grad_bw2 is not None:
                grad_bw2 = grad_bw2 * norm_factor * grad_output

        # Return grads for: q, r, cell, amp, bw2, volume_val, diag_sum_val,
        #                   cubes2_phi_max, n_dl, r_c, b, order,
        #                   xi, remove_self, self_coeff, norm_factor
        return (grad_q_out, grad_r, None, grad_amp, grad_bw2, None, None,
                None, None, None, None, None,
                None, None, None, None)


# ── Main FFT solver ──

def compute_cubes2_fft(
    q: torch.Tensor,
    r: torch.Tensor,
    cell: torch.Tensor,
    amp: torch.Tensor,
    bw2: torch.Tensor,
    volume: torch.Tensor,
    diag_sum: torch.Tensor,
    n_dl: Optional[float] = None,
    cubes2_phi_max: Optional[float] = None,
    r_c: Optional[float] = None,
    b: float = 2.0,
    xi: float = XI_4,
    order: int = 4,
    remove_self_interaction: bool = True,
    self_coeff: float = 0.0,
    norm_factor: float = 1.0,
    compute_force: bool = False,
    compute_virial: bool = False,
    compute_dq: bool = False,
) -> Dict[str, Optional[torch.Tensor]]:
    """Compute SOG long-range energy/forces via CubeS₂ + PyTorch FFT.

    All operations are vectorized and autograd-compatible.
    Forces computed via explicit path: ik·G(k)·ρ(k) → irfftn → interpolate.

    Parameters
    ----------
    cubes2_phi_max : float, optional
        φ = Δ/r_c grid control (recommended). Auto-defaults from Predescu 2020
        Table III when neither this nor n_dl is given.
    n_dl : float, optional
        Legacy grid density (deprecated). Use cubes2_phi_max instead.
    r_c : float, optional
        Real-space cutoff radius, required for φ-based grid sizing.
    b : float
        Geometric base, used for φ_max table lookup.
    """
    device = r.device
    dtype = r.dtype

    if q.dim() > 1:
        q = q.reshape(-1)

    # ── Grid sizing (φ-based or n_dl-based) ──
    with torch.no_grad():
        box_norms = torch.norm(cell, dim=1)
        lx = float(box_norms[0].item())
        ly = float(box_norms[1].item())
        lz = float(box_norms[2].item())
        nx, ny, nz, k_sq_max = _resolve_grid(
            lx, ly, lz,
            n_dl=n_dl,
            cubes2_phi_max=cubes2_phi_max,
            r_c=r_c,
            b=b,
            spline_order=order,
            min_grid=8,
        )

    # ── Fractional coords ──
    cell_inv = torch.linalg.inv(cell)
    r_frac = r @ cell_inv
    r_frac = torch.remainder(r_frac, 1.0)

    # ── k-space grid + spectral kernel ──
    # Reciprocal-lattice k² (correct for triclinic cells).
    # FFT mode (mx,my,mz) has physical k = 2π * [mx,my,mz] @ cell_inv
    mx2 = torch.arange(0, nx // 2 + 1, device=device, dtype=torch.float64)
    my2 = torch.cat([
        torch.arange(0, ny // 2 + 1, device=device, dtype=torch.float64),
        torch.arange(-ny // 2 + 1, 0, device=device, dtype=torch.float64),
    ]) if ny > 1 else torch.zeros(1, device=device, dtype=torch.float64)
    mz2 = torch.cat([
        torch.arange(0, nz // 2 + 1, device=device, dtype=torch.float64),
        torch.arange(-nz // 2 + 1, 0, device=device, dtype=torch.float64),
    ]) if nz > 1 else torch.zeros(1, device=device, dtype=torch.float64)
    MX2, MY2, MZ2 = torch.meshgrid(mx2, my2, mz2, indexing="ij")
    two_pi2 = 2.0 * math.pi
    KX2 = two_pi2 * (MX2 * cell_inv[0, 0] + MY2 * cell_inv[1, 0] + MZ2 * cell_inv[2, 0])
    KY2 = two_pi2 * (MX2 * cell_inv[0, 1] + MY2 * cell_inv[1, 1] + MZ2 * cell_inv[2, 1])
    KZ2 = two_pi2 * (MX2 * cell_inv[0, 2] + MY2 * cell_inv[1, 2] + MZ2 * cell_inv[2, 2])
    k_sq = (KX2 ** 2 + KY2 ** 2 + KZ2 ** 2).permute(2, 1, 0)

    amp_dev = amp.to(device=device, dtype=dtype)
    bw2_dev = bw2.to(device=device, dtype=dtype)
    kfac_fft = (
        amp_dev.view(1, 1, 1, -1)
        * torch.exp(-0.5 * bw2_dev.view(1, 1, 1, -1) * k_sq.unsqueeze(-1))
    ).sum(dim=-1)

    # k=0 exclusion (matching C++ spectral_kernel: if !(ksq>0) return 0)
    kfac_fft[0, 0, 0] = 0.0

    # k_sq_max mask: grid Nyquist for φ-based, (2π/n_dl)² for legacy n_dl
    kfac_fft = kfac_fft.masked_fill(k_sq > k_sq_max, 0.0)

    # ── Green function via analytic variance-subtraction (paper Eq. 70) ──
    # CubeS₂ spreading approximates an ideal Gaussian of variance σ_s²=(ξ₀Δ)²
    # per axis; the deconvolution is analytic: G(k)=K(k²)·exp(+Σ_α σ_{s,α}² k_α²)
    # = Σ_m amp_m exp(-½(bw_m − 2σ_s²)k²) (isotropic grid). No division by the
    # spline influence → stable, O(Δ^{2ν}) accurate, matches the direct k-sum.
    # (The SPME-division form K/|Φ|² is unstable for the non-convolutional CubeS₂
    #  window and is not used.)
    sig_sx2 = (xi * lx / nx) ** 2
    sig_sy2 = (xi * ly / ny) ** 2
    sig_sz2 = (xi * lz / nz) ** 2
    kx2_p = (KX2 ** 2).permute(2, 1, 0)
    ky2_p = (KY2 ** 2).permute(2, 1, 0)
    kz2_p = (KZ2 ** 2).permute(2, 1, 0)
    deconv = torch.exp(sig_sx2 * kx2_p + sig_sy2 * ky2_p + sig_sz2 * kz2_p)
    green_k = kfac_fft * deconv
    green_k[0, 0, 0] = 0.0
    green_k = green_k.masked_fill(k_sq > k_sq_max, 0.0)

    # rfftn weight: mx>0 modes represent both +k_x and -k_x.
    _rfft_w2 = torch.ones(1, 1, nx // 2 + 1, dtype=dtype, device=device)
    if nx % 2 == 0:
        _rfft_w2[:, :, 1:nx // 2] = 2.0
    else:
        _rfft_w2[:, :, 1:] = 2.0

    # FFT grid scaling factors
    N3_val = float(nx * ny * nz)
    s2_val = 1.0 / (N3_val * N3_val)
    scaleinv_val = 1.0 / N3_val
    volume_val = volume.item() if isinstance(volume, torch.Tensor) else float(volume)

    # Self-interaction: Σ K(k²) / (2V) over k≠0
    diag_sum_fft = (_rfft_w2 * kfac_fft).sum() / (2.0 * volume_val)

    # ── Spread + FFT + Energy (rho_scale=N/V) ──
    # ρ(k) ≈ (N/V)·S(k)·exp(-½σ_s²k²) (CubeS₂ ≈ Gaussian spread). With the
    # variance-subtraction Green function G=K·exp(+σ_s²k²), the σ_s² factors
    # cancel: E = V/(2N²)·Σ w·G·|ρ|² = (1/2V)·Σ w·K·|S|².
    rho_grid = cubes2_spread(q, r_frac, nx, ny, nz, xi=xi, order=order,
                              rho_scale=N3_val/volume_val)
    rho_k = torch.fft.rfftn(rho_grid)
    rho_sq = rho_k.real**2 + rho_k.imag**2

    energy = 0.5 * volume_val * s2_val * (_rfft_w2 * green_k * rho_sq).sum()
    if remove_self_interaction:
        energy = energy - (q * q).sum() * diag_sum_fft
    # Real-space self-energy (fastsog convention, applied before norm_factor)
    if self_coeff != 0.0:
        energy = energy - (q * q).sum() * self_coeff
    energy = energy * norm_factor

    result: Dict[str, Optional[torch.Tensor]] = {
        "energy": energy, "forces": None, "virial": None, "dq": None,
    }

    # ── Explicit forces: ik·(1/N)·G(k)·ρ(k) → irfftn → interpolate ──
    if compute_force or compute_virial or compute_dq:
        conv_k = (scaleinv_val * green_k).to(dtype=torch.complex128) * rho_k.to(dtype=torch.complex128)

        KX3 = KX2.permute(2, 1, 0).to(dtype=torch.complex128)
        KY3 = KY2.permute(2, 1, 0).to(dtype=torch.complex128)
        KZ3 = KZ2.permute(2, 1, 0).to(dtype=torch.complex128)

    if compute_force or compute_virial:
        grad_kx = 1j * KX3 * conv_k
        grad_ky = 1j * KY3 * conv_k
        grad_kz = 1j * KZ3 * conv_k

        force_grid_x = torch.fft.irfftn(grad_kx, s=(nz, ny, nx))
        force_grid_y = torch.fft.irfftn(grad_ky, s=(nz, ny, nx))
        force_grid_z = torch.fft.irfftn(grad_kz, s=(nz, ny, nx))

        force_grid = torch.stack(
            [force_grid_x, force_grid_y, force_grid_z], dim=0,
        ).to(dtype=dtype)

        force = cubes2_interpolate(q, r_frac, force_grid, nx, ny, nz, xi=xi, order=order)
        # Force scaling: conv_k uses scaleinv=1/N, force = -q * IFFT(ik·conv_k) * N
        force = force * N3_val * norm_factor
        result["forces"] = force

        if compute_virial:
            result["virial"] = torch.einsum("ni,nj->ij", force, r)

    # ── ∂E/∂q: potential at atoms via G·ρ/V → irfftn → interpolate ──
    if compute_dq:
        phi_k = conv_k / volume_val
        phi_grid = torch.fft.irfftn(
            phi_k.to(dtype=torch.complex128), s=(nz, ny, nx),
        ).to(dtype=dtype)
        dq = cubes2_interpolate_potential(
            r_frac, phi_grid, nx, ny, nz, xi=xi, order=order,
        )
        dq = dq * N3_val * norm_factor
        # Self-interaction corrections
        if remove_self_interaction:
            dq = dq - 2.0 * q * diag_sum_fft * norm_factor
        if self_coeff != 0.0:
            dq = dq - 2.0 * q * self_coeff * norm_factor
        result["dq"] = dq

    return result
