"""QuadS (Midtown quadrature splines) — separable 1-D charge-assignment weights.

Reference: Predescu, Bergdorf, Shaw, J. Chem. Phys. 153, 224117 (2020), Appendix A
           (explicit weights); mirror of papers/midtown/midtown-sog.md §A.2 (order 4)
           and §A.3 (order 6), verified against the defining linear system (Eq. 12).

QuadS is the SEPARABLE (tensor-product, CubeS_∞) midtown spline: the 3-D transfer
function is an outer product S_ξ(s,i)=∏_α S_ξ(s_α,i_α), so a node at offset
(i₁,i₂,i₃) carries weight c_{i₁}(θ₁)·c_{i₂}(θ₂)·c_{i₃}(θ₃).  Because it is separable,
its Fourier influence |Ŵ(k)|²=∏_α|Ŵ_α(k_α)|² factorizes and its SPME-style
deconvolution (Form A) is stable — the property CubeS₂ (non-separable) lacks.

Order 2ν has 2ν nodes per axis at offsets 1−ν … ν (ν=order/2):
  order 4 (ν=2): offsets [-1,0,1,2]   → 64 nodes/atom,  ξ○ = 1/√3        (C²)
  order 6 (ν=3): offsets [-2,-1,0,1,2,3] → 216 nodes/atom, ξ○ = 0.72879488 (C⁰)

The reflection identity c_i(θ)=c_{1−i}(1−θ) generates the i≤0 weights from the
explicit c_1…c_ν given in §A; this module expands everything into per-offset
monomial coefficient arrays (source of truth for both weight evaluation and the
analytic influence function).
"""

from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
import torch

# ── Optimal ξ○ (Table I) ──
QUADS_XI_4: float = 1.0 / math.sqrt(3.0)   # ≈ 0.5773502691896258  (C² continuous, 4k order)
QUADS_XI_6: float = 0.72879488             # C⁰ continuous, 4k+2 order (Eq. 18 root)

_SUPPORTED_ORDERS = (4, 6)


def quads_xi(order: int) -> float:
    """Optimal ξ○ for the given QuadS order (Table I / §A)."""
    if order == 4:
        return QUADS_XI_4
    if order == 6:
        return QUADS_XI_6
    raise ValueError(f"Unsupported QuadS order: {order}. Supported: {_SUPPORTED_ORDERS}.")


def quads_num_nodes_1d(order: int) -> int:
    """Nodes per axis = 2ν = order."""
    if order in _SUPPORTED_ORDERS:
        return order
    raise ValueError(f"Unsupported QuadS order: {order}. Supported: {_SUPPORTED_ORDERS}.")


def quads_num_nodes(order: int) -> int:
    """3-D nodes per atom = (2ν)³."""
    return quads_num_nodes_1d(order) ** 3


def quads_offsets(order: int) -> List[int]:
    """1-D grid offsets 1−ν … ν."""
    nu = order // 2
    return list(range(1 - nu, nu + 1))


# ── Base weight polynomials c_i(θ), i = 1 … ν (§A.2 / §A.3) ──
# Each returned as an ascending-power coefficient array [a0, a1, …, a_{2ν-1}]
# so that c_i(θ) = Σ_p a_p θ^p.

def _base_polys(xi: float, order: int) -> Dict[int, np.ndarray]:
    """Explicit c_1 … c_ν as ascending monomial-coefficient arrays."""
    x2 = xi * xi
    x4 = x2 * x2
    if order == 4:
        # §A.2
        c1 = [x2 / 2.0, -(3.0 * x2 - 2.0) / 2.0, 1.0 / 2.0, -1.0 / 2.0]
        c2 = [0.0, (3.0 * x2 - 1.0) / 6.0, 0.0, 1.0 / 6.0]
        return {1: np.array(c1), 2: np.array(c2)}
    if order == 6:
        # §A.3
        c1 = [-(3.0 * x4 - 4.0 * x2) / 6.0,
              (5.0 * x4 - 7.0 * x2 + 4.0) / 4.0,
              -(3.0 * x2 - 2.0) / 3.0,
              (10.0 * x2 - 7.0) / 12.0,
              -1.0 / 6.0,
              1.0 / 12.0]
        c2 = [(3.0 * x4 - x2) / 24.0,
              -(5.0 * x4 - 7.0 * x2 + 2.0) / 8.0,
              (6.0 * x2 - 1.0) / 24.0,
              -(10.0 * x2 - 7.0) / 24.0,
              1.0 / 24.0,
              -1.0 / 24.0]
        c3 = [0.0,
              (15.0 * x4 - 15.0 * x2 + 4.0) / 120.0,
              0.0,
              (2.0 * x2 - 1.0) / 24.0,
              0.0,
              1.0 / 120.0]
        return {1: np.array(c1), 2: np.array(c2), 3: np.array(c3)}
    raise ValueError(f"Unsupported QuadS order: {order}. Supported: {_SUPPORTED_ORDERS}.")


def _reflect_poly(coeff: np.ndarray) -> np.ndarray:
    """Coefficients of p(1−θ) given ascending coefficients of p(θ).

    p(1−θ) = Σ_p c_p (1−θ)^p = Σ_j θ^j (−1)^j Σ_{p≥j} c_p·C(p,j).
    """
    n = len(coeff)
    out = np.zeros(n)
    for j in range(n):
        s = 0.0
        for p in range(j, n):
            s += coeff[p] * math.comb(p, j)
        out[j] = ((-1) ** j) * s
    return out


_OFFSET_POLY_CACHE: Dict = {}


def quads_offset_polys(xi: float, order: int) -> Dict[int, np.ndarray]:
    """Per-offset monomial-coefficient arrays for ALL 2ν offsets (source of truth).

    offset i∈{1…ν}: base c_i.   offset i∈{1−ν…0}: c_i(θ)=c_{1−i}(1−θ) via reflection.
    Returns {offset: coeff_array[2ν]}.
    """
    key = (round(xi, 12), order)
    if key in _OFFSET_POLY_CACHE:
        return _OFFSET_POLY_CACHE[key]
    base = _base_polys(xi, order)
    nu = order // 2
    polys: Dict[int, np.ndarray] = {}
    for i in range(1, nu + 1):
        polys[i] = base[i]
    for i in range(1 - nu, 1):          # i = 1−ν … 0
        polys[i] = _reflect_poly(base[1 - i])
    _OFFSET_POLY_CACHE[key] = polys
    return polys


def quads_1d_weights(theta: torch.Tensor, xi: float, order: int) -> torch.Tensor:
    """Vectorized 1-D QuadS weights at fractional offsets θ∈[0,1).

    Args:
        theta: [N] fractional coordinate (s − ⌊s⌋).
    Returns:
        [N, 2ν] weights, column k = weight of grid offset quads_offsets(order)[k].
    """
    polys = quads_offset_polys(xi, order)
    offs = quads_offsets(order)
    deg = order  # number of monomial terms = 2ν
    # Powers of theta: [N, deg]
    powers = torch.stack([theta ** p for p in range(deg)], dim=1)
    # Coefficient matrix [deg, 2ν] in offset order
    coeff = torch.tensor(
        np.stack([polys[i] for i in offs], axis=1),
        dtype=theta.dtype, device=theta.device,
    )
    return powers @ coeff  # [N, 2ν]


# ── partition-of-unity / sanity self-check (used by tests) ──
def _check_partition_of_unity(order: int, n: int = 11) -> float:
    xi = quads_xi(order)
    th = torch.linspace(0.0, 1.0, n, dtype=torch.float64)
    w = quads_1d_weights(th, xi, order)  # [n, 2ν]
    return float((w.sum(dim=1) - 1.0).abs().max())
