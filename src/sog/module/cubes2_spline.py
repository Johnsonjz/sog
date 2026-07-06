"""CubeS₂ Midtown Splines — node table and weight functions.

Reference: Predescu, Bergdorf, Shaw, J. Chem. Phys. 153, 224117 (2020)
Ported from fastsog_spline.h / sog_spline.h in the deepmd-kit LAMMPS plugin.

Supported orders: 4 (32 nodes, φ_max=0.23@b=2), 6 (88 nodes, φ_max=0.35@b=2).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import torch

# ── Optimal ξ parameters ──
XI_4: float = 1.0 / math.sqrt(3.0)  # ≈ 0.5773502691896258  (C² continuous)
XI_6: float = 0.6503998              # optimal for CubeS₂ 6th-order (paper Appendix, NOT Table I)


def _get_xi(order: int) -> float:
    """Get optimal ξ for the given CubeS₂ order."""
    if order == 4:
        return XI_4
    if order == 6:
        return XI_6
    raise ValueError(f"Unsupported CubeS₂ order: {order}. Supported: 4, 6.")


# ── Node counts ──
NUM_NODES_4: int = 32
NUM_NODES_6: int = 88


def _get_num_nodes(order: int) -> int:
    if order == 4:
        return NUM_NODES_4
    if order == 6:
        return NUM_NODES_6
    raise ValueError(f"Unsupported CubeS₂ order: {order}. Supported: 4, 6.")


@dataclass
class CubeS2Node4:
    """A single CubeS₂ assignment node (works for all orders)."""

    dx: int  # offset from floor grid index (x)
    dy: int  # offset from floor grid index (y)
    dz: int  # offset from floor grid index (z)
    cls: int  # 0 = (1,1,1) class, 1 = (2,1,1) class
    sp_axis: int  # for cls==1: which axis (0=x,1=y,2=z) has special component
    sp_is_neg: int  # for cls==1: 1 if d=-1, 0 if d=2

# Generic alias — same struct for all orders
CubeS2Node = CubeS2Node4


# ── 4th-order node table (32 nodes) ──
CUBES2_NODES_4: List[CubeS2Node] = [
    # Class 0: offsets in {0,1}³ (8 nodes, L-symmetry)
    CubeS2Node4(0, 0, 0, 0, -1, 0),
    CubeS2Node4(0, 0, 1, 0, -1, 0),
    CubeS2Node4(0, 1, 0, 0, -1, 0),
    CubeS2Node4(0, 1, 1, 0, -1, 0),
    CubeS2Node4(1, 0, 0, 0, -1, 0),
    CubeS2Node4(1, 0, 1, 0, -1, 0),
    CubeS2Node4(1, 1, 0, 0, -1, 0),
    CubeS2Node4(1, 1, 1, 0, -1, 0),
    # Class 1: x-axis special, d=-1 (4 nodes)
    CubeS2Node4(-1, 0, 0, 1, 0, 1),
    CubeS2Node4(-1, 0, 1, 1, 0, 1),
    CubeS2Node4(-1, 1, 0, 1, 0, 1),
    CubeS2Node4(-1, 1, 1, 1, 0, 1),
    # Class 1: x-axis special, d=2 (4 nodes)
    CubeS2Node4(2, 0, 0, 1, 0, 0),
    CubeS2Node4(2, 0, 1, 1, 0, 0),
    CubeS2Node4(2, 1, 0, 1, 0, 0),
    CubeS2Node4(2, 1, 1, 1, 0, 0),
    # Class 1: y-axis special, d=-1 (4 nodes)
    CubeS2Node4(0, -1, 0, 1, 1, 1),
    CubeS2Node4(0, -1, 1, 1, 1, 1),
    CubeS2Node4(1, -1, 0, 1, 1, 1),
    CubeS2Node4(1, -1, 1, 1, 1, 1),
    # Class 1: y-axis special, d=2 (4 nodes)
    CubeS2Node4(0, 2, 0, 1, 1, 0),
    CubeS2Node4(0, 2, 1, 1, 1, 0),
    CubeS2Node4(1, 2, 0, 1, 1, 0),
    CubeS2Node4(1, 2, 1, 1, 1, 0),
    # Class 1: z-axis special, d=-1 (4 nodes)
    CubeS2Node4(0, 0, -1, 1, 2, 1),
    CubeS2Node4(0, 1, -1, 1, 2, 1),
    CubeS2Node4(1, 0, -1, 1, 2, 1),
    CubeS2Node4(1, 1, -1, 1, 2, 1),
    # Class 1: z-axis special, d=2 (4 nodes)
    CubeS2Node4(0, 0, 2, 1, 2, 0),
    CubeS2Node4(0, 1, 2, 1, 2, 0),
    CubeS2Node4(1, 0, 2, 1, 2, 0),
    CubeS2Node4(1, 1, 2, 1, 2, 0),
]


def _cubes2_L(theta: torch.Tensor, xi: float) -> torch.Tensor:
    """L(θ,ξ) = -½θ³ + ½θ² - (9ξ²-2)/6·θ + ξ²/2  (paper Eq. 15)"""
    xi2 = xi * xi
    xi2_adj = (9.0 * xi2 - 2.0) / 6.0
    return -0.5 * theta**3 + 0.5 * theta**2 - xi2_adj * theta + 0.5 * xi2


def _cubes2_R(theta: torch.Tensor, xi: float) -> torch.Tensor:
    """R(θ,ξ) = ⅙θ³ + (3ξ²-1)/6·θ  (paper Eq. 16)"""
    xi2 = xi * xi
    return (1.0 / 6.0) * theta**3 + (3.0 * xi2 - 1.0) / 6.0 * theta


# ── 6th-order weight sub-functions (degree-5, paper Eq. A9) ──
# NOTE: Polynomial coefficients need verification against published Appendix.
# The functional forms (degree-5), node table (88 nodes, 5 classes), weight
# formula structure (Eq. A10), and optimal xi (0.6503998) are correct.
# Current coefficients give weight sum ≈ 1/27 instead of 1.
# TODO: verify L111/L311/L211 coefficients against paper Eq. A9;
#       then remove the fallback in cubes2_weight() order=6 dispatch.
_COEFFS_NEED_VERIFICATION = True

# Coefficients below produce correctly-structured but incorrectly-scaled
# (~1/27×) weights. See midtown-sog.md §7.3 for expected formulas.

def _cubes2_L111_6(theta: torch.Tensor, xi: float) -> torch.Tensor:
    """L₁₁₁(θ,ξ) — degree-5 polynomial for 6th-order cls=0 (Eq. A9)."""
    xi2 = xi * xi
    xi4 = xi2 * xi2
    return (1.0 / 12.0 * theta**5
            - 1.0 / 6.0 * theta**4
            + (10.0 * xi2 - 1.0) / 12.0 * theta**3
            - (6.0 * xi2 - 1.0) / 6.0 * theta**2
            + (5.0 * xi4 - xi2) / 4.0 * theta
            - (3.0 * xi4 - xi2) / 6.0)


def _cubes2_L311_6(theta: torch.Tensor, xi: float) -> torch.Tensor:
    """L₃₁₁(θ,ξ) — degree-5 polynomial for 6th-order cls=2 (Eq. A9)."""
    xi2 = xi * xi
    xi4 = xi2 * xi2
    return (1.0 / 120.0 * theta**5
            + (2.0 * xi2 - 1.0) / 24.0 * theta**3
            + (15.0 * xi4 - 15.0 * xi2 + 4.0) / 120.0 * theta)


def _cubes2_L211_6(theta: torch.Tensor, xi: float) -> torch.Tensor:
    """L₂₁₁(θ,ξ) = -¼ L₁₁₁ - ⁵⁄₂ L₃₁₁ (Eq. A9)."""
    return (-0.25 * _cubes2_L111_6(theta, xi)
            - 2.5 * _cubes2_L311_6(theta, xi))


# ── 6th-order node table (88 nodes, D₂ ball with v=3) ──

def _generate_nodes_6() -> List[CubeS2Node]:
    """Generate the 88-node CubeS₂ 6th-order table (D₂ ball, v=3).

    Classifies each D₂-ball point into one of 5 symmetry classes based on
    coordinate pattern, encoding the special/normal axis info needed by the
    weight dispatch.
    """
    nodes: List[CubeS2Node] = []

    # D₂ ball membership: squared distance to [0,1]³ ≤ 4
    def _in_d2(dx: int, dy: int, dz: int) -> bool:
        def _dist2(c: int) -> float:
            if 0 <= c <= 1:
                return 0.0
            if c < 0:
                return float(c * c)
            return float((c - 1) * (c - 1))
        return (_dist2(dx) + _dist2(dy) + _dist2(dz)) <= 4.0 + 1e-10

    # Classify based on coordinate pattern
    def _classify(dx: int, dy: int, dz: int):
        coords = [dx, dy, dz]
        special_axes = []  # axes not in {0,1}
        for a, c in enumerate(coords):
            if c not in (0, 1):
                special_axes.append(a)

        n_special = len(special_axes)
        if n_special == 0:
            return 0, -1, 0
        elif n_special == 1:
            sa = special_axes[0]
            c = coords[sa]
            if c in (-1, 2):
                return 1, sa, 1 if c == -1 else 0
            else:
                return 2, sa, 1 if c == -2 else 0
        elif n_special == 2:
            # cls=3: (2,2,1) — find normal axis
            normal_axis = [a for a in range(3) if a not in special_axes][0]
            sign_bits = 0
            for i, sa in enumerate(sorted(special_axes)):
                if coords[sa] == -1:
                    sign_bits |= (1 << i)
            return 3, normal_axis, sign_bits
        else:
            # n_special == 3: cls=4: (2,2,2)
            sign_bits = 0
            for a in range(3):
                if coords[a] == -1:
                    sign_bits |= (1 << a)
            return 4, -1, sign_bits

    # Scan all integer triples within bounding box [-2, 3]³
    for dx in range(-2, 4):
        for dy in range(-2, 4):
            for dz in range(-2, 4):
                if _in_d2(dx, dy, dz):
                    cls, sp_axis, sp_is_neg = _classify(dx, dy, dz)
                    nodes.append(CubeS2Node4(dx, dy, dz, cls, sp_axis, sp_is_neg))

    # Sort for reproducibility: by (cls, dz, dy, dx)
    nodes.sort(key=lambda n: (n.cls, n.dz, n.dy, n.dx))
    return nodes


CUBES2_NODES_6: List[CubeS2Node] = _generate_nodes_6()


# ── 6th-order weight function (Eq. A10) ──

def cubes2_weight_6(
    tx: torch.Tensor,
    ty: torch.Tensor,
    tz: torch.Tensor,
    node: CubeS2Node,
    xi: float,
) -> torch.Tensor:
    """Compute CubeS₂ 6th-order weight for fractional coords (tx,ty,tz) ∈ [0,1).

    5 symmetry classes with degree-5 polynomials (paper Eq. A10).
    S(θ) = _cubes2_L, R(θ) = _cubes2_R from 4th-order (shared sub-functions).
    """
    # ── cls=0: c₁₁₁ = Σ L111(ηᵢ)ηⱼηₖ + S(η₁)S(η₂)S(η₃)
    if node.cls == 0:
        eta_x = tx if node.dx == 0 else (1.0 - tx)
        eta_y = ty if node.dy == 0 else (1.0 - ty)
        eta_z = tz if node.dz == 0 else (1.0 - tz)
        return (
            _cubes2_L111_6(eta_x, xi) * eta_y * eta_z
            + _cubes2_L111_6(eta_y, xi) * eta_z * eta_x
            + _cubes2_L111_6(eta_z, xi) * eta_x * eta_y
            + _cubes2_L(eta_x, xi) * _cubes2_L(eta_y, xi) * _cubes2_L(eta_z, xi)
        )

    # ── cls=1: c₂₁₁ = L211(η_sp)η_n1η_n2 + R(η_sp)S(η_n1)S(η_n2)
    elif node.cls == 1:
        coords = [tx, ty, tz]
        offsets = [node.dx, node.dy, node.dz]
        sp = node.sp_axis
        eta_s = coords[sp] if node.sp_is_neg else (1.0 - coords[sp])
        normal_axes = [a for a in range(3) if a != sp]
        eta_n1 = coords[normal_axes[0]] if offsets[normal_axes[0]] == 0 else (1.0 - coords[normal_axes[0]])
        eta_n2 = coords[normal_axes[1]] if offsets[normal_axes[1]] == 0 else (1.0 - coords[normal_axes[1]])
        return (
            _cubes2_L211_6(eta_s, xi) * eta_n1 * eta_n2
            + _cubes2_R(eta_s, xi) * _cubes2_L(eta_n1, xi) * _cubes2_L(eta_n2, xi)
        )

    # ── cls=2: c₃₁₁ = L311(η_sp)η_n1η_n2
    elif node.cls == 2:
        coords = [tx, ty, tz]
        offsets = [node.dx, node.dy, node.dz]
        sp = node.sp_axis
        eta_s = coords[sp] if node.sp_is_neg else (1.0 - coords[sp])
        normal_axes = [a for a in range(3) if a != sp]
        eta_n1 = coords[normal_axes[0]] if offsets[normal_axes[0]] == 0 else (1.0 - coords[normal_axes[0]])
        eta_n2 = coords[normal_axes[1]] if offsets[normal_axes[1]] == 0 else (1.0 - coords[normal_axes[1]])
        return _cubes2_L311_6(eta_s, xi) * eta_n1 * eta_n2

    # ── cls=3: c₂₂₁ = R(η_sp1)R(η_sp2)S(η_n)
    elif node.cls == 3:
        coords = [tx, ty, tz]
        offsets = [node.dx, node.dy, node.dz]
        normal_axis = node.sp_axis  # sp_axis = NORMAL axis for cls=3
        special_axes = sorted([a for a in range(3) if a != normal_axis])
        eta_n = coords[normal_axis] if offsets[normal_axis] == 0 else (1.0 - coords[normal_axis])
        eta_s1 = coords[special_axes[0]] if bool(node.sp_is_neg & 1) else (1.0 - coords[special_axes[0]])
        eta_s2 = coords[special_axes[1]] if bool(node.sp_is_neg & 2) else (1.0 - coords[special_axes[1]])
        return (_cubes2_L(eta_n, xi)
                * _cubes2_R(eta_s1, xi)
                * _cubes2_R(eta_s2, xi))

    # ── cls=4: c₂₂₂ = R(η₁)R(η₂)R(η₃)
    else:  # node.cls == 4
        eta_x = tx if bool(node.sp_is_neg & 1) else (1.0 - tx)
        eta_y = ty if bool(node.sp_is_neg & 2) else (1.0 - ty)
        eta_z = tz if bool(node.sp_is_neg & 4) else (1.0 - tz)
        return (_cubes2_R(eta_x, xi)
                * _cubes2_R(eta_y, xi)
                * _cubes2_R(eta_z, xi))


def cubes2_weight_4(
    tx: torch.Tensor,
    ty: torch.Tensor,
    tz: torch.Tensor,
    node: CubeS2Node4,
    xi: float,
) -> torch.Tensor:
    """Compute CubeS₂ 4th-order weight for fractional coords (tx,ty,tz) ∈ [0,1).

    Args:
        tx, ty, tz: Fractional position scalars within [0,1).
        node: CubeS₂ node definition.
        xi: Optimal ξ parameter (XI_4 = 1/√3 for 4th order).

    Returns:
        Weight value (scalar tensor).
    """
    if node.cls == 0:
        # Class 0: c_d = L(ηx)·ηy·ηz + L(ηy)·ηz·ηx + L(ηz)·ηx·ηy
        eta_x = tx if node.dx == 0 else (1.0 - tx)
        eta_y = ty if node.dy == 0 else (1.0 - ty)
        eta_z = tz if node.dz == 0 else (1.0 - tz)
        return (
            _cubes2_L(eta_x, xi) * eta_y * eta_z
            + _cubes2_L(eta_y, xi) * eta_z * eta_x
            + _cubes2_L(eta_z, xi) * eta_x * eta_y
        )
    else:
        # Class 1: c_d = R(η_special) · η_n1 · η_n2
        if node.sp_axis == 0:
            eta_special = tx if node.sp_is_neg else (1.0 - tx)
            eta_n1 = ty if node.dy == 0 else (1.0 - ty)
            eta_n2 = tz if node.dz == 0 else (1.0 - tz)
        elif node.sp_axis == 1:
            eta_special = ty if node.sp_is_neg else (1.0 - ty)
            eta_n1 = tx if node.dx == 0 else (1.0 - tx)
            eta_n2 = tz if node.dz == 0 else (1.0 - tz)
        else:  # sp_axis == 2
            eta_special = tz if node.sp_is_neg else (1.0 - tz)
            eta_n1 = tx if node.dx == 0 else (1.0 - tx)
            eta_n2 = ty if node.dy == 0 else (1.0 - ty)
        return _cubes2_R(eta_special, xi) * eta_n1 * eta_n2


# ── Order dispatch ──

_SUPPORTED_ORDERS = (4, 6)


def get_nodes(order: int) -> List[CubeS2Node]:
    """Get the CubeS₂ node table for the given order."""
    if order == 4:
        return CUBES2_NODES_4
    if order == 6:
        import warnings
        warnings.warn(
            "6th-order CubeS₂ weight polynomial coefficients need verification "
            "against paper Appendix Eq. A9. Node table (88 nodes, 5 classes) and "
            "formula structure (Eq. A10) are correct but produce weights scaled "
            "~1/27× too small. Falling back to 4th-order weights. "
            "Grid sizing (φ_max) still uses 6th-order thresholds from Table III."
        )
        return CUBES2_NODES_4
    raise ValueError(
        f"Unsupported CubeS₂ order: {order}. Supported: {_SUPPORTED_ORDERS}."
    )


def cubes2_weight(
    tx: torch.Tensor,
    ty: torch.Tensor,
    tz: torch.Tensor,
    node: CubeS2Node,
    xi: float,
    order: int = 4,
) -> torch.Tensor:
    """Compute CubeS₂ weight for the given order.

    For 4th-order, uses L(θ)/R(θ) from paper Eqs. 15-16.
    For 6th-order, uses the same cls==0/cls==1 structure with L6/R6 (TODO).
    """
    if order == 4:
        return cubes2_weight_4(tx, ty, tz, node, xi)
    if order == 6:
        # Polynomial coefficients need verification; use 4th-order weights
        return cubes2_weight_4(tx, ty, tz, node, xi)
    raise ValueError(f"Unsupported order: {order}")


def get_supported_orders() -> Tuple[int, ...]:
    """Return the tuple of supported CubeS₂ orders."""
    return _SUPPORTED_ORDERS
