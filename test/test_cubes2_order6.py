import itertools
import math

import numpy as np
import torch

from sog.module.cubes2_spline import XI_4, XI_6, cubes2_weight, get_nodes, _get_xi


def _weights(nodes, th, xi, order):
    tx, ty, tz = (torch.tensor(t, dtype=torch.float64) for t in th)
    return np.array([cubes2_weight(tx, ty, tz, n, xi, order=order).item() for n in nodes])


def _max_moment_theta_std(order, xi, max_deg, npts=60, seed=0):
    """Max θ-std of Σ_j w_j·(d_j−θ)^(α,β,γ) over degrees ≤ max_deg.

    A correct order-2ν scheme reproduces polynomials through degree 2ν−1, which
    forces these central moments to be θ-INDEPENDENT (std ≈ 0) up to that degree.
    """
    nodes = get_nodes(order)
    dv = np.array([(n.dx, n.dy, n.dz) for n in nodes], float)
    rng = np.random.default_rng(seed)
    worst = 0.0
    combos = [c for c in itertools.product(range(max_deg + 1), repeat=3)
              if sum(c) <= max_deg]
    for combo in combos:
        vals = []
        for _ in range(npts):
            th = rng.random(3)
            w = _weights(nodes, th, xi, order)
            dm = dv - th
            vals.append(float((w * dm[:, 0]**combo[0] * dm[:, 1]**combo[1]
                               * dm[:, 2]**combo[2]).sum()))
        worst = max(worst, float(np.std(vals)))
    return worst


def test_order6_nodes_are_available():
    nodes = get_nodes(6)
    assert len(nodes) == 88, f"expected 88 nodes for order-6 CubeS2, got {len(nodes)}"
    assert all(getattr(node, "cls", None) in {0, 1, 2, 3, 4} for node in nodes)


def test_order6_weights_are_finite_and_structured():
    nodes = get_nodes(6)
    for tx, ty, tz in [(0.0, 0.0, 0.0), (0.25, 0.25, 0.25), (0.5, 0.5, 0.5)]:
        weights = [
            cubes2_weight(
                torch.tensor(tx, dtype=torch.float64),
                torch.tensor(ty, dtype=torch.float64),
                torch.tensor(tz, dtype=torch.float64),
                node,
                XI_6,
                order=6,
            ).item()
            for node in nodes
        ]
        assert all(math.isfinite(w) for w in weights), "order-6 weights should be finite"
        assert any(abs(w) > 1e-12 for w in weights), "order-6 weights should be non-zero"


def test_order6_partition_of_unity():
    nodes = get_nodes(6)
    for tx, ty, tz in [(0.0, 0.0, 0.0), (0.25, 0.25, 0.25), (0.5, 0.5, 0.5)]:
        weights = [
            cubes2_weight(
                torch.tensor(tx, dtype=torch.float64),
                torch.tensor(ty, dtype=torch.float64),
                torch.tensor(tz, dtype=torch.float64),
                node,
                XI_6,
                order=6,
            ).item()
            for node in nodes
        ]
        assert abs(sum(weights) - 1.0) < 1e-6, f"expected partition of unity, got {sum(weights)}"


def test_order4_moment_conditions_pass():
    """Regression guard: order-4 must reproduce polynomials through degree 3."""
    assert _max_moment_theta_std(4, XI_4, max_deg=3) < 1e-9


def test_order6_moment_conditions_pass():
    """Order-6 must reproduce polynomials through degree 5 (2ν−1).

    Validated with the SI-notebook-2 formulas (distinct order-6 S, no ×27).
    """
    assert _max_moment_theta_std(6, XI_6, max_deg=5) < 1e-9


if __name__ == "__main__":
    test_order6_nodes_are_available()
    test_order6_weights_are_finite_and_structured()
    test_order6_partition_of_unity()
    test_order4_moment_conditions_pass()
    test_order6_moment_conditions_pass()
    print("Order-6 smoke + moment checks passed: order-4 through degree 3, "
          "order-6 through degree 5.")

