"""φ-convergence tests for the CubeS₂ FFT path (force accuracy vs φ = Δ/r_c).

1. ``test_phi_ladder_converges_to_direct`` — for 100/500-atom neutral random
   systems, the FFT forces converge to the direct k-sum as φ decreases; at the
   Table III default ``φ_max = 0.23`` the agreement is already tight.
2. ``test_convergence_order`` — the same-system force error follows a high
   power of Δ (order-4 CubeS₂; measured p ≈ 7 at these resolutions — the
   spline is super-convergent in this regime).
3. ``test_training_scale_agreement`` — 192-atom water-scale box, b=1.63,
   r_c=6.0 Å (σ = r_c/RCUT_TO_SIGMA on both sides — the two paths must share
   the SAME kernel; mixing an explicit σ with r_c silently changes it).

All assertions are pinned a few× above the measured values with the measured
numbers recorded in the assert messages; update both together.
"""

import math
import warnings

import torch

torch.set_default_dtype(torch.float64)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from sog.module.gaussian import Gaussian  # noqa: E402
from sog.module.cubes2_fft import (  # noqa: E402
    _INFLUENCE_CACHE,
    _compute_phi_max,
    _compute_grid_from_phi,
)

PHI_LADDER = [0.50, 0.35, 0.23, 0.15, 0.10, 0.07, 0.05, 0.03]


def make_neutral_system(n_points, L=30.0):
    """Charge-neutral random points in a cubic box L×L×L."""
    r = torch.rand(n_points, 3) * L
    q = torch.randn(n_points // 2) * 0.5
    q = torch.cat([q, -q])
    if n_points % 2 == 1:
        q = torch.cat([q, torch.zeros(1)])
    perm = torch.randperm(n_points)
    r, q = r[perm], q[perm]
    return r, q, torch.eye(3).unsqueeze(0) * L


def _force(g, q, r, cell):
    r2 = r.clone().requires_grad_(True)
    e = g(q, r2, cell)
    f = -torch.autograd.grad(e, r2, grad_outputs=torch.ones_like(e))[0]
    return f, e


def _fft_force(phi, q, r, cell, b, rcut, nlayers=1):
    _INFLUENCE_CACHE.clear()
    g = Gaussian(cubes2_phi_max=phi, b=b, rcut=rcut, nlayers=nlayers,
                 use_cubes2_fft=True, remove_self_interaction=True)
    return _force(g, q, r, cell)


def _direct_force(q, r, cell, b, rcut, n_dl, nlayers=1):
    g = Gaussian(n_dl=n_dl, b=b, rcut=rcut, nlayers=nlayers,
                 use_cubes2_fft=False, use_nufft=False,
                 remove_self_interaction=True)
    return _force(g, q, r, cell)


def test_phi_ladder_converges_to_direct():
    """FFT force error shrinks monotonically along the φ ladder; the Table III
    default (φ_max=0.23) is already tight."""
    torch.manual_seed(42)
    phi_max = _compute_phi_max(2.0, 4)
    assert abs(phi_max - 0.23) < 0.02, (
        f"Table III φ_max for (b=2, order-4) changed: {phi_max:.4f}"
    )

    for n_pts in (100, 500):
        r, q, cell = make_neutral_system(n_pts)
        f_dir, _ = _direct_force(q, r, cell, 2.0, 10.0, n_dl=2.0)

        rmses = []
        for phi in PHI_LADDER:
            f_fft, _ = _fft_force(phi, q, r, cell, 2.0, 10.0)
            rmse = (f_fft - f_dir).norm().item() / max(f_dir.norm().item(), 1e-30)
            rmses.append(rmse)
            if abs(phi - phi_max) < 0.01:
                assert rmse <= 5e-3, (
                    f"N={n_pts}: rmse at φ_max={phi_max} is {rmse:.4f} "
                    f"(measured 0.0012-0.0014; threshold 5e-3)"
                )
        # monotone (within small numerical slack), and a real dynamic range
        for a, b_ in zip(rmses, rmses[1:]):
            assert b_ <= a * 1.05 + 1e-12, (
                f"N={n_pts}: rmse not decreasing along the φ ladder: {rmses}"
            )
        assert rmses[0] > 10.0 * rmses[-1], (
            f"N={n_pts}: coarse-to-fine rmse gain only {rmses[0] / rmses[-1]:.1f}x: "
            f"{rmses}"
        )


def test_convergence_order():
    """Force error vs Δ on the SAME system follows a high power of Δ.

    The original script compared a stale fine-grid reference (different random
    system) — fixed here by computing both on one system.  Measured p ≈ 7 for
    order-4 CubeS₂ at these resolutions (super-convergent regime).
    """
    torch.manual_seed(42)
    r, q, cell = make_neutral_system(500)

    f_fine, _ = _fft_force(0.03, q, r, cell, 2.0, 10.0)
    errs = []
    for phi in (0.23, 0.15, 0.10, 0.07, 0.05):
        f, _ = _fft_force(phi, q, r, cell, 2.0, 10.0)
        cos = torch.cosine_similarity(f_fine.flatten(), f.flatten(), dim=0).item()
        errs.append((phi * 10.0, 1.0 - cos))  # (Δ, err)

    for (d1, e1), (d2, e2) in zip(errs, errs[1:]):
        assert d2 < d1 and e2 < e1, f"not converging: {errs}"

    (d_coarse, e_coarse), (d_fine, e_fine) = errs[0], errs[-1]
    p = math.log(e_coarse / e_fine) / math.log(d_coarse / d_fine)
    assert 3.0 <= p <= 10.0, (
        f"estimated convergence order p={p:.2f} outside [3, 10] "
        f"(measured ≈7.5; errors {[f'{e:.2e}' for _, e in errs]})"
    )


def test_training_scale_agreement():
    """192-atom, 12.44 Å box, b=1.63, r_c=6.0 Å — both paths share σ via r_c.

    NOTE: giving ``rcut`` OVERRIDES an explicit ``sigma``
    (σ = r_c·nlayers/RCUT_TO_SIGMA, gaussian.py); the two sides of this
    comparison must therefore both be built with r_c=6.0 Å (σ≈3.0162).
    """
    torch.manual_seed(42)
    L = 12.44
    r_c = 6.0
    r = torch.rand(192, 3) * L
    q = torch.randn(96) * 0.5
    q = torch.cat([q, -q])
    cell = torch.eye(3).unsqueeze(0) * L

    f_dir, e_dir = _direct_force(q, r, cell, 1.63, r_c, n_dl=2.0)

    phi_auto = _compute_phi_max(1.63, 4)
    f_fft, e_fft = _fft_force(phi_auto, q, r, cell, 1.63, r_c)
    cos = torch.cosine_similarity(f_dir.flatten(), f_fft.flatten(), dim=0).item()
    e_rel = abs(e_fft.item() - e_dir.item()) / max(abs(e_dir.item()), 1e-30)

    assert e_rel < 1e-3, (
        f"energy rel {e_rel:.2e} at φ_auto={phi_auto:.4f} (measured 2.8e-4)"
    )
    assert cos > 0.9999, f"force cosine {cos:.8f} (measured ≈1.00000000)"

    # refinement must improve the energy agreement
    _, e_finer = _fft_force(0.03, q, r, cell, 1.63, r_c)
    e_rel_finer = abs(e_finer.item() - e_dir.item()) / max(abs(e_dir.item()), 1e-30)
    assert e_rel_finer < e_rel, (
        f"refining φ must reduce the error: {e_rel:.2e} -> {e_rel_finer:.2e}"
    )
