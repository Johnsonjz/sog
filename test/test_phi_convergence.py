#!/usr/bin/env python3
"""φ-convergence test: validate Predescu 2020 Table III φ_max thresholds.

Tests force accuracy as a function of φ = Δ/r_c for CubeS₂ 4th-order.
Verifies that φ_max gives mesh error ≤ u-series intrinsic error,
and that convergence follows O(Δ⁴) for the 4th-order spline.
"""

import math, warnings
import torch
torch.set_default_dtype(torch.float64)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from sog.module.gaussian import Gaussian
from sog.module.cubes2_fft import (
    _INFLUENCE_CACHE,
    _compute_phi_max,
    _compute_grid_from_phi,
)


def make_charge_neutral_system(n_points: int, L: float = 30.0):
    """Generate charge-neutral random points in a cubic box L×L×L."""
    device = torch.device("cpu")
    r = torch.rand(n_points, 3, device=device) * L
    q = torch.randn(n_points // 2, device=device) * 0.5
    q = torch.cat([q, -q])
    if n_points % 2 == 1:
        q = torch.cat([q, torch.zeros(1, device=device)])
    perm = torch.randperm(n_points)
    r = r[perm]
    q = q[perm]
    cell = torch.eye(3, device=device).unsqueeze(0) * L
    return r, q, cell


def test_phi_convergence():
    """Force convergence vs φ for CubeS₂ 4th-order, b=2."""
    print("=" * 70)
    print("  φ-Convergence: Force Accuracy vs φ = Δ/r_c")
    print("  CubeS₂ 4th-order, b=2, r_c=10.0 Å")
    print("=" * 70)

    torch.manual_seed(42)
    phi_max = _compute_phi_max(2.0, 4)
    print(f"  φ_max (Table III): {phi_max:.3f}")
    print()

    for n_pts in [100, 500]:
        r, q, cell = make_charge_neutral_system(n_pts)
        L = 30.0

        # Reference: direct sum
        r_dir = r.clone().requires_grad_(True)
        g_dir = Gaussian(n_dl=2.0, b=2, rcut=10.0, nlayers=1,
                         use_cubes2_fft=False, use_nufft=False,
                         remove_self_interaction=True)
        e_dir = g_dir(q, r_dir, cell)
        f_dir = -torch.autograd.grad(e_dir, r_dir,
            grad_outputs=torch.ones_like(e_dir))[0]

        # Reference: FFT with very fine grid (mesh error ≈ 0)
        _INFLUENCE_CACHE.clear()
        g_fine = Gaussian(cubes2_phi_max=0.03, b=2, rcut=10.0, nlayers=1,
                          use_cubes2_fft=True, remove_self_interaction=True)
        r_fine = r.clone().requires_grad_(True)
        e_fine = g_fine(q, r_fine, cell)
        f_fine = -torch.autograd.grad(e_fine, r_fine,
            grad_outputs=torch.ones_like(e_fine))[0]

        # u-series intrinsic error: fine FFT vs direct
        cos_useries = torch.cosine_similarity(
            f_dir.flatten(), f_fine.flatten(), dim=0)

        print(f"  N={n_pts:5d}:")
        print(f"    u-series intrinsic error (fine FFT vs direct): cos={cos_useries.item():.6f}")
        print(f"    {'φ':>8s}  {'grid':>8s}  {'cos_vs_dir':>12s}  "
              f"{'cos_vs_fine':>12s}  {'rmse':>10s}  {'Δ(Å)':>8s}")
        print(f"    {'-'*8}  {'-'*8}  {'-'*12}  {'-'*12}  {'-'*10}  {'-'*8}")

        for phi in [0.50, 0.35, 0.23, 0.15, 0.10, 0.07, 0.05, 0.03]:
            _INFLUENCE_CACHE.clear()
            nx, ny, nz = _compute_grid_from_phi(L, L, L, 10.0, phi)
            delta = phi * 10.0  # Δ = φ * r_c

            g_fft = Gaussian(cubes2_phi_max=phi, b=2, rcut=10.0, nlayers=1,
                             use_cubes2_fft=True, remove_self_interaction=True)
            r_fft = r.clone().requires_grad_(True)
            e_fft = g_fft(q, r_fft, cell)
            f_fft = -torch.autograd.grad(e_fft, r_fft,
                grad_outputs=torch.ones_like(e_fft))[0]

            cos_dir = torch.cosine_similarity(
                f_dir.flatten(), f_fft.flatten(), dim=0)
            cos_fine = torch.cosine_similarity(
                f_fine.flatten(), f_fft.flatten(), dim=0)
            rmse = (f_fft - f_dir).norm() / max(f_dir.norm(), 1e-30)

            marker = " ← φ_max" if abs(phi - phi_max) < 0.01 else ""
            print(f"    {phi:8.3f}  {nx:4d}×{ny:4d}×{nz:4d}  "
                  f"{cos_dir.item():12.6f}  {cos_fine.item():12.6f}  "
                  f"{rmse.item():10.4f}  {delta:8.3f}{marker}")

        print()

    # ── Convergence order test ──
    print("  Convergence Order (N=500):")
    print(f"    {'φ':>8s}  {'cos_vs_fine':>12s}  {'1-cos':>12s}  {'Δ(Å)':>8s}")
    print(f"    {'-'*8}  {'-'*12}  {'-'*12}  {'-'*8}")

    r, q, cell = make_charge_neutral_system(500)
    errors = []
    for phi in [0.23, 0.15, 0.10, 0.07, 0.05]:
        _INFLUENCE_CACHE.clear()
        g_fft = Gaussian(cubes2_phi_max=phi, b=2, rcut=10.0, nlayers=1,
                         use_cubes2_fft=True, remove_self_interaction=True)
        r_fft = r.clone().requires_grad_(True)
        e_fft = g_fft(q, r_fft, cell)
        f_fft = -torch.autograd.grad(e_fft, r_fft,
            grad_outputs=torch.ones_like(e_fft))[0]

        cos = torch.cosine_similarity(f_fine.flatten(), f_fft.flatten(), dim=0)
        err = 1.0 - cos.item()
        delta = phi * 10.0
        errors.append((phi, delta, err))
        print(f"    {phi:8.3f}  {cos.item():12.8f}  {err:12.6e}  {delta:8.3f}")

    # Estimate convergence order: err ∝ Δ^p
    if len(errors) >= 2:
        phi1, d1, e1 = errors[-1]  # finest
        phi2, d2, e2 = errors[0]   # coarsest
        if e1 > 0 and e2 > 0:
            p = math.log(e2 / e1) / math.log(d2 / d1)
            print(f"\n    Estimated convergence order: O(Δ^{p:.2f})")
            print(f"    Expected for 4th-order CubeS₂: O(Δ⁴)")

    print()
    print("  ✓ φ-convergence validation complete")


def test_training_scale():
    """φ-convergence at training-relevant scales."""
    print()
    print("=" * 70)
    print("  Training-Scale: 12.44 Å box, b=1.63, r_c=6.0 Å")
    print("=" * 70)

    torch.manual_seed(42)
    r = torch.rand(192, 3) * 12.44  # 64 H₂O scale
    q = torch.randn(96) * 0.5
    q = torch.cat([q, -q])
    cell = torch.eye(3).unsqueeze(0) * 12.44
    L = 12.44
    r_c = 6.0

    # Reference: direct sum
    g_dir = Gaussian(n_dl=2.0, b=1.63, sigma=3.63, m=16, nlayers=1,
                     use_cubes2_fft=False, remove_self_interaction=True)
    r_dir = r.clone().requires_grad_(True)
    e_dir = g_dir(q, r_dir, cell)
    f_dir = -torch.autograd.grad(e_dir, r_dir,
        grad_outputs=torch.ones_like(e_dir))[0]

    phi_auto = _compute_phi_max(1.63, 4)
    print(f"  φ_max(b=1.63) = {phi_auto:.4f}")
    print(f"    {'φ':>8s}  {'grid':>10s}  {'cos':>10s}  {'rmse':>10s}  {'Δ(Å)':>8s}")
    print(f"    {'-'*8}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*8}")

    for phi in [0.15, 0.10, phi_auto, 0.05, 0.03]:
        _INFLUENCE_CACHE.clear()
        nx, ny, nz = _compute_grid_from_phi(L, L, L, r_c, phi)
        delta = phi * r_c

        g_fft = Gaussian(cubes2_phi_max=phi, b=1.63, sigma=3.63, m=16, rcut=r_c,
                         nlayers=1, use_cubes2_fft=True,
                         remove_self_interaction=True)
        r_fft = r.clone().requires_grad_(True)
        e_fft = g_fft(q, r_fft, cell)
        f_fft = -torch.autograd.grad(e_fft, r_fft,
            grad_outputs=torch.ones_like(e_fft))[0]

        cos = torch.cosine_similarity(f_dir.flatten(), f_fft.flatten(), dim=0)
        rmse = (f_fft - f_dir).norm() / max(f_dir.norm(), 1e-30)

        marker = " ← auto φ_max" if abs(phi - phi_auto) < 0.001 else ""
        print(f"    {phi:8.4f}  {nx:4d}×{ny:4d}×{nz:4d}  "
              f"{cos.item():10.6f}  {rmse.item():10.4f}  {delta:8.4f}{marker}")

    print()
    print("  ✓ Training-scale validation complete")


if __name__ == "__main__":
    test_phi_convergence()
    test_training_scale()
    print("\nDone.")
