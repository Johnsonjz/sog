#!/usr/bin/env python3
"""Small-scale test: SOG (FFT, φ=auto) vs direct sum — accuracy + speed.

Tests water 64H₂O (192 atoms) and NaCl (512 ions) at training-relevant scales.
No GPU required.
"""

import math, time, warnings
import torch
torch.set_default_dtype(torch.float64)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from sog.module.gaussian import Gaussian
from sog.module.cubes2_fft import _INFLUENCE_CACHE, _compute_phi_max, _compute_grid_from_phi


def test_water_64():
    """Water 64H₂O: 192 atoms, ~12.44 Å box, b=1.63, r_c=6.0 Å."""
    print("=" * 70)
    print("  Water 64H₂O (192 atoms, 12.44 Å box)")
    print("  b=1.63, sigma=3.63, M=16, r_c=6.0 Å")
    print("=" * 70)

    torch.manual_seed(42)
    L = 12.44
    r_c = 6.0
    b_val = 1.63
    r = torch.rand(192, 3) * L
    q = torch.randn(96) * 0.5
    q = torch.cat([q, -q])
    cell = torch.eye(3).unsqueeze(0) * L

    # ── Direct reference ──
    g_dir = Gaussian(n_dl=2.0, b=b_val, sigma=3.63, m=16, nlayers=1,
                     use_cubes2_fft=False, remove_self_interaction=True)
    t0 = time.perf_counter()
    r_dir = r.clone().requires_grad_(True)
    e_dir = g_dir(q, r_dir, cell)
    f_dir = -torch.autograd.grad(e_dir, r_dir,
        grad_outputs=torch.ones_like(e_dir))[0]
    t_dir = (time.perf_counter() - t0) * 1000

    phi_auto = _compute_phi_max(b_val, 4)
    print(f"  φ_max(b={b_val}) = {phi_auto:.4f}")
    print(f"  Direct: E={e_dir.item():.4f}  |f|={f_dir.norm().item():.2f}  {t_dir:.1f}ms")

    # ── FFT with auto φ ──
    nx, ny, nz = _compute_grid_from_phi(L, L, L, r_c, phi_auto)
    _INFLUENCE_CACHE.clear()
    g_fft = Gaussian(cubes2_phi_max=phi_auto, b=b_val, sigma=3.63, m=16,
                     rcut=r_c, nlayers=1, use_cubes2_fft=True,
                     remove_self_interaction=True)
    t0 = time.perf_counter()
    r_fft = r.clone().requires_grad_(True)
    e_fft = g_fft(q, r_fft, cell)
    f_fft = -torch.autograd.grad(e_fft, r_fft,
        grad_outputs=torch.ones_like(e_fft))[0]
    t_fft = (time.perf_counter() - t0) * 1000

    cos = torch.cosine_similarity(f_dir.flatten(), f_fft.flatten(), dim=0)
    e_rel = abs(e_fft.item() - e_dir.item()) / max(abs(e_dir.item()), 1e-30) * 100
    rmse = (f_fft - f_dir).norm() / max(f_dir.norm(), 1e-30)

    print(f"  FFT φ=auto: grid={nx}³  E={e_fft.item():.4f}  "
          f"cos={cos.item():.6f}  e_rel={e_rel:.2f}%  rmse={rmse.item():.4f}  "
          f"{t_fft:.1f}ms  speedup={t_dir/t_fft:.1f}x")
    print()


def test_nacl_512():
    """NaCl 512 ions: 256 Na + 256 Cl, ~24 Å box, b=2, r_c=10.0 Å."""
    print("=" * 70)
    print("  NaCl 512 ions (256 Na + 256 Cl, 24 Å box)")
    print("  b=2.0, M=12, r_c=10.0 Å")
    print("=" * 70)

    torch.manual_seed(42)
    L = 24.0
    r_c = 10.0
    b_val = 2.0
    n = 512
    r = torch.rand(n, 3) * L
    q = torch.ones(n)
    q[:n//2] = 1.0   # Na+
    q[n//2:] = -1.0  # Cl-
    cell = torch.eye(3).unsqueeze(0) * L

    # ── Direct reference ──
    g_dir = Gaussian(n_dl=2.0, b=b_val, rcut=r_c, nlayers=1,
                     use_cubes2_fft=False, remove_self_interaction=True)
    t0 = time.perf_counter()
    r_dir = r.clone().requires_grad_(True)
    e_dir = g_dir(q, r_dir, cell)
    f_dir = -torch.autograd.grad(e_dir, r_dir,
        grad_outputs=torch.ones_like(e_dir))[0]
    t_dir = (time.perf_counter() - t0) * 1000

    phi_auto = _compute_phi_max(b_val, 4)
    print(f"  φ_max(b={b_val}) = {phi_auto:.3f}")
    print(f"  Direct: E={e_dir.item():.4f}  |f|={f_dir.norm().item():.2f}  {t_dir:.1f}ms")

    # ── FFT with auto φ ──
    nx, ny, nz = _compute_grid_from_phi(L, L, L, r_c, phi_auto)
    _INFLUENCE_CACHE.clear()
    g_fft = Gaussian(cubes2_phi_max=phi_auto, b=b_val, rcut=r_c, nlayers=1,
                     use_cubes2_fft=True, remove_self_interaction=True)
    t0 = time.perf_counter()
    r_fft = r.clone().requires_grad_(True)
    e_fft = g_fft(q, r_fft, cell)
    f_fft = -torch.autograd.grad(e_fft, r_fft,
        grad_outputs=torch.ones_like(e_fft))[0]
    t_fft = (time.perf_counter() - t0) * 1000

    cos = torch.cosine_similarity(f_dir.flatten(), f_fft.flatten(), dim=0)
    e_rel = abs(e_fft.item() - e_dir.item()) / max(abs(e_dir.item()), 1e-30) * 100
    rmse = (f_fft - f_dir).norm() / max(f_dir.norm(), 1e-30)

    print(f"  FFT φ=auto: grid={nx}³  E={e_fft.item():.4f}  "
          f"cos={cos.item():.6f}  e_rel={e_rel:.2f}%  rmse={rmse.item():.4f}  "
          f"{t_fft:.1f}ms  speedup={t_dir/t_fft:.1f}x")
    print()


def test_speed_vs_n():
    """Speed scaling: direct vs FFT for increasing N."""
    print("=" * 70)
    print("  Speed Scaling: Direct vs CubeS₂+FFT (30³ box, b=2, r_c=10)")
    print("=" * 70)
    print(f"  {'N':>6s}  {'Dir(ms)':>10s}  {'FFT(ms)':>10s}  {'Speedup':>8s}  "
          f"{'cos':>10s}")

    n_warmup, n_timing = 2, 3
    L, r_c, b_val = 30.0, 10.0, 2.0

    for n_pts in [100, 500, 2000, 5000]:
        torch.manual_seed(42)
        r = torch.rand(n_pts, 3) * L
        q = torch.randn(n_pts // 2) * 0.5
        q = torch.cat([q, -q])
        if n_pts % 2 == 1:
            q = torch.cat([q, torch.zeros(1)])
        perm = torch.randperm(n_pts)
        r = r[perm]; q = q[perm]
        cell = torch.eye(3).unsqueeze(0) * L

        # Direct
        g_dir = Gaussian(n_dl=2.0, b=b_val, rcut=r_c, nlayers=1,
                         use_cubes2_fft=False, remove_self_interaction=True)
        for _ in range(n_warmup):
            g_dir(q, r, cell)
        t0 = time.perf_counter()
        for _ in range(n_timing):
            g_dir(q, r, cell)
        t_d = (time.perf_counter() - t0) / n_timing * 1000

        # FFT φ=0.15 (good force accuracy)
        g_fft = Gaussian(cubes2_phi_max=0.15, b=b_val, rcut=r_c, nlayers=1,
                         use_cubes2_fft=True, remove_self_interaction=True)
        for _ in range(n_warmup):
            _INFLUENCE_CACHE.clear()
            r_grad = r.clone().requires_grad_(True)
            e = g_fft(q, r_grad, cell)
            f = -torch.autograd.grad(e, r_grad,
                grad_outputs=torch.ones_like(e))[0]
        t0 = time.perf_counter()
        for _ in range(n_timing):
            _INFLUENCE_CACHE.clear()
            r_grad = r.clone().requires_grad_(True)
            e = g_fft(q, r_grad, cell)
            f = -torch.autograd.grad(e, r_grad,
                grad_outputs=torch.ones_like(e))[0]
        t_f = (time.perf_counter() - t0) / n_timing * 1000

        # Accuracy
        g_ref = Gaussian(n_dl=2.0, b=b_val, rcut=r_c, nlayers=1,
                         use_cubes2_fft=False, remove_self_interaction=True)
        r_ref = r.clone().requires_grad_(True)
        e_ref = g_ref(q, r_ref, cell)
        f_ref = -torch.autograd.grad(e_ref, r_ref,
            grad_outputs=torch.ones_like(e_ref))[0]

        _INFLUENCE_CACHE.clear()
        r_fft = r.clone().requires_grad_(True)
        e_f = g_fft(q, r_fft, cell)
        f_f = -torch.autograd.grad(e_f, r_fft,
            grad_outputs=torch.ones_like(e_f))[0]
        cos = torch.cosine_similarity(f_ref.flatten(), f_f.flatten(), dim=0)

        sp = t_d / t_f if t_f > 0 else float('inf')
        print(f"  {n_pts:6d}  {t_d:10.1f}  {t_f:10.1f}  {sp:7.1f}x  "
              f"{cos.item():10.6f}")

    print()


if __name__ == "__main__":
    test_water_64()
    test_nacl_512()
    test_speed_vs_n()
    print("Done.")
