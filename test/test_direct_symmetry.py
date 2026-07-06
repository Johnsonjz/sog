#!/usr/bin/env python3
"""Test Fourier space symmetry optimization in _compute_periodic_direct.

Compares the old (full k-sphere) and new (half-sphere × 2) implementations
for correctness and measures the speedup.
"""

import math
import time
import warnings
import torch

warnings.filterwarnings("ignore", category=DeprecationWarning)
torch.set_default_dtype(torch.float64)

from sog.module.gaussian import Gaussian


def compute_direct_old(kvec, kfac_flat, r_raw, q, volume, mask):
    """Old implementation: compute over full k-sphere (no symmetry)."""
    kvec = kvec[mask]
    kfac_flat = kfac_flat[mask]

    k_dot_r = torch.matmul(r_raw, kvec.transpose(0, 1))
    cos_kr = torch.cos(k_dot_r)
    sin_kr = torch.sin(k_dot_r)

    s_real = (q.unsqueeze(2) * cos_kr.unsqueeze(1)).sum(dim=0)
    s_imag = (q.unsqueeze(2) * sin_kr.unsqueeze(1)).sum(dim=0)
    s_sq = s_real.square() + s_imag.square()

    return (kfac_flat.unsqueeze(0) * s_sq).sum() / (2.0 * volume)


def compute_direct_new(kvec, kfac_flat, r_raw, q, volume, mask):
    """New implementation: half-sphere symmetry, multiply by 2."""
    kvec = kvec[mask]
    kfac_flat = kfac_flat[mask]

    # Half-sphere filter
    half_mask = (
        (kvec[:, 0] > 0)
        | ((kvec[:, 0] == 0) & (kvec[:, 1] > 0))
        | ((kvec[:, 0] == 0) & (kvec[:, 1] == 0) & (kvec[:, 2] > 0))
    )
    kvec = kvec[half_mask]
    kfac_flat = kfac_flat[half_mask]

    k_dot_r = torch.matmul(r_raw, kvec.transpose(0, 1))
    cos_kr = torch.cos(k_dot_r)
    sin_kr = torch.sin(k_dot_r)

    s_real = (q.unsqueeze(2) * cos_kr.unsqueeze(1)).sum(dim=0)
    s_imag = (q.unsqueeze(2) * sin_kr.unsqueeze(1)).sum(dim=0)
    s_sq = s_real.square() + s_imag.square()

    return 2.0 * (kfac_flat.unsqueeze(0) * s_sq).sum() / (2.0 * volume)


def _get_state(g: Gaussian, r_raw, q, cell_now):
    """Extract k-space state from a Gaussian instance.

    _prepare_triclinic_state expects cell_now as [3,3] (unbatched).
    """
    if cell_now.dim() == 3:
        cell_now = cell_now[0]
    state = g._prepare_triclinic_state(r_raw, q, cell_now, compute_spectral=True)
    return state


def test_water_192(benchmark_calls: int = 500):
    """Water 64 H₂O: 192 atoms, 12.44 Å cubic box."""
    print("=" * 70)
    print("  Water 64H₂O (192 atoms, 12.44 Å box, single-channel q)")
    print("=" * 70)

    torch.manual_seed(42)
    L = 12.44
    r = torch.rand(192, 3) * L
    q = torch.randn(192, 1) * 0.5
    cell = torch.eye(3).unsqueeze(0) * L

    results = {}
    for n_dl in [1.5, 2.0, 3.0, 5.0]:
        g = Gaussian(n_dl=n_dl, b=2.0, sigma=3.017, m=12, nlayers=1,
                     use_cubes2_fft=False, remove_self_interaction=True)

        state = _get_state(g, r, q, cell)
        kvec = state["g_cart"].reshape(3, -1).transpose(0, 1)
        kfac_flat = state["kfac"].reshape(-1)
        mask = state["k_mode_mask"].reshape(-1)
        volume = state["volume"]

        n_k = mask.sum().item()

        # --- Correctness ---
        e_old = compute_direct_old(kvec.clone(), kfac_flat.clone(),
                                   r, q, volume, mask)
        e_new = compute_direct_new(kvec.clone(), kfac_flat.clone(),
                                   r, q, volume, mask)
        diff = abs(e_old.item() - e_new.item())
        rel = diff / max(abs(e_old.item()), 1e-30)

        # --- Also verify against actual method ---
        e_method = g._compute_periodic_direct(
            r, q, state["g_cart"], state["kfac"], volume, state["k_mode_mask"])
        diff_method = abs(e_method.item() - e_old.item())
        rel_method = diff_method / max(abs(e_old.item()), 1e-30)

        # --- Speed ---
        kvec_clone = kvec.clone()
        kfac_clone = kfac_flat.clone()
        t0 = time.perf_counter()
        for _ in range(benchmark_calls):
            _ = compute_direct_old(kvec_clone, kfac_clone, r, q, volume, mask)
        t_old = (time.perf_counter() - t0) / benchmark_calls * 1000

        t0 = time.perf_counter()
        for _ in range(benchmark_calls):
            _ = compute_direct_new(kvec_clone, kfac_clone, r, q, volume, mask)
        t_new = (time.perf_counter() - t0) / benchmark_calls * 1000

        speedup = t_old / t_new if t_new > 0 else float("inf")

        print(f"  n_dl={n_dl:>5.1f}  n_k={n_k:>5d}  "
              f"E_old={e_old.item():.6f}  |dE|={diff:.2e}  rel={rel:.2e}")
        print(f"           "
              f"E_method={e_method.item():.6f}  |dE_vs_method|={diff_method:.2e}")
        print(f"           "
              f"t_old={t_old:.3f}ms  t_new={t_new:.3f}ms  speedup={speedup:.2f}x")
        print()

        results[n_dl] = {
            "n_k": n_k,
            "diff": diff,
            "rel": rel,
            "t_old_ms": t_old,
            "t_new_ms": t_new,
            "speedup": speedup,
        }

    return results


def test_au_mgo_110(benchmark_calls: int = 500):
    """Au-MgO 110 atoms, orthorhombic cell ~9×9×26 Å, multi-channel q."""
    print("=" * 70)
    print("  Au-MgO (110 atoms, orthorhombic, 3-channel latent charge)")
    print("=" * 70)

    torch.manual_seed(123)
    # Approximate cell from Au-MgO data
    Lx, Ly, Lz = 8.9, 8.9, 26.4
    r = torch.rand(110, 3) * torch.tensor([Lx, Ly, Lz])
    q = torch.randn(110, 3) * 0.3  # 3-channel latent charge
    cell = torch.diag(torch.tensor([Lx, Ly, Lz])).unsqueeze(0)

    results = {}
    for n_dl in [1.5, 2.0, 3.0, 5.0]:
        g = Gaussian(n_dl=n_dl, b=2.0, sigma=3.017, m=12, nlayers=1,
                     use_cubes2_fft=False, remove_self_interaction=True)

        state = _get_state(g, r, q, cell)
        kvec = state["g_cart"].reshape(3, -1).transpose(0, 1)
        kfac_flat = state["kfac"].reshape(-1)
        mask = state["k_mode_mask"].reshape(-1)
        volume = state["volume"]

        n_k = mask.sum().item()

        # --- Correctness ---
        e_old = compute_direct_old(kvec.clone(), kfac_flat.clone(),
                                   r, q, volume, mask)
        e_new = compute_direct_new(kvec.clone(), kfac_flat.clone(),
                                   r, q, volume, mask)
        diff = abs(e_old.item() - e_new.item())
        rel = diff / max(abs(e_old.item()), 1e-30)

        # --- Also verify against actual method ---
        e_method = g._compute_periodic_direct(
            r, q, state["g_cart"], state["kfac"], volume, state["k_mode_mask"])
        diff_method = abs(e_method.item() - e_old.item())
        rel_method = diff_method / max(abs(e_old.item()), 1e-30)

        # --- Speed ---
        kvec_clone = kvec.clone()
        kfac_clone = kfac_flat.clone()
        t0 = time.perf_counter()
        for _ in range(benchmark_calls):
            _ = compute_direct_old(kvec_clone, kfac_clone, r, q, volume, mask)
        t_old = (time.perf_counter() - t0) / benchmark_calls * 1000

        t0 = time.perf_counter()
        for _ in range(benchmark_calls):
            _ = compute_direct_new(kvec_clone, kfac_clone, r, q, volume, mask)
        t_new = (time.perf_counter() - t0) / benchmark_calls * 1000

        speedup = t_old / t_new if t_new > 0 else float("inf")

        print(f"  n_dl={n_dl:>5.1f}  n_k={n_k:>5d}  "
              f"E_old={e_old.item():.6f}  |dE|={diff:.2e}  rel={rel:.2e}")
        print(f"           "
              f"E_method={e_method.item():.6f}  |dE_vs_method|={diff_method:.2e}")
        print(f"           "
              f"t_old={t_old:.3f}ms  t_new={t_new:.3f}ms  speedup={speedup:.2f}x")
        print()

        results[n_dl] = {
            "n_k": n_k,
            "diff": diff,
            "rel": rel,
            "t_old_ms": t_old,
            "t_new_ms": t_new,
            "speedup": speedup,
        }

    return results


def test_edge_cases():
    """Edge cases: very few k-points, no k-points."""
    print("=" * 70)
    print("  Edge Cases")
    print("=" * 70)

    torch.manual_seed(999)
    L = 10.0
    r = torch.rand(10, 3) * L
    q = torch.randn(10, 1)
    cell = torch.eye(3).unsqueeze(0) * L

    # Large n_dl → very few k-points (sphere of radius 2π/n_dl is small)
    for n_dl in [5.0, 8.0]:
        g = Gaussian(n_dl=n_dl, b=2.0, sigma=3.017, m=12, nlayers=1,
                     use_cubes2_fft=False, remove_self_interaction=True)

        state = _get_state(g, r, q, cell)
        kvec = state["g_cart"].reshape(3, -1).transpose(0, 1)
        kfac_flat = state["kfac"].reshape(-1)
        mask = state["k_mode_mask"].reshape(-1)
        volume = state["volume"]

        n_k = mask.sum().item()
        e_old = compute_direct_old(kvec.clone(), kfac_flat.clone(),
                                   r, q, volume, mask)
        e_new = compute_direct_new(kvec.clone(), kfac_flat.clone(),
                                   r, q, volume, mask)
        diff = abs(e_old.item() - e_new.item()) if n_k > 0 else 0.0

        status = "OK" if diff < 1e-12 else "FAIL"
        print(f"  n_dl={n_dl}  n_k={n_k:>3d}  E={e_old.item():.6e}  "
              f"|dE|={diff:.2e}  [{status}]")

    print()


def main():
    print("Fourier Space Symmetry Optimization Test")
    print("=" * 70)
    print()

    all_pass = True

    # --- Run tests ---
    results_water = test_water_192(benchmark_calls=500)
    results_au_mgo = test_au_mgo_110(benchmark_calls=500)
    test_edge_cases()

    # --- Summary ---
    print("=" * 70)
    print("  Summary")
    print("=" * 70)

    for name, results in [("Water 192", results_water), ("Au-MgO 110", results_au_mgo)]:
        print(f"\n  {name}:")
        for n_dl, r in results.items():
            ok = "OK" if r["diff"] < 1e-12 else "FAIL"
            print(f"    n_dl={n_dl:<5.1f}  n_k={r['n_k']:>5d}  "
                  f"|dE|={r['diff']:.2e}  "
                  f"speedup={r['speedup']:.2f}x  [{ok}]")
            if r["diff"] >= 1e-12:
                all_pass = False

    print()
    if all_pass:
        print("  ALL TESTS PASSED ✓")
    else:
        print("  SOME TESTS FAILED ✗")

    return all_pass


if __name__ == "__main__":
    main()
