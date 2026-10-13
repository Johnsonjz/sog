"""Half-sphere symmetry in ``Gaussian._compute_periodic_direct``.

Asserts the new (half-sphere ×2) k-space implementation reproduces the old
(full k-sphere) energy exactly — ``|dE| < 1e-12``, the original script's own
criterion — across ``n_dl`` on two systems (water 192, Au-MgO 110 with
3-channel charges), and that ``Gaussian._compute_periodic_direct`` itself
agrees with the same reference.  Timing is printed for diagnostics only
(machine-dependent, not asserted; tune with ``SOG_BENCH_CALLS``).
"""

import os
import time
import warnings

import torch

warnings.filterwarnings("ignore", category=DeprecationWarning)
torch.set_default_dtype(torch.float64)

from sog.module.gaussian import Gaussian  # noqa: E402

BENCH_CALLS = int(os.environ.get("SOG_BENCH_CALLS", "20"))
TOL_E = 1e-12          # the original script's own pass criterion (absolute)


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


def _check_system(r, q, cell, label):
    """One identical comparison across n_dl; returns the per-n_dl results."""
    results = {}
    for n_dl in [1.5, 2.0, 3.0, 5.0]:
        g = Gaussian(n_dl=n_dl, b=2.0, sigma=3.017, m=12, nlayers=1,
                     use_cubes2_fft=False, remove_self_interaction=True)

        state = _get_state(g, r, q, cell)
        kvec = state["g_cart"].reshape(3, -1).transpose(0, 1)
        kfac_flat = state["kfac"].reshape(-1)
        mask = state["k_mode_mask"].reshape(-1)
        volume = state["volume"]

        n_k = int(mask.sum().item())

        # --- Correctness: half-sphere (×2) must equal the full-sphere sum ---
        e_old = compute_direct_old(kvec.clone(), kfac_flat.clone(),
                                   r, q, volume, mask)
        e_new = compute_direct_new(kvec.clone(), kfac_flat.clone(),
                                   r, q, volume, mask)
        diff = abs(e_old.item() - e_new.item())
        assert diff < TOL_E, (
            f"{label} n_dl={n_dl}: half-sphere energy differs from the "
            f"full-sphere reference by {diff:.3e} (n_k={n_k}, "
            f"E_old={e_old.item():.10f}, E_new={e_new.item():.10f})"
        )

        # --- The production method must use the same value ---
        e_method = g._compute_periodic_direct(
            r, q, state["g_cart"], state["kfac"], volume, state["k_mode_mask"])
        diff_method = abs(e_method.item() - e_old.item())
        assert torch.isfinite(e_method), f"{label} n_dl={n_dl}: non-finite method energy"
        assert diff_method < TOL_E, (
            f"{label} n_dl={n_dl}: _compute_periodic_direct differs from the "
            f"full-sphere reference by {diff_method:.3e}"
        )

        # --- Timing (informational only) ---
        kvec_clone = kvec.clone()
        kfac_clone = kfac_flat.clone()
        t0 = time.perf_counter()
        for _ in range(BENCH_CALLS):
            _ = compute_direct_old(kvec_clone, kfac_clone, r, q, volume, mask)
        t_old = (time.perf_counter() - t0) / BENCH_CALLS * 1000

        t0 = time.perf_counter()
        for _ in range(BENCH_CALLS):
            _ = compute_direct_new(kvec_clone, kfac_clone, r, q, volume, mask)
        t_new = (time.perf_counter() - t0) / BENCH_CALLS * 1000
        speedup = t_old / t_new if t_new > 0 else float("inf")

        print(f"  {label} n_dl={n_dl:>5.1f}  n_k={n_k:>7d}  "
              f"|dE|={diff:.2e}  |dE_method|={diff_method:.2e}  "
              f"t_old={t_old:.3f}ms t_new={t_new:.3f}ms speedup={speedup:.2f}x")

        results[n_dl] = {"n_k": n_k, "diff": diff, "diff_method": diff_method,
                         "t_old_ms": t_old, "t_new_ms": t_new, "speedup": speedup}
    return results


def test_water_192():
    """Water 64 H₂O: 192 atoms, 12.44 Å cubic box, single-channel q."""
    torch.manual_seed(42)
    L = 12.44
    r = torch.rand(192, 3) * L
    q = torch.randn(192, 1) * 0.5
    cell = torch.eye(3).unsqueeze(0) * L
    _check_system(r, q, cell, "water192")


def test_au_mgo_110():
    """Au-MgO 110 atoms, orthorhombic cell ~9×9×26 Å, multi-channel q."""
    torch.manual_seed(123)
    Lx, Ly, Lz = 8.9, 8.9, 26.4
    r = torch.rand(110, 3) * torch.tensor([Lx, Ly, Lz])
    q = torch.randn(110, 3) * 0.3  # 3-channel latent charge
    cell = torch.diag(torch.tensor([Lx, Ly, Lz])).unsqueeze(0)
    _check_system(r, q, cell, "au-mgo110")


def test_edge_cases():
    """Edge cases: very few k-points (large n_dl), incl. the empty-k limit."""
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

        n_k = int(mask.sum().item())
        e_old = compute_direct_old(kvec.clone(), kfac_flat.clone(),
                                   r, q, volume, mask)
        e_new = compute_direct_new(kvec.clone(), kfac_flat.clone(),
                                   r, q, volume, mask)
        diff = abs(e_old.item() - e_new.item()) if n_k > 0 else 0.0
        assert torch.isfinite(e_old) and torch.isfinite(e_new), (
            f"n_dl={n_dl}: non-finite energy (n_k={n_k})"
        )
        assert diff < TOL_E, (
            f"n_dl={n_dl}: |dE|={diff:.3e} (n_k={n_k}, E={e_old.item():.6e})"
        )
