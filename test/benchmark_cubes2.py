#!/usr/bin/env python3
"""Benchmark: CubeS₂+FFT vs Direct path — accuracy & speed for MLIP training."""

import math
import time
import torch
from sog.module.gaussian import Gaussian
from sog.module.cubes2_fft import compute_cubes2_fft, _INFLUENCE_CACHE


def make_system(n_molecules, device="cpu", dtype=torch.float64):
    """Create water-like system."""
    n_atoms = n_molecules * 3
    density = 0.033  # atoms/Å³
    volume = n_atoms / density
    L = volume ** (1.0 / 3.0)
    cell = torch.eye(3, dtype=dtype, device=device).unsqueeze(0) * L  # [1, 3, 3]
    r = torch.rand(n_atoms, 3, dtype=dtype, device=device) * L
    q = torch.randn(n_atoms, dtype=dtype, device=device) * 0.5
    return r, q, cell


def benchmark_accuracy(n_molecules=64, n_dl=1.0):
    """Compare energy and forces between direct and FFT paths."""
    print(f"\n{'='*70}")
    print(f"  Accuracy Benchmark: {n_molecules*3} atoms, n_dl={n_dl}")
    print(f"{'='*70}")

    r, q, cell = make_system(n_molecules)
    r.requires_grad_(True)

    # Direct path
    g_dir = Gaussian(n_dl=n_dl, use_nufft=False, remove_self_interaction=True)
    e_dir = g_dir(q, r, cell)
    f_dir = -torch.autograd.grad(e_dir, r,
                                  grad_outputs=torch.ones_like(e_dir),
                                  create_graph=True, retain_graph=True)[0]

    # FFT path — use compute_bundle with explicit force
    _INFLUENCE_CACHE.clear()
    g_fft = Gaussian(n_dl=n_dl, use_cubes2_fft=True, use_nufft=False,
                     remove_self_interaction=True)
    bundle = g_fft.compute_bundle(q, r, cell, compute_force=True, compute_virial=False)
    e_fft = bundle["energy"]
    f_fft = bundle["forces"]
    # Note: f_fft may be None if explicit path not available — skip force comparison in that case

    # Compare
    e_rel = abs(e_fft.item() - e_dir.item()) / max(abs(e_dir.item()), 1e-30) * 100
    f_rmse = (f_fft - f_dir).norm() / max(f_dir.norm(), 1e-30)
    f_max = (f_fft - f_dir).abs().max() / max(f_dir.abs().max(), 1e-30)

    print(f"  Energy direct: {e_dir.item():.6f}")
    print(f"  Energy FFT:    {e_fft.item():.6f}")
    print(f"  Energy rel diff: {e_rel:.2f}%")
    print(f"  Force RMSE / |F|: {f_rmse.item():.4f}")
    print(f"  Force max / max|F|: {f_max.item():.4f}")

    return e_rel, f_rmse.item(), f_max.item()


def benchmark_speed():
    """Compare speed for increasing system sizes."""
    print(f"\n{'='*70}")
    print(f"  Speed Benchmark: Direct vs CubeS₂+FFT")
    print(f"{'='*70}")
    print(f"  {'Natoms':>8s}  {'Direct(ms)':>12s}  {'FFT(ms)':>12s}  {'Speedup':>8s}  {'E rel%':>8s}")

    n_warmup = 2
    n_timing = 5

    for n_mol in [64, 256, 512, 1024, 1728]:
        n_atoms = n_mol * 3
        r, q, cell = make_system(n_mol)

        n_dl = 2.0 if n_atoms <= 768 else 3.0  # coarser for larger systems

        g_dir = Gaussian(n_dl=n_dl, use_nufft=False, remove_self_interaction=True)
        _INFLUENCE_CACHE.clear()
        g_fft = Gaussian(n_dl=n_dl, use_cubes2_fft=True, use_nufft=False,
                         remove_self_interaction=True)

        # Warmup
        for _ in range(n_warmup):
            g_dir(q, r, cell)
            g_fft(q, r, cell)

        # Time direct
        t0 = time.perf_counter()
        for _ in range(n_timing):
            g_dir(q, r, cell)
        t_dir = (time.perf_counter() - t0) / n_timing * 1000

        # Time FFT
        t0 = time.perf_counter()
        for _ in range(n_timing):
            _INFLUENCE_CACHE.clear()
            g_fft(q, r, cell)
        t_fft = (time.perf_counter() - t0) / n_timing * 1000

        # Accuracy
        e_dir = g_dir(q, r, cell)
        e_fft = g_fft(q, r, cell)
        e_rel = abs(e_fft.item() - e_dir.item()) / max(abs(e_dir.item()), 1e-30) * 100

        speedup = t_dir / t_fft if t_fft > 0 else float('inf')
        print(f"  {n_atoms:8d}  {t_dir:12.1f}  {t_fft:12.1f}  {speedup:7.2f}x  {e_rel:7.2f}%")


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(42)

    benchmark_accuracy(64, n_dl=1.0)
    benchmark_accuracy(64, n_dl=0.5)
    benchmark_speed()
    print(f"\n{'='*70}")
    print("  Done.")
    print(f"{'='*70}")
