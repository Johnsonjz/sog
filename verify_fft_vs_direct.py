#!/usr/bin/env python3
"""Verify CubeS2 FFT == direct k-sum for arbitrary Sum-of-Gaussians kernels.

Validates the generalized-midtown-splines theorem (Task 2):
  E_mesh = (1/2V) Σ_{k≠0} K(k²)/|Φ|² |ρ|²  ==  (1/2V) Σ_{k≠0} K(k²)|S|²  (direct)
for any {amp_m, bw_m}, not just the geometric u-series.

Two kernels tested:
  1. Geometric (u-series):  amp/bw from (b, sigma, M) RBSOG convention
  2. Trained (non-geometric): amp/bw from the trained checkpoint (first 4 modified)
"""
import sys
sys.path.insert(0, "/root/code/sog/src")

import math
import numpy as np
import torch

torch.set_default_dtype(torch.float64)

E2_PER_ANGSTROM_TO_EV = 14.3996454784255

from sog.module.gaussian import Gaussian

# ── SOG parameters ──
B = 1.6297670882677646
SIGMA = 2.180230445405648
M = 12
RCUT = 5.0
XI4 = 2.0 / math.sqrt(15.0)  # CubeS2 order-4 optimal

# Trained amp/bandwidth (internal convention: amp=raw*bw², bw=σ²b^2m), from checkpoint
AMP_TRAINED = np.array([1.05852560e+01, 5.87989331e+01, 1.81409068e+02, 5.32747690e+02,
                        1.45220412e+03, 3.85725858e+03, 1.02454217e+04, 2.72132823e+04,
                        7.22823085e+04, 1.91991986e+05, 5.09957740e+05, 1.35451954e+06])
BW_TRAINED = np.array([7.77989631e+00, 2.89900118e+01, 5.64525176e+01, 1.20836924e+02,
                       2.36598317e+02, 6.28434810e+02, 1.66921132e+03, 4.43366022e+03,
                       1.17764256e+04, 3.12798441e+04, 8.30836690e+04, 2.20681920e+05])


def geometric_amp_bw(b, sigma, M):
    """RBSOG geometric (internal) amp/bw:  bw[m]=σ²b^2m,  amp[m]=4π ln b · bw[m]."""
    coef = 4.0 * math.pi * math.log(b)
    bw = np.array([sigma**2 * b**(2*m) for m in range(M)])
    amp = coef * bw
    return amp, bw


def make_gaussian(amp, bw, use_fft, phi_max=None, n_dl=None):
    """Build a Gaussian with external internal-mode amp/bw."""
    return Gaussian(
        amp=torch.tensor(amp),
        bandwidth=torch.tensor(bw),
        kernel_param_mode="internal",
        kernel_tensor_mode="external",
        remove_self_interaction=True,
        use_nufft=False,
        use_cubes2_fft=use_fft,
        norm_factor=E2_PER_ANGSTROM_TO_EV,
        trainable=False,
        b=B,
        rcut=RCUT if use_fft else None,   # rcut needed for φ-grid sizing
        cubes2_phi_max=phi_max,
        cubes2_order=4,
        n_dl=n_dl,
    )


def make_test_system(n_mol=64, box_l=None, seed=0):
    """Neutral water-like system: n_mol 'molecules', each with charges summing to 0."""
    rng = np.random.default_rng(seed)
    if box_l is None:
        box_l = (n_mol ** (1/3)) * 3.1  # ~water density
    coords = rng.uniform(0, box_l, size=(n_mol * 3, 3))
    # neutral charges: for each molecule O=-0.8, H=+0.4, H=+0.4
    q = np.tile([-0.8, 0.4, 0.4], n_mol)
    # add small random perturbation but keep per-frame neutral
    pert = rng.normal(0, 0.1, size=q.shape)
    pert -= pert.mean()
    q = q + pert
    q -= q.mean()  # enforce exact neutrality
    box = np.diag([box_l, box_l, box_l])
    return coords, q, box, box_l


def energy_direct(amp, bw, coords, q, box, n_dl):
    """Direct k-sum energy via Gaussian (use_cubes2_fft=False)."""
    g = make_gaussian(amp, bw, use_fft=False, n_dl=n_dl)
    r = torch.tensor(coords)
    qt = torch.tensor(q).reshape(-1, 1)
    cell = torch.tensor(box).unsqueeze(0)
    batch = torch.zeros(len(q), dtype=torch.int64)
    e = g.compute_bundle(q=qt, r=r, cell=cell, batch=batch)["energy"]
    return float(e.sum().item())


def energy_fft(amp, bw, coords, q, box, phi_max):
    """CubeS2 FFT energy via Gaussian (use_cubes2_fft=True)."""
    g = make_gaussian(amp, bw, use_fft=True, phi_max=phi_max)
    r = torch.tensor(coords)
    qt = torch.tensor(q).reshape(-1, 1)
    cell = torch.tensor(box).unsqueeze(0)
    batch = torch.zeros(len(q), dtype=torch.int64)
    e = g.compute_bundle(q=qt, r=r, cell=cell, batch=batch)["energy"]
    return float(e.sum().item())


def validity_check(bw, phi_max, box_l):
    """Check min-bandwidth condition bw_min > 2σ_s², σ_s = ξ₀·Δ, Δ = φ·rcut."""
    delta = phi_max * RCUT
    sigma_s = XI4 * delta
    bw_min = float(np.min(bw))
    ok = bw_min > 2.0 * sigma_s**2
    return ok, bw_min, 2.0 * sigma_s**2, delta


def run_case(name, amp, bw, coords, q, box, box_l):
    print(f"\n{'='*66}\n{name}\n{'='*66}")
    print(f"  amp[:4] = {np.array2string(amp[:4], precision=3)}")
    print(f"  bw[:4]  = {np.array2string(bw[:4], precision=3)}")
    print(f"  bw_min = {np.min(bw):.4f}  (σ_min = {math.sqrt(np.min(bw)):.4f} Å)")

    # Direct: converge by shrinking n_dl (larger k_max)
    print("\n  --- Direct k-sum convergence (smaller n_dl → larger k_max) ---")
    e_direct_prev = None
    e_direct_conv = None
    for n_dl in [2.0, 1.0, 0.7, 0.5]:
        e = energy_direct(amp, bw, coords, q, box, n_dl)
        dd = "" if e_direct_prev is None else f"  Δ={e-e_direct_prev:+.3e}"
        print(f"    n_dl={n_dl:<5} k_max={2*math.pi/n_dl:6.3f}  E={e:14.6f} eV{dd}")
        e_direct_prev = e
        e_direct_conv = e

    # FFT: converge by shrinking phi_max (finer grid)
    print("\n  --- CubeS2 FFT convergence (smaller φ → finer grid) ---")
    e_fft_conv = None
    for phi in [0.23, 0.15, 0.10, 0.065]:
        ok, bwmin, bound, delta = validity_check(bw, phi, box_l)
        e = energy_fft(amp, bw, coords, q, box, phi)
        flag = "" if ok else "  [!] bw_min ≤ 2σ_s² (invalid grid)"
        print(f"    φ={phi:<6} Δ={delta:5.3f}  E={e:14.6f} eV{flag}")
        e_fft_conv = e

    rel = abs(e_fft_conv - e_direct_conv) / max(abs(e_direct_conv), 1e-30)
    print(f"\n  Converged direct = {e_direct_conv:14.6f} eV")
    print(f"  Converged FFT    = {e_fft_conv:14.6f} eV")
    print(f"  |ΔE|/|E| = {rel:.3e}  →  {'✓ PASS (<1e-3)' if rel < 1e-3 else '✗ FAIL'}")
    return rel


def main():
    coords, q, box, box_l = make_test_system(n_mol=64, seed=1)
    print(f"Test system: {len(q)} charges, box = {box_l:.2f}³ Å³, "
          f"Σq = {q.sum():.2e} (neutral)")

    amp_geo, bw_geo = geometric_amp_bw(B, SIGMA, M)

    rel1 = run_case("KERNEL 1: Geometric u-series (b=1.63, σ=2.18, M=12)",
                    amp_geo, bw_geo, coords, q, box, box_l)
    rel2 = run_case("KERNEL 2: Trained (non-geometric, first 4 modified)",
                    AMP_TRAINED, BW_TRAINED, coords, q, box, box_l)

    print(f"\n{'='*66}\nSUMMARY\n{'='*66}")
    print(f"  Geometric kernel: |ΔE|/|E| = {rel1:.3e}  {'✓' if rel1<1e-3 else '✗'}")
    print(f"  Trained kernel:   |ΔE|/|E| = {rel2:.3e}  {'✓' if rel2<1e-3 else '✗'}")
    if rel1 < 1e-3 and rel2 < 1e-3:
        print("\n  ✓ THEOREM VERIFIED: CubeS2 FFT == direct k-sum for BOTH geometric")
        print("    and arbitrary (trained) amp/bw. Midtown-splines generalizes.")
    else:
        print("\n  ✗ Mismatch — investigate grid sizing / influence function.")


if __name__ == "__main__":
    main()
