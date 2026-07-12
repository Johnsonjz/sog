#!/usr/bin/env python3
"""Verify the phi_max general method reproduces the paper's Table III b=2/1.63 anchors,
and calibrate the (C_nu, p_nu) constants (CubeS2) for the sog.cpp closed-form. CPU-only."""
import sys, math
sys.path.insert(0, "/root/code/sog/src"); sys.path.insert(0, "/root/code/sog")
import numpy as np, torch
torch.set_default_dtype(torch.float64)
from verify_phi_max import energy_fft, reference_direct, sigma_min
from verify_fft_vs_direct import geometric_amp_bw, RCUT, M
from verify_cons_phi import AMP_CONS, BW_CONS
from fit_phi_max_law import load_water192
from phi_max_rule import calibrate, phi_max_from_law, validity_phi

COORDS, Q, BOX, _ = load_water192()

def rel(amp, bw, order, phi):
    e_ref = reference_direct(amp, bw, COORDS, Q, BOX)
    return abs(energy_fft(amp, bw, COORDS, Q, BOX, float(phi), order=order) - e_ref) / abs(e_ref)

# ── Part A: paper u-series kernels (C1/C2 continuity fixes r_c/sigma) ──
USERIES = {"b=2":  (2.0,                 RCUT / 1.989),   # sigma = r_c/1.989 = 2.514, C1
           "b=1.63": (1.6297670882677646, RCUT / 2.752)}  # sigma = r_c/2.752 = 1.817, C2
ANCHOR = {("b=2", 4): 0.23, ("b=1.63", 4): 0.065, ("b=2", 6): 0.35, ("b=1.63", 6): 0.16}
print("== Part A: reproduce Table III anchors on real 192-water (u-series kernels) ==")
print(f"  {'kernel':8} {'order':>5} {'sigma_min':>9} {'anchor phi':>10} {'rel@anchor(=eps tier)':>21} {'method phi(eps)':>15}")
tiers = {}
for kn, (b, sig) in USERIES.items():
    amp, bw = geometric_amp_bw(b, sig, M)
    smin = sigma_min(bw)
    for order in (4, 6):
        anc = ANCHOR[(kn, order)]
        eps = rel(amp, bw, order, anc)            # the intrinsic-error tier the anchor targets
        law = calibrate(amp, bw, COORDS, Q, BOX, RCUT, order)
        phi_m = phi_max_from_law(smin, RCUT, order, eps, law)   # method should return ~anchor
        tiers[(kn, order)] = eps
        ok = abs(phi_m - anc) / anc < 0.15
        print(f"  {kn:8} {order:>5} {smin:>9.3f} {anc:>10.3f} {eps:>21.2e} {phi_m:>15.4f}  {'OK' if ok else 'X'}")
print("  two-tier check (b=2 looser than b=1.63):",
      "OK" if tiers[("b=2",4)] > tiers[("b=1.63",4)] and tiers[("b=2",6)] > tiers[("b=1.63",6)] else "X")

# ── Part B: CubeS2 (C_nu, p_nu) constants for sog.cpp, calibrated on cons/water ──
print("\n== Part B: CubeS2 (C_nu, p_nu) for cons kernel on real 192-water (for sog.cpp) ==")
for order in (4, 6):
    law = calibrate(AMP_CONS, BW_CONS, COORDS, Q, BOX, RCUT, order, fix_p=order)  # fix nominal p
    lawf = calibrate(AMP_CONS, BW_CONS, COORDS, Q, BOX, RCUT, order)              # free-fit
    print(f"  order-{order}: fixed-p={order} -> C={law['C']:.4e}   |  free-fit p={lawf['p']:.2f} C={lawf['C']:.4e}")
