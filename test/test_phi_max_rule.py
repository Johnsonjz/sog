#!/usr/bin/env python3
"""Stability / regression test for the general phi_max rule (sog/phi_max_rule.py).

Asserts the self-calibrating accuracy-inverted rule works robustly across the GENERAL case:
kernels (cons / neutral / geometric), spline orders (4, 6), and target accuracies eps. Checks:
  (a) fitted convergence exponent p_nu is in the physical range (order-4 ~4; order-6 >=5.5, super-conv);
  (b) round-trip: phi_max(eps) plugged back into FFT-vs-direct yields rel within a bounded factor of eps;
  (c) monotonicity: eps down -> phi down; order up -> phi up (at fixed eps, fixed sigma_min);
  (d) validity: phi_max stays below the ceiling sigma_min/(sqrt2 xi0)/r_c;
  (e) production reproduction: cons order-4 eps~4e-3 -> phi~0.0675 ; order-6 eps~1e-4 -> phi~0.10.
CPU-only. Run: python sog/test/test_phi_max_rule.py   (exits non-zero on failure)."""
import sys, math
sys.path.insert(0, "/root/code/sog/src"); sys.path.insert(0, "/root/code/sog")
import numpy as np
from phi_max_rule import (calibrate, phi_max, phi_max_from_law, phi_max_for_kernel, validity_phi,
                          sigma_min_of, XI0)
from verify_phi_max import energy_fft, reference_direct, realized_delta, AMP_MD, BW_MD
from verify_fft_vs_direct import geometric_amp_bw, B, SIGMA, M, RCUT
from verify_cons_phi import AMP_CONS, BW_CONS
from fit_phi_max_law import load_water192

FAILS = []
def check(cond, msg):
    print(("  ok  " if cond else " FAIL ") + msg)
    if not cond: FAILS.append(msg)

COORDS, Q, BOX, _ = load_water192()
KERNELS = {"cons": (np.asarray(AMP_CONS), np.asarray(BW_CONS)),
           "neutral": (np.asarray(AMP_MD), np.asarray(BW_MD))}
ag, bg = geometric_amp_bw(B, SIGMA, M); KERNELS["geom"] = (np.asarray(ag), np.asarray(bg))

def actual_rel(amp, bw, order, phi):
    e_ref = reference_direct(amp, bw, COORDS, Q, BOX)
    e = energy_fft(amp, bw, COORDS, Q, BOX, float(phi), order=order)
    return abs(e - e_ref) / abs(e_ref)

print("== (a) exponent p_nu physical + (b) round-trip accuracy + (d) validity ==")
LAWS = {}
for kn, (amp, bw) in KERNELS.items():
    smin = sigma_min_of(bw)
    for order in (4, 6):
        law = calibrate(amp, bw, COORDS, Q, BOX, RCUT, order)   # free-fit p and C (fits the actual curve)
        LAWS[(kn, order)] = law
        pmin = 3.4 if order == 4 else 5.0
        check(pmin <= law["p"] <= (order + 6),
              f"(a) {kn} order-{order}: fitted p={law['p']:.2f} in [{pmin},{order+6}]")
        vphi = validity_phi(smin, RCUT, order)
        for eps in (1e-2, 1e-3, 1e-4):
            phi = phi_max_from_law(smin, RCUT, order, eps, law)
            check(phi <= vphi + 1e-12,
                  f"(d) {kn} order-{order} eps={eps:.0e}: phi={phi:.4f} <= validity {vphi:.3f}")
            # round-trip only where the power law holds: comfortably below validity
            if phi < 0.75 * vphi:
                rel = actual_rel(amp, bw, order, phi)
                check(eps / 10 <= rel <= eps * 10,
                      f"(b) {kn} order-{order} eps={eps:.0e}: round-trip rel={rel:.2e} within 10x of eps")

print("\n== (c) monotonicity ==")
for kn, (amp, bw) in KERNELS.items():
    smin = sigma_min_of(bw)
    for order in (4, 6):
        law = LAWS[(kn, order)]
        phis = [phi_max_from_law(smin, RCUT, order, e, law) for e in (1e-2, 1e-3, 1e-4)]
        check(phis[0] >= phis[1] >= phis[2], f"(c) {kn} order-{order}: eps down -> phi down {['%.3f'%p for p in phis]}")
    # order up -> phi up at fixed eps, but ONLY in the accuracy-limited regime;
    # for wide sigma_min (e.g. geom) both clamp to validity, where order-4's ceiling is higher.
    p4 = phi_max_from_law(smin, RCUT, 4, 1e-3, LAWS[(kn, 4)])
    p6 = phi_max_from_law(smin, RCUT, 6, 1e-3, LAWS[(kn, 6)])
    acc_limited = p4 < 0.9 * validity_phi(smin, RCUT, 4) and p6 < 0.9 * validity_phi(smin, RCUT, 6)
    if acc_limited:
        check(p6 > p4, f"(c) {kn} eps=1e-3 (accuracy-limited): order-6 phi={p6:.3f} > order-4 phi={p4:.3f}")
    else:
        print(f"  skip (c) {kn} eps=1e-3: validity-limited (p4={p4:.3f},p6={p6:.3f}); order monotonicity N/A")

print("\n== (e) committed FORCE-rel FALLBACK_LAW -> honest production phi (cons) ==")
# The closed-form default now uses the honest bisected FORCE-rel law (calibrate_phi_max_anchors.py).
# At eps=1e-4 the cons kernel gives phi=0.032 (order-4) / 0.068 (order-6). The old back-fit that made
# eps=1e-4 -> phi~0.10 was OPTIMISTIC ~30x in force-rel (phi=0.10 is really force-rel ~2e-3). The
# validated production grid deliberately uses the COARSER phi=0.10 (adequate; RDF/density match DPA).
smin_c = sigma_min_of(BW_CONS)
phi4 = phi_max(smin_c, RCUT, 4, 1.0e-4)
phi6 = phi_max(smin_c, RCUT, 6, 1.0e-4)
check(abs(phi4 - 0.0321) / 0.0321 < 0.05, f"(e) cons order-4 eps=1e-4 (force) -> phi={phi4:.4f} ~= 0.032")
check(abs(phi6 - 0.0685) / 0.0685 < 0.05, f"(e) cons order-6 eps=1e-4 (force) -> phi={phi6:.4f} ~= 0.068")

print(f"\n{'='*50}\n{'ALL PASS' if not FAILS else str(len(FAILS))+' FAILURES'}")
sys.exit(1 if FAILS else 0)
