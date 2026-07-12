#!/usr/bin/env python3
"""Held-out verification of the committed FORCE-rel phi_max law (phi_max_rule.FALLBACK_LAW / sog.cpp).

Generates FRESH random systems and kernels that were NOT used to fit the law
(calibrate_phi_max_anchors.py used seeds 11/1/7 and kernels cons/neutral/geom1.5/geom2.18). For each
(system, kernel, order) we PREDICT phi via the committed closed form at eps=1e-4, let the CubeS2 FFT
resolve the actual grid at that phi, and MEASURE the true FFT-vs-direct FORCE relative error on that
realized grid. The formula is correct iff the measured force-rel lands near the 1e-4 target.

Checks:
  (v1) per point: realized-grid force-rel <= 5e-4  (the law must never badly UNDER-resolve);
  (v2) aggregate: MEDIAN force-rel at predicted phi(1e-4) is within ~3x of 1e-4 (robust to the
       integer-grid rounding that makes some small-box points over-resolved);
  (v3) monotonicity: phi(1e-3) > phi(1e-4) and rel(phi(1e-3)) > rel(phi(1e-4));
  (v4) validity: predicted phi stays below the ceiling sigma_min/(sqrt2 xi0)/r_c.
CPU-only. Run: python sog/test/test_phi_max_verify.py   (exits non-zero on failure)."""
import sys, math
sys.path.insert(0, "/root/code/sog/src"); sys.path.insert(0, "/root/code/sog")
import numpy as np
import torch
torch.set_default_dtype(torch.float64)

from sog.module.gaussian import Gaussian
from verify_fft_vs_direct import (make_test_system, geometric_amp_bw,
                                  E2_PER_ANGSTROM_TO_EV as NORM, B, RCUT, SIGMA, M)
from verify_cons_phi import AMP_CONS, BW_CONS
from verify_phi_max import realized_delta
from phi_max_rule import phi_max, sigma_min_of, validity_phi

FAILS = []
def check(cond, msg):
    print(("  ok  " if cond else " FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


def _build(amp, bw, phi, order, fft):
    return Gaussian(amp=torch.tensor(np.asarray(amp)), bandwidth=torch.tensor(np.asarray(bw)),
                    kernel_param_mode="internal", kernel_tensor_mode="external",
                    remove_self_interaction=True, use_nufft=False, use_cubes2_fft=fft,
                    norm_factor=NORM, trainable=False, b=B, rcut=RCUT,
                    cubes2_phi_max=phi, cubes2_order=order, n_dl=(None if fft else 0.5))


def _force(g, coords, q, box):
    r = torch.tensor(coords, dtype=torch.float64, requires_grad=True)
    qt = torch.tensor(q, dtype=torch.float64).reshape(-1, 1)
    cell = torch.tensor(box, dtype=torch.float64).unsqueeze(0)
    batch = torch.zeros(len(q), dtype=torch.int64)
    E = g.compute_bundle(q=qt, r=r, cell=cell, batch=batch)["energy"].sum()
    return (-torch.autograd.grad(E, r)[0]).detach().cpu().numpy()


def force_rel(amp, bw, coords, q, box, phi, order, F_ref):
    F = _force(_build(amp, bw, phi, order, True), coords, q, box)
    return float(np.linalg.norm(F - F_ref) / (np.linalg.norm(F_ref) + 1e-30))


# HELD-OUT systems (seeds/sizes disjoint from the calibration panel) and kernels.
SYSTEMS = [("H64s21", 64, 21), ("H72s33", 72, 33), ("H80s42", 80, 42)]
KERNELS = [("cons", np.asarray(AMP_CONS), np.asarray(BW_CONS))]
for tag, sig in [("geom1.8", 1.8), ("geom2.5", 2.5)]:
    a, b = geometric_amp_bw(B, sig, M)
    KERNELS.append((tag, np.asarray(a), np.asarray(b)))

EPS = 1e-4
print(f"Held-out verification of committed FORCE-rel phi_max law at eps={EPS:.0e}\n")
all_rel = []
for sid, nmol, seed in SYSTEMS:
    coords, q, box, L = make_test_system(n_mol=nmol, seed=seed)
    box = np.asarray(box); lx = float(box[0, 0])
    for kid, amp, bw in KERNELS:
        smin = sigma_min_of(bw)
        F_ref = _force(_build(amp, bw, 0.1, 4, False), coords, q, box)   # converged direct k-sum force
        for order in (4, 6):
            phi = phi_max(smin, RCUT, order, EPS)
            _, nx = realized_delta(phi, lx, order)
            rel = force_rel(amp, bw, coords, q, box, phi, order, F_ref)
            all_rel.append(rel)
            fac = rel / EPS
            print(f"  {sid:8s} {kid:8s} o{order}: phi(1e-4)={phi:.4f} -> grid nx={nx:3d}  "
                  f"force-rel={rel:.2e}  ({fac:4.1f}x target)")
            check(rel <= 5e-4, f"(v1) {sid}/{kid}/o{order}: realized force-rel {rel:.2e} <= 5e-4")
            check(phi <= validity_phi(smin, RCUT, order) + 1e-12,
                  f"(v4) {sid}/{kid}/o{order}: phi={phi:.4f} <= validity")

# (v2) aggregate median within ~3x of target
med = float(np.median(all_rel))
print(f"\n  median realized force-rel = {med:.2e}  (target {EPS:.0e}, factor {med/EPS:.1f}x)")
check(EPS / 3 <= med <= EPS * 3, f"(v2) median force-rel {med:.2e} within 3x of {EPS:.0e}")

# (v3) monotonicity on the cons kernel / first held-out system
print("\n== (v3) monotonicity (cons, H64s21) ==")
coords, q, box, L = make_test_system(n_mol=64, seed=21); box = np.asarray(box)
F_ref = _force(_build(AMP_CONS, BW_CONS, 0.1, 4, False), coords, q, box)
smin = sigma_min_of(BW_CONS)
for order in (4, 6):
    p3 = phi_max(smin, RCUT, order, 1e-3); p4 = phi_max(smin, RCUT, order, 1e-4)
    r3 = force_rel(AMP_CONS, BW_CONS, coords, q, box, p3, order, F_ref)
    r4 = force_rel(AMP_CONS, BW_CONS, coords, q, box, p4, order, F_ref)
    check(p3 > p4 and r3 > r4,
          f"(v3) o{order}: phi(1e-3)={p3:.4f}>phi(1e-4)={p4:.4f} and rel {r3:.2e}>{r4:.2e}")

print(f"\n{'='*54}\n{'ALL PASS' if not FAILS else str(len(FAILS))+' FAILURES'}")
sys.exit(1 if FAILS else 0)
