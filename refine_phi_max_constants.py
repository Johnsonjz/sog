#!/usr/bin/env python3
"""Refine the sog.cpp phi_max formula constants (C_nu, p_nu) using bisection.

The formula stays closed-form: phi_max = (eps/C_nu)^(1/p_nu) * sigma_min/r_c. But instead of fitting
(C_nu, p_nu) to the noisy log-log rel(phi) curve (which gave off-nominal slopes p=3.48/6.57), we:
  1. seed initial phi guesses from the CURRENT formula,
  2. BISECT the measured rel(phi)=eps on the true FFT-vs-direct curve (force-rel metric) to pin the
     authoritative phi_max at each target eps (no fit scatter),
  3. refit (C_nu, p_nu) to those authoritative (Delta/sigma_min, eps) points.
CPU-only. Metric = force rel by default (MD accuracy is force-driven).

NOTE (2026-07-10): this single-kernel/single-box refiner is SUPERSEDED by
calibrate_phi_max_anchors.py, which bisects a PANEL of random systems x kernels (the collapse of
Delta/sigma_min across kernels is what justifies the closed form). The live sog.cpp / FALLBACK_LAW
constants come from that panel: order-6 {C=1.681e-2, p=6.533}, order-4 {C=4.465e-2, p=3.956}."""
import sys, math
sys.path.insert(0, "/root/code/sog/src"); sys.path.insert(0, "/root/code/sog")
import numpy as np, torch
torch.set_default_dtype(torch.float64)
from sog.module.gaussian import Gaussian
from verify_fft_vs_direct import E2_PER_ANGSTROM_TO_EV as NORM, B, RCUT
from verify_cons_phi import AMP_CONS, BW_CONS
from fit_phi_max_law import load_water192
from phi_max_rule import sigma_min_of, validity_phi, phi_max_from_law, XI0

COORDS, Q, BOX, _ = load_water192()
CELL = torch.tensor(BOX, dtype=torch.float64).unsqueeze(0)
QT = torch.tensor(Q, dtype=torch.float64).reshape(-1, 1)
BATCH = torch.zeros(len(Q), dtype=torch.int64)


def _gaussian(amp, bw, phi, order, fft):
    return Gaussian(amp=torch.tensor(np.asarray(amp)), bandwidth=torch.tensor(np.asarray(bw)),
                    kernel_param_mode="internal", kernel_tensor_mode="external",
                    remove_self_interaction=True, use_nufft=False, use_cubes2_fft=fft,
                    norm_factor=NORM, trainable=False, b=B, rcut=RCUT,
                    cubes2_phi_max=phi, cubes2_order=order, n_dl=(None if fft else 0.5))


def _ef(g, want_force):
    r = torch.tensor(COORDS, dtype=torch.float64, requires_grad=want_force)
    E = g.compute_bundle(q=QT, r=r, cell=CELL, batch=BATCH)["energy"].sum()
    if not want_force:
        return float(E), None
    F = -torch.autograd.grad(E, r)[0]
    return float(E), F.detach().numpy()


_DIRECT = {}   # cache the direct k-sum reference (phi/order-INDEPENDENT: no grid)


def _direct_ref(amp, bw, want_force):
    key = ("F" if want_force else "E")
    if key not in _DIRECT:
        _DIRECT[key] = _ef(_gaussian(amp, bw, 0.1, 4, False), want_force)
    return _DIRECT[key]


def rel_at(amp, bw, phi, order, metric):
    if metric == "energy":
        e_ref, _ = _direct_ref(amp, bw, False)
        e, _ = _ef(_gaussian(amp, bw, phi, order, True), False)
        return abs(e - e_ref) / abs(e_ref)
    _, F_ref = _direct_ref(amp, bw, True)
    _, F = _ef(_gaussian(amp, bw, phi, order, True), True)
    return float(np.linalg.norm(F - F_ref) / (np.linalg.norm(F_ref) + 1e-30))


def bisect_phi_max(amp, bw, rcut, order, eps, metric="force", iters=28):
    """Largest phi with rel(phi) <= eps (rel increases with phi). Seed bracket from validity."""
    smin = sigma_min_of(bw)
    lo, hi = 0.02, 0.95 * validity_phi(smin, rcut, order)
    # guard: if even hi is below eps, return hi (accuracy not the binding constraint)
    if rel_at(amp, bw, hi, order, metric) <= eps:
        return hi
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if rel_at(amp, bw, mid, order, metric) <= eps:
            lo = mid
        else:
            hi = mid
    return lo


def refit(amp, bw, rcut, order, eps_list, metric):
    smin = sigma_min_of(bw)
    xs, ys = [], []
    for eps in eps_list:
        phi = bisect_phi_max(amp, bw, rcut, order, eps, metric)
        ds = phi * rcut / smin
        xs.append(ds); ys.append(eps)
        print(f"    eps={eps:.0e} -> bisect phi={phi:.4f}  (D/sig={ds:.3f})")
    p, logC = np.polyfit(np.log(xs), np.log(ys), 1)   # eps = C * (D/sig)^p
    return math.exp(logC), p


if __name__ == "__main__":
    metric = "force"
    print(f"Refining phi_max constants on cons kernel / 192-water, metric={metric}")
    eps_list = {4: [3e-2, 1e-2, 3e-3, 1e-3], 6: [1e-2, 3e-3, 1e-3, 3e-4]}
    for order in (4, 6):
        print(f"\n== order-{order} ==")
        C, p = refit(AMP_CONS, BW_CONS, RCUT, order, eps_list[order], metric)
        cur_C = 1.681e-2 if order == 6 else 4.465e-2   # live sog.cpp (honest force-rel, panel-refit)
        cur_p = 6.533 if order == 6 else 3.956
        print(f"  REFINED (force-rel): C={C:.4e}  p={p:.3f}   [current sog.cpp: C={cur_C:.3e} p={cur_p}]")
        # sanity: production phi from refined law at a target force eps
        for eps in (1e-3, 2e-3):
            phi = phi_max_from_law(sigma_min_of(BW_CONS), RCUT, order, eps, {"C": C, "p": p})
            print(f"    refined formula: eps_force={eps:.0e} -> phi={phi:.4f}")
