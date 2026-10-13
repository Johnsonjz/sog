"""Held-out verification of the committed FORCE-rel phi_max law
(sog.tools.phi_max_rule.FALLBACK_LAW / sog.cpp).

Generates FRESH random systems and kernels that were NOT used to fit the law
(calibrate_phi_max_anchors.py used seeds 11/1/7 and kernels
cons/neutral/geom1.5/geom2.18).  For each (system, kernel, order) we PREDICT
phi via the committed closed form at eps=1e-4, let the CubeS2 FFT resolve the
actual grid at that phi, and MEASURE the true FFT-vs-direct FORCE relative
error on that realized grid.  The formula is correct iff the measured
force-rel lands near the 1e-4 target.

Checks:
  (v1) per point: realized-grid force-rel <= 5e-4 (never badly UNDER-resolve);
  (v2) aggregate: MEDIAN force-rel at predicted phi(1e-4) is within ~3x of
       1e-4 (robust to integer-grid rounding over-resolving small boxes);
  (v3) monotonicity: phi(1e-3) > phi(1e-4) and rel(phi(1e-3)) > rel(phi(1e-4));
  (v4) validity: predicted phi stays below the ceiling
       sigma_min/(sqrt2 xi0)/r_c.

CPU-only, minutes-scale (marked slow)::

    pytest test/test_phi_max_verify.py
"""

import numpy as np
import pytest
import torch

from sog.module.gaussian import Gaussian
from sog.tools.verify_fft_vs_direct import (
    make_test_system,
    geometric_amp_bw,
    E2_PER_ANGSTROM_TO_EV as NORM,
    B,
    RCUT,
    SIGMA,
    M,
)
from sog.tools.verify_cons_phi import AMP_CONS, BW_CONS
from sog.tools.verify_phi_max import realized_delta
from sog.tools.phi_max_rule import phi_max, sigma_min_of, validity_phi


torch.set_default_dtype(torch.float64)

EPS = 1e-4

# HELD-OUT systems (seeds/sizes disjoint from the calibration panel).
SYSTEMS = [("H64s21", 64, 21), ("H72s33", 72, 33), ("H80s42", 80, 42)]


def _build(amp, bw, phi, order, fft):
    return Gaussian(
        amp=torch.tensor(np.asarray(amp)),
        bandwidth=torch.tensor(np.asarray(bw)),
        kernel_param_mode="internal",
        kernel_tensor_mode="external",
        remove_self_interaction=True,
        use_nufft=False,
        use_cubes2_fft=fft,
        norm_factor=NORM,
        trainable=False,
        b=B,
        rcut=RCUT,
        cubes2_phi_max=phi,
        cubes2_order=order,
        n_dl=(None if fft else 0.5),
    )


def _force(g, coords, q, box):
    r = torch.tensor(coords, dtype=torch.float64, requires_grad=True)
    qt = torch.tensor(q, dtype=torch.float64).reshape(-1, 1)
    cell = torch.tensor(box, dtype=torch.float64).unsqueeze(0)
    batch = torch.zeros(len(q), dtype=torch.int64)
    E = g.compute_bundle(q=qt, r=r, cell=cell, batch=batch)["energy"].sum()
    return (-torch.autograd.grad(E, r)[0]).detach().cpu().numpy()


def _force_rel(amp, bw, coords, q, box, phi, order, F_ref):
    F = _force(_build(amp, bw, phi, order, True), coords, q, box)
    return float(np.linalg.norm(F - F_ref) / (np.linalg.norm(F_ref) + 1e-30))


@pytest.fixture(scope="module")
def held_out_points():
    """One dict per (system, kernel, order): predicted phi, realized grid, rel."""
    kernels = [("cons", np.asarray(AMP_CONS), np.asarray(BW_CONS))]
    for tag, sig in [("geom1.8", 1.8), ("geom2.5", 2.5)]:
        a, b = geometric_amp_bw(B, sig, M)
        kernels.append((tag, np.asarray(a), np.asarray(b)))

    points = []
    for sid, nmol, seed in SYSTEMS:
        coords, q, box, _L = make_test_system(n_mol=nmol, seed=seed)
        box = np.asarray(box)
        lx = float(box[0, 0])
        for kid, amp, bw in kernels:
            smin = sigma_min_of(bw)
            # converged direct k-sum force reference
            F_ref = _force(_build(amp, bw, 0.1, 4, False), coords, q, box)
            for order in (4, 6):
                phi = phi_max(smin, RCUT, order, EPS)
                _, nx = realized_delta(phi, lx, order)
                rel = _force_rel(amp, bw, coords, q, box, phi, order, F_ref)
                points.append(
                    dict(sid=sid, kid=kid, order=order, smin=smin,
                         phi=phi, nx=nx, rel=rel)
                )
    return points


def test_v1_realized_force_rel_budget(held_out_points):
    """(v1) per point: the law must never badly UNDER-resolve."""
    bad = [p for p in held_out_points if p["rel"] > 5e-4]
    assert not bad, "(v1) realized force-rel must be <= 5e-4:\n" + "\n".join(
        f"  {p['sid']}/{p['kid']}/o{p['order']}: grid nx={p['nx']:3d} "
        f"force-rel={p['rel']:.2e}" for p in bad
    )


def test_v2_median_within_3x_target(held_out_points):
    """(v2) median realized force-rel within ~3x of the 1e-4 target."""
    med = float(np.median([p["rel"] for p in held_out_points]))
    assert EPS / 3 <= med <= EPS * 3, (
        f"(v2) median force-rel {med:.2e} (factor {med / EPS:.1f}x) "
        f"not within 3x of {EPS:.0e}"
    )


def test_v3_monotonicity_in_eps():
    """(v3) phi(1e-3) > phi(1e-4) and rel(phi(1e-3)) > rel(phi(1e-4))."""
    coords, q, box, _L = make_test_system(n_mol=64, seed=21)
    box = np.asarray(box)
    F_ref = _force(_build(AMP_CONS, BW_CONS, 0.1, 4, False), coords, q, box)
    smin = sigma_min_of(BW_CONS)
    for order in (4, 6):
        p3 = phi_max(smin, RCUT, order, 1e-3)
        p4 = phi_max(smin, RCUT, order, 1e-4)
        r3 = _force_rel(AMP_CONS, BW_CONS, coords, q, box, p3, order, F_ref)
        r4 = _force_rel(AMP_CONS, BW_CONS, coords, q, box, p4, order, F_ref)
        assert p3 > p4 and r3 > r4, (
            f"(v3) o{order}: phi(1e-3)={p3:.4f} > phi(1e-4)={p4:.4f} and "
            f"rel {r3:.2e} > {r4:.2e} must both hold"
        )


def test_v4_validity_ceiling(held_out_points):
    """(v4) predicted phi stays below sigma_min/(sqrt2 xi0)/r_c."""
    for p in held_out_points:
        vphi = validity_phi(p["smin"], RCUT, p["order"])
        assert p["phi"] <= vphi + 1e-12, (
            f"(v4) {p['sid']}/{p['kid']}/o{p['order']}: phi={p['phi']:.4f} "
            f"> validity {vphi:.4f}"
        )
