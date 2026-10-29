"""Stability / regression tests for the general phi_max rule (sog.tools.phi_max_rule).

Asserts the self-calibrating accuracy-inverted rule works robustly across the
GENERAL case: kernels (cons / neutral / geometric), spline orders (4, 6), and
target accuracies eps. Checks:
  (a) fitted convergence exponent p_nu is in the physical range (order-4 ~4;
      order-6 >=5.5, super-conv);
  (b) round-trip: phi_max(eps) plugged back into FFT-vs-direct yields rel
      within a bounded factor of eps;
  (c) monotonicity: eps down -> phi down; order up -> phi up (accuracy-limited);
  (d) validity: phi_max stays below the ceiling sigma_min/(sqrt2 xi0)/r_c;
  (e) reference law reproduction: cons order-4 eps=1e-4 -> phi~0.0267;
      order-6 eps=1e-4 -> phi~0.0602.

CPU-only, minutes-scale (marked slow)::

    pytest test/test_phi_max_rule.py
"""

import numpy as np
import pytest

from sog.tools.phi_max_rule import (
    calibrate,
    phi_max,
    phi_max_from_law,
    sigma_min_of,
    validity_phi,
)
from sog.tools.verify_phi_max import energy_fft, reference_direct, AMP_MD, BW_MD
from sog.tools.verify_fft_vs_direct import geometric_amp_bw, B, SIGMA, M, RCUT
from sog.tools.verify_cons_phi import AMP_CONS, BW_CONS
from sog.tools.fit_phi_max_law import load_water192



@pytest.fixture(scope="module")
def panel():
    coords, q, box, _ = load_water192()
    kernels = {
        "cons": (np.asarray(AMP_CONS), np.asarray(BW_CONS)),
        "neutral": (np.asarray(AMP_MD), np.asarray(BW_MD)),
    }
    ag, bg = geometric_amp_bw(B, SIGMA, M)
    kernels["geom"] = (np.asarray(ag), np.asarray(bg))
    return coords, q, box, kernels


@pytest.fixture(scope="module")
def laws(panel):
    """One free-fit (p, C) law per (kernel, order) — the expensive calibration."""
    coords, q, box, kernels = panel
    out = {}
    for kn, (amp, bw) in kernels.items():
        for order in (4, 6):
            out[(kn, order)] = calibrate(amp, bw, coords, q, box, RCUT, order)
    return out


def _actual_rel(amp, bw, coords, q, box, order, phi):
    e_ref = reference_direct(amp, bw, coords, q, box)
    e = energy_fft(amp, bw, coords, q, box, float(phi), order=order)
    return abs(e - e_ref) / abs(e_ref)


def test_exponent_physical_range(laws, panel):
    """(a) fitted convergence exponent p_nu is in the physical range."""
    coords, q, box, kernels = panel
    for kn, (amp, bw) in kernels.items():
        for order in (4, 6):
            law = laws[(kn, order)]
            pmin = 3.4 if order == 4 else 5.0
            assert pmin <= law["p"] <= order + 6, (
                f"(a) {kn} order-{order}: fitted p={law['p']:.2f} "
                f"outside [{pmin},{order + 6}]"
            )


def test_validity_ceiling_and_roundtrip(laws, panel):
    """(d) phi <= validity ceiling; (b) round-trip rel within 10x of eps."""
    coords, q, box, kernels = panel
    for kn, (amp, bw) in kernels.items():
        smin = sigma_min_of(bw)
        for order in (4, 6):
            law = laws[(kn, order)]
            vphi = validity_phi(smin, RCUT, order)
            for eps in (1e-2, 1e-3, 1e-4):
                phi = phi_max_from_law(smin, RCUT, order, eps, law)
                assert phi <= vphi + 1e-12, (
                    f"(d) {kn} order-{order} eps={eps:.0e}: phi={phi:.4f} "
                    f"> validity {vphi:.3f}"
                )
                # round-trip only where the power law holds: comfortably below
                # the validity ceiling.
                if phi < 0.75 * vphi:
                    rel = _actual_rel(amp, bw, coords, q, box, order, phi)
                    assert eps / 10 <= rel <= eps * 10, (
                        f"(b) {kn} order-{order} eps={eps:.0e}: "
                        f"round-trip rel={rel:.2e} outside 10x of eps"
                    )


def test_monotonicity_in_eps(laws, panel):
    """(c) eps down -> phi down."""
    coords, q, box, kernels = panel
    for kn, (amp, bw) in kernels.items():
        smin = sigma_min_of(bw)
        for order in (4, 6):
            law = laws[(kn, order)]
            phis = [
                phi_max_from_law(smin, RCUT, order, e, law)
                for e in (1e-2, 1e-3, 1e-4)
            ]
            assert phis[0] >= phis[1] >= phis[2], (
                f"(c) {kn} order-{order}: eps down must lower phi, got "
                f"{[round(p, 4) for p in phis]}"
            )


def test_order_monotonicity_when_accuracy_limited(laws, panel):
    """(c) order up -> phi up, but ONLY in the accuracy-limited regime.

    For wide sigma_min (e.g. geom) both orders clamp to the validity ceiling,
    where order-4's ceiling is higher — there is nothing to check.
    """
    coords, q, box, kernels = panel
    for kn, (amp, bw) in kernels.items():
        smin = sigma_min_of(bw)
        p4 = phi_max_from_law(smin, RCUT, 4, 1e-3, laws[(kn, 4)])
        p6 = phi_max_from_law(smin, RCUT, 6, 1e-3, laws[(kn, 6)])
        acc_limited = (
            p4 < 0.9 * validity_phi(smin, RCUT, 4)
            and p6 < 0.9 * validity_phi(smin, RCUT, 6)
        )
        if acc_limited:
            assert p6 > p4, (
                f"(c) {kn} eps=1e-3 (accuracy-limited): order-6 phi={p6:.3f} "
                f"not > order-4 phi={p4:.3f}"
            )


def test_committed_fallback_law_gives_expected_production_phi():
    """(e) the committed FORCE-rel FALLBACK_LAW reproduces the bisected anchors.

    At eps=1e-4 the cons kernel gives phi=0.0267 (order-4) / 0.0602 (order-6)
    with the paper's reference coefficients (C4=1.0e-1, C6=2.4e-2), adopted in
    deepmd-kit-LR @ a266642d2 so this rule matches sog.cpp and the in-graph op.
    """
    smin_c = sigma_min_of(BW_CONS)
    phi4 = phi_max(smin_c, RCUT, 4, 1.0e-4)
    phi6 = phi_max(smin_c, RCUT, 6, 1.0e-4)
    assert abs(phi4 - 0.0267) / 0.0267 < 0.05, (
        f"(e) cons order-4 eps=1e-4 (force) -> phi={phi4:.4f}, expected ~0.0267"
    )
    assert abs(phi6 - 0.0602) / 0.0602 < 0.05, (
        f"(e) cons order-6 eps=1e-4 (force) -> phi={phi6:.4f}, expected ~0.0602"
    )
