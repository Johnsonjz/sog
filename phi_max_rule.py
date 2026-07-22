#!/usr/bin/env python3
"""General phi_max determination for SOG particle-mesh electrostatics.

The Midtown-splines paper has NO closed-form for phi_max = Delta/r_c: Table III is an empirical
lookup by (spline family, order 2nu, u-series b), calibrated as "grid error <= the u-series' own
intrinsic residual" -- undefined for a trained/arbitrary SOG. This module replaces that with a
principled, self-calibrating rule grounded in the measured grid-error law

        rel(phi) ~= C_nu * (Delta/sigma_min)^p_nu ,   Delta = phi * r_c,   sigma_min = sqrt(min beta),

where p_nu is the (system-robust) convergence order (~4 for order-4; ~7-8 for order-6, which
super-converges o(Delta^6) since nu=3 is odd) and C_nu is a kernel/system-dependent prefactor.
Because C_nu varies ~100x across kernels/systems (fit_phi_max_law.py), we do NOT hardcode a single
constant: we MEASURE the actual kernel's (C_nu, p_nu) via a few FFT-vs-direct evaluations on a
representative config, then invert the law for a target relative accuracy eps:

        phi_max = (eps / C_nu)^(1/p_nu) * sigma_min / r_c ,   clamped below the validity ceiling
        phi_valid = sigma_min / (sqrt(2) * xi0) / r_c        (on-grid variance sigma_k^2 > 0).

The closed-form default (phi_max()/FALLBACK_LAW) is now calibrated on FORCE-rel by direct bisection
(calibrate_phi_max_anchors.py): at eps=1e-4 the cons order-6 kernel gives phi~0.068 (NOT 0.10 -- the
old 0.10 anchor was a back-fit; phi=0.10 is really force-rel ~2e-3). calibrate()/phi_max_for_kernel()
below still self-calibrate a per-kernel ENERGY-rel law (a looser, cancellation-prone metric; prefer the
force-rel FALLBACK_LAW for grid sizing). The paper's 0.23/0.065 (order-4) and 0.35/0.16 (order-6) are
the two u-series intrinsic-error tiers (b=2 C1 ~1/r_c^2 ; b~1.63 C2 ~1/r_c^3).
CPU-only. Depends on the sog Python package (verify_phi_max helpers)."""
import sys, math
sys.path.insert(0, "/root/code/sog/src"); sys.path.insert(0, "/root/code/sog")
import numpy as np

# xi0 (spreading width constant) per CubeS2 order -- from sog_spline.h (kCubes2Xi4/6)
XI0 = {4: 0.5773502691896258, 6: 0.6503998764035732}

# Grid-error law  rel(phi) = C_nu * (Delta/sigma_min)^{2nu} , where 2nu is the spline order.
# The EXPONENT IS ENFORCED FROM THEORY: Proposition 5 establishes the leading error is O(Delta^{2nu}),
# so p_nu = 2nu (4 for order-4, 6 for order-6).  Only the prefactor C_nu is calibrated — by refitting
# the bisection anchors (calibrate_phi_max_anchors.py, phi_max_anchors.json) with p FIXED at 2nu,
# taking the MEDIAN C over a panel of random SOG kernels x random charge configurations.
#
# **CubeS₂ (Form B) vs QuadS (Form A) — separate C_nu (§9.5 / Table 9.3).** The law captures the
# *combined* charge-spreading + Green-function error. For CubeS₂ (Form B), the variance-subtraction
# Green function contributes a leading O(Δ^{2ν}) term, so C_nu ≳ 10⁻². For QuadS (Form A), the exact
# window-influence division K/|Ŵ|² eliminates this term entirely — only bare charge-spreading
# interpolation error remains, giving C_nu ~2×10⁻⁴ (100–500× smaller). The constants below are
# calibrated from the python FFT ε-ladder sweep on random water-like systems (n_dl=0.5 reference):
#   - CubeS₂  (Form B): pooled-median bisection (calibrate_phi_max_anchors.py)
#   - QuadS   (Form A): upper-bound from measured F_rel ≤ 1×10⁻⁴ at all grids (Δ/σ_min ≤ 0.95);
#     true C likely smaller but our n_dl=0.5 reference cannot resolve below ~9×10⁻⁵.
#
# Force-rel is canonical: it is what MD is driven by, it matches the C++ auto-derive (sog.cpp), and it
# is the metric where the single-parameter (Delta/sigma_min) law collapses (CV ~5% across kernels;
# energy-rel scatters 15-30% and floors out).  The old back-fit (p ~3.96/6.53, C tuned to reproduce
# phi=0.10 at eps=1e-4) was optimistic ~30x in force-rel; the true force-rel at phi=0.10 (order-6 cons)
# is ~2e-3.  With honest p=2nu, the median-refit C gives phi(eps=1e-4) = 0.032 (order-4) / 0.066
# (order-6) on the cons kernel (sigma_min=0.750 A).  Refit date: 2026-07-13; split date: 2026-07-22.
CUBES2_LAW = {4: {"p": 4, "C": 4.953e-2}, 6: {"p": 6, "C": 1.377e-2}}
QUADS_LAW  = {4: {"p": 4, "C": 2.0e-4},  6: {"p": 6, "C": 2.0e-4}}
# Legacy alias — kept for backward compatibility; new code should use the spline-specific dicts.
FALLBACK_LAW = CUBES2_LAW


def sigma_min_of(bandwidth):
    return math.sqrt(float(np.min(np.asarray(bandwidth))))


def validity_phi(sigma_min, rcut, order):
    """phi at the validity ceiling Delta_max = sigma_min/(sqrt2 xi0) (paper Eq. 23)."""
    return sigma_min / (math.sqrt(2.0) * XI0[order]) / rcut


def phi_max_from_law(sigma_min, rcut, order, eps, law):
    """Invert rel=C*(Delta/sigma_min)^p for the target eps, clamp below validity."""
    ds = (eps / law["C"]) ** (1.0 / law["p"])          # Delta/sigma_min at target accuracy
    phi = ds * sigma_min / rcut
    return min(phi, 0.95 * validity_phi(sigma_min, rcut, order))


def calibrate(amp, bandwidth, coords, q, box, rcut, order, fix_p=None):
    """Measure (C, p) of the grid-error law for THIS kernel on THIS config via FFT-vs-direct.
    If fix_p is given, only C is fit (more stable when few clean points, esp. order-6)."""
    from verify_phi_max import energy_fft, reference_direct, realized_delta
    amp = np.asarray(amp); bandwidth = np.asarray(bandwidth)
    e_ref = reference_direct(amp, bandwidth, coords, q, box)
    smin = sigma_min_of(bandwidth); lx = float(box[0, 0])
    xs, ys, seen = [], [], set()
    phi_lo = 0.5 * validity_phi(smin, rcut, order)     # scan below validity
    for phi in np.linspace(0.35 * phi_lo, phi_lo, 20):
        d, nx = realized_delta(float(phi), lx, order)
        if nx in seen:
            continue
        seen.add(nx)
        rel = abs(energy_fft(amp, bandwidth, coords, q, box, float(phi), order=order) - e_ref) / abs(e_ref)
        if 1e-8 < rel < 2e-1:
            xs.append(d / smin); ys.append(rel)
    if len(xs) < 3:
        raise RuntimeError(f"calibrate: too few clean points ({len(xs)}) for order {order}")
    lx_, ly_ = np.log(xs), np.log(ys)
    if fix_p is not None:
        p = float(fix_p); C = math.exp(float(np.mean(ly_ - p * lx_)))
    else:
        p, logC = np.polyfit(lx_, ly_, 1); C = math.exp(logC)
    return {"p": float(p), "C": float(C), "npts": len(xs), "sigma_min": smin}


def phi_max(sigma_min, rcut=5.0, order=4, eps=1e-4, law=None):
    """Closed-form phi_max from the (fallback or supplied) FORCE-rel error law. Simple default.

    Default eps=1e-4 is a genuine target FORCE relative accuracy: for the order-6 CubeS2
    conservative kernel (sigma_min=0.750, r_c=5) the honest FALLBACK_LAW gives phi_max~=0.068
    (grid ~75x150x150 on the 6144 box). NOTE: the validated production run used the coarser
    phi=0.10 (grid 50x100x100), which is force-rel ~2e-3 -- adequate for MD but NOT 1e-4; the
    old '1e-4 -> 0.10' mapping was a back-fit (see FALLBACK_LAW note / calibrate_phi_max_anchors.py)."""
    return phi_max_from_law(sigma_min, rcut, order, eps, law or FALLBACK_LAW[order])


def phi_max_for_kernel(amp, bandwidth, coords, q, box, rcut=5.0, order=4, eps=1e-4, fix_p=None):
    """Principled path: self-calibrate the actual kernel/config, then invert for eps."""
    law = calibrate(amp, bandwidth, coords, q, box, rcut, order, fix_p=fix_p)
    smin = sigma_min_of(bandwidth)
    return phi_max_from_law(smin, rcut, order, eps, law), law


if __name__ == "__main__":
    # Self-check: the self-calibrated rule reproduces the cons production phi choices.
    from verify_cons_phi import AMP_CONS, BW_CONS
    from fit_phi_max_law import load_water192
    coords, q, box, L = load_water192()
    smin = sigma_min_of(BW_CONS)
    print(f"cons kernel sigma_min={smin:.3f} A, r_c=5.0, real 192-water box")
    # calibrate each order once, then show the eps -> phi mapping (energy rel)
    for order, fixp, prod_phi in [(4, 4, 0.0675), (6, 6, 0.10)]:
        law = calibrate(AMP_CONS, BW_CONS, coords, q, box, 5.0, order, fix_p=fixp)
        print(f"\norder-{order}: measured law rel = {law['C']:.3e}*(D/sig)^{law['p']:.0f} "
              f"(n={law['npts']}), validity phi={validity_phi(smin,5,order):.3f}")
        for eps in (4e-3, 1e-3, 1e-4):
            phi = phi_max_from_law(smin, 5.0, order, eps, law)
            print(f"    eps={eps:.0e} -> phi_max={phi:.4f}  (D/sig={phi*5/smin:.3f})")
        # which eps reproduces the production phi?
        ds = prod_phi * 5.0 / smin
        eps_prod = law["C"] * ds ** law["p"]
        print(f"    => production phi={prod_phi} corresponds to energy-rel eps={eps_prod:.1e}")

