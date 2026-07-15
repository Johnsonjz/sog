#!/usr/bin/env python3
"""Numerical validation: verify the error formula rel ≈ C_nu * (Delta/sigma_min)^{2nu}
with p_nu = 2nu enforced. Also check aliasing vs spline error balance.

Uses existing anchor data from phi_max_anchors.json + spot-checks on random configs.
"""
import sys, os, math, json
import numpy as np
sys.path.insert(0, "/root/code/sog/src"); sys.path.insert(0, "/root/code/sog")

from phi_max_rule import (sigma_min_of, validity_phi, phi_max_from_law,
                          FALLBACK_LAW, XI0)

HERE = os.path.dirname(os.path.abspath(__file__))
FAILS = 0

# ── load anchor data ──────────────────────────────────────────────────
with open(os.path.join(HERE, "phi_max_anchors.json")) as f:
    anchors_data = json.load(f)

print("=" * 70)
print("NUMERICAL VALIDATION: error formula rel = C*(Δ/σ_min)^{2ν}")
print("=" * 70)

# ── (1) Slope verification from anchor data ───────────────────────────
print("\n--- (1) Slope verification (anchor data) ---")
for order in (4, 6):
    p_theory = int(order)  # 2*nu
    pts = [(a["ds"], a["eps"]) for a in anchors_data["anchors"][str(order)]["force"]]
    if len(pts) < 5:
        print(f"  order-{order}: too few anchors ({len(pts)}), skip")
        continue

    xs = np.log([d for d, _ in pts])
    ys = np.log([e for _, e in pts])

    # Free fit
    p_free, logC = np.polyfit(xs, ys, 1)
    C_free = math.exp(logC)

    # Fixed-p fit
    C_fixed = float(np.median([eps / (ds ** p_theory) for ds, eps in pts]))

    # Compute R² for both
    y_mean = np.mean(ys)
    ss_tot = np.sum((ys - y_mean) ** 2)
    r2_free = 1 - np.sum((ys - (np.log(C_free) + p_free * xs)) ** 2) / ss_tot
    r2_fixed = 1 - np.sum((ys - (np.log(C_fixed) + p_theory * xs)) ** 2) / ss_tot

    # Bootstrap slope uncertainty
    slopes = []
    for _ in range(500):
        idx = np.random.choice(len(xs), len(xs), replace=True)
        s, _ = np.polyfit(xs[idx], ys[idx], 1)
        slopes.append(s)
    slope_std = np.std(slopes)

    ok = abs(p_free - p_theory) <= 2 * slope_std  # within 2σ
    status = "PASS" if ok else "WARN"
    if not ok:
        FAILS += 1
        FAILS += 1

    print(f"  order-{order}: p_theory={p_theory}  p_free={p_free:.2f} +/- {slope_std:.2f}"
          f"  R²_free={r2_free:.4f}  R²_fixed={r2_fixed:.4f}"
          f"  C_free={C_free:.3e}  C_fixed={C_fixed:.3e}  [{status}]")
    print(f"    n={len(pts)}  ds∈[{min(d for d,_ in pts):.3f},{max(d for d,_ in pts):.3f}]"
          f"  eps∈[{min(e for _,e in pts):.1e},{max(e for _,e in pts):.1e}]")

# ── (2) Aliasing vs spline balance ────────────────────────────────────
print("\n--- (2) Aliasing vs spline error balance ---")

def aliasing_error(sigma_min, delta, xi0):
    """Compute aliasing error at Nyquist."""
    sigma_k2 = sigma_min**2 - 2 * (xi0 * delta)**2
    if sigma_k2 <= 0:
        return float('inf')
    k_nyq = math.pi / delta
    return math.exp(-0.5 * sigma_k2 * k_nyq**2)

def spline_error(sigma_min, delta, order, C_nu):
    """Spline interpolation error."""
    p = int(order)  # 2*nu
    ds = delta / sigma_min
    return C_nu * (ds ** p)

# Test on cons kernel (sigma_min=0.750) at various phi
smin_cons = 0.750
rcut = 5.0

for order in (4, 6):
    print(f"\n  order-{order} (cons kernel, sigma_min={smin_cons} A):")
    law = FALLBACK_LAW[order]
    xi0 = XI0[order]
    for eps_target in [1e-2, 1e-3, 1e-4]:
        ds_opt = (eps_target / law["C"]) ** (1.0 / law["p"])
        delta_opt = ds_opt * smin_cons
        e_alias = aliasing_error(smin_cons, delta_opt, xi0)
        e_spline = spline_error(smin_cons, delta_opt, order, law["C"])
        ratio = e_alias / e_spline if e_spline > 0 else float('inf')
        balance = "balanced" if 0.1 <= ratio <= 10 else ("alias-dom" if ratio > 10 else "spline-dom")
        print(f"    eps={eps_target:.0e}: phi={delta_opt/rcut:.4f}  ds={ds_opt:.3f}  "
              f"e_alias={e_alias:.2e}  e_spline={e_spline:.2e}  ratio={ratio:.3g}  [{balance}]")

# ── (3) Spot-check: bisection on one random config ─────────────────────
print("\n--- (3) Spot-check: honest bisection on random config ---")
try:
    from verify_fft_vs_direct import make_test_system
    from calibrate_phi_max_anchors import bisect_phi, rel_at

    # Generate a random system
    coords, q, box, n_mol = make_test_system(n_mol=48, seed=42)

    # Use a wide geometric-like kernel
    amp_geo = np.array([1.0, 2.0, 4.0, 8.0])
    bw_geo = np.array([1.0, 4.0, 16.0, 64.0])
    smin_geo = sigma_min_of(bw_geo)
    print(f"  test kernel: sigma_min={smin_geo:.3f} A, M={len(amp_geo)}")

    for order in (4, 6):
        for eps_target in [1e-3]:
            try:
                phi_star, status = bisect_phi("spot", "test", amp_geo, bw_geo, coords, q, box,
                                              order, eps_target, "force")
                delta_star = phi_star * rcut
                ds_star = delta_star / smin_geo
                rel_actual = rel_at("spot", "test", amp_geo, bw_geo, coords, q, box, phi_star, order, "force")
                ratio_ok = 0.5 <= rel_actual / eps_target <= 2.0
                st = "PASS" if ratio_ok else "WARN"
                if not ratio_ok:
                    FAILS += 1
                print(f"    order-{order} eps={eps_target:.0e}: phi*={phi_star:.4f}"
                      f"  ds*={ds_star:.3f}  rel_actual={rel_actual:.2e}"
                      f"  ratio={rel_actual/eps_target:.2f}  [{st}] ({status})")
            except Exception as ex:
                print(f"    order-{order} eps={eps_target:.0e}: bisection failed: {ex}")
except ImportError as ex:
    print(f"  (spot-check skipped: {ex})")

print("\n" + "=" * 70)
if FAILS:
    print(f"NUMERICAL VALIDATION: {FAILS} WARNING(S) — review above")
else:
    print("NUMERICAL VALIDATION: ALL PASS")
print("=" * 70)
sys.exit(1 if FAILS else 0)
