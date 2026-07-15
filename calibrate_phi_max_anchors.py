#!/usr/bin/env python3
"""Honest calibration of the sog phi_max grid-sizing law by DIRECT bisection.

The production law  phi_max = (eps/C_nu)^(1/p_nu) * sigma_min/r_c  had its (C_nu, p_nu)
*back-fit* so that eps=1e-4 reproduces the production phi=0.10 (order-6 cons). That never
verified the reverse claim: that the TRUE FFT-vs-direct relative error at phi=0.10 actually
equals 1e-4. This script does the honest thing:

  1. HEADLINE: measure the real FFT-vs-direct rel at phi=0.10 (order-6 cons kernel), BOTH
     energy- and force-rel, on the real 192-water box AND on random systems; and BISECT the
     phi that actually achieves rel=1e-4.
  2. PANEL: over a set of small RANDOM test systems x kernels of varied amp/bandwidth (varied
     sigma_min), bisect the true rel(phi)=eps curve at an eps-ladder that INCLUDES 1e-4 -> the
     real anchors phi*(eps). No power-law extrapolation.
  3. COLLAPSE: the closed form presumes rel depends only on Delta/sigma_min. Tabulate the
     bisected Delta/sigma_min at each eps across all (system,kernel); small scatter => the law
     is justified and its (C_nu,p_nu) are the pooled log-log fit; large scatter => report that
     the single-parameter law is only approximate.
  4. REFIT + report old-vs-new (C_nu, p_nu) and old-vs-new phi(1e-4); dump phi_max_anchors.json.

Force-rel is canonical (matches the C++ auto-derive at sog.cpp:911-912 and is what MD is driven
by); energy-rel is reported alongside for transparency. CPU-only. Run with the dp env python.
"""
import sys, os, json, math
sys.path.insert(0, "/root/code/sog/src"); sys.path.insert(0, "/root/code/sog")
import numpy as np
import torch
torch.set_default_dtype(torch.float64)

from sog.module.gaussian import Gaussian
from sog.module.cubes2_fft import _resolve_grid
from verify_fft_vs_direct import (make_test_system, geometric_amp_bw,
                                  E2_PER_ANGSTROM_TO_EV as NORM, B, RCUT, SIGMA, M)
from verify_cons_phi import AMP_CONS, BW_CONS, BOXP
from verify_phi_max import AMP_MD, BW_MD, realized_delta
from phi_max_rule import sigma_min_of, validity_phi, phi_max_from_law, XI0, FALLBACK_LAW

HERE = os.path.dirname(os.path.abspath(__file__))

# Current C++ auto-derive constants (force-rel) from sog.cpp:911-912, for the report.
CPP_FORCE = {4: {"C": 1.90e-3, "p": 3.69}, 6: {"C": 2.10e-3, "p": 7.59}}


# ── low-level FFT / direct energy+force for an arbitrary (kernel, config) ────────────────
def _build(amp, bw, phi, order, fft):
    return Gaussian(amp=torch.tensor(np.asarray(amp)), bandwidth=torch.tensor(np.asarray(bw)),
                    kernel_param_mode="internal", kernel_tensor_mode="external",
                    remove_self_interaction=True, use_nufft=False, use_cubes2_fft=fft,
                    norm_factor=NORM, trainable=False, b=B, rcut=RCUT,
                    cubes2_phi_max=phi, cubes2_order=order, n_dl=(None if fft else 0.5))


def _ef(g, coords, q, box, want_force):
    r = torch.tensor(coords, dtype=torch.float64, requires_grad=want_force)
    qt = torch.tensor(q, dtype=torch.float64).reshape(-1, 1)
    cell = torch.tensor(box, dtype=torch.float64).unsqueeze(0)
    batch = torch.zeros(len(q), dtype=torch.int64)
    E = g.compute_bundle(q=qt, r=r, cell=cell, batch=batch)["energy"].sum()
    if not want_force:
        return float(E), None
    F = -torch.autograd.grad(E, r)[0]
    return float(E), F.detach().cpu().numpy()


_DIRECT = {}   # (sysid, kerid, 'E'|'F') -> (E_ref, F_ref) ; direct k-sum is grid-independent


def _direct(sysid, kerid, amp, bw, coords, q, box, want_force):
    key = (sysid, kerid, "F" if want_force else "E")
    if key not in _DIRECT:
        _DIRECT[key] = _ef(_build(amp, bw, 0.1, 4, False), coords, q, box, want_force)
    return _DIRECT[key]


def rel_at(sysid, kerid, amp, bw, coords, q, box, phi, order, metric):
    """True FFT-vs-direct relative error at a continuous phi (energy or force)."""
    wf = (metric == "force")
    e_ref, F_ref = _direct(sysid, kerid, amp, bw, coords, q, box, wf)
    e, F = _ef(_build(amp, bw, phi, order, True), coords, q, box, wf)
    if metric == "energy":
        return abs(e - e_ref) / abs(e_ref)
    return float(np.linalg.norm(F - F_ref) / (np.linalg.norm(F_ref) + 1e-30))


def bisect_phi(sysid, kerid, amp, bw, coords, q, box, order, eps, metric, iters=20):
    """Largest phi with rel(phi) <= eps (rel increases with phi). Returns (phi, status)
    status in {'ok','validity','floor'} — 'ok' anchors are accuracy-limited and fit-worthy."""
    smin = sigma_min_of(bw)
    lo0, hi = 0.02, 0.95 * validity_phi(smin, RCUT, order)
    lo = lo0
    if rel_at(sysid, kerid, amp, bw, coords, q, box, hi, order, metric) <= eps:
        return hi, "validity"          # accuracy isn't the binding constraint at this eps
    if rel_at(sysid, kerid, amp, bw, coords, q, box, lo, order, metric) > eps:
        return lo, "floor"             # even the coarsest floor exceeds eps
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if rel_at(sysid, kerid, amp, bw, coords, q, box, mid, order, metric) <= eps:
            lo = mid
        else:
            hi = mid
    return lo, "ok"


# ── panel of random systems and varied kernels ──────────────────────────────────────────
def make_systems():
    specs = [("R48s11", 48, 11), ("R64s1", 64, 1), ("R64s7", 64, 7)]
    out = []
    for sid, nmol, seed in specs:
        coords, q, box, L = make_test_system(n_mol=nmol, seed=seed)
        out.append((sid, coords, q, np.asarray(box), L))
    return out


def make_kernels():
    """Kernels spanning sigma_min ~ 0.75 .. 2.18 A (cons, MD-neutral, and geometric at 2 sigma)."""
    ker = [("cons", np.asarray(AMP_CONS), np.asarray(BW_CONS)),
           ("neutral", np.asarray(AMP_MD), np.asarray(BW_MD))]
    for tag, sig in [("geom1.5", 1.5), ("geom2.18", SIGMA)]:
        a, b = geometric_amp_bw(B, sig, M)
        ker.append((tag, np.asarray(a), np.asarray(b)))
    return ker


EPS_LADDER = {4: [1e-2, 3e-3, 1e-3, 3e-4, 1e-4],
              6: [1e-2, 3e-3, 1e-3, 3e-4, 1e-4]}


def cons_prod_grid(phi, order):
    nx, ny, nz, _ = _resolve_grid(BOXP[0], BOXP[1], BOXP[2], None, phi, RCUT, b=B, spline_order=order)
    return nx, ny, nz, nx * ny * nz


def refit(ds_eps):
    """(ds, eps) points -> (C, p) with eps = C*(ds)^p via log-log least squares."""
    xs = np.log([d for d, _ in ds_eps]); ys = np.log([e for _, e in ds_eps])
    p, logC = np.polyfit(xs, ys, 1)
    return math.exp(logC), float(p)


def refit_fixed_p(ds_eps, p_theory):
    """(ds, eps) points -> C only, with p = p_theory (2*nu) enforced by theory.
    Refit via: C = median(eps / ds^{p_theory})."""
    C_vals = [eps / (ds ** p_theory) for ds, eps in ds_eps]
    return float(np.median(C_vals))


# ── main ────────────────────────────────────────────────────────────────────────────────
def headline():
    print("=" * 78)
    print("HEADLINE — is the true FFT-vs-direct rel at phi=0.10 (order-6 cons) really 1e-4?")
    print("=" * 78)
    try:
        from fit_phi_max_law import load_water192
        cw, qw, bw_box, Lw = load_water192()
        boxes = [("water192", cw, qw, np.asarray(bw_box))]
    except Exception as ex:
        print(f"  (load_water192 unavailable: {ex})")
        boxes = []
    cr, qr, br, _ = make_test_system(n_mol=64, seed=1)
    boxes.append(("random64", cr, qr, np.asarray(br)))
    for sid, c, q, box in boxes:
        for metric in ("force", "energy"):
            r010 = rel_at(sid, "cons", AMP_CONS, BW_CONS, c, q, box, 0.10, 6, metric)
            phi_e4, st = bisect_phi(sid, "cons", AMP_CONS, BW_CONS, c, q, box, 6, 1e-4, metric)
            d010 = 0.10 * RCUT / sigma_min_of(BW_CONS)
            print(f"  [{sid:9s} {metric:6s}] rel(phi=0.10) = {r010:.3e}   "
                  f"(target 1e-4 => {'OPTIMISTIC by' if r010>1e-4 else 'conservative,'} "
                  f"{r010/1e-4:.1f}x)   phi@1e-4 = {phi_e4:.4f} [{st}]")
    g = cons_prod_grid(0.10, 6)
    print(f"  cons order-6 phi=0.10 on prod box {BOXP} -> grid {g[0]}x{g[1]}x{g[2]} = {g[3]:,}")
    print()


def main():
    headline()
    systems = make_systems()
    kernels = make_kernels()
    anchors = {4: {"force": [], "energy": []}, 6: {"force": [], "energy": []}}
    per_kernel = {}     # (order, metric, kerid) -> {eps: (phi, ds, status)}
    print("=" * 78)
    print("PANEL — bisected phi*(eps) across random systems x kernels")
    print("=" * 78)
    for order in (4, 6):
        for metric in ("force", "energy"):
            print(f"\n-- order-{order}  metric={metric} --")
            print(f"   {'kernel':9s} {'sig_min':>7s} {'system':8s} " +
                  " ".join(f"{e:>9.0e}" for e in EPS_LADDER[order]) + "   (Delta/sigma_min)")
            for kid, amp, bw in kernels:
                smin = sigma_min_of(bw)
                # average anchor across systems, per eps
                ds_by_eps = {e: [] for e in EPS_LADDER[order]}
                for sid, c, q, box, L in systems:
                    row = []
                    for eps in EPS_LADDER[order]:
                        phi, st = bisect_phi(sid, kid, amp, bw, c, q, box, order, eps, metric)
                        ds = phi * RCUT / smin
                        row.append((eps, phi, ds, st))
                        if st == "ok":
                            ds_by_eps[eps].append(ds)
                            anchors[order][metric].append({"kernel": kid, "system": sid,
                                                           "eps": eps, "phi": phi, "ds": ds})
                    # compact per-system print of Delta/sigma_min
                    print(f"   {kid:9s} {smin:7.3f} {sid:8s} " +
                          " ".join(f"{ds:9.3f}" for _, _, ds, _ in row) +
                          "   " + "".join("." if st == "ok" else st[0] for _, _, _, st in row))
                per_kernel[(order, metric, kid)] = ds_by_eps
    # COLLAPSE: at each eps, spread of Delta/sigma_min across all ok anchors
    print("\n" + "=" * 78)
    print("COLLAPSE CHECK — Delta/sigma_min at fixed eps should be ~constant if law holds")
    print("=" * 78)
    for order in (4, 6):
        for metric in ("force", "energy"):
            print(f"\n-- order-{order}  metric={metric} --")
            for eps in EPS_LADDER[order]:
                vals = [a["ds"] for a in anchors[order][metric] if abs(a["eps"] - eps) < 1e-30]
                if vals:
                    mu, sd = np.mean(vals), np.std(vals)
                    print(f"   eps={eps:.0e}: n={len(vals):2d}  Delta/sig = {mu:.3f} +/- {sd:.3f}"
                          f"  (CV {100*sd/mu:4.1f}%)  range [{min(vals):.3f}, {max(vals):.3f}]")
                else:
                    print(f"   eps={eps:.0e}: no accuracy-limited anchors (all validity/floor)")
    # REFIT + report (both free-p and theory-enforced p=2ν)
    print("\n" + "=" * 78)
    print("REFIT — pooled (C_nu, p_nu) from bisected anchors  vs  current constants")
    print("=" * 78)
    smin_cons = sigma_min_of(BW_CONS)
    out = {"metric_canonical": "force", "anchors": anchors, "refit": {}}
    for metric in ("force", "energy"):
        print(f"\n### metric = {metric}{'  (CANONICAL)' if metric=='force' else ''}")
        for order in (4, 6):
            pts = [(a["ds"], a["eps"]) for a in anchors[order][metric]]
            if len(pts) < 3:
                print(f"  order-{order}: too few anchors ({len(pts)})"); continue
            # Free-p fit (for diagnostics only)
            C, p = refit(pts)
            # Theory-enforced fit: p = 2*nu, C = median(eps/ds^{2nu})
            p_theory = int(order)  # 4 or 6 = 2*nu
            C_fixed = refit_fixed_p(pts, p_theory)
            out["refit"][f"{metric}_o{order}"] = {"C": C, "p": p, "C_fixed_p": C_fixed,
                                                    "p_fixed": p_theory, "npts": len(pts)}
            cur_fallback = FALLBACK_LAW[order]
            phi_free = phi_max_from_law(smin_cons, RCUT, order, 1e-4, {"C": C, "p": p})
            phi_fixed = phi_max_from_law(smin_cons, RCUT, order, 1e-4, {"C": C_fixed, "p": p_theory})
            phi_old = phi_max_from_law(smin_cons, RCUT, order, 1e-4, cur_fallback)
            print(f"  order-{order}: FREE-fit C={C:.3e} p={p:.3f}    "
                  f"THEORY (p={p_theory}) C={C_fixed:.3e}    "
                  f"old C={cur_fallback['C']:.3e} p={cur_fallback['p']:.3f}   (n={len(pts)})")
            print(f"           phi(1e-4): free={phi_free:.4f}  theory={phi_fixed:.4f}  old={phi_old:.4f}"
                  f"  (cons sigma_min={smin_cons:.3f} A)")
    with open(os.path.join(HERE, "phi_max_anchors.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    print(f"\nwrote {os.path.join(HERE, 'phi_max_anchors.json')}")


if __name__ == "__main__":
    main()
