#!/usr/bin/env python3
"""Fit the SOG particle-mesh error law  rel = C_nu * (Delta/sigma_min)^p_nu  per spline order,
across kernels (cons / neutral / geometric u-series) and systems (dense toy vs real 192-water),
to calibrate a GENERAL phi_max rule  phi_max = (eps/C_nu)^(1/p_nu) * sigma_min/r_c.
The paper has NO phi_max formula (Table III is empirical); this builds the accuracy-inverted rule.
CPU-only. Run with dp_dev python."""
import sys, math
sys.path.insert(0, "/root/code/sog/src"); sys.path.insert(0, "/root/code/sog")
import numpy as np, torch
torch.set_default_dtype(torch.float64)
from verify_phi_max import energy_fft, reference_direct, sigma_min, realized_delta, AMP_MD, BW_MD
from verify_fft_vs_direct import geometric_amp_bw, make_test_system, B, SIGMA, M, RCUT
from verify_cons_phi import AMP_CONS, BW_CONS


def load_water192():
    path = "/root/code/deepmd-example/water-scan0/lmp/replicate/data/water_charge_192.lmp"
    coords, types, L = [], [], None
    inatoms = False
    for line in open(path):
        if "xhi" in line: L = float(line.split()[1])
        if line.strip().startswith("Atoms"): inatoms = True; continue
        p = line.split()
        if inatoms and len(p) >= 6 and p[0].isdigit():
            types.append(int(p[1])); coords.append([float(p[3]), float(p[4]), float(p[5])])
    coords = np.array(coords); types = np.array(types)
    q = np.where(types == 1, -0.2, 0.1).astype(float)   # O=-0.2, H=+0.1 -> neutral
    return coords, q, np.diag([L, L, L]).astype(float), L


def fit_law(amp, bw, coords, q, box, order, phis):
    amp = np.asarray(amp); bw = np.asarray(bw)
    e_ref = reference_direct(amp, bw, coords, q, box)
    smin = sigma_min(bw); lx = float(box[0, 0])
    xs, ys, seen = [], [], set()
    for phi in phis:
        d, nx = realized_delta(float(phi), lx, order)
        if nx in seen: continue
        seen.add(nx)
        e = energy_fft(amp, bw, coords, q, box, float(phi), order=order)
        rel = abs(e - e_ref) / abs(e_ref)
        if 1e-7 < rel < 2e-1:               # clean asymptotic window
            xs.append(d / smin); ys.append(rel)
    if len(xs) < 3:
        return None
    p, logC = np.polyfit(np.log(xs), np.log(ys), 1)
    return p, math.exp(logC), smin, len(xs)


KERNELS = {"cons": (AMP_CONS, BW_CONS), "neutral": (AMP_MD, BW_MD),
           "geom_b2": geometric_amp_bw(B, SIGMA, M)}


def main():
    toy = make_test_system(n_mol=64, seed=1)[:3]
    w192 = load_water192()[:3]
    print(f"RCUT={RCUT}  B={B:.4f}")
    for sysname, sysc in [("toy(dense)", toy), ("water192(real)", w192)]:
        for order, phis in [(4, np.linspace(0.05, 0.24, 16)), (6, np.linspace(0.07, 0.22, 16))]:
            print(f"\n== system={sysname}  order={order} ==   (rel = C*(D/sig)^p)")
            for kn, (amp, bw) in KERNELS.items():
                r = fit_law(amp, bw, *sysc, order, phis)
                if r:
                    p, C, smin, n = r
                    ds_1e3 = (1e-3 / C) ** (1.0 / p)
                    print(f"  {kn:8s} sig_min={smin:.3f}  p={p:5.2f}  C={C:.3e}  "
                          f"(D/sig@1e-3={ds_1e3:.3f}, phi@1e-3={ds_1e3*smin/RCUT:.4f})  n={n}")


if __name__ == "__main__":
    main()

