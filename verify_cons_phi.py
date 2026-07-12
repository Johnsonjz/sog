#!/usr/bin/env python3
"""A-lever validation (CPU, no GPU): for the CONSERVATIVE kernel (sigma_min=0.750),
find the order-6 CubeS2 phi that matches the current order-4 phi=0.0675 accuracy,
and quantify the real-box grid-point reduction. Also probe separable QuadS if available.

Reuses sog/verify_phi_max.py helpers. Run with dp_dev python.
"""
import sys, math
sys.path.insert(0, "/root/code/sog/src")
sys.path.insert(0, "/root/code/sog")
import numpy as np
import torch
torch.set_default_dtype(torch.float64)

from verify_phi_max import (energy_fft, reference_direct, sigma_min, realized_delta)
from verify_fft_vs_direct import make_test_system, RCUT, B
from sog.module.cubes2_fft import _resolve_grid

# ── CONSERVATIVE kernel: verbatim amp/bandwidth from sog_k0_cons_kspace.in ──
AMP_CONS = np.array([28.931973, 82.540521, 212.78413, 560.83623, 1452.2041, 3857.2586,
                     10245.422, 27213.282, 72282.308, 191991.99, 509957.74, 1354519.5])
BW_CONS = np.array([0.56216432, 5.9422127, 28.477837, 69.351734, 236.59701, 628.43481,
                    1669.2113, 4433.6602, 11776.426, 31279.844, 83083.669, 220681.92])

# real production box (water_2x4x4): 24.8894 x 49.7788 x 49.7788
BOXP = (24.8894, 49.7788, 49.7788)


def prod_grid(phi, order):
    nx, ny, nz, _ = _resolve_grid(BOXP[0], BOXP[1], BOXP[2], None, phi, RCUT, b=B, spline_order=order)
    return nx, ny, nz, nx * ny * nz


def main():
    coords, q, box, box_l = make_test_system(n_mol=64, seed=1)
    smin = sigma_min(BW_CONS)
    e_ref = reference_direct(AMP_CONS, BW_CONS, coords, q, box)
    print(f"CONS kernel: bw_min={float(np.min(BW_CONS)):.5f}  sigma_min={smin:.4f} A  r_c={RCUT}  b={B:.4f}")
    print(f"toy system: {len(q)} charges, box {box_l:.2f} A (dense -> conservative upper-bound on rel err)")
    print(f"validity ceilings: phi_valid(o4)={smin/(math.sqrt(2)*0.5773502691896258)/RCUT:.4f}  "
          f"phi_valid(o6)={smin/(math.sqrt(2)*0.6503998764035732)/RCUT:.4f}")

    def rel(phi, order):
        e = energy_fft(AMP_CONS, BW_CONS, coords, q, box, phi, order=order)
        return abs(e - e_ref) / abs(e_ref)

    # baseline: current production order-4 phi=0.0675
    r4 = rel(0.0675, 4)
    g4 = prod_grid(0.0675, 4)
    print(f"\n[baseline] order-4 phi=0.0675 : rel={r4:.3e}  Delta/sig={0.0675*RCUT/smin:.3f}  "
          f"prod grid {g4[0]}x{g4[1]}x{g4[2]} = {g4[3]:,}")

    print(f"\n{'order':>5} {'phi':>6} {'D/sig':>6} {'rel_toy':>10} | {'prod grid':>14} {'pts':>10} {'vs o4-base':>9}")
    for order, phis in [(4, [0.0675, 0.09]),
                        (6, [0.09, 0.10, 0.105, 0.11, 0.12, 0.13, 0.14, 0.16])]:
        for phi in phis:
            r = rel(phi, order)
            nx, ny, nz, pts = prod_grid(phi, order)
            print(f"{order:>5} {phi:>6.3f} {phi*RCUT/smin:>6.3f} {r:>10.3e} | "
                  f"{nx:>3}x{ny:>3}x{nz:>3}   {pts:>10,} {g4[3]/pts:>8.2f}x")

    # find coarsest order-6 phi whose rel <= order-4 phi=0.0675 rel
    best = None
    for phi in np.arange(0.20, 0.06, -0.005):
        if rel(float(phi), 6) <= r4:
            best = float(phi); break
    if best:
        g6 = prod_grid(best, 6)
        print(f"\n[match] coarsest order-6 phi with rel<=order-4(0.0675): phi={best:.3f}  "
              f"rel={rel(best,6):.3e}")
        print(f"        prod grid {g6[0]}x{g6[1]}x{g6[2]} = {g6[3]:,}  -> {g4[3]/g6[3]:.2f}x fewer points vs current")
        print(f"        (scatter nodes 88 vs 32 = 2.75x; net kspace ~ {g4[3]/g6[3]/2.75*32/32:.1f}x on grid-dominated cost)")


if __name__ == "__main__":
    main()
