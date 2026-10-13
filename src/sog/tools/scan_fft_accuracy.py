#!/usr/bin/env python3
"""Scan FFT accuracy vs direct with CONSISTENT kernel parameters.

Key insight: both direct and FFT Gaussian MUST use the same rcut
so sigma = rcut*nlayers/RCUT_TO_SIGMA (fastsog convention) in both cases.

Both paths use same k_sq_max from n_dl. Difference = mesh discretization + influence fn.
"""

import math
import torch
import numpy as np
torch.set_default_dtype(torch.float64)

from sog.module.gaussian import Gaussian
from sog.module.cubes2_fft import _compute_phi_max, _compute_grid_from_phi, _INFLUENCE_CACHE

RCUT_TO_SIGMA = 1.9892536839080267


def main():
    print("=" * 80)
    print("  FFT Accuracy Scan: Consistent kernel params (same rcut for both)")
    print("=" * 80)

    torch.manual_seed(42)

    # Systems
    Lx, Ly, Lz = 8.9, 8.9, 26.4
    r_au = torch.rand(110, 3) * torch.tensor([Lx, Ly, Lz])
    q_au = torch.randn(110, 3) * 0.3
    cell_au = torch.diag(torch.tensor([Lx, Ly, Lz])).unsqueeze(0)

    Lw = 12.44
    r_w = torch.rand(192, 3) * Lw
    q_w = torch.randn(192, 1) * 0.5
    cell_w = torch.eye(3).unsqueeze(0) * Lw

    systems = [
        ("Au-MgO 110at", r_au, q_au, cell_au, Lx, Ly, Lz),
        ("Water 192at", r_w, q_w, cell_w, Lw, Lw, Lw),
    ]

    for b_val, b_label in [(2.0, "b=2.0"), (1.6297670882677647, "b≈1.63")]:
        sigma_val = 6.0 / RCUT_TO_SIGMA
        bw0 = sigma_val ** 2
        eps = 1e-6
        ln_1e = -math.log(eps)
        k_sq_eps = 2 * ln_1e / bw0
        n_dl_eps = 2 * math.pi / math.sqrt(k_sq_eps)

        phi_auto = _compute_phi_max(b_val, 4)
        print(f"\n{'='*80}")
        print(f"  {b_label}: σ={sigma_val:.4f}Å, bw[0]={bw0:.3f}Å²")
        print(f"  ε=1e-6 → k_sq_max={k_sq_eps:.2f}Å⁻², n_dl={n_dl_eps:.3f}Å")
        print(f"  auto φ_max = {phi_auto:.4f}")
        print(f"{'='*80}")

        # n_dl scan from ε-based down to very fine
        n_dl_vals = np.unique(np.round(
            np.logspace(np.log10(0.3), np.log10(n_dl_eps), 15), 4
        ))[::-1]

        for sys_name, r, q, cell, lx, ly, lz in systems:
            # ── Direct reference at finest n_dl ──
            n_dl_fine = n_dl_vals[0]
            g_ref = Gaussian(
                n_dl=n_dl_fine, b=b_val, m=12, rcut=6.0, nlayers=1,
                use_cubes2_fft=False, remove_self_interaction=True,
            )
            e_ref = float(g_ref(q, r, cell).item())

            print(f"\n  --- {sys_name} (ref n_dl={n_dl_fine:.3f}, E_ref={e_ref:.6f}) ---")
            print(f"    {'n_dl':>8s}  {'k_sq':>8s}  {'E_dir':>14s}  {'E_fft':>14s}  "
                  f"{'|dE|':>10s}  {'|dE/E|':>10s}  {'Φ grid':>10s}")
            print(f"    {'-'*8}  {'-'*8}  {'-'*14}  {'-'*14}  "
                  f"{'-'*10}  {'-'*10}  {'-'*10}")

            for n_dl in n_dl_vals:
                # Both use rcut=6.0 → same sigma, same amp/bw!
                _INFLUENCE_CACHE.clear()
                g_dir = Gaussian(
                    n_dl=n_dl, b=b_val, m=12, rcut=6.0, nlayers=1,
                    use_cubes2_fft=False, remove_self_interaction=True,
                )
                g_fft = Gaussian(
                    n_dl=n_dl, b=b_val, m=12, rcut=6.0, nlayers=1,
                    use_cubes2_fft=True, remove_self_interaction=True,
                )

                e_dir = float(g_dir(q, r, cell).item())
                e_fft = float(g_fft(q, r, cell).item())
                k_sq = (2 * math.pi / n_dl) ** 2

                dE = abs(e_fft - e_dir)
                rel = dE / max(abs(e_dir), 1e-30)

                nk_x = max(1, int(lx / n_dl))
                nk_y = max(1, int(ly / n_dl))
                nk_z = max(1, int(lz / n_dl))
                g_str = f"{2*nk_x+1}×{2*nk_y+1}×{2*nk_z+1}"

                note = ""
                if rel < 1e-6: note = " ★1e-6"
                elif rel < 1e-4: note = " <1e-4"

                print(f"    {n_dl:8.4f}  {k_sq:8.2f}  {e_dir:14.6f}  {e_fft:14.6f}  "
                      f"{dE:10.3e}  {rel:10.2e}  {g_str:>10s}{note}")

            # Find n_dl giving 1e-6 match to reference
            print(f"\n    Convergence to E_ref={e_ref:.6f}:")
            for n_dl in n_dl_vals:
                g_check = Gaussian(
                    n_dl=n_dl, b=b_val, m=12, rcut=6.0, nlayers=1,
                    use_cubes2_fft=True, remove_self_interaction=True,
                )
                e_check = float(g_check(q, r, cell).item())
                rel_ref = abs(e_check - e_ref) / max(abs(e_ref), 1e-30)
                if rel_ref < 1e-6:
                    print(f"    n_dl ≤ {n_dl:.4f} → FFT matches finest direct to 1e-6")
                    break
            else:
                print(f"    ⚠ finest n_dl={n_dl_vals[-1]:.4f} still above 1e-6")

    print(f"\n{'='*80}")
    print("  Scan complete.")


if __name__ == "__main__":
    main()
