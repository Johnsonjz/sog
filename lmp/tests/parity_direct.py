"""Numeric parity check: LAMMPS `kspace_style sog ... direct yes` vs the Python sog lib.

Both sides run the EXACT k-sum on the same frozen kernel
(SogKernel(r_cut=6.0, b=2.0, m=12)) with the k-set pinned to the same mode set
(LAMMPS `direct yes 1.5708`; Python `n_dl=4.0` -> nk=2/axis and the same
spherical cutoff -> 16 half-sphere modes), so any discrepancy is a convention
bug, not a grid-resolution difference.

Standalone usage (from this directory, after running parity_direct.in)::

    lmp -in parity_direct.in -log parity_direct.log
    python parity_direct.py                    # exits non-zero on failure

Via pytest it is driven by test/test_lammps_plugin.py (set SOG_LMP_BINARY).

Reference values (this machine, v30Mar2026 binary): energy diff 4e-16 eV,
max force diff 1.3e-15 relative, pressure tensor ~1e-15 relative.  Thresholds
here are 1e-8 relative — a pass proves the conventions agree; a ~3.3e-8
failure signature means the unit constants differ (see below).

Unit constants: this check pins the PYTHON side to the LAMMPS tree's metal
values `qqr2e = 14.399645` (update.cpp:198) and `nktv2p = 1.6021765e6`
(update.cpp:194) — the older CODATA pair.  The sog lib defaults to the newer
`14.3996454784255`, which differs by 3.32e-8 relative and would fail these
thresholds.  Override for a different LAMMPS build::

    SOG_QQR2E=14.3996454784255 SOG_NKTV2P=1.602176634e6 python parity_direct.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]  # repo root (…/sog)
sys.path.insert(0, str(ROOT / "src"))
from sog import Sog, SogKernel  # noqa: E402

QQR2E = float(os.environ.get("SOG_QQR2E", "14.399645"))
NKTV2P = float(os.environ.get("SOG_NKTV2P", "1.6021765e6"))

# Frozen kernel exported by SogKernel(r_cut=6.0, b=2.0, m=12) — must match the
# kspace_style line in parity_direct.in.
FROZEN_AMP = [78.80229949951172, 316.9695129394531, 1267.8780517578125,
              5071.51220703125, 20286.048828125, 81144.1953125,
              324576.78125, 1298307.125, 5193228.5, 20772914.0,
              83091656.0, 332366624.0]
FROZEN_BW2 = [9.097501754760742, 36.39000701904297, 145.56002807617188,
              582.2401123046875, 2328.96044921875, 9315.841796875,
              37263.3671875, 149053.46875, 596213.875, 2384855.5,
              9539422.0, 38157688.0]
KMAX = 1.5708          # `direct yes 1.5708`  ==  2*pi/n_dl with n_dl = 4.0
N_DL = 4.0

TOL_REL = 1e-8         # comfortably above fp noise (1e-15), below the
                       # 3.3e-8 unit-constant mismatch signature

# Raw LAMMPS virial order for `compute pressure` in this tree.
PRESS_NAMES = ["xx", "yy", "zz", "xy", "xz", "yz"]


# ── LAMMPS side parsers ───────────────────────────────────────────────────


def parse_dump(path):
    """Return (q, r, f, L) from a parity.dump (id type q x y z fx fy fz)."""
    box, atoms, n_atoms = None, [], None
    with open(path) as fh:
        lines = iter(fh)
        for line in lines:
            if "ITEM: NUMBER OF ATOMS" in line:
                n_atoms = int(next(lines))
            elif "ITEM: BOX BOUNDS" in line:
                box = [[float(x) for x in next(lines).split()] for _ in range(3)]
            elif "ITEM: ATOMS" in line:
                for _ in range(n_atoms):
                    t = next(lines).split()
                    atoms.append([float(x) for x in t])
    if box is None or len(atoms) != n_atoms:
        raise ValueError(f"malformed dump {path}: {len(atoms)} atoms, box {box}")
    atoms.sort(key=lambda a: a[0])
    q = torch.tensor([a[2] for a in atoms], dtype=torch.float64)
    r = torch.tensor([[a[3], a[4], a[5]] for a in atoms], dtype=torch.float64)
    f = torch.tensor([[a[6], a[7], a[8]] for a in atoms], dtype=torch.float64)
    L = torch.tensor([box[i][1] - box[i][0] for i in range(3)],
                     dtype=torch.float64)
    return q, r, f, L


def parse_thermo(log_path):
    """Return (E_lmp, c_p[6]) from the step-0 thermo line of a LAMMPS log.

    NOTE: this LAMMPS build's compute pressure passes the kspace virial
    through in raw LAMMPS virial order (xx, yy, zz, xy, xz, yz) — NOT the
    documented (pyz, pxz, pxy) order — so c_p[3]=P_xy, c_p[4]=P_xz,
    c_p[5]=P_yz.
    """
    thermo = None
    with open(log_path) as fh:
        lines = fh.read().splitlines()
    for i, line in enumerate(lines):
        if line.strip().startswith("Step") and "PotEng" in line:
            for cand in lines[i + 1:]:
                cand = cand.strip()
                if cand and not cand.startswith("Loop"):
                    thermo = cand.split()
                    break
            break
    if thermo is None:
        raise ValueError(f"no thermo line found in {log_path}")
    if int(thermo[0]) != 0:
        raise ValueError(f"expected step 0 thermo, got {thermo}")
    # step pe press c_p[1..6]
    return float(thermo[1]), [float(x) for x in thermo[3:9]]


# ── Python side ───────────────────────────────────────────────────────────


def python_reference(q, r_lmp, L):
    """Python E_lr, forces and pressure tensor on the dumped geometry.

    The virial is the COMPLETE strain derivative: the box is differentiated at
    fixed scaled coordinates, which delivers both PBC pressure terms (Σ F⊗r
    and the box-deformation term) at once.  This mirrors what sog.cpp computes
    analytically for the k-space part.
    """
    # Build the kernel from the SAME frozen constants the LAMMPS input uses —
    # never regenerate it from (rcut, b, m) here: that computation is
    # environment-dependent (it follows the global torch default dtype, and a
    # float64-recomputed kernel differs from the float32-rounded exported
    # constants by ~5e-8, which this checker would report as a failure).
    kernel = SogKernel(b=2.0, sigma=3.0162065545167596, m=12)
    kernel.bind_external(torch.tensor(FROZEN_AMP, dtype=torch.float64),
                         torch.tensor(FROZEN_BW2, dtype=torch.float64),
                         amp_is_internal=True, bandwidth_is_squared=True)
    model = Sog({"use_atomwise": False, "n_dl": N_DL,
                 "remove_self_interaction": False,
                 "norm_factor": QQR2E}, kernel=kernel)

    s = (r_lmp / L).clone().requires_grad_(True)          # scaled coords leaf
    h = torch.diag(L).unsqueeze(0).requires_grad_(True)   # [1,3,3] orthorhombic
    out = model(positions=s @ h[0], cell=h, latent_charges=q,
                compute_energy=True)
    E_py = out["E_lr"].sum()
    E_py.backward()
    # dE/ds_ij = dE/dr_ij * h_jj  ->  f = -dE/dr = -(dE/ds) / h_jj
    f_py = -s.grad / L
    g_h = h.grad[0]
    P_py = -(1.0 / torch.det(h[0])) * g_h @ h[0].T        # eV/Å³
    return E_py.item(), f_py, P_py


def explicit_ref_energy(r_in, L, q):
    """Hand-built C++ compute_direct formula on the identically enumerated
    k-set — pinpoints WHICH side carries a discrepancy."""
    import math

    amp = torch.tensor(FROZEN_AMP, dtype=torch.float64)
    bw2 = torch.tensor(FROZEN_BW2, dtype=torch.float64)
    dk = 2.0 * float(torch.pi) / float(L[0])
    ncap = math.ceil(KMAX / dk)
    e = torch.zeros((), dtype=torch.float64)
    for ix in range(0, ncap + 1):
        for iy in range(-ncap, ncap + 1):
            for iz in range(-ncap, ncap + 1):
                if ix == 0 and iy == 0 and iz == 0:
                    continue
                if ix == 0 and iy < 0:
                    continue
                if ix == 0 and iy == 0 and iz < 0:
                    continue
                kv = torch.tensor([ix * dk, iy * dk, iz * dk],
                                  dtype=torch.float64)
                if (kv * kv).sum() > KMAX * KMAX:
                    continue
                kr = r_in @ kv
                S_re = (q * torch.cos(kr)).sum()
                S_im = (q * torch.sin(kr)).sum()
                K = (amp * torch.exp(-0.5 * bw2 * (kv * kv).sum())).sum()
                e = e + K * (S_re * S_re + S_im * S_im)
    return QQR2E / torch.prod(L) * e


# ── comparison ────────────────────────────────────────────────────────────


def compare(dump_path, log_path):
    """Run the full comparison; returns a dict of measured differences.

    Keys: energy_abs, energy_rel, force_rel, press_rel (max over components),
    plus the raw arrays for reporting.
    """
    q, r_lmp, f_lmp, L = parse_dump(dump_path)
    E_lmp, c_p = parse_thermo(log_path)

    E_py, f_py, P_py = python_reference(q, r_lmp, L)
    E_ref = explicit_ref_energy(r_lmp, L, q).item()

    d_force = (f_py - f_lmp).abs()
    f_scale = f_lmp.abs().max().item()
    if f_scale < 1e-9:
        raise AssertionError(
            "LAMMPS forces are ~zero — degenerate/symmetric geometry; the "
            "displacement in parity_direct.in must be present."
        )

    P_py6 = (P_py * NKTV2P)
    P_py6 = [P_py6[0][0].item(), P_py6[1][1].item(), P_py6[2][2].item(),
             P_py6[0][1].item(), P_py6[0][2].item(), P_py6[1][2].item()]
    d_press = [abs(a - b) for a, b in zip(P_py6, c_p)]
    press_rel = max(d / max(abs(b), 1.0) for d, b in zip(d_press, c_p))

    return {
        "natoms": int(q.numel()),
        "box": L.tolist(),
        "E_lmp": E_lmp, "E_py": E_py, "E_ref": E_ref,
        "energy_abs": abs(E_lmp - E_py),
        "energy_rel": abs(E_lmp - E_py) / max(abs(E_lmp), 1e-6),
        "force_abs": d_force.max().item(),
        "force_scale": f_scale,
        "force_rel": d_force.max().item() / f_scale,
        "P_lmp": c_p, "P_py": P_py6, "P_diff": d_press,
        "press_rel": press_rel,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dump", nargs="?", default="parity.dump")
    ap.add_argument("--log", default="parity_direct.log")
    args = ap.parse_args(argv)

    res = compare(args.dump, args.log)

    print(f"box = {res['box']}, natoms = {res['natoms']}")
    print("\n=== ENERGY (eV) ===")
    print(f"LAMMPS pe     : {res['E_lmp']:.12f}")
    print(f"Python  E_lr  : {res['E_py']:.12f}")
    print(f"explicit ref  : {res['E_ref']:.12f}")
    print(f"|lmp - py|    : {res['energy_abs']:.3e}  (rel {res['energy_rel']:.3e})")

    print("\n=== FORCES (eV/A) ===")
    print(f"max |dF|      : {res['force_abs']:.3e}")
    print(f"max |F_lmp|   : {res['force_scale']:.3e}")
    print(f"rel           : {res['force_rel']:.3e}")

    print(f"\n=== PRESSURE TENSOR (bar), {PRESS_NAMES} ===")
    for name, py, lmp, d in zip(PRESS_NAMES, res["P_py"], res["P_lmp"],
                                res["P_diff"]):
        print(f"P_{name}: py {py:12.4f}  lmp {lmp:12.4f}  diff {d:9.3e}")

    checks = [("energy", res["energy_rel"], TOL_REL),
              ("force", res["force_rel"], TOL_REL),
              ("pressure", res["press_rel"], TOL_REL)]
    print()
    ok = True
    for name, val, tol in checks:
        good = val < tol
        ok &= good
        print(f"{'PASS' if good else 'FAIL'} {name:9s} rel {val:.3e} "
              f"(tol {tol:.0e})")
    if not ok:
        print("\nHint: a uniform ~3.3e-8 / ~8.4e-8 residual is the LAMMPS "
              "tree's qqr2e / nktv2p constant mismatch — see the module "
              "docstring for SOG_QQR2E / SOG_NKTV2P.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
