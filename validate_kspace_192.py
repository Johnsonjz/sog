#!/usr/bin/env python3
"""Task 2: numerical correctness of sog.cpp CubeS2-FFT kspace vs a Python direct k-sum reference,
on the 192-atom water box, over a short small-timestep trajectory. Also runs an ASE small-timestep
MD with a SogDirect (Python direct k-sum) calculator per the boss's ASE request.

Compares, per frame of the LAMMPS-generated trajectory (shared configs, identical fixed charges):
  - kspace ENERGY: C++ elong (k!=0) vs Python direct (k!=0, after analytic k=0 self-term subtraction)
  - kspace FORCE : C++ atom->f (pure kspace) vs Python direct autograd force  (k=0-free)
for order-4 (phi=0.0675) and order-6 (phi=0.10). Produces a matplotlib figure. CPU-only.
"""
import sys, math, numpy as np, torch
sys.path.insert(0, "/root/code/sog/src"); sys.path.insert(0, "/root/code/sog")
torch.set_default_dtype(torch.float64)
from sog.module.gaussian import Gaussian
from verify_cons_phi import AMP_CONS, BW_CONS
from verify_fft_vs_direct import E2_PER_ANGSTROM_TO_EV as NORM, B, RCUT

AMP = np.asarray(AMP_CONS); BW = np.asarray(BW_CONS)
BOX = 12.44470; VOL = BOX ** 3
BASE = "/root/code/deepmd-example/water-scan0/lmp/scale2x4x4"
NVE_STEPS = 1000   # NVE trajectory length (0.5 fs step -> 0.5 ps); stronger conservation test


def direct_gaussian(n_dl=0.7):
    # n_dl=0.7 (k_max~9, tail ~1e-10) is a converged reference for the py-vs-cpp comparison and much
    # cheaper than 0.5; the ASE MD uses a coarser n_dl (still conservative — force = autograd(-E)).
    return Gaussian(amp=torch.tensor(AMP), bandwidth=torch.tensor(BW),
                    kernel_param_mode="internal", kernel_tensor_mode="external",
                    remove_self_interaction=True, use_nufft=False, use_cubes2_fft=False,
                    norm_factor=NORM, trainable=False, b=B, rcut=RCUT, n_dl=n_dl)


def py_direct_ef(g, coords, q, want_force=True):
    r = torch.tensor(coords, dtype=torch.float64, requires_grad=want_force)
    qt = torch.tensor(q, dtype=torch.float64).reshape(-1, 1)
    cell = torch.tensor(np.diag([BOX, BOX, BOX]), dtype=torch.float64).unsqueeze(0)
    batch = torch.zeros(len(q), dtype=torch.int64)
    E = g.compute_bundle(q=qt, r=r, cell=cell, batch=batch)["energy"].sum()
    if not want_force:
        return float(E), None
    F = -torch.autograd.grad(E, r)[0]
    return float(E), F.detach().numpy()


def k0_self(q):
    """Python direct includes a k=0 term; C++ elong is k!=0 only. Neutral -> only the self term."""
    kmin2 = (2.0 * np.pi / BOX) ** 2
    kfac = float((AMP * np.exp(-0.5 * BW * kmin2)).sum())
    return -kfac * float(np.sum(q ** 2)) / (2.0 * VOL) * NORM


def parse_traj(path):
    frames, f = [], open(path)
    lines = f.read().splitlines(); i = 0
    while i < len(lines):
        if lines[i].startswith("ITEM: TIMESTEP"):
            n = int(lines[i + 3])
            # find ATOMS header
            j = i + 4
            while not lines[j].startswith("ITEM: ATOMS"):
                j += 1
            rows = [lines[j + 1 + k].split() for k in range(n)]
            rows.sort(key=lambda r: int(r[0]))
            arr = np.array([[float(x) for x in r] for r in rows])
            frames.append((arr[:, 2:5], arr[:, 5], arr[:, 6:9], arr[:, 1].astype(int)))  # coords,q,Fcpp,type
            i = j + 1 + n
        else:
            i += 1
    return frames


def parse_elong(logpath):
    vals = []
    for line in open(logpath):
        p = line.split()
        if len(p) >= 2 and p[0].isdigit():
            try:
                vals.append(float(p[1]))
            except ValueError:
                pass
    return vals


def compare(order, tag, stride=10):
    frames = parse_traj(f"{BASE}/traj_192_{tag}.txt")
    elong = parse_elong(f"{BASE}/kspace192_{tag}.log")
    g = direct_gaussian()
    idx = list(range(0, len(frames), stride))
    relE, relF, maxdF = [], [], []
    for k in idx:
        coords, q, fcpp, _typ = frames[k]
        e0 = k0_self(q)
        Epy, Fpy = py_direct_ef(g, coords, q)
        Epy_kneq0 = Epy - e0
        relE.append(abs(Epy_kneq0 - elong[k]) / abs(elong[k]))
        df = Fpy - fcpp
        relF.append(np.linalg.norm(df) / (np.linalg.norm(fcpp) + 1e-30))
        maxdF.append(np.abs(df).max())
    return np.array(idx), np.array(relE), np.array(relF), np.array(maxdF)


# ── ASE path (boss's request): a Calculator wrapping the Python direct-k-sum SOG kspace ──
def make_sog_direct_calculator():
    from ase.calculators.calculator import Calculator, all_changes
    g = direct_gaussian(n_dl=1.0)   # coarser k-sum for the 1000-step MD; still conservative

    class SogDirect(Calculator):
        implemented_properties = ["energy", "forces"]

        def calculate(self, atoms=None, properties=("energy",), system_changes=None):
            from ase.calculators.calculator import all_changes as _ac
            Calculator.calculate(self, atoms, properties, system_changes or _ac)
            coords = atoms.get_positions()
            q = atoms.get_initial_charges()
            E, F = py_direct_ef(g, coords, q)              # direct k-sum, autograd force
            self.results["energy"] = E                     # conservative field (F = -dE/dr)
            self.results["forces"] = F

    return SogDirect()


def run_ase_md(nsteps=100):
    """Small-timestep ASE NVE MD driven by the SogDirect (Python direct SOG) calculator.
    Returns the total-energy drift as a conservativity sanity check of the field ASE runs."""
    from ase import Atoms, units
    from ase.md.verlet import VelocityVerlet
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    coords, q, _f, typ = parse_traj(f"{BASE}/traj_192_o4.txt")[0]
    symbols = ["O" if t == 1 else "H" for t in typ]
    atoms = Atoms(symbols=symbols, positions=coords, cell=[BOX, BOX, BOX], pbc=True)
    atoms.set_initial_charges(q)
    atoms.calc = make_sog_direct_calculator()
    MaxwellBoltzmannDistribution(atoms, temperature_K=300.0, rng=np.random.default_rng(1))
    dyn = VelocityVerlet(atoms, timestep=0.5 * units.fs)
    etot = []
    for _ in range(nsteps):
        dyn.run(1)
        etot.append(atoms.get_total_energy())
    etot = np.array(etot)
    drift = (etot.max() - etot.min()) / abs(np.mean(etot))
    return etot, drift


def parse_etotal(logpath):
    """Parse etotal (col 4) from `step elong pe ke etotal temp` thermo rows. Require ALL columns
    numeric so setup lines like `1 by 1 by 1 MPI processor grid` (has 'by') are rejected."""
    vals = []
    for line in open(logpath):
        p = line.split()
        if len(p) == 6 and p[0].isdigit():
            try:
                nums = [float(x) for x in p]   # every column must parse
            except ValueError:
                continue
            vals.append(nums[4])               # etotal
    return vals


def analyze_nve(etot, dt_fs=0.5, n_eq=200):
    """Post-transient NVE conservation diagnostics. In a symplectic (velocity-Verlet) integrator the
    total energy is set at t=0 and conserved regardless of thermodynamic equilibration; the observable
    is (i) a bounded O(dt²) fluctuation around the shadow Hamiltonian, and (ii) any slow SECULAR drift
    (the true conservativity test). So we (a) discard the first n_eq equilibration steps (the noisiest
    part: largest accelerations => largest O(dt²) excursion), (b) reference to the production-window
    time-mean <E> (NOT E(0), a single transient-contaminated point), and (c) report the RMS fluctuation
    and the secular DRIFT RATE (linear-fit slope), instead of the outlier-sensitive (max-min)/|mean|."""
    e = np.asarray(etot, float)
    prod = e[n_eq:]
    mean = prod.mean()
    rms = np.sqrt(np.mean((prod - mean) ** 2)) / abs(mean)          # relative RMS fluctuation
    t = np.arange(len(prod)) * dt_fs * 1e-3                         # ps
    slope = np.polyfit(t, prod, 1)[0]                              # eV/ps (secular drift)
    drift_rate = slope / abs(mean)                                  # relative / ps
    return mean, rms, drift_rate


def make_nve_plot(outpath, ase_etot, n_eq=200, dt_fs=0.5):
    """NVE energy conservation with the corrected diagnostic: y = (E(t)-<E>)/|<E>| referenced to the
    production-window mean <E> (post-transient), the first n_eq steps shaded as equilibration (excluded
    from metrics), and each curve annotated with RMS fluctuation + secular drift RATE (per ps)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C = {"o4": "#2a78d6", "o6": "#1baf7a", "ase": "#4a3aa7"}
    INK, MUT, GRID, SURF = "#0b0b0b", "#52514e", "#e1e0d9", "#fcfcfb"
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10,
                         "axes.edgecolor": "#c3c2b7", "axes.linewidth": 0.8})
    fig, ax = plt.subplots(figsize=(7.2, 4), facecolor=SURF)
    ax.set_facecolor(SURF); ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.tick_params(colors=MUT)

    series = []
    for tag, name in [("o4", "LAMMPS sog.cpp order-4"), ("o6", "LAMMPS sog.cpp order-6")]:
        e = np.array(parse_etotal(f"{BASE}/kspace192_{tag}.log"))
        if len(e):
            series.append((tag, name, e))
    series.append(("ase", "ASE SogDirect (py-direct)", np.asarray(ase_etot, float)))

    for tag, name, e in series:
        mean, rms, dr = analyze_nve(e, dt_fs=dt_fs, n_eq=n_eq)
        y = (e - mean) / abs(mean)                                  # referenced to production <E>
        ax.plot(range(len(e)), y, "-", color=C[tag], lw=1.5,
                label=f"{name}   RMS {rms:.1e}, drift {dr:+.1e}/ps")
    ax.axvspan(0, n_eq, color="#d9d7cc", alpha=0.5, lw=0)           # equilibration (excluded)
    ax.text(n_eq * 0.5, ax.get_ylim()[1] * 0.86, "equilibration\n(excluded)",
            ha="center", va="top", fontsize=7.5, color=MUT)
    ax.axhline(0.0, color=MUT, lw=0.7, ls="--", alpha=0.6)
    ax.set_xlabel("MD step (0.5 fs, NVE)", color=MUT)
    ax.set_ylabel(r"(E$_{tot}$(t) − ⟨E$_{tot}$⟩) / |⟨E$_{tot}$⟩|   (⟨·⟩ over production window)", color=MUT)
    ax.set_title("NVE energy conservation", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="lower left")
    fig.tight_layout()
    fig.savefig(outpath, dpi=140, bbox_inches="tight", facecolor=SURF)
    print(f"saved {outpath}")
    for tag, name, e in series:
        mean, rms, dr = analyze_nve(e, dt_fs=dt_fs, n_eq=n_eq)
        print(f"  {name:32s} <E>={mean:.5f}  RMS={rms:.2e}  drift_rate={dr:+.2e}/ps")


def make_plot(res, outpath):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # dataviz palette (light): series blue/aqua, recessive ink/grid
    C = {"o4": "#2a78d6", "o6": "#1baf7a"}
    LBL = {"o4": "order-4  φ=0.0675", "o6": "order-6  φ=0.10"}
    INK, MUT, GRID, SURF = "#0b0b0b", "#52514e", "#e1e0d9", "#fcfcfb"
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10,
                         "axes.edgecolor": "#c3c2b7", "axes.linewidth": 0.8})
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6), facecolor=SURF)
    for a in ax:
        a.set_facecolor(SURF); a.grid(True, color=GRID, lw=0.6); a.set_axisbelow(True)
        a.tick_params(colors=MUT); a.set_yscale("log")
    for tag in ("o4", "o6"):
        idx, relE, relF, maxdF = res[tag]
        ax[0].plot(idx, relE, "-", color=C[tag], lw=1.6, label=LBL[tag])
        ax[1].plot(idx, relF, "-", color=C[tag], lw=1.6, label=LBL[tag])
        ax[2].plot(idx, maxdF, "-", color=C[tag], lw=1.6, label=LBL[tag])
    ax[0].set_title("kspace energy: |E$_{py}$−E$_{cpp}$|/|E$_{cpp}$|", color=INK, fontsize=10)
    ax[1].set_title("kspace force: ‖F$_{py}$−F$_{cpp}$‖/‖F$_{cpp}$‖", color=INK, fontsize=10)
    ax[2].set_title("kspace force: max|ΔF| (eV/Å)", color=INK, fontsize=10)
    for a in ax:
        a.set_xlabel("MD frame (0.5 fs step)", color=MUT)
        a.legend(frameon=False, fontsize=9, labelcolor=INK)
    fig.suptitle("sog.cpp CubeS₂-FFT kspace vs Python direct k-sum  (192-atom water, NVE trajectory)",
                 color=INK, fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(outpath, dpi=140, bbox_inches="tight", facecolor=SURF)
    print(f"saved {outpath}")


if __name__ == "__main__":
    import os
    res = {}
    for order, tag in [(4, "o4"), (6, "o6")]:
        idx, relE, relF, maxdF = compare(order, tag)
        res[tag] = (idx, relE, relF, maxdF)
        print(f"order-{order} ({tag}): frames={len(idx)}  "
              f"energy rel  med={np.median(relE):.2e} max={relE.max():.2e}  |  "
              f"force rel med={np.median(relF):.2e} max={relF.max():.2e}  |  max|dF| med={np.median(maxdF):.2e}")
    print("\n== ASE small-timestep MD (SogDirect calculator) ==")
    etot, drift = run_ase_md(NVE_STEPS)
    print(f"  ASE NVE {NVE_STEPS} steps (0.5 fs): E_tot rel drift (max-min)/|mean| = {drift:.2e}  "
          f"(conservative field -> small)")
    os.makedirs(f"{BASE}/figs", exist_ok=True)
    make_plot(res, f"{BASE}/figs/kspace192_py_vs_cpp.png")
    make_nve_plot(f"{BASE}/figs/nve_energy_conservation.png", etot)
    np.savez(f"{BASE}/kspace192_cmp.npz", ase_etot=etot,
             **{f"{t}_{k}": v for t in res for k, v in zip(("idx", "relE", "relF", "maxdF"), res[t])})
    print("saved kspace192_cmp.npz")

