# 08 — Proving a port is correct

"Compiles and runs" is not evidence. This page lists the checks that *are*,
with the reference numbers a correct port reproduces.

## 1. Python ↔ C++ parity on the exact direct k-sum

The strongest cross-language test: both sides sum the same k-vectors, so any
difference is a convention bug, not grid error.

```bash
cd sog/lmp/tests
lmp -in parity_direct.in -log parity_direct.log   # needs the plugin built (native mode ok)
python parity_direct.py                           # prints PASS/FAIL, exit code 0/1
# or: SOG_LMP_BINARY=/path/to/lmp pytest test/test_lammps_plugin.py
```

Setup (both files, read them): 32-atom rocksalt, random-displaced (see below),
frozen kernel `SogKernel(r_cut=6.0, b=2.0, m=12)`, `pair_style zero` so
`pe == elong`, k-set pinned to the same 16 half-sphere modes on both sides.

**Measured result** (2026-10-13, LAMMPS v30Mar2026 integrated binary):

| Quantity | Difference | Relative |
|---|---|---|
| Energy (`pe` vs `E_lr`) | 4.4e-16 eV (on 1.25 eV) | machine precision |
| Forces (max over 32 atoms) | 6.1e-16 eV/Å (on max \|F\| 0.48 eV/Å) | 1.3e-15 |
| Pressure tensor, all 6 components | ≤ 1.8e-11 bar (on ~10⁴ bar) | ~1e-15 |

Three traps this test was built to expose — expect to hit all three in a port:

- **LAMMPS unit constants.** This tree's metal units use
  `qqr2e = 14.399645` and `nktv2p = 1.6021765e6` (older CODATA values), while
  the Python lib defaults to `norm_factor = 14.3996454784255`. Mismatch shows
  as a *uniform* 3.3225e-8 relative error on energy **and** forces with the
  same factor — tell-tale signature. Pin the Python side to the binary's
  constants for parity work (the script does).
- **Matching k-sets.** `direct yes kmax` with `kmax = 2π/n_dl` gives exactly
  the Python `nk = floor(L/n_dl)` spherical cutoff. Other `kmax` values give a
  different mode set → differences that are *not* bugs.
- **Symmetry-zero forces.** Every rocksalt site is an inversion center; at
  perfect geometry all forces are ~1e-15 and the force check is vacuous.
  `parity_direct.in` applies `displace_atoms all random 0.15 ...` for this
  reason. If your test system is symmetric, break it.

Also mind this tree's `compute_p` component order: raw
`[xx yy zz xy xz yz]` from the thermo vector (verified against the source, not
the manual's `pyz pxz pxy` ordering) — label accordingly.

## 2. Finite-difference force check (any port, any path)

Displace one atom by ±δ (δ ≈ 1e-4 Å), re-run `run 0`, central-difference the
`pe`, compare to that atom's force:

```
F_i ≈ −(E(+δ) − E(−δ)) / (2δ)
```

Pass criterion: relative agreement ~1e-5 or better with a converged k-space
(`phi_max` small enough that mesh error ≪ FD error — tighten until the number
stops moving). This is the **only** check that validates the whole
`vpot → VJP → f_ext` chain including signs, so run it on the response port
before any long MD.

Virial version: strain the box at fixed scaled coordinates and compare
`∂E/∂ε` against `compute p` (component mapping above). Two rules, both
learned the hard way:

- The reference derivative must be the **full** strain derivative (both
  pressure terms: `Σ F⊗r` and the box-deformation term `(∂U/∂h)|_r·hᵀ`) —
  differentiating the energy w.r.t. `h` at fixed scaled coordinates delivers
  both at once. Reconstructing the k-space virial from forces alone
  (`Σ r⊗F`) is incomplete by construction — `sog.cpp` says so verbatim
  ("A Σ r⊗F formulation is therefore INCOMPLETE") — and an `F⊗r`-only
  reference would validate the same mistake twice.
- For the charge-response part, an `F⊗r` response virial is complete only for
  charge graphs that take no explicit box input (see 06 §"The math"); if
  yours does, the FD-vs-reference check here is exactly what catches the
  missing term.

## 3. Accuracy scans (mesh paths)

For the CubeS₂/FFT path (production), the family is:

- `phi_max` ↓ → grid ↑ → energy error ↓, cost ↑. Scans in the φ_max notes /
  `sog.tools` calibrate the bisected φ_max values (Predescu 2020 Table III
  baseline; the auto-default guarantees mesh error ≤ u-series error).
- Cross-check any mesh setting against `direct yes` on a small cell — the
  direct path is the reference; keep the difference below your target.

Reference points from this repo's test suite: CubeS₂+FFT vs direct on a 12-atom
cell ≈ 0.74 % energy difference at the default `phi_max` (fine for MD, not for
parity work — use `direct` for parity).

## 4. Degradation checks (no-graph frameworks)

- Native build + bare `fix sresp all sog/response`: must print the one-time
  warning and contribute zero (executed: `native_smoke.in`, clean 2-step run).
- Latent-charge build with `fused yes` and no retained graph: must error
  loudly, not drop the short-range force.
- Multi-rank: mesh path is single-rank-correct by design (documented limit);
  the `direct` path reduces structure factors — use it for any MPI check.

## 5. Known-broken / avoid

- `BEC.forward(output_index=...)` raises `IndexError` (empirically reproduced
  2026-10-13; `sog/module/bec.py:53-62` selects a component but the subsequent
  grad/phase combine still assumes the full 3-vector). Use `output_index=None`
  — the full BEC — in ports. Not fixed in this pass; flagged only.
- `use_finufft` is accepted and ignored by the C++ parser.
- Non-canonical batches produce silently wrong energies; `validate_batch` is
  the guard (02).

## 6. Running the test suite

The tooling is plain pytest from the repo root (`pytest.ini` declares the
markers; repo-root `conftest.py` bootstraps `src/` for fresh checkouts):

```bash
pytest                            # whole suite (~2 min, CPU-only)
pytest -m lammps                  # LAMMPS integration only (needs SOG_LMP_BINARY)
SOG_LMP_BINARY=/path/to/lmp pytest test/test_lammps_plugin.py
```

Layout and conventions: `test/README.md`. The LAMMPS inputs + checker live in
`lmp/tests/` (`lmp/tests/README.md`).

Regression history:

| Run | Result |
|---|---|
| Before the refactor | 26 passed, 7 failed (stale call signatures), 2 script-style files not collectable (the `phi_max` pair) |
| After the repairs | 7 stale failures repaired; new tests added (kernel/export/forward/validation); the `phi_max` pair converted to real pytest tests (calibration + held-out) |
| Note | `test_deepmd_alignment` *skips*: the deepmd reference bridge is stale against this deepmd-kit (`SOGEnergyModel._compute_sog_frame_correction_bundle` does not exist) — a bridge gap, not a SOG regression |

## 7. The `sog.v2` deployment layer

New tests live in `test/v2/` (markers `v2`, `torch_op`, `cuda`, `aoti`):

| File | Check |
|---|---|
| `test_import_isolated.py` | `import sog` does not import `sog.v2`; `sog.v2` imports without CUDA/op; legacy API intact |
| `test_exportable_kernel.py` | analytic `v/F/vir6` equal autograd (≤1e-10); `E = ½Σvq`; batch invariance; `make_fx`-traceable |
| `test_external_kspace_algebra.py` | energy identity, seed semantics, vir6 mappings, global-mean neutralization + ghost zeroing, `eager_efv` vs FD |
| `test_kspace_op.py` | schema, grid equals the φ_max rule, op vs Python CubeS₂ + direct reference, fp32/triclinic rejected |
| `test_neutralize_vjp.py` | exact VJP equals explicit autograd; ghost rows zero; idempotent registration |
| `test_export_pipeline.py` | full-LR trace/export, Mode-A artifacts, metadata/zip layout, AOTI compile+load |

Measured on the 64-atom/12 Å fixture: op vs Python CubeS₂ energy identical
(1e-16), force ≈ 3–5e-4 relative, virial ≈ 1e-4 relative (mesh
discretization); op vs the static direct reference agrees at the same level.
The AOTI smoke compiles with `gcc-toolset-13` and matches eager to 1e-8.

**Known issue found while validating:** the legacy `Gaussian` CubeS₂ autograd
virial does not include the self-interaction strain term (toggling
`remove_self_interaction` leaves it unchanged).  The deployed C++ mesh and the
in-graph op use the complete `W_ks + W_self` form; the `sog.v2` reference
implements the same complete form and was verified against autograd to 1e-12.
Use the reference/op for virial comparisons, not the legacy autograd path.

Run the deployment tests explicitly:

```bash
pytest test/v2 -q                       # skips op/AOTI when unavailable
SOG_KS_OP_LIB=/path/libsog_kspace_op.so pytest -m torch_op
PATH=/opt/rh/gcc-toolset-13/root/usr/bin:$PATH pytest -m aoti
```
