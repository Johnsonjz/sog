# `test/` — Python test suite

Plain pytest, run from the repo root:

```bash
pytest                            # the whole suite (~2 min on CPU)
pytest test/test_sog_kernel.py -v # one file
```

Markers are declared in `pytest.ini`:

| Marker | Meaning |
|---|---|
| `lammps` | needs the LAMMPS binary + plugin — **auto-skips** unless `SOG_LMP_BINARY` is set (see `test_lammps_plugin.py`) |

## Conventions

- One file per subject; pure pytest — no import-time execution, no
  `sys.exit`, no `sys.path` hacks (the repo-root `conftest.py` puts `src/` on
  the path for fresh checkouts).
- Test-only helpers come from `sog.tools.*` (shipped with the package), not
  from ad-hoc copies.
- Numbers under test and the validation methodology are documented in
  `docs/08-validation.md` — keep the two in sync.
- `test_data`-style fixtures must live next to the package data they need
  (e.g. `sog/tools/data/water_charge_192.lmp`), never at absolute paths.

## What's covered

| Area | Files |
|---|---|
| Kernel parameters / LAMMPS export / state dict | `test_sog_kernel.py`, `test_sog_kernel_modes.py` |
| Training integration contract (`build_sog`, batch/charge validation) | `test_integration.py` |
| Solver paths (CubeS₂ FFT vs direct, orders, timing) | `test_cubes2_fft.py`, `test_cubes2_order6.py`, `test_direct_symmetry.py`, `test_small_scale.py` |
| Forces/virial vs finite differences & autograd | `test_force_fd_realspace.py`, `test_explicit_alignment.py`, `test_gaussian_triclinic.py` |
| φ_max grid rule (calibration, held-out verification) | `test_phi_max_rule.py`, `test_phi_max_verify.py`, `test_phi_convergence.py` |
| DeepMD cross-checks | `test_deepmd_alignment.py`, `test_gaussian_external_kernel_tensors.py` |
| LAMMPS plugin (skippable) | `test_lammps_plugin.py` (drives `lmp/tests/`) |
| PyTorch-2.x deployment layer (`sog.v2`) | `test/v2/` — op, neutralization VJP, export/AOTI, algebra (see `docs/09`–`11`) |
