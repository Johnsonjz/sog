# 04 — Parameter export: training → MD

The handoff is one `kspace_style sog ...` input line (coupling #4). Get the
conventions wrong here and the MD silently runs a *different* kernel than the
one you trained — no error, just wrong physics.

## The exporter

```python
from sog import SogKernel

k = SogKernel(r_cut=6.0, b=2.0, m=12, cubes2_phi_max=0.23)
print(k.to_lammps_input())
# kspace_style sog amp 78.80... 316.96... ... bandwidth 9.097... ... m 12 b 2.0 \
#     sigma 3.0162065545167596 spline cubes2_4 remove_self_interaction yes phi_max 0.23
```

(Without `cubes2_phi_max`/`n_dl` the line has no grid keyword and the solver
auto-sizes the mesh — fine for a first run, pin it for production.)

| Method | Returns |
|---|---|
| `to_lammps_dict(include_grid=True)` | dict with `amp`, `bandwidth`, `m`, `b`, `sigma`, `spline`, `remove_self_interaction`, `phi_max`/`n_dl`, plus deprecated aliases |
| `to_lammps_tokens(...)` | the keyword/value token list (for composing your own line) |
| `to_lammps_input(...)` | the full `kspace_style sog ...` line |
| `to_state_dict()` / `from_state_dict()` | versioned hyperparameters + tensors for archival |

## Unit conventions (the traps)

| Quantity | Convention |
|---|---|
| `amp` | the **internal** amplitude — exactly `amp_internal()` |
| `bandwidth` | **bw² — squared**, exactly `bandwidth_sq()` (nm²/Å² as the lib works in) |
| `m` | lowercase keyword for the term count (uppercase `M` exists only in the legacy positional header) |
| grid control | `phi_max` (floats). `cubes2_phi_max` is emitted by `to_lammps_dict()` as a same-value alias for old dict consumers, but the **C++ plugin rejects it** — never write it into a LAMMPS input |
| `remove_self_interaction` | exporter default `True` (emits `yes`); the C++ parser's default is `no` — **always pass the keyword explicitly** |
| `sigma` | `r_cut·nlayers/RCUT_TO_SIGMA`, informational when `amp`/`bandwidth` are given |

`to_lammps_input()` emits the pure keyword form (starts with `amp`). This is
deliberate: a numeric first token is parsed by C++ as the legacy positional
header `accuracy b sigma M [n_dl]` and misreads the rest of the line.

## The complete, canonical handoff

```python
k = trained_model.as_kernel()              # or SogKernel.adopt(model.gaussian)
with open("kernel.state", "w") as f:       # archive (versioned format)
    import json; json.dump(k.to_state_dict(), f, default=str)
print(k.to_lammps_input())                 # paste into the LAMMPS input
```

```lammps
plugin load /path/to/libsog_lmp.so
kspace_style sog amp <M floats> bandwidth <M floats²> m 12 b 2.0 sigma 3.0162 \
    spline cubes2_4 remove_self_interaction yes phi_max 0.23
```

Consistency checks before you trust an MD run:

1. `Σ amp` should be positive and large (it is `K(0)`, the k→0 kernel value).
2. `bandwidth[0] > 0` and the sequence geometric within float precision.
3. The same kernel's Python `E_lr` on the MD start geometry equals the LAMMPS
   `elong` when the k-set is matched — this is the acceptance test in 08.

## What can silently differ

| Trap | Effect |
|---|---|
| `remove_self_interaction` omitted | C++ drops the `diag_sum` self term (`no`), Python included it (`True`) → constant offset per q² |
| `cubes2_phi_max` written by hand | hard error at parser (`Unknown sog option`) |
| bandwidth exported **unsquared** | the whole kernel shape changes; energies still finite |
| legacy positional header written with a stray leading number | parser consumes `amp` as `b` → `sog requires b > 0` |
| `norm_factor` mismatch (14.3996454784255 vs a binary's 14.399645) | uniform 3.32e-8 relative scale on E and F |
| different `n_dl`/`kmax` than the test used | a mesh/truncation difference, not a convention bug — match k-sets before concluding anything |

## Versioned state dict

`to_state_dict()` writes `_bw_format_version = 2`, meaning the stored
`bandwidth` is **pre-square**. `from_state_dict()` up-converts a v1 payload
(squared, unlabelled) via `sqrt(clamp(bw, min=0))` — float-close, not
bit-exact. When sharing kernel files across teams, include the version marker.

## Checklist

- [ ] Line starts with `amp` (no leading number).
- [ ] `amp` is internal, `bandwidth` is squared — compare against
      `amp_internal()`/`bandwidth_sq()` of the same trained object.
- [ ] `remove_self_interaction` written explicitly and equal on both sides.
- [ ] `phi_max` (or `n_dl`) chosen for the target MD resolution; recorded.
- [ ] A 0-step LAMMPS run on a test geometry reproduces Python `E_lr` (see 08).

## Deployment artifacts (`.pt2`)

Besides the `kspace_style sog ...` line used by the old chain, the training
parameters can be frozen into AOTInductor artifacts for the two deployment
chains (`sog.v2.export`).  Names, metadata flags and ABIs are in
[10-export-and-artifacts.md](10-export-and-artifacts.md); the conventions that
still apply are the same:

- the solver consumes `amp` (internal) and `bandwidth` **squared**;
- grid coefficients: see [09-in-graph-kspace.md](09-in-graph-kspace.md)
  §Grid sizing — do not restate them here.
