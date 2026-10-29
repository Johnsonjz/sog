# Sync note: `sog/` ↔ `deepmd-kit-LR`

The SOG code in this repository mirrors the active development tree
`/data/home/public/jiangzhen/dp/deepmd-kit-LR` (branch `feat/sezm-sog`).
**LR is the source of truth**; this repository vendors/ports the generic parts
and keeps an explicit record of what is copied verbatim, what is patched, and
what is deliberately deferred.

## Pinned upstream

```
repo:   /data/home/public/jiangzhen/dp/deepmd-kit-LR
branch: feat/sezm-sog
commit: a266642d2ae797ef818575d50dbc4c43b8e3d7ca
```

Update the pin only together with `scripts/sync_from_lr.sh` and a full test run.

## Vendored byte-for-byte (never hand-edit)

| File here | Source in LR |
|---|---|
| `lmp/sog_gpu.cu` | `source/lmp/sog_gpu.cu` |
| `lmp/sog_gpu.cuh` | `source/lmp/sog_gpu.cuh` |
| `lmp/sog_spline.h` | `source/lmp/sog_spline.h` |
| `lmp/sog_charge_source.h` | `source/lmp/sog_charge_source.h` |
| `op/sog_kspace/sog_kspace_op.cc` | `source/op/pt/sog_kspace/sog_kspace_op.cc` |
| `op/sog_kspace/sog_green.h` | `source/op/pt/sog_kspace/sog_green.h` |

The op library is built from `op/CMakeLists.txt` and compiles the vendored
`lmp/sog_gpu.cu` unchanged, so the in-graph solver reproduces the deployed
discretization exactly.

Grid sizing follows the same phi_max rule in all three places — `op/sog_kspace/
sog_green.h` (in-graph op), `lmp/sog.cpp` (old chain) and the Python
`sog.tools.phi_max_rule` — using the paper's reference coefficients
(C₄ = 1.0e-1, C₆ = 2.4e-2, adopted in LR `a266642d2`).  Keep them in lockstep.

## Patched (LR behaviour + model-agnostic abstraction)

| File here | Difference |
|---|---|
| `lmp/sog.cpp`, `lmp/sog.h` | `SogChargeSource` indirection (`set_charge_source` / `get_charge_source`); no `pair_deepmd` include |
| `lmp/fix_sog_response.cpp/.h` | talks only to `SogChargeResponse` (adapters in `lmp/adapters/`); `thermo_virial = 1` added |
| `lmp/sog_charge_response.h` | extended contract documentation (complete box-strain derivative) |

The patch is `lmp/patches/sog-model-agnostic.diff` (regenerate against the
pinned SHA; `git apply --check` must pass on an unmodified LR tree).

## Deliberately deferred (P2 debt)

These LR features are **not** in this repository yet; the old chain remains
unchanged and functional without them:

- `box_tol` keyword (NPT green-table reuse);
- PPPM-style distributed mesh (`compute_distributed`, `Grid3d`/`Remap`);
- Kokkos device k-space (`kspace_style sog/kk`);
- the device charge-response branch (`compute_charge_response_device`);
- `make_comm_nlist` / multi-rank response artifacts;
- solver-side global latent-charge neutralization inside `sog.cpp`.

New Python-side work lives in the isolated `sog.v2` package and does not touch
`lmp/`'s behaviour beyond the items above.

## How to sync

```bash
bash scripts/sync_from_lr.sh            # copies the vendored files
git diff --stat                         # review
# rebuild + test (see lmp/README.md and docs/08-validation.md)
```

The script refuses to modify the LR tree; it only copies out of it.

## The `build_input_nlist` subtlety (unchanged)

`deepmd::InputNlist` needs the comm-brick layout (`nswap`, `sendnum`, …),
which `CommBrickDeepMD` exposes only to `friend class PairDeepMD` and
`friend class FixSOGResponse` (`deepmd-kit/source/lmp/pair_deepmd.h`).  The
generic adapter cannot read it, so the access lives in
`FixSOGResponse::build_input_nlist()` and the DeepMD adapter calls it through a
back-pointer.  Keep that split when syncing, or widen the friendship upstream.
