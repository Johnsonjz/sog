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
commit: 156a84d5baeb8bfb5c2f79ac156f3fe6913906d4
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

**The op now requires MPI** (`solve_auto`'s distributed slab path calls
`MPI_Alltoallv`): configure with `-DSOG_WITH_MPI=ON` and an MPI whose
`MPI_Comm` is a pointer (OpenMPI / Intel MPI).  `MPICH`, whose `MPI_Comm` is
`int`, does not compile the vendored `set_comm` cast.  `sog_gpu.cu` reaches
NCCL via `dlopen`, so both builds link `${CMAKE_DL_LIBS}`.

**Self-energy is closed-form** (since `0c3e8186c`, the `156a84d5b` pin):
`E_self = ½·Σq²·g(0)` with `g(0) = Σ_m amp_m (2π bw_m)^(-3/2)`.  It is exactly
`∝ q²` (volume-extensive) and has **zero** strain derivative (no self virial).
The old mesh Riemann sum `Σ_k K/V` was neither, so `sog_gpu_setup` dropped its
two self tables (now 3 pointers) and `lmp/sog.cpp`'s CPU/direct paths no longer
accumulate `diag_sum` / self-stress.  `lmp/sog_gpu_stub.cpp` mirrors the 3-arg
`setup`.

Grid sizing follows the same phi_max rule in all three places — `op/sog_kspace/
sog_green.h` (in-graph op), `lmp/sog.cpp` (old chain) and the Python
`sog.tools.phi_max_rule` — using the paper's reference coefficients
(C₄ = 1.0e-1, C₆ = 2.4e-2).  Keep them in lockstep.

## Patched (LR behaviour + model-agnostic abstraction)

| File here | Difference |
|---|---|
| `lmp/sog.cpp`, `lmp/sog.h` | `SogChargeSource` indirection (`set_charge_source` / `get_charge_source`); no `pair_deepmd` include |
| `lmp/fix_sog_response.cpp/.h` | talks only to `SogChargeResponse` (adapters in `lmp/adapters/`); `thermo_virial = 1` added |
| `lmp/sog_charge_response.h` | extended contract documentation (complete box-strain derivative) |

The patch is `lmp/patches/sog-model-agnostic.diff` (regenerate against the
pinned SHA; `git apply --check` must pass on an unmodified LR tree).

## Deliberately deferred (P2 debt)

These LR features are **not** in the `lmp/sog.cpp` old-chain copy yet (the
in-graph `op/` **does** have the distributed slab mesh, vendored from LR):

- the PPPM-style `compute_distributed` mesh inside `lmp/sog.cpp`
  (`Grid3d`/`Remap`); the standalone op has its own slab solver;
- Kokkos device k-space (`kspace_style sog/kk`);
- the device charge-response branch (`compute_charge_response_edges_gpu`);
- `make_comm_nlist` / multi-rank response artifacts;
- solver-side global latent-charge neutralization inside `lmp/sog.cpp`.

`box_tol` and triclinic acceptance are already present in this fork's
`lmp/sog.cpp`.  New Python-side work lives in the isolated `sog.v2` package and
does not touch `lmp/`'s behaviour beyond the items above.

## Sync history

- `156a84d5b` (2026-11-05): self-energy closed form + zero self virial; op
  gains the distributed slab mesh (MPI required) and compact half-spectrum
  green tables; `sog_gpu_setup` 3-arg; NCCL via `dlopen`.

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
