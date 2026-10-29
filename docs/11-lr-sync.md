# 11 — Mirroring the active tree (deepmd-kit-LR)

SOG's active development happens in
`/data/home/public/jiangzhen/dp/deepmd-kit-LR` (branch `feat/sezm-sog`).
**That tree is the source of truth.**  This repository vendors the generic
parts and records exactly what it took, what it patched, and what it
deliberately left behind.

## The pin

```
repo:   /data/home/public/jiangzhen/dp/deepmd-kit-LR
branch: feat/sezm-sog
commit: a266642d2ae797ef818575d50dbc4c43b8e3d7ca
```

`sog/lmp/SYNC_NOTE.md` is the authoritative record; `scripts/sync_from_lr.sh`
enforces the same pin and warns (non-interactively it continues) when the LR
tree has moved.

## Vendored (byte-for-byte; never hand-edit)

```
lmp/sog_gpu.cu          ← source/lmp/sog_gpu.cu
lmp/sog_gpu.cuh         ← source/lmp/sog_gpu.cuh
lmp/sog_spline.h        ← source/lmp/sog_spline.h
lmp/sog_charge_source.h ← source/lmp/sog_charge_source.h
op/sog_kspace/sog_kspace_op.cc ← source/op/pt/sog_kspace/sog_kspace_op.cc
op/sog_kspace/sog_green.h      ← source/op/pt/sog_kspace/sog_green.h
```

The op builds these together with the vendored `sog_gpu.cu`, so the in-graph
solver is the deployed discretization, not a re-derivation.

## Patched (LR behaviour + the model-agnostic abstraction)

```
lmp/sog.cpp, lmp/sog.h          SogChargeSource indirection; no pair_deepmd include
lmp/fix_sog_response.cpp/.h     talks only to SogChargeResponse; thermo_virial = 1
lmp/sog_charge_response.h       extended contract documentation
```

The patch is `lmp/patches/sog-model-agnostic.diff` (regenerate against the pin;
`git apply --check` must pass on an unmodified LR tree).

## Deliberately deferred (P2 debt)

The standalone plugin does **not** yet carry these LR features; the old chain
keeps working without them:

- `box_tol` (NPT green-table reuse) — the op has it, `lmp/sog.cpp` does not;
- PPPM-style distributed charge mesh (`compute_distributed`, `Grid3d`/`Remap`);
- Kokkos device k-space (`kspace_style sog/kk`);
- the device charge-response branch and `make_comm_nlist`;
- solver-side global neutralization inside `sog.cpp`.

Document them as gaps rather than silently diverging.  The new Python work in
`sog.v2` is additive and does not change `lmp/` behaviour.

## Grid coefficients: never re-type them

The `phi_max` rule and its coefficient values are stated once in
[09-in-graph-kspace.md](09-in-graph-kspace.md) §Grid sizing.  They appear in
`op/sog_kspace/sog_green.h`, `lmp/sog.cpp` and `src/sog/tools/phi_max_rule.py`;
this mirror must keep all three in lockstep — copy from that single source,
never edit one in isolation.

## Sync procedure

```bash
bash scripts/sync_from_lr.sh       # copies the vendored files, never touches LR
git diff --stat                    # review
git apply --check lmp/patches/sog-model-agnostic.diff   # if you re-apply upstream
# rebuild and test:
cmake --build op/build -j                       # in-graph op
cmake --build lmp/build -j                      # standalone plugin
pytest test/ -q
```

When the pin moves:

1. read `git -C <LR> log --oneline <old>..<new> -- source/op/pt/sog_kspace
   source/lmp/sog_gpu.cu source/lmp/sog_gpu.cuh source/lmp/sog.cpp`;
2. if the SOG numerical code changed, update the pin + `SYNC_NOTE.md` +
   `scripts/sync_from_lr.sh` together and rerun the op ↔ reference parity tests
   (`test/v2/test_kspace_op.py`);
3. if only DeepMD glue changed, no action here.

## What lives where (so a sync does not drop the abstraction)

| Concern | Standalone `sog` | DeepMD tree |
|---|---|---|
| Op + device kernels | yes (vendored, built here) | yes (built in-tree) |
| Python algebra / export helpers | `sog.v2` | model-specific call sites |
| `SogChargeSource` / `SogChargeResponse` | yes (used by the fix) | source-only adaptation (fix still DeepMD-coupled) |
| Graph ABI / message passing / pair / fix glue | protocols only | implementations |
| LAMMPS kspace (old chain) | abstracted copy | deployed copy (+ box_tol, distributed, Kokkos) |

The direction is one-way for now: LR → sog.  If the generic pieces in `sog.v2`
ever become the upstream for LR, that is a separate decision recorded in the
sync note.
