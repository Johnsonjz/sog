# SOG integration guide

SOG is long-range electrostatics for MLIPs: a Sum-of-Gaussians (u-series)
decomposition solved in k-space, with latent (learned) per-atom charges. This
directory is written to be handed to an **AI coding agent** that is porting SOG
into a new framework — every porting page is a specification: exact interface,
invariants, minimal working example, failure modes, checklist.

## What SOG needs from your framework

Exactly four coupling points. Each has one interface and one contract:

| # | Coupling | Interface | Direction |
|---|---|---|---|
| 1 | Kernel parameters (`b`, σ, `m`, `n_dl`, `phi_max`, `amp`, `bw²`) | `sog.SogKernel` | framework → SOG |
| 2 | Per-atom **latent charges** | `LatentChargeProvider` (Python) / `SogChargeSource` (C++) | framework → SOG |
| 3 | **∂q/∂r and ∂q/∂h** (the charge-response VJP) | autograd (Python) / `SogChargeResponse` (C++) | SOG → framework |
| 4 | Parameter export training→MD | `SogKernel.to_lammps_dict()` | SOG → your LAMMPS input |

Read [01-overview.md](01-overview.md) first — one page of physics plus the
data-flow picture.

## Pick your path

**Training only** (add `E_lr` to your loss; MD handled elsewhere):
[02-python-api.md](02-python-api.md) → [03-training-adapter.md](03-training-adapter.md)
→ [04-parameter-export.md](04-parameter-export.md)

**MD only** (you already have trained kernel parameters):
[04-parameter-export.md](04-parameter-export.md) → [05-lammps-plugin.md](05-lammps-plugin.md)
→ [06-cpp-porting.md](06-cpp-porting.md) (only if your charges are latent)

**Both** (full port): read 01 → 02 → 03 → 04 → 05 → 06 in order, then
[07-pipeline.md](07-pipeline.md) and prove it with [08-validation.md](08-validation.md).

## How to use these docs with an AI agent

1. Point the agent at `01-overview.md` + the one porting page for its task
   (`03` for training, `06` for the LAMMPS side) **and at your model's source**.
2. Ask it to produce a **plan** naming, for each coupling point, the function
   in *your* code that implements it. The checklists at the end of 03/06 are
   the acceptance criteria.
3. Have it run the checks in [08-validation.md](08-validation.md) — numerical
   parity, not "it compiles". The reference numbers in that file are what a
   correct port reproduces.

Context the agent must be told explicitly (not derivable from your code):

- The C++ side's `amp` is the **internal** amplitude and `bandwidth` is
  **bw²** (squared), while the keyword form for the term count is lowercase
  `m` and the grid keyword is `phi_max` (not `cubes2_phi_max`).
- `remove_self_interaction` defaults **differently** on the two sides
  (Python `True`, C++ parser `no`).
- The charge-response fix (`fix sog/response`) is what turns `v_i = ∂E_k/∂q_i`
  into the conservative force; the framework supplies `∂q/∂r` via its own
  autograd. Do not hand-derive this from the pair style.

## Reference implementations

| What | Where |
|---|---|
| DeepMD training adapter | `dp/deepmd-kit/deepmd/pt/model/model/sog_model.py`, `task/lr_fitting.py` |
| MACE training adapter | `mace/mace/modules/extensions.py` |
| DeepMD MD-side adapter (C++) | `sog/lmp/adapters/deepmd_charge_response.*` |
| DeepMD worked example (build + input files) | `dp/dp_example/water/lmp/SOG_WORKFLOW.md` |
| Plugin reference manual | `sog/lmp/README.md` |
| Canonical sync patch (deepmd-kit) | `sog/lmp/SYNC_NOTE.md` + `patches/` |
