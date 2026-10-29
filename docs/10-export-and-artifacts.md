# 10 — Exporting artifacts: two chains, one library

An MLIP that adopts SOG has **two** deployment choices.  The library supports
both; the framework owns the pair/fix glue and the graph ABI.

## Mode A vs Mode B

| | **Mode A — pair + kspace + fix** (2F+1B) | **Mode B — full-LR single graph** (1F+1B) |
|---|---|---|
| LAMMPS input | `pair ... latent_charge_to_q yes` + `kspace_style sog` + `fix sog/response` | `pair ... full_lr yes` + `kspace_style none` |
| Artifacts | `sog_energy_only.pt2`, `latent_charge_response.pt2` (+`_with_comm`), `sog_response_combined.pt2` | `sog_full_lr.pt2` (12 inputs), `sog_full_lr_wc.pt2` (20 inputs) |
| Model passes | 2 forwards + 1 backward | 1 forward + 1 backward |
| 12 288 atoms (RTX 4090) | 259.5 ms | **181.4 ms** (−30%) |
| Solver location | LAMMPS C++ (`kspace sog`) | inside the artifact (`sogkspace::solve_auto`) |
| Needs CUDA op | no | yes |
| Works with any pair | yes | requires the pair to run the full-LR artifact |
| Multi-rank | LAMMPS kspace/comm handles it; response needs the with-comm artifact | with-comm full-LR artifact (20 inputs) |

The −30 % is not the solver: the k-space solve is ~1.7 ms of either step.  The
difference is the **duplicated model forward** in Mode A (70.3 ms = 27 % at
12 288 atoms).  Choose Mode A when the pair cannot own the LR graph (no
CUDA op, existing deployment, LAMMPS-managed MPI); choose Mode B when it can.

Both modes compute the same numbers: DeepMD measured Mode-A fused vs non-fused
agreement to 4 digits, and Mode-B vs the dense PyTorch model to
E 2.5e-4 eV / F 4.3e-5 / V 1.3e-3 (the mesh's own accuracy).

## Artifact names and metadata

All entries live under `model/extra/` inside the `.pt2` zip:

| Kind | File | Metadata flag |
|---|---|---|
| `energy_only` | `sog_energy_only.pt2` | (with `response_combined`) `has_fused_response_artifacts` |
| `response_combined` | `sog_response_combined.pt2` | ↑ |
| `response` | `latent_charge_response.pt2` | `has_charge_response_artifact` |
| `response_wc` | `latent_charge_response_with_comm.pt2` | `has_charge_response_with_comm_artifact` |
| `full_lr` | `sog_full_lr.pt2` | `has_full_lr_artifact` |
| `full_lr_wc` | `sog_full_lr_wc.pt2` | `has_full_lr_artifact_wc` |

Output-key records: `fused_energy_only_output_keys`,
`fused_combined_output_keys`, `charge_response_output_keys`,
`full_lr_output_keys`.  The flag names match the DeepMD freeze exactly.

```python
from sog.v2.export import build_metadata, install_artifacts, read_metadata

meta = build_metadata({
    "full_lr": ("energy", "force", "virial"),
    "energy_only": ("energy", "latent_charge"),
    "response_combined": ("energy", "force", "virial"),
})
install_artifacts("model.pt2", {"full_lr": flr_bytes,
                                "energy_only": eo_bytes,
                                "response_combined": rc_bytes}, metadata=meta)
print(read_metadata("model.pt2")["has_full_lr_artifact"])
```

`install_artifacts` rewrites the archive atomically (no duplicate zip entries)
and merges `model/extra/metadata.json`.

## The protocols a framework implements

`src/sog/v2/export/abi.py` defines three structural protocols.  A framework
implements only what its chain needs:

```python
class FullLRTraceable(Protocol):        # Mode B
    def full_lr_sample_inputs(self, *, n_node, device, dtype) -> tuple: ...
    def full_lr_dynamic_shapes(self, *sample_inputs) -> tuple | None: ...
    def trace_full_lr(self, *inputs, tracing_mode="symbolic") -> nn.Module: ...

class EnergyOnlyTraceable(Protocol):    # Mode A pair
    def energy_only_sample_inputs(...) -> tuple: ...
    def trace_energy_only(...) -> nn.Module: ...

class ResponseTraceable(Protocol):      # Mode A fix
    def response_sample_inputs(...) -> tuple: ...
    def trace_response(...) -> nn.Module: ...
```

The SOG contract is only the **positions and ordering** of the SOG-specific
pieces.  The canonical DeepMD ABIs are recorded as constants:

```
SINGLE_RANK_FULL_LR_INPUTS = 10 graph tensors + ("coord","box")          # 12
WITH_COMM_FULL_LR_INPUTS   = 10 graph tensors + 8 comm tensors + coord + box  # 20
FULL_LR_OUTPUT_KEYS        = ("energy","force","virial")
```

(The DeepMD with-comm on-disk ABI has three extra `None` conditioning slots,
so it is 23 inputs; they are not part of the SOG contract.)

## Tracing and exporting

```python
from sog.v2.export import trace_with_make_fx, export_program, compile_artifact

inputs = model.full_lr_sample_inputs(n_node=8)          # framework-defined tensors
traced = trace_with_make_fx(model.full_lr_callable, inputs,
                            tracing_mode="symbolic")
exported = export_program(traced, inputs, dynamic_shapes=None)
payload = compile_artifact(exported, "sog_full_lr.pt2",
                           sample_inputs=inputs, cuda=False,
                           source_fn=model.full_lr_callable)
```

Pipeline (mirrors the DeepMD freeze):

1. `make_fx(..., tracing_mode="symbolic", _allow_non_fake_inputs=True)`;
2. `torch.export.export(..., strict=False, prefer_deferred_runtime_asserts_\
over_guards=True)` (the second kwarg exists only on torch ≥ 2.6; the helper
passes it when available);
3. strip deferred shape assertions (`aten._assert_scalar` → `True`);
4. optional `move_to_device_pass`;
5. AOTInductor compilation.

Dynamic shapes: the **node and edge axes** may be symbolic; the frame axis is
static (the full-LR ABI is single-frame) and `box` is static.  The traced
function must have an explicit parameter list (make_fx/Dynamo cannot re-export
a `*args` GraphModule).

## Torch-version differences

| | torch 2.5 | torch ≥ 2.6 / 2.11 |
|---|---|---|
| Packaging API | `torch._export.aot_compile` → `.so` | `aoti_compile_and_package` → `.pt2` zip |
| Loading | `torch._export.aot_load(so, device)` | `aoti_load_package(pt2)` |
| `prefer_deferred_runtime_asserts_over_guards` | absent | present |
| Inductor keys `scalar_asserts`/`size_asserts`/`assert_indirect_indexing` | absent (filtered out) | present |

`sog.v2.export.aoti` handles all of this: `compile_artifact` picks the API,
filters unknown Inductor keys, and `load_package` dispatches on what is
installed.  The DeepMD C++ loader consumes the `.pt2` zip, so producing
Mode-B artifacts for that consumer requires torch ≥ 2.6.

## Compiler / toolchain

- `CXX` must be `g++` (not `mpiCC`); check with `sog.v2.export.check_toolchain()`.
- g++ ≥ 9 for libtorch; the freeze toolchain must match the *loading*
  environment (the AOTI wrapper is compiled at freeze time).  On this machine:
  `PATH=/opt/rh/gcc-toolset-13/root/usr/bin:$PATH` (and matching `CXX`).
- Inductor configs (`default_inductor_configs`): `scalar_asserts=False`,
  `size_asserts=False`, `assert_indirect_indexing=False`, and on CUDA
  `realize_opcount_threshold=0` (prevents NaN in the backward fusion).

## Failure modes (learned the hard way)

| Symptom | Cause |
|---|---|
| Energy right, force wrong | a **value-only op on the response path** without an autograd rule (`neutralize`); see 09 |
| Force/virial off by `n_local/nnode` | seed divided by total nodes instead of owned count |
| Response force half-size | differentiating `½·sum(vq)` instead of `sum(vq)` (v is frozen) |
| Grid differs from the deployed chain | unsquared `bandwidth` passed to the solver, or phi_max coefficients out of sync |
| `.pt2` loads but pressure is wrong by ~38 % | virial component order: op gives `(xx,yy,zz,xy,xz,yz)`; deepmd wants row-major 3×3 |
| AOTI artifact dlopen crashes | frozen with a different toolchain than the loader |

The reference fixture (`test/fixtures/tiny_charge_model.py`) implements all
three protocols with closed-form physics and is the smallest working example.
