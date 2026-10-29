# SOG

SOG is a lightweight, framework-agnostic **long-range electrostatics** library
for ML interatomic potentials.  A model learns a per-atom latent charge `q_i`; a
k-space solver computes `E_k = ½ Σ_i v_i q_i` with `v_i = ∂E_k/∂q_i`; the
per-atom potential `v` is fed back through the framework's autograd to produce
the conservative force `F_j = −Σ_i v_i ∂q_i/∂r_j` and the box virial.  The
`k = 0` mode is dropped, so charges must be zero-mean ("neutralized") — a gauge
condition, not an option.

The training-side API (`sog`, `SogKernel`, `Gaussian`) is unchanged.  The
PyTorch-2.x deployment layer lives in the isolated `sog.v2` subpackage (the
legacy package never imports it).

## Start here

| If you want to… | Read |
|---|---|
| **Port SOG into another MLIP** (any framework) | the [port section](#port-sog-into-your-framework-for-agents) below, then [`docs/`](docs/README.md) |
| Drive an **AI agent** to do the port | the [port section](#port-sog-into-your-framework-for-agents) + [`example/mace/`](example/mace/) |
| Just **train** with SOG | [`docs/02`](docs/02-python-api.md) → [`03`](docs/03-training-adapter.md) → [`04`](docs/04-parameter-export.md) |
| Just **run MD** with a trained kernel | [`docs/04`](docs/04-parameter-export.md) → [`05`](docs/05-lammps-plugin.md) |
| **Deploy under PyTorch 2.x** (`sog.v2`) | [`docs/09`](docs/09-in-graph-kspace.md) → [`10`](docs/10-export-and-artifacts.md) → [`11`](docs/11-lr-sync.md) |

---

## Port SOG into your framework (for agents)

SOG touches exactly **four coupling points**.  A port names, for each, the
function/class in the target code that implements it — there is no "just add it"
shortcut.

| # | Coupling | SOG interface | Direction |
|---|---|---|---|
| 1 | Kernel parameters (`amp`, `bw²`, `b`, `σ`, `m`, `n_dl`, `phi_max`) | `sog.SogKernel` | framework → SOG |
| 2 | Per-atom **latent charges** `q` | `LatentChargeProvider` / `SogChargeSource` | framework → SOG |
| 3 | **∂q/∂r and ∂q/∂h** (charge-response VJP) | autograd / `SogChargeResponse` | SOG → framework |
| 4 | Parameter export (training → MD) | `SogKernel.to_lammps_dict()`, `.pt2` metadata | SOG → your MD input |

### Port scope → deliverable

| Scope | You deliver | Read |
|---|---|---|
| Training | `E_lr` in the loss; gradients reach `q`, kernel params, backbone | [`01`](docs/01-overview.md) → [`02`](docs/02-python-api.md) → [`03`](docs/03-training-adapter.md) → [`04`](docs/04-parameter-export.md) |
| MD — fixed charges | `amp`/`bandwidth` + the LAMMPS plugin | [`04`](docs/04-parameter-export.md) → [`05`](docs/05-lammps-plugin.md) |
| MD — latent charges | C++ `SogChargeSource` + `SogChargeResponse` adapters | [`05`](docs/05-lammps-plugin.md) → [`06`](docs/06-cpp-porting.md) |
| Deployment (PyTorch 2.x) | the three traceable protocols + `.pt2` artifacts | [`09`](docs/09-in-graph-kspace.md) → [`10`](docs/10-export-and-artifacts.md) → [`11`](docs/11-lr-sync.md) |

Every port is accepted on **numerical parity**, not "it compiles": run the checks
in [`docs/08-validation.md`](docs/08-validation.md).

### Deployment protocols (`sog.v2`)

| Protocol | Chain |
|---|---|
| `FullLRTraceable` | **Mode B** — 1F+1B single graph |
| `EnergyOnlyTraceable` | **Mode A** — pair forward |
| `ResponseTraceable` | **Mode A** — fix VJP |

**Mode A** = `pair + kspace + fix` (2F+1B; the solver stays in LAMMPS).  **Mode
B** = `pair ... full_lr yes` + `kspace_style none` (1F+1B; the
`sogkspace::solve_auto` op solves *inside* the exported graph and removes the
duplicated model forward).  Method signatures, the two ABIs and the measured
numbers are in [`docs/10`](docs/10-export-and-artifacts.md); the protocols are
defined in [`src/sog/v2/export/abi.py`](src/sog/v2/export/abi.py).

Reference implementations, smallest first:

| What | Where |
|---|---|
| Closed-form, all three protocols (start here) | [`test/fixtures/tiny_charge_model.py`](test/fixtures/tiny_charge_model.py) |
| MACE-shaped migration walkthrough (runnable) | [`example/mace/`](example/mace/) |
| Production reference (DeepMD) | `deepmd-kit-LR` → `deepmd/pt_expt/model/make_model.py` |

### How to drive an agent

1. Give it the framework name, the model source tree, and the scope above.
2. Require a **plan** naming, for each coupling point, the exact function/class
   in the target code.  Reject "implement SOG" without owners.
3. Require the acceptance checklists in [`docs/03`](docs/03-training-adapter.md)
   (training) and [`docs/06`](docs/06-cpp-porting.md) (MD).
4. Require **numerical proof** from [`docs/08`](docs/08-validation.md);
   "it compiles" is not acceptance.

### Non-obvious facts (state these — not derivable from the code)

- `amp` is the **internal** amplitude; `bandwidth` is **bw²** (squared).  The
  C++ term-count keyword is lowercase `m`; the grid keyword is `phi_max` (not
  `cubes2_phi_max`).
- `remove_self_interaction` defaults differ: Python `True`, C++ parser `no`.
  Emit the keyword explicitly on the C++ side.
- The response seed is `Σ v·q`, **not** `½ Σ v·q` (`v` is frozen; the `½`
  belongs to the reported energy only).
- Divide the seed by the **owned** node count (`n_local`), not the total — else
  multi-rank force/pressure are off by `n_local/nnode`.
- Virial order: the op returns `(xx,yy,zz,xy,xz,yz)`; DeepMD wants row-major
  3×3 → map `{0,4,8,1,2,5}`.  Wrong order = pressure off by ~38 %.
- A value-only op on the response path with no autograd rule gives "energy
  right, force wrong" — the `neutralize` VJP ([`docs/09`](docs/09-in-graph-kspace.md)).
- `CXX` must be plain `g++` (not `mpiCC`); the freeze toolchain must equal the
  load toolchain or the AOTI artifact crashes on `dlopen`.
- The transferable unit is **the op + the algebra + the export machinery**; the
  *model adapter* and the *MD-engine adapter* are per-framework.  DeepMD's
  12/20-input ABI is **not** portable.

### Layout

```
src/sog/            legacy Python API (Sog, SogKernel, Gaussian) — training side
src/sog/v2/         PyTorch-2.x deployment layer (isolated)
  exportable.py     trace-safe pure-PyTorch solver (no-CUDA fallback)
  external_kspace.py  E=½Σvq, seed, vir6, rank-share algebra
  kspace_op.py      sogkspace op loader + exact neutralize VJP
  export/           abi (protocols) · trace · aoti · artifacts
op/                 standalone CUDA op build (sogkspace::solve_auto)
lmp/                LAMMPS plugin (kspace sog, fix sog/response, adapters)
docs/               01–08 integration specs · 09–11 deployment
example/mace/       MACE → sog.v2 migration walkthrough (runnable)
test/               pytest; test/v2/ covers the deployment layer
```

Run the suite with `pytest` from the repo root (the root `conftest.py` puts
`src/` on the path for a fresh checkout).

---

## Kernel parameterization (important)

The Gaussian kernel internally uses:

$$
kfac = \sum_m amp^{\text{internal}}_m\,\exp\left(-\tfrac{1}{2}bw^2_m k^2\right)
$$

`Sog` takes two mode keys — `kernel_param_mode` (`"raw"`/`"internal"`: is
`bandwidth` un-squared or already `bw²`?) and `kernel_tensor_mode`
(`"owned"`/`"external"`/`"auto"`: who owns the kernel tensors?).  Defaults are
`"raw"` + `"owned"`; legacy bool keys are mapped automatically.  Full semantics:
[`docs/02`](docs/02-python-api.md).

## Install

```bash
pip install -e .
```

## Quick start

```python
import torch
from sog import Sog

model = Sog({"use_atomwise": False})

# `positions` needs grad when the solver path has no analytic forces
# (the energy is differentiated); `cell` likewise for the virial.
r = torch.rand(16, 3, requires_grad=True)
q = torch.rand(16) - 0.5
q = q - q.mean()
cell = torch.eye(3).unsqueeze(0) * 12.0

out = model(
    positions=r,
    cell=cell,
    latent_charges=q,
    compute_energy=True,
    compute_force=True,
)

print(out["E_lr"], out["forces"].shape)
```

## Parameter examples

### 1) Standard MACE/MACESOG-style usage (raw parameters)

```python
from sog import Sog

model = Sog(
        {
                "use_atomwise": False,
                "kernel_param_mode": "raw",
                "kernel_tensor_mode": "owned",
        }
)
```

### 2) External upstream parameters (DeepMD-style bridge)

Use this when upstream already keeps `amp_internal` and `bw2` and you want gradients
to flow directly to upstream tensors.

```python
from sog import Sog

model = Sog(
        {
                "use_atomwise": False,
                "amp": amp_internal_tensor,
                "bandwidth": bw2_tensor,
                "kernel_param_mode": "internal",
                "kernel_tensor_mode": "external",
                "trainable_kernel": False,
        }
)
```

## MACESOG call correctness notes

`MACESOG` in `mace/modules/extensions.py` initializes SOG as:

```python
self.sog = Sog(sog_arguments=sog_arguments, r_cut=sog_arguments.get("r_cut", None))
```

This is correct with the current API:

- If `sog_arguments` does not specify mode keys, SOG defaults to
    `kernel_param_mode="raw"`, `kernel_tensor_mode="owned"`.
- If needed, `MACESOG` users can pass mode keys directly through `sog_arguments`
    without code changes in `MACESOG`.

## Cross-implementation check (SOG vs DeepMD)

Run the same-input comparison script against the DeepMD reference implementation:

```bash
PYTHONPATH=src conda run -n dp_devel python example/compare_with_deepmd_sog.py \
    --deepmd-source ../dp_pt/dp_devel/deepmd-kit-devel
```

You can also set `DEEPMD_SOURCE_DIR` instead of passing `--deepmd-source`.
