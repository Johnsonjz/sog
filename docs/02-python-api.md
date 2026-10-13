# 02 — Python API: `Sog`, `SogKernel`, `Gaussian`

Three objects; know which one you need:

| Object | Role | You touch it when… |
|---|---|---|
| `sog.Sog` | the model: `(positions, cell, latent_charges) → {"E_lr", ...}` | always (training call site) |
| `sog.SogKernel` | kernel-parameter owner + export/state-dict (couplings #1, #4) | building/exporting a kernel; binding your framework's tensors |
| `sog.module.gaussian.Gaussian` | the solver core (mesh/NUFFT/direct paths) | rarely — only if you drive it directly (DeepMD's fitting net does) |

`Gaussian` is what `Sog` wraps; a port that only needs energies should never
call it directly.

## `Sog` — the model

```python
from sog import Sog

model = Sog({"use_atomwise": False, "r_cut": 6.0, "b": 2.0, "m": 12})
out = model(positions=r,      # [N, 3]
            cell=cell,        # [nf, 3, 3], or None for non-periodic
            latent_charges=q, # [N, nq]
            batch=batch,      # [N] int64, canonical layout (see validate_batch)
            compute_energy=True)
E_lr = out["E_lr"]            # [nf]
```

`forward` keys (additive — only `E_lr`/`latent_charges`/`BEC` always present):

| Key | When | Meaning |
|---|---|---|
| `E_lr` | `compute_energy=True` | long-range energy `[nf]` |
| `latent_charges` | always | the charges echoed back (mean-removed if `charge_neutral`) |
| `BEC` | `compute_bec=True` | Born effective charges |
| `forces` | `compute_force=True` | `[N, 3]` explicit forces, or `None` if the path has no analytic derivatives |
| `virial` | `compute_virial=True` | `[nf, 3, 3]` or `None` |
| `used_explicit_derivatives` | with force/virial | whether the values are analytic or autograd |

**Critical rule:** `forces=None` (or `used_explicit_derivatives=False`) does
**not** mean "no derivatives". Every path is built from differentiable ops, so
the correct fallback is autograd through `E_lr`. Do not treat `None` as an
error and do not silently skip the force.

Construction arguments come from `sog_arguments` (dict), matching the DeepMD
config keys; `kernel=` takes a `SogKernel`. `Sog.as_kernel()` exposes a
read-only kernel view of an already-built `Sog` (no parameter duplication).

## `SogKernel` — parameters and export

```python
from sog import SogKernel

k = SogKernel(r_cut=6.0, b=2.0, m=12)          # owned parameters (nn.Parameter)
k = SogKernel(b=2.0, sigma=3.0, m=12)          # or derived from (b, σ, m)
k.init_from_rcut(6.0, nlayers=2)               # regenerate from a cutoff
```

| Method | Purpose |
|---|---|
| `amp_internal()` | internal amplitude tensor (what C++ calls `amp`) |
| `bandwidth_sq()` | **squared** bandwidths, never negative (what C++ calls `bandwidth`) |
| `init_from_rcut(r_cut, nlayers, b=, m=)` | regenerate σ/`bw²`/amp; raises in external mode |
| `bind_external(amp, bw, amp_is_internal=True, bandwidth_is_squared=False)` | bind your framework's tensors by reference — gradients reach them |
| `SogKernel.adopt(gaussian)` | read-only view over an existing `Gaussian` |
| `to_lammps_dict()` / `to_lammps_tokens()` / `to_lammps_input()` | coupling #4 (see 04) |
| `to_state_dict()` / `from_state_dict()` | versioned round-trip (`_bw_format_version`) |

Two tensor modes, reusing the existing semantics:

- `owned` (default): kernel creates `nn.Parameter`s — pre-square bandwidth
  (`_bandwidth_is_squared=False`), the negativity-safe choice.
- `external`: binds caller tensors; `bandwidth_sq()` squares unless
  `bandwidth_is_squared=True`. Nothing is registered, so no duplicate
  `state_dict` keys.

Composition with `Gaussian` needs no changes to `Gaussian`: `Sog` builds it in
external mode with `trainable=False`.

## `integration.py` — the wiring helpers

```python
from sog.integration import (
    build_sog, lr_energy, validate_batch, validate_charges,
    LatentChargeProvider, AtomwiseChargeProvider, ChargeNeutralityPolicy,
)
```

- `build_sog(mode="owned"|"external", r_cut=..., amp=..., bandwidth=..., ...)`
  — one call for either integration mode (see 03).
- `lr_energy(sog, positions, cell, latent_charges, batch=None, ...)` — the
  Mode-A helper: validates charges/batch, returns `E_lr [nf]`. Add it to your
  total energy; **do not detach the charges**.
- `validate_batch(batch, n_atoms)` — enforces the canonical layout SOG's fast
  paths assume (contiguous per-frame blocks, ids `0..nf−1`, no empty frames).
  A violation otherwise yields *wrong energies*, not an error. Call it at your
  integration boundary.
- `validate_charges(q, n_atoms, n_channels)` — shape/finiteness/channel count.
- `LatentChargeProvider` — a `typing.Protocol`: your framework implements
  `n_channels` + `__call__(...) -> [N, nq]`. SOG never inspects the descriptor.
- `AtomwiseChargeProvider(n_in, n_out, n_layers, n_hidden)` — reference
  provider (MLP over descriptors); also the drop-in when you already have
  per-atom descriptors.
- `ChargeNeutralityPolicy(k0_excluded=True, hard_in_model=False, soft_lambda=None)`
  — records which of the three neutrality mechanisms are active:
  k=0 always excluded (no switch), hard per-frame mean subtraction
  (`Sog(charge_neutral=True)`), soft penalty (`charge_neutral_lambda`).

## Failure modes to know

| Symptom | Cause |
|---|---|
| energies subtly wrong on batches | non-canonical `batch` — call `validate_batch` |
| force mismatch with FD | `forces` came back `None` and autograd was skipped |
| shape error in `build_mlp`/`Atomwise` | `n_hidden` must have `n_layers−1` entries |
| BEC shape/index errors | `compute_bec` with `output_index` is broken (raises; see 08) — use the full BEC |
| kernel not trainable | `kernel_tensor_mode="external"` with non-differentiable inputs |
