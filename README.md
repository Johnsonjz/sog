# SOG

SOG is a lightweight long-range interaction plugin library for ML interatomic potentials.

This first version follows the LES-style package layout and uses a Gaussian reciprocal-space core:

- Core module: `src/sog/module/gaussian.py`
- Top-level API: `src/sog/sog.py`
- Optional BEC output: `src/sog/module/bec.py`

## Kernel Parameterization (Important)

The Gaussian kernel internally uses:

$$
kfac = \sum_m amp^{\text{internal}}_m\,\exp\left(-\tfrac{1}{2}bw2_m\,k^2\right)
$$

`Sog` accepts two concise mode keys to describe how your input tensors are interpreted:

- `kernel_param_mode`: how `amp` and `bandwidth` are interpreted
    - `"raw"` (default): input `amp` is raw coefficient and input `bandwidth` is $bw$.
        The library converts internally to $(amp \cdot bw^2, bw^2)$.
    - `"internal"`: input `amp` is already internal amplitude and input `bandwidth` is already $bw^2$.
- `kernel_tensor_mode`: how Gaussian kernel tensors are owned/bound
    - `"owned"` (default): Gaussian owns `nn.Parameter` tensors.
    - `"external"`: Gaussian directly uses caller tensors (useful when gradients must flow to upstream parameters).
    - `"auto"`: choose external binding when `trainable_kernel=False` and differentiable tensor inputs are detected.

### Backward Compatibility

Older bool-style keys are still accepted and mapped internally:

- `amp_is_internal` + `bandwidth_is_squared` -> `kernel_param_mode`
- `use_external_kernel_tensors` -> `kernel_tensor_mode`

Prefer new mode keys for clarity.

## Install

```bash
pip install -e .
```

## Quick start

```python
import torch
from sog import Sog

model = Sog({"use_atomwise": False})

r = torch.rand(16, 3)
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

## Parameter Examples

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

## MACESOG Call Correctness Notes

`MACESOG` in `mace/modules/extensions.py` initializes SOG as:

```python
self.sog = Sog(sog_arguments=sog_arguments, r_cut=sog_arguments.get("r_cut", None))
```

This is correct with the current API:

- If `sog_arguments` does not specify mode keys, SOG defaults to
    `kernel_param_mode="raw"`, `kernel_tensor_mode="owned"`.
- If needed, `MACESOG` users can now pass mode keys directly through `sog_arguments`
    without code changes in `MACESOG`.

## Cross-implementation check (SOG vs deepmd)

Run the same-input comparison script against deepmd reference implementation:

```bash
PYTHONPATH=src conda run -n dp_devel python example/compare_with_deepmd_sog.py \
    --deepmd-source ../dp_pt/dp_devel/deepmd-kit-devel
```

You can also set `DEEPMD_SOURCE_DIR` instead of passing `--deepmd-source`.
