# SOG

SOG is a lightweight long-range interaction plugin library for ML interatomic potentials.

This first version follows the LES-style package layout and uses a Gaussian reciprocal-space core:

- Core module: `src/sog/module/gaussian.py`
- Top-level API: `src/sog/sog.py`
- Optional BEC output: `src/sog/module/bec.py`

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

## Cross-implementation check (SOG vs deepmd)

Run the same-input comparison script against deepmd reference implementation:

```bash
PYTHONPATH=src conda run -n dp_devel python example/compare_with_deepmd_sog.py \
    --deepmd-source ../dp_pt/dp_devel/deepmd-kit-devel
```

You can also set `DEEPMD_SOURCE_DIR` instead of passing `--deepmd-source`.
