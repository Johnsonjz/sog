# `sog.tools` — tooling shipped with the package

Importable helpers used by the test suite and by users validating a kernel.
Nothing here is imported by `import sog` itself — import the submodule you
need explicitly:

```python
from sog.tools.phi_max_rule import phi_max          # grid rule (φ_max from ε)
from sog.tools.verify_phi_max import energy_fft     # FFT-vs-direct spot checks
from sog.tools.fit_phi_max_law import load_water192 # 192-atom reference box
```

| Module | Purpose |
|---|---|
| `phi_max_rule` | The general φ_max accuracy-inverted rule (and its calibrated anchors for CubeS₂ 4/6, QuadS) — the same closed form `sog.cpp`/`SogKernel` use. |
| `verify_phi_max`, `verify_fft_vs_direct`, `verify_cons_phi` | FFT-vs-direct energy/force verification on toy and water panels; shared kernels (`AMP_CONS`, `AMP_MD`, geometric) and `make_test_system`. |
| `fit_phi_max_law`, `calibrate_phi_max_anchors`, `refine_phi_max_constants`, `validate_error_formula`, `verify_anchors` | Calibration chain for the φ_max law (minimal-surface `calibrate()` fits, bisection anchors, error-formula validation). |
| `validate_kspace_192` | C++-vs-Python kspace comparison over a 192-water trajectory (reads an external run dir: set `SOG_SCALE_DIR`). |
| `scan_fft_accuracy`, `compare_fft_forms`, `verify_order6` | Grid/accuracy scans, spline comparisons. |
| `compare` | Minimal external-kernel binding check (internal amp + squared bw). |
| `check_deepmd`, `simulate_mace_sog` | Framework-side smoke scripts (DeepMD eval loading; MACE-style Sog call). |
| `benchmark_cubes2` | CubeS₂+FFT vs direct benchmark (accuracy/speed). |

Notes

- The 192-atom water box used by the φ_max panels ships in `data/`
  (override with `SOG_WATER192_DATA`). Charges are the fixed convention
  O = −0.2, H = +0.1.
- A few calibration scripts still execute at import time and are meant to be
  run as scripts (they print study output); everything the test suite imports
  is import-safe.
