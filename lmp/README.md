# SOG KSpace Plugin for LAMMPS

Standalone LAMMPS plugin providing `kspace_style sog` — long-range electrostatics
via Sum-of-Gaussians (u-series) decomposition with CubeS₂ Midtown spline charge
assignment and FFT-based Poisson solver.

## Build

```bash
cmake -S . -B build \
  -DLAMMPS_SOURCE_ROOT=/path/to/lammps_stable \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
# → build/libsog_lmp.so
```

Requires LAMMPS headers (v30Mar2026 or compatible) and an FFT library (FFTW3 or MKL).

## Usage

```lammps
plugin load /path/to/libsog_lmp.so

kspace_style sog 1e-6 \
    cubes2_phi_max 0.23 \
    spline cubes2_4 \
    b 2.0 sigma 2.18 M 12 \
    amp <M values> bandwidth <M values> \
    remove_self_interaction yes

pair_style deepmd model.pth latent_charge_to_q yes
```

### Key Parameters

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `accuracy` | float | (required) | Target force accuracy |
| `cubes2_phi_max` | float | auto | φ = Δ/r_c grid control. Auto from Predescu 2020 Table III |
| `n_dl` | float | 1.0 | Legacy grid density (deprecated, use cubes2_phi_max) |
| `spline` | keyword | bspline | `bspline`, `cubes2_4`, or `cubes2_6` |
| `b` | float | 1.6298 | Geometric base for bandwidths |
| `sigma` | float | 2.1802 | Base bandwidth (Å) |
| `M` | int | 12 | Number of Gaussian levels |
| `amp` | list | auto | Kernel amplitudes (M values) |
| `bandwidth` | list | auto | Kernel bandwidths squared (M values) |

### φ_max Reference (Predescu 2020 Table III)

| Spline | Nodes | φ_max (b=2) | φ_max (b≈1.63) |
|--------|-------|-------------|-----------------|
| cubes2_4 | 32 | 0.23 | 0.065 |
| cubes2_6 | 88 | 0.35 | 0.160 |

φ = Δ / r_c. Smaller φ → finer grid → higher accuracy, slower.
Auto-default uses the table value (guarantees mesh error ≤ u-series error).

## Feature Comparison

| Feature | mesh FFT | FINUFFT |
|---------|----------|---------|
| MPI parallel | single-rank | single-rank |
| Triclinic | ✗ | ✗ |
| CubeS₂ splines | ✓ (cubes2_4, cubes2_6) | N/A |
| B-spline (5th) | ✓ | N/A |
| Speed | fast (FFT) | moderate |

## See Also

- Python library: `pip install sog` → training with autograd
- sog-lmp-report: `/data/home/public/jiangzhen/dp/dp_example/water/lmp/sog-lmp-report/`
- Reference: Predescu, Bergdorf, Shaw, J. Chem. Phys. 153, 224117 (2020)
