# 09 — The in-graph k-space op (`sog.v2.kspace_op`)

The deployment pattern that the active development tree (deepmd-kit-LR) uses
puts the reciprocal-space solve **inside the exported model graph** as an
opaque PyTorch custom op.  This page is the contract: what the op does, how to
build/load it, and what a consumer must respect.

Everything here lives in the isolated `sog.v2` package; the legacy `sog`
package does not import it.

## Files

| Path | Role |
|---|---|
| `op/sog_kspace/sog_kspace_op.cc` | op registration + `solve_auto` / `neutralize` / `qmean` / `setup` / `destroy` / `last_grid` / `set_comm` |
| `op/sog_kspace/sog_green.h` | grid sizing (`phi_max` rule) + CubeS₂ green tables |
| `lmp/sog_gpu.cu`, `lmp/sog_gpu.cuh` | the device kernels (compiled unchanged by the op) |
| `op/CMakeLists.txt` | standalone CUDA/torch build → `libsog_kspace_op.so` |
| `src/sog/v2/kspace_op.py` | loader + typed wrappers + exact neutralization VJP |

## Build

```bash
cmake -S op -B op/build \
  -DTORCH_DIR=$(python -c "import torch; print(torch.utils.cmake_prefix_path)")/Torch \
  -DCUDAToolkit_ROOT=/usr/local/cuda-12.4 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.4/bin/nvcc \
  -DCMAKE_CXX_COMPILER=$(command -v g++) \
  -DCMAKE_CUDA_HOST_COMPILER=$(command -v g++) \
  -DCMAKE_BUILD_TYPE=Release
cmake --build op/build -j
# → op/build/libsog_kspace_op.so
```

Rules:

- **ABI = libtorch's.**  This is *not* the LAMMPS plugin: do not apply
  `SOG_GLIBCXX_OLD_ABI`.  Use `${TORCH_CXX_FLAGS}` (the CMake project does).
- `CXX` must be a plain `g++`, never `mpiCC`/`mpicxx`.
- `SOG_WITH_MPI=ON` compiles the replicated-grid `MPI_Allreduce` path; without
  it the op solves single-rank.
- The kernels need `atomicAdd(double)` → CUDA arch ≥ sm_60 when building the
  LAMMPS plugin (`lmp/CMakeLists.txt` defaults to 70).

## Loading

`src/sog/v2/kspace_op.py` resolves the library in this order:

1. an explicit argument to `load_op_library(path)`;
2. `$SOG_KS_OP_LIB`;
3. `sog/v2/_lib/libsog_kspace_op.so` (packaged install);
4. in-repo builds: `op/build/libsog_kspace_op.so`, `build_op/...`.

`load_op_library()` is idempotent and returns `False` when nothing is found;
`require_op()` raises `OpUnavailableError` with build instructions.  A library
found but broken raises — it is never silently ignored.  Importing `sog.v2`
does **not** load anything.

## Op schemas

```
sogkspace::solve_auto(int h, Tensor q, Tensor pos, Tensor box,
                      Tensor amp, Tensor bandwidth,
                      float rcut, int spline, float phi_accuracy,
                      float box_tol, float qscale, float self_coeff,
                      bool remove_self) -> Tensor[]        # (v, F_dir, vir6)
sogkspace::neutralize(Tensor q, Tensor n_local) -> Tensor   # value-only
sogkspace::qmean(Tensor q, Tensor n_local) -> Tensor        # value-only
sogkspace::setup(int h, int nx, int ny, int nz, Tensor ge, Tensor gf,
                 Tensor gs, Tensor gv, Tensor gsv) -> ()
sogkspace::destroy(int h) -> ()
sogkspace::last_grid(int h) -> Tensor
sogkspace::set_comm(int comm_ptr) -> ()
```

Conventions:

- `q` is a **1-D** charge vector `[N]`; `pos` is `(N,3)` absolute Cartesian
  coordinates in `[0, L)` with `boxlo = 0`; `box` is `(9,)` row-major `h`
  (`[xx,xy,xz,yx,yy,yz,zx,zy,zz]`) or `(3,3)`; **orthorhombic only**.
- `amp` is the internal amplitude and `bandwidth` is **squared** (gb²), exactly
  `SogKernel.amp_internal()` / `bandwidth_sq()`.
- All tensors are float64 on CUDA.  fp32 raises; a tilted box raises.
- `qscale` is the Coulomb constant (default `14.3996454784255`; LAMMPS metal
  units use `qqr2e = 14.3996` — a 3.3e-8 relative difference).
- Outputs: `v` (per-atom potential `∂E_k/∂q`), `F_dir` (`−∂E_k/∂r` at fixed
  q), `vir6` in LAMMPS order `(xx, yy, zz, xy, xz, yz)`.

## Grid sizing (keep three implementations in lockstep)

```
Δ = φ_max · r_cut
φ_max = min( (ε/C_ν)^{1/p_ν} · σ_min / r_cut ,  0.95 · σ_min/(√2·ξ0)/r_cut )
n_α = max(8, ⌈L_α/Δ⌉), rounded up to a 2/3/5-smooth number
```

CubeS₂ order-4: `C_ν = 1.0e-1, p_ν = 4`; order-6: `C_ν = 2.4e-2, p_ν = 6`
(the paper's reference coefficients, adopted in deepmd-kit-LR `a266642d2`).
The same rule is implemented in `sog.tools.phi_max_rule`, `op/sog_kspace/
sog_green.h` and `lmp/sog.cpp`; changing one without the others makes the
chains disagree.

## `box_tol`

`solve_auto` caches the green tables and the box they were built for; a
relative box drift ≤ `box_tol` (default `1e-3`) with an unchanged grid reuses
them.  This is what keeps NPT from rebuilding a 125³ table every step (≈60 ms).
Use `0` only for exact A/B comparisons: the comparison is `drift <= tol`, so
`0` still reuses when the box is bit-identical.

## Share semantics (multi-rank)

- `E_k = ½ Σ_local v q` — this rank's share;
- `F_dir` — per local node;
- `vir6` — the global value divided by the rank count (so summing over ranks
  recovers the total).

The fused seed divides by the **owned** count, never the total node count
(doing the latter rescales force and virial by `n_local/nnode`).

## Neutralization and its exact VJP

The solver drops k=0, so charges must be zero-mean.  `neutralize` projects onto
the **global owned mean** in one Allreduce and zeroes ghost rows:

```
qn_i = owned_i · (q_i − mean_owned(q)),   mean_owned = (1/N_owned) Σ_{owned} q
```

It is a value-only CUDA kernel.  `sog.v2.kspace_op.register_neutralize_autograd()`
registers the exact VJP:

```
dL/dq_i = owned_i · (g_i − mean_owned(g)),   g = ∂L/∂qn (= v)
```

via `torch.library.register_autograd(..., setup_context=...)`; the backward
calls only `qmean` + elementwise ops, so make_fx / `torch.export` / AOTI can
trace it.  **Never** implement it as a C++ `torch::autograd::Function` (it
executes on fake tensors during tracing).  Without this rule the fused
`sum(v*q)` silently loses the `−mean` term: energy stays right, the force is
wrong.

## Python usage

```python
from sog import SogKernel
from sog.v2.external_kspace import ExternalKSpace, KSpaceSettings

k = SogKernel(r_cut=5.0, b=1.6297670882677646, m=12)
ek = ExternalKSpace(k, backend="op")           # or "auto" / "reference"
sol = ek.solve(q, pos, box)                    # SogSolution(qn, v, F_dir, vir6)
E_k = sol.energy()                             # ½ Σ v q
seed = ek.fused_seed(sol, n_owned=n_local)     # inject into the differentiated energy
```

If CUDA or the op is unavailable, use `backend="reference"` (pure PyTorch,
CPU-traceable) or let `"auto"` fall back — see
[10-export-and-artifacts.md](10-export-and-artifacts.md).
