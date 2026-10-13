# SOG LAMMPS plugin

A standalone LAMMPS plugin (`libsog_lmp.so`) providing two styles:

| Style | Purpose |
|---|---|
| `kspace_style sog` | Long-range electrostatics via Sum-of-Gaussians (u-series) decomposition with CubeS₂ Midtown spline charge assignment and an FFT Poisson solver. Also runs an exact direct k-sum (`direct yes`) for validation. |
| `fix ID group sog/response` | The charge-response correction for **latent-charge MLIPs**: adds `F_j = −Σ_i v_i·∂q_i/∂r_j` and its virial using the per-atom potential `v_i = ∂E_k/∂q_i` that the kspace solver produces. (PBC pressure has two terms — `Σ F⊗r` plus the energy's box-deformation derivative; the k-space's box term is computed analytically inside `sog.cpp`, never as `r⊗F`.) Framework-agnostic — see [the two contracts](#the-two-mlip-contracts). |

Neither style knows any model by name. Everything model-specific sits behind two
abstract classes (`SogChargeSource`, `SogChargeResponse`) whose adapters live in
`adapters/`. See `docs/06-cpp-porting.md` for how to write an adapter for a new
framework.

## Files

| File | Origin |
|---|---|
| `sog.cpp`, `sog.h` | Synced from `deepmd-kit/source/lmp/`, + charge-source indirection |
| `sog_spline.h`, `sog_gpu.cu`, `sog_gpu.cuh` | Byte-identical to `deepmd-kit/source/lmp/` |
| `sog_gpu_stub.cpp` | CPU-only stand-ins (clear error if `use_gpu yes` is reached) |
| `fix_sog_response.cpp/.h` | Rewritten to talk only to `SogChargeResponse` |
| `sog_charge_source.h`, `sog_charge_response.h` | The two abstract contracts (new) |
| `adapters/deepmd_charge_source.h` | `pair_style deepmd` as a charge source |
| `adapters/deepmd_charge_response.cpp/.h` | DeepMD VJP adapter (`SOG_WITH_DEEPMD`) |
| `adapters/native_charge_response.h` | Documented no-op (fixed-charge models) |
| `sogplugin.cpp` | `lammpsplugin_init` registration for `plugin load` |

The diff against the canonical deepmd-kit sources is recorded in
[`SYNC_NOTE.md`](SYNC_NOTE.md) (with a `patch.diff`; not applied to deepmd-kit).

## Build

Requirements:

- LAMMPS **source tree** matching the binary that will load the plugin
  (headers only; the FFT3d signature and `LAMMPS_VERSION_NUMBER` must match).
  Tested against LAMMPS v30Mar2026.
- C++17 compiler. **The libstdc++ ABI must match the target binary** — see below.
- No MPI *link* dependency: a plugin resolves MPI symbols from the LAMMPS
  binary at runtime, so only headers are needed (falls back to LAMMPS' own
  `src/STUBS` serial `mpi.h` when no MPI dev install exists).

### Native mode (default)

```bash
cmake -S . -B build \
  -DLAMMPS_SOURCE_ROOT=/path/to/lammps \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -j          # → build/libsog_lmp.so
```

Native mode provides: `kspace_style sog` (all charge paths), and
`fix sog/response` backed by `NativeChargeResponse` (a documented no-op for
fixed-charge models; warns once).

### DeepMD mode

Adds the `pair_style deepmd` charge-source and charge-response adapters:

```bash
cmake -S . -B build_deepmd \
  -DLAMMPS_SOURCE_ROOT=/path/to/lammps \
  -DSOG_WITH_DEEPMD=ON \
  -DDEEPMD_SOURCE_ROOT=/path/to/deepmd-kit/source \
  -DDEEPMD_CC_LIBRARY=/path/to/build/api_cc/libdeepmd_cc.so \
  -DTORCH_DIR=/path/to/libtorch/share/cmake/Torch \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build_deepmd -j
```

This requires **this fork's** deepmd-kit: the adapters call the C++-API
extensions `set_retain_charge_graph`, `compute_charge_response_cached`,
`compute_charge_only_forward`, `compute_combined_response` (see
`api_cc/include/DeepPot.h`), which do not exist in a stock deepmd-kit C-API
build.

### Build knobs that must match the target binary

| Option | Default | Notes |
|---|---|---|
| `SOG_GLIBCXX_OLD_ABI` | `ON` | `-D_GLIBCXX_USE_CXX11_ABI=0`. The integrated LAMMPS binary here was built with gcc-toolset-13 + the old ABI; a stock conda-gcc LAMMPS build uses the new ABI → set `OFF`. A mismatch shows up as `undefined symbol ...Ss...` (std::string in old mangling) at `plugin load`. |
| `FFT` | `KISS` | Must match the target binary's FFT backend. `KISS` links nothing (kissfft lives inside the LAMMPS binary); `FFTW3`/`MKL` add an external dependency. |
| `SOG_WITH_GPU` | `OFF` | Compiles `sog_gpu.cu` (raw CUDA + cuFFT) instead of the stub. |
| `LAMMPS_SMALLBIG` | forced | Must match the target LAMMPS build's size setting. |

## Usage

### Frozen kernel (production path)

```lammps
plugin load /path/to/libsog_lmp.so

kspace_style sog amp <M values> bandwidth <M values^2> \
    spline cubes2_4 remove_self_interaction yes phi_max 0.23 m 12

pair_style deepmd model.pth latent_charge_to_q yes
fix sresp all sog/response model model.pth
```

`amp`/`bandwidth` are the **internal** amplitude and the **squared**
bandwidth (C++ computes `Σ_m amp[m]·exp(−0.5·bandwidth[m]·k²)`), exactly what
`sog.SogKernel.to_lammps_dict()` emits. Both keywords must be given together.

### Auto-generated kernel (legacy positional header)

```lammps
kspace_style sog 1e-6 2.0 3.016 12 n_dl 4.0 spline cubes2_4
```

The leading positional `accuracy b sigma M [n_dl]` header is optional and only
used when the kernel must be **auto-generated** from `(b, σ, M)`. It is
detected by whether `arg[0]` parses as a number, so it must come first. With
`amp`+`bandwidth` supplied these scalars are vestigial (accuracy feeds only
auto-`n_dl`, `b`/`σ` only the unused `w0` and pair compatibility).

### Charge-response fix

```lammps
fix sresp all sog/response [model <path>] [fused yes|no] [pair_deepmd_index <n>]
```

| Keyword | Meaning |
|---|---|
| `model <path>` | TorchScript model path. **Required in deepmd builds**; ignored by the native adapter. |
| `fused yes/no` | The pair ran an energy-only forward (its short-range force was dropped); one combined backward yields the TOTAL short-range + response force. No fallback: a missing retained graph is a hard error. |
| `pair_deepmd_index <n>` | Which `pair_style deepmd` instance to use (default: first). |

In a native build the bare `fix sresp all sog/response` form is accepted:
`NativeChargeResponse` warns once and contributes a zero correction — the
explicit "this framework has no differentiable charge graph" case.

## `kspace_style sog` keyword reference

Defaults are the `settings()` values; the full parser is `sog.cpp:410-656`.

| Keyword | Values | Default | Meaning |
|---|---|---|---|
| `amp` | list of floats | — (required unless auto-generating) | Internal kernel amplitudes `amp[m]`. Stops at the next keyword token. |
| `bandwidth` | list of floats | — | **Squared** bandwidths `bandwidth[m]` (Å²), same length as `amp`. |
| `m` | int ≥ 1 | 1 | Number of Gaussian terms. Overridden by `amp.size()` when `amp` is given. **Lowercase** — `M` is only the positional header's 4th token. |
| `b` | float > 0 | 1.0 | Geometric base for bandwidths (auto-generation only). |
| `sigma` | float > 0 | 1.0 | Base bandwidth σ (Å) (auto-generation only). |
| `n_dl` | float > 0 | auto | k-space cutoff `k_max = 2π/n_dl` (Å). Auto = `π√2·σ/√(ln(1/accuracy))`. |
| `spline` | `bspline`(=5), `bspline_4`, `bspline_6`, `cubes2_4`, `cubes2_6`, `quads_4`, `quads_6` | `cubes2_4` | Charge-assignment scheme. |
| `grid_method` | `sog_bandwidth`, `pppm_legacy` | `sog_bandwidth` | Grid-sizing rule. |
| `phi_max` | float > 0 | auto (Predescu 2020 Table III) | φ = Δ/r_c grid control for CubeS₂/QuadS. Smaller = finer. |
| `phi_accuracy` | float > 0 | auto per order | Target relative energy accuracy ε for the φ_max general method. |
| `mesh_oversample` | float ≥ 1.0 | 1.5 | Mesh padding factor. |
| `mesh_alias_extent` | int ≥ 1 | 8 | Alias-truncation extent (mesh periods). |
| `remove_self_interaction` | `yes/no/1/0/on/off/true/false` | `no` | Subtract the `diag_sum` self term (gates it on the mesh, GPU and `direct` paths alike). **The parser default is `no` while the Python kernel's default is `True`** — `to_lammps_tokens()` always emits the keyword, so a hand-written input that omits it silently drops a term the training side included. |
| `direct` | `yes [kmax]`, `no` | `no` | Exact reciprocal-space sum, no mesh/FFT. `kmax` overrides the auto cutoff (σ_min tail ≲ 1e-12). `kmax = 2π/n_dl` gives the identical mode set to the Python lib. |
| `use_gpu` | `yes/no` | `no` | Plugin-internal CUDA path (`SOG_WITH_GPU` builds only; otherwise a clear abort if reached). |
| `use_finufft` | (any) | — | Accepted for compatibility; ignored. |

Positional header (optional, must be first, only if `arg[0]` is numeric):
`accuracy b sigma M [n_dl]` — note the **uppercase** `M` here; the keyword form
is lowercase `m`.

**Trap:** the pre-2026 keyword `cubes2_phi_max` is *not* accepted — it falls
through to `Unknown sog option`. Use `phi_max`. (`SogKernel.to_lammps_tokens()`
emits the correct name.)

## The two MLIP contracts

Coupling #2 — what the MLIP **gives** SOG (`sog_charge_source.h`):

```cpp
class SogChargeSource {
  virtual int nchannels() const;         // <=1 → use the native atom->q path
  virtual const double *charges() const; // [i*nchannels + ch], i in [0, nlocal)
};
```

Resolution order in `SOGKSpace::compute()`: (1) a source registered via
`SOGKSpace::set_charge_source()`; (2) the deepmd adapter (if compiled in);
(3) `atom->q` — which **any** pair style that writes charges satisfies. The
deepmd adapter guards against `dcharge_multi` being empty (model with
`latent_charge_to_q` off), falling back to channel 1.

Coupling #3 — what SOG **asks back** (`sog_charge_response.h`):

```cpp
class SogChargeResponse {
  virtual double energy_unit_factor() const;  // MLIP energy → LAMMPS
  virtual double force_unit_factor() const;   // MLIP force  → LAMMPS
  virtual int nchannels() const;
  virtual bool fused_short_range() const;
  virtual void response(std::vector<double> &f_ext,   // out [nall*3]
                        std::vector<double> &virial9, // out [9]
                        const double *v_per_atom,     // in [nlocal*nchannels]
                        int nlocal, int nall, int nchannels) = 0;
};
```

`v_per_atom` is passed in MLIP energy units (`v_i = ∂E_k/∂q_i` from the kspace);
outputs come back in MLIP units and `FixSOGResponse` converts them. The fix
keeps every LAMMPS-native duty: locating `kspace_style sog`, enabling
`set_want_potential`, `reverse_comm` of the ghost-inclusive force, `atom->f`
accumulation, and `v_tally` of the 9-component virial.

For DeepMD, `∂q/∂r` and `∂q/∂h` come from Torch via VJP on the pair's retained
forward graph (`compute_charge_response_cached`), with a standalone
forward+backward fallback. The fallback's `deepmd::InputNlist` is assembled by
`FixSOGResponse::build_input_nlist` — the only context that is a friend of
`CommBrickDeepMD` (whose protected comm layout the nlist needs), because
deepmd-kit must not be modified. A new framework supplies its own
`SogChargeResponse` adapter; see `docs/06-cpp-porting.md`.

## Known limits

- **Orthorhombic boxes only** — `init()` rejects triclinic
  (`sog.cpp:792`), 2D, and non-periodic boxes.
- **Single-rank mesh path**: the charge mesh is spread from `nlocal` with no
  reverse-comm, so the FFT/mesh kspace is correct on one MPI rank only (MPI is
  used for energy/virial reductions). The `direct` path reduces its structure
  factors across ranks but is the validation path, not the fast one.
- **`use_gpu yes`** needs an `SOG_WITH_GPU` build; the CPU build's stub prints
  a clear error instead of failing with a dynamic-linker message.
- Plugin `gs_style`/`plugin load` requires a LAMMPS ≥ v30Mar2026 binary with
  plugin support compiled in.

## Tests

Portable inputs + checker in [`tests/`](tests/README.md) (native smoke; direct
k-sum Python↔LAMMPS parity). From the repo root:

```bash
SOG_LMP_BINARY=/path/to/lmp pytest test/test_lammps_plugin.py
```

Auto-skips when no LAMMPS binary is configured; the inputs also run standalone
(`cd tests && lmp -in parity_direct.in … && python parity_direct.py`).

## See also

- Python training side / kernel export: [`../README.md`](../README.md),
  `SogKernel.to_lammps_input()`
- Integration guide (AI-consumable specs): [`../docs/`](../docs/)
- DeepMD worked example:
  `/data/home/public/jiangzhen/dp/dp_example/water/lmp/SOG_WORKFLOW.md`
- Reference: Predescu, Bergdorf, Shaw, J. Chem. Phys. 153, 224117 (2020)
