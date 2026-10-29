# 01 — What SOG is, and where it plugs in

## The physics in one page

An MLIP gives you short-range interactions and per-atom **latent charges**
q_i (learned, not fixed). The electrostatic energy those charges would have —
the long-range part, Coulomb in the far field — is computed as an Ewald-type
reciprocal-space sum whose kernel is a sum of Gaussians:

```
K(k²) = Σ_m  amp[m] · exp(−½ · bandwidth[m] · k²)          (squared bandwidths)

E_lr  = norm_factor/(2V) · Σ_{k≠0} K(k²) |ρ(k)|²  −  self-energy
   ρ(k) = Σ_i q_i e^{−i k·r_i}
```

Key facts:

- **u-series decomposition**: the sum of Gaussians *is* the erfc-splitting of
  1/r, fitted rather than truncated. `m` terms, geometric in bandwidth with
  base `b`. The default construction derives all parameters from a real-space
  cutoff: `σ = r_cut · nlayers / RCUT_TO_SIGMA` (RCUT_TO_SIGMA ≈ 1.98925...),
  `bw²[0] = σ²`, `bw²[m] = bw²[m−1]·b²`, and amplitudes
  `amp[0] = 4π·ln(b)·w0·σ²`, geometric thereafter.
- **norm_factor** = 14.3996454784255 eV·Å/e² (e²/4πε₀). A LAMMPS build may use
  the older `qqr2e = 14.399645` — a 3.3e-8 relative difference; see 08.
- **k=0 is always excluded.** For a net-charged system that term is silently
  gone, hence the neutrality machinery in 02.
- The charges are **differentiable**. The conservative force therefore has a
  term the pair style cannot produce: `F_j = −Σ_i v_i · ∂q_i/∂r_j`, with
  `v_i = ∂E_k/∂q_i` computed by the kspace solver. That is coupling #3, and it
  is why the C++ side needs an abstract charge-response interface rather than a
  fixed potential.

Everything is differentiable end-to-end (PyTorch), so training needs no
special library: `E_lr` is just another term in the loss, with gradients to
`q`, to the kernel parameters, and through `∂q/∂r` to the model.

## Data flow

```
                     ┌──────────────── your MLIP ────────────────┐
   positions, cell   │  descriptor/backbone  →  charge head      │
        ─────────────┼──────────────────────────────────────────►│─┐
                     │                                           │ │ q_i  [N, nq]
                     └───────────────────────────────────────────┘ │
                                                                   ▼
   coupling #1: kernel parameters (SogKernel) ──────────────►  ┌─────────┐
   coupling #4: to_lammps_dict() ──► MD input                 │   SOG   │
                                                              │  k-space│──► E_lr
                     ┌─────────────────────────────────────────┤ solver  │
   coupling #3:      │  v_i = ∂E_k/∂q_i  ────► your autograd ──┤         │
   ∂q/∂r, ∂q/∂h ◄────┘  (force/virial, incl. the response)    └─────────┘
```

In training (Python) couplings #1–#3 all ride on one autograd graph: SOG
returns `E_lr` (and optionally explicit `forces`/`virial`), the framework's own
backward produces everything.

In MD (LAMMPS) the graph lives in the C++/Torch boundary:

- **Python side** (once): export the frozen kernel as `kspace_style sog amp ...
  bandwidth ...` (coupling #4).
- **C++ side** (per step): the pair style writes the latent charges
  (coupling #2, `SogChargeSource`); the kspace computes `v_i`; the fix asks the
  framework for the VJP at fixed `v_i` and adds the resulting force
  (coupling #3, `SogChargeResponse`).

## Solver paths (what the numbers come from)

| Path | Where | Use |
|---|---|---|
| CubeS₂ Midtown spline + FFT mesh | C++ default (`spline cubes2_4`); Python `use_cubes2_fft` | production MD |
| B-spline charge assignment | C++ `spline bspline_4/_5/_6` | legacy comparison |
| Exact direct k-sum | C++ `direct yes [kmax]`; Python `n_dl`/no-FFT paths | **validation** — the cross-language reference |
| NUFFT / QuadS | Python training extras | experiments |

Python and C++ agree on the direct path to machine precision when the k-sets
and unit constants match (verified: 4e-16 eV on energy, 1.3e-15 relative on
forces — see 08).

## What SOG is *not*

- Not a pair style: it never touches the short-range force (except in `fused`
  mode where it *completes* it via the pair's graph).
- Not an Ewald splitter: no `g_ewald`, no real-space erfc corrections in the
  pair; `g_ewald` is set only for pair-style compatibility.
- Not restricted to DeepMD: after the abstraction, the only framework-named
  code is in `lmp/adapters/` behind `#ifdef SOG_WITH_DEEPMD`.

## Deployment chains (PyTorch 2.x)

For inference/export there are two equivalent chains: **Mode A** (2F+1B, solver
in LAMMPS) and **Mode B** (1F+1B, solver in-graph).  The physics and the kernel
parameters are identical — only the number of model passes and where the
reciprocal solve lives change.  See [10](10-export-and-artifacts.md) (chains,
artifacts) and [09](09-in-graph-kspace.md) (in-graph op).  `sog.v2` is an
isolated subpackage: nothing here changes the legacy `Sog`/`Gaussian`
behaviour.
