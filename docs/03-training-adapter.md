# 03 — Port the training side

Goal: your MLIP's total energy gains a term `E_lr` computed by SOG from the
model's own latent charges, with gradients flowing correctly to every
trainable thing (charges, kernel parameters, backbone).

## The contract your framework must satisfy

1. It produces per-atom **latent charges** `q ∈ [N, nq]` from `(positions,
   cell, ...)` through differentiable ops. If the charges are fixed, stop —
   you want the MD-side native path, not this document.
2. Its batch layout matches SOG's canonical convention: frames as contiguous
   blocks, ids `0..nf−1` (call `validate_batch` at the boundary).
3. It adds SOG's `E_lr` (or binds SOG's explicit `forces`/`virial`) without
   detaching the charges. Detaching kills coupling #3.

## Two integration modes

### Mode A — `owned` (framework owns nothing; SOG owns the kernel)

The pattern MACE uses (`mace/mace/modules/extensions.py`): construct one `Sog`
inside your module, add `E_lr` to your total energy, and let your own autograd
produce forces/virial.

```python
from sog.integration import build_sog, lr_energy, validate_batch

class MyLongRange(nn.Module):
    def __init__(self, r_cut, n_channels=1):
        super().__init__()
        self.n_channels = n_channels
        self.sog = build_sog(mode="owned", r_cut=r_cut, n_channels=n_channels)

    def forward(self, positions, cell, charges, batch):
        # `charges` MUST still be attached to the graph that produced them.
        e_lr = lr_energy(self.sog, positions, cell, charges, batch,   # [nf]
                         n_channels=self.n_channels)
        return e_lr          # caller adds it to the total energy
```

In `train_step`, `(E_total).backward()` then delivers:
`∂E_lr/∂q → ∂q/∂(your params)` automatically, plus `∂E_lr/∂r` (which contains
the charge-response force) and `∂E_lr/∂h` when `cell` is a leaf.

### Mode B — `external` (the framework owns the kernel tensors)

The pattern DeepMD's `sog_model.py` uses: the fitting net itself holds
trainable `amp`/`bandwidth` (its `SOGEnergyFittingNet`), and SOG binds them so
one backward updates them.

```python
from sog.integration import build_sog

sog = build_sog(mode="external", r_cut=self.r_cut, n_channels=self.n_channels,
                amp=self.amp, bandwidth=self.bandwidth,
                amp_is_internal=True,      # C++/DeepMD convention
                bandwidth_is_squared=False) # pre-square, negativity-safe
```

Invariants specific to Mode B:

- `bind_external` **replaces** any owned parameters; only one graph exists.
- `.to(device)`/`.cuda()` does not move the bound attributes (plain tensor
  attributes, not buffers) — SOG re-syncs cheaply on each `forward` via a
  version counter, so bind → move is safe.
- `init_from_rcut` refuses to overwrite externally bound tensors (correct: it
  would clobber your trainable parameter).
- Your optimizer must still be constructed over *your* parameters — that is
  the point of the mode.

## Worked shape: descriptor → charge head

Minimal, complete, framework-neutral:

```python
from sog.integration import AtomwiseChargeProvider, build_sog, validate_charges

class MyModel(nn.Module):
    def __init__(self, desc_dim, r_cut):
        super().__init__()
        self.charge = AtomwiseChargeProvider(n_in=desc_dim, n_out=1)
        self.sog = build_sog(mode="owned", r_cut=r_cut, n_channels=1)

    def forward(self, desc, positions, cell, batch):
        q = self.charge(desc, batch)                # [N, 1], attached
        validate_charges(q, positions.shape[0], 1)
        e_lr = self.sog(positions=positions, cell=cell,
                        latent_charges=q, batch=batch,
                        compute_energy=True)["E_lr"]
        return e_lr
```

Then, in the loss:

```python
loss = ((model(...) + e_short - e_ref) ** 2).mean()
loss.backward()      # gradients reach the charge head AND the kernel params
```

DeepMD's variant additionally trains against forces/virial with
`compute_force`/`compute_virial`; remember the `None`/autograd fallback rule
from 02.

## Invariants (any mode)

- Charges passed to SOG must be the *same tensor* the energy was learned from —
  no `float()` casts, no `.detach()`, no re-computation with different numerics.
- dtypes: match `positions` and `cell` (float64 recommended for kernel work and
  tests; training may run float32 — but see neutrality rounding in 02).
- One `Sog` instance per model (the solver caches per-batch state; sharing
  across models with different cells is fine — it is keyed on box + n_dl — but
  sharing across dtypes/devices is not).
- Charge neutrality is a training-time decision, not a solver switch: k=0 is
  always excluded, so a net-charged model silently loses that term. Use
  `ChargeNeutralityPolicy` and be explicit.
- `r_cut` given to `build_sog` must be the cutoff the charges were trained
  with; it fixes σ and hence the whole kernel.

## Checklist (acceptance for a training port)

- [ ] `LatentChargeProvider` (or equivalent) implemented; `n_channels` correct.
- [ ] `validate_batch` called at the boundary; a deliberately scrambled batch
      raises (test it).
- [ ] `E_lr` added to the loss without detaching charges.
- [ ] One backward step moves **both** the charge-head parameters and (Mode B)
      your `amp`/`bandwidth` — assert `grad is not None and grad.abs().sum()>0`.
- [ ] Forces: either use `compute_force=True` and compare to autograd, or rely
      on autograd — but handle the `forces=None` fallback explicitly.
- [ ] Kernel parity: your construction reproduces
      `SogKernel(r_cut=...).amp_internal()/bandwidth_sq()` bit-for-bit
      (see `test/test_sog_kernel.py::test_kernel_matches_gaussian_rcut_convention`).
- [ ] Export step works and matches the C++ side (04).
