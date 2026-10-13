# 07 — End-to-end pipeline

Train → export → LAMMPS → MD → validate. The DeepMD-specific full walkthrough
(with DFT data, training configs and long MD) is
`dp/dp_example/water/lmp/SOG_WORKFLOW.md` — reference it for the production
recipe. This page is the framework-neutral skeleton, and marks which steps were
executed in this repo as a live check.

## Stage 1 — Train (Python)

```python
# fragment — your training loop, any framework
model = MyModel(desc_dim, r_cut=6.0)                 # charge head + SOG (see 03)
for batch in loader:
    E_short = model.short_range(batch)
    E_lr    = model.long_range(batch)                # = SOG E_lr, charges attached
    loss    = ((E_short + E_lr - E_ref) ** 2).mean()
    loss.backward()                                  # reaches charges AND kernel params
```

Invariants: charges never detached; `validate_batch` at the boundary; kernel
parameters move (assert their `.grad`).

## Stage 2 — Export the frozen kernel (Python)

```python
from sog import SogKernel

k = model.long_range.sog.as_kernel()                 # or the kernel you trained
print(k.to_lammps_input())
```

Copy the line it prints. Do not hand-write `amp`/`bandwidth` — see 04 for the
squared/internal conventions and the `phi_max`/`m` naming traps.

## Stage 3 — LAMMPS input

```lammps
units metal
atom_style charge
boundary p p p
read_data system.data                 # or create_atoms + fix shake etc.

pair_style <your model>               # must write atom->q (or implement SogChargeSource)
pair_coeff * *

plugin load /path/to/libsog_lmp.so
kspace_style sog amp <M> bandwidth <M²> m 12 b 2.0 sigma 3.016 \
    spline cubes2_4 remove_self_interaction yes phi_max 0.23

fix sresp all sog/response model model.pth          # only for latent-charge models

fix nvt all nvt temp 300 300 0.1
thermo 100
run 100000
```

Practical notes, in order of how often they bite:

1. `plugin load` must precede `kspace_style`/fix names.
2. `remove_self_interaction` explicit (side defaults differ).
3. `restart` files do not carry plugin styles: re-issue `plugin load`,
   `kspace_style`, `fix` after `read_restart`.
4. Latent-charge models: without `fix sog/response` the dynamics is
   non-conservative (the `∂q/∂r` force is missing) — energies look plausible
   and drift.

## Stage 4 — Validate before trusting MD

Run the 0-step cross-language check (this is scripted in this repo):

```bash
cd sog/lmp/tests
lmp -in parity_direct.in -log parity_direct.log   # kspace only, pair_style zero, direct k-sum
python parity_direct.py                           # PASS/FAIL vs the Python lib (exit code)
# or, from the repo root:
SOG_LMP_BINARY=/path/to/lmp pytest test/test_lammps_plugin.py
```

Expected (measured, see 08): energy agreement ~1e-16 eV, max force difference
~1e-15 relative, pressure tensor ~1e-11 bar. If you see ~1e-8-relative
discrepancies, check the LAMMPS binary's `qqr2e`/`nktv2p` constants first (08).

Then one short MD leg with dump output, and the FD force check from 08 to
validate the response chain.

## Stage 5 — Production MD and analysis

Storage this repo's workflows use:

| Artifact | Example location |
|---|---|
| MD runs | `dp/dp_example/water/lmp/` (see SOG_WORKFLOW.md for the naming) |
| Trajectories/logs | beside the input, one directory per leg |
| Analysis scripts | same tree; every input/output referenced by absolute path |

Analysis checklist (physics, not code):

- energy drift per ps for NVE (should be ~0 for a conservative setup);
- RDF/structure against the training reference;
- pressure stationarity in NPT, and **zero long-term drift** of the box
  (the classic symptom of a missing response term is slow shear/volume creep);
- if available, a short `pppm` cross-run: thermodynamics should agree within
  its own accuracy budget.

## What was executed in this repo (vs. referenced)

| Stage | Status |
|---|---|
| kernel construction, export, format round-trip | executed — `pytest test/test_sog_kernel.py` |
| external binding + grads (Mode B) | executed — `test_external_binding_reaches_framework_tensors` |
| plugin load + native run (fixed charges) | executed — `lmp/tests/native_smoke.in`, 2 steps clean |
| export → LAMMPS direct k-sum → Python parity | executed — `lmp/tests/parity_direct.py` (numbers in 08) |
| full DFT-data training + production MD | reference — `dp/dp_example/water/lmp/SOG_WORKFLOW.md` |
