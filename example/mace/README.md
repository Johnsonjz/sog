# MACE → SOG: migration protocol and methods

This directory shows **how a MACE-based model is migrated onto the `sog.v2`
deployment protocols** so it can be exported to `.pt2` artifacts for Mode A
(`pair + kspace + fix`, 2F+1B) or Mode B (`full_lr` single graph, 1F+1B).

It is the concrete MACE instance of the procedure in the README port section
([`../../README.md#port-sog-into-your-framework-for-agents`](../../README.md#port-sog-into-your-framework-for-agents))
and [`../../docs/10-export-and-artifacts.md`](../../docs/10-export-and-artifacts.md).
The runnable reference is [`mace_sog_deployment.py`](mace_sog_deployment.py).

---

## 0. Where MACE's SOG already lives (training side)

`MACESOG` in `mace/mace/modules/extensions.py` is the **training-side** port.
It already implements the training scope
([`../../docs/03-training-adapter.md`](../../docs/03-training-adapter.md)) and
needs no change:

```python
self.sog = Sog(sog_arguments=sog_arguments, r_cut=sog_arguments.get("r_cut", None))
self.sog_readouts = nn.ModuleList(...)      # one charge readout per MACE readout

# forward(): the three coupling points
node_qs = sog_readout(node_feats, node_heads)        # coupling #2: q from node features
sog_q = torch.sum(torch.stack(node_qs_list, dim=1), dim=1)
sog_result = self.sog(latent_charges=sog_q, positions=positions,
                      cell=cell_sog, batch=data["batch"], compute_energy=True)
total_energy += sog_result["E_lr"]                   # E_lr enters the loss attached
```

`MACESOG` is **Mode A `owned`**: `Sog` owns the kernel, MACE's own autograd
produces the forces and virial because `sog_q` is still attached to `node_feats`
and hence to `positions`.  That is all a *training* port needs.

The deployment port below is a **separate, additive step**: it re-exposes the
same computation as the three `sog.v2` traceable protocols so it can be
`make_fx`-traced and AOTI-compiled.

---

## 1. The mapping: MACE concept → SOG contract → protocol method

| MACE concept (in `MACESOG.forward`) | SOG coupling | Protocol | Method to implement |
|---|---|---|---|
| `node_feats` from interactions | descriptor → charge head | — | (your graph; see §0/§1) |
| `sog_readout(node_feats, node_heads)` → `node_qs` | #2 latent charges `q` | all | charge head |
| `node_es` from `readouts` → `inter_e` | short-range energy `E_SR` | all | SR readout |
| `self.sog(...)` → `E_lr` | #1 kernel, #3 VJP | Mode A/B | solver call |
| `total_energy += E_lr` | — | — | energy composition |
| `get_outputs(energy=inter_e + sog_energy, ...)` | forces/virial | — | autograd (Mode A) |

Key structural fact: MACE's force/virial come from **one autograd backward over
`edge_vec`/`positions`** (`get_outputs`).  Both Mode A and Mode B reuse that
single backward — they differ only in *where the k-space solve lives*.

---

## 2. Mode A — `pair + kspace + fix` (2F+1B)

The solver stays in LAMMPS; MACE emits charges and the response VJP.

Implement **two** protocols:

### `EnergyOnlyTraceable` — the pair forward

```python
def energy_only(self, atype, n_node, n_local, edge_index, edge_vec, edge_mask,
                destination_order, destination_row_ptr, source_order,
                source_row_ptr, coord, box):
    # Run the MACE backbone on the traced graph, exactly as MACESOG.forward does,
    # but WITHOUT the k-space term. Return the short-range energy and the RAW
    # (unnormalised) latent charge; the LAMMPS kspace solver neutralises it.
    e_sr = ...                                  # scalar (1,)
    q_raw = ...                                 # (nnode, 1)
    return {"energy": e_sr, "latent_charge": q_raw}
```

### `ResponseTraceable` — the fix VJP

```python
def response(self, ..., coord, box, v_charge):
    # v_charge = ∂E_k/∂q from the LAMMPS kspace solve (per atom).
    # Rebuild q, keep it attached to the graph, and differentiate
    #   seed = sum(v_charge * q)  ->  F_resp, and its F (x) r virial.
    q = ...                                     # attached to edge_vec/positions
    f_resp = -autograd.grad((v_charge * q).sum(), positions_leaf)
    w_resp = f_resp.transpose @ positions
    return {
        "latent_charge": q.detach(),
        "charge_response": f_resp,
        "charge_response_virial": w_resp.reshape(9),
    }
```

> **The seed is `sum(v·q)`, never `½·sum(v·q)`** — `v` is frozen in this
> artifact, so the `½` belongs to the reported energy of the pair forward only.

Artifacts produced: `sog_energy_only.pt2` + `latent_charge_response.pt2`
(+`_with_comm` for np>1), consumed by `fix sog/response`.  See
[`../../docs/10`](../../docs/10-export-and-artifacts.md).

---

## 3. Mode B — `full_lr` single graph (1F+1B)

Faster (−30 % at 12 288 atoms) because the k-space solve moves *inside* the
graph: one forward + one backward produces `F_SR + F_resp` and the complete
box virial.  Implement **one** protocol:

### `FullLRTraceable` — the whole chain

```python
def full_lr(self, atype, n_node, n_local, edge_index, edge_vec, edge_mask,
            destination_order, destination_row_ptr, source_order,
            source_row_ptr, coord, box):
    # 1. MACE backbone -> e_sr, q_raw
    e_sr, q_raw = ...                       # as in MACESOG.forward
    # 2. neutralise (GLOBAL owned mean; zero ghost rows)
    qn = torch.ops.sogkspace.neutralize(q_raw, n_local)
    # 3. in-graph solve: (v, F_dir, vir6)
    v, f_dir, vir6 = torch.ops.sogkspace.solve_auto(
        1, qn.reshape(-1), coord.reshape(-1, 3), box.reshape(-1),
        amp, bandwidth_sq, rcut, 4, 1.0e-4, 1.0e-3, 14.3996454784255, 0.0, True)
    # 4. ONE backward: seed = sum(v*qn)  (NOT 0.5*seed)
    seed = (v.reshape(-1) * qn.reshape(-1)).sum()
    # 5. compose: energy = e_sr + 0.5*seed ; force = F_sr + F_resp + F_dir ;
    #    virial = SR + F_resp (x) r + vir6  (complete strain form, no extra F(x)r)
    ...
    return {"energy": ..., "force": ..., "virial": ...}
```

The seed must be divided by the **owned** node count (`n_local`), not `nnode`:
`seed_norm = n_local`.  On a single rank they are equal; multi-rank they are not.

Artifacts produced: `sog_full_lr.pt2` (12 inputs) and `sog_full_lr_wc.pt2`
(20 inputs, with the 8 comm tensors).  Consumed by `pair ... full_lr yes` +
`kspace_style none`.  Requires the CUDA `sogkspace` op.

---

## 4. Step-by-step migration

1. **Keep `MACESOG` unchanged** — it is the training port and the source of the
   charge head and SR readout.  Do not fork it.
2. **Extract the traced callable.**  MACE is `compile_mode("script")`; for
   `make_fx`/`torch.export` you need a plain `nn.Module` whose `forward` has an
   explicit parameter list (no `*args`, no `dict` input).  Wrap the MACE
   graph-input ABI (10 tensors) + `coord` + `box` exactly as
   [`tiny_charge_model.py`](../../test/fixtures/tiny_charge_model.py) does.
3. **Implement the protocol methods.**  For Mode A: `EnergyOnlyTraceable` +
   `ResponseTraceable`.  For Mode B: `FullLRTraceable`.  Add the
   `full_lr_callable` / `response_callable` properties (torch-2.5 AOTI
   re-exports a callable, not a make_fx GraphModule).
4. **Export.**  `trace_with_make_fx` → `export_program` → `compile_artifact`;
   write metadata with `build_metadata` / `install_artifacts`
   (see [`../../docs/10`](../../docs/10-export-and-artifacts.md)).
5. **Validate** against [`../../docs/08-validation.md`](../../docs/08-validation.md):
   Mode B vs the dense PyTorch model (E ~2.5e-4 eV, F ~4.3e-5, V ~1.3e-3), and
   Mode A fused vs non-fused to 4 digits.

---

## 5. MACE-specific gotchas (observed in the MACE tree)

- **`compile_mode("script")` vs tracing**: the deployed `MACESOG` is
  TorchScript-compiled.  `make_fx` traces *Python* execution, so the export
  wrapper must call the eager `nn.Module`, not the scripted one.
- **`get_outputs(energy=inter_e + sog_energy, ...)` excludes `e0`.**  MACE
  differentiates only the position-dependent part; the constant atomic energies
  `e0` are added afterwards.  An export wrapper must reproduce this split or the
  force gains a spurious contribution.
- **Charge readout is per-head and summed**: `sog_q = Σ_heads node_qs`.  Keep
  the sum in the graph before neutralisation; neutralise the summed charge.
- **Non-periodic frames**: `MACESOG` zeroes the cell block for non-PBC graphs
  (`cell_sog`).  The exported artifact only handles periodic boxes (the SOG op
  requires a cell); a non-PBC deployment must take the real-space fallback path
  (`Sog`, not `sog.v2`), which is outside the Mode-A/B contract.
- **fp64**: the `sogkspace` op solves in double precision.  MACE training often
  runs fp32; the export wrapper must upcast `coord`/`edge_vec` to fp64 for the
  solve and cast the outputs back.

---

## 6. Acceptance checklist (MACE deployment port)

- [ ] `MACESOG` untouched; the deployment wrapper is additive.
- [ ] The wrapper's `forward` has an explicit parameter list (10 graph tensors +
      `coord` + `box`), matching the Mode-A/B ABI.
- [ ] `trace_*` output keys equal the ABI constants
      (`FULL_LR_OUTPUT_KEYS` / `ENERGY_ONLY_OUTPUT_KEYS` / `RESPONSE_OUTPUT_KEYS`).
- [ ] Mode B energy = `e_sr + 0.5·seed`; force = `F_sr + F_resp + F_dir`;
      virial uses the **complete** strain form (no extra `Σ F⊗r`).
- [ ] Seed is `Σ v·q` (not `½Σ v·q`) and divided by `n_local`.
- [ ] `E_k = ½Σ v·q` holds to 1e-12 (the quadratic identity).
- [ ] Mode B vs dense PyTorch: E/F/V within the mesh's own accuracy.
- [ ] `check_toolchain()` clean; freeze toolchain == load toolchain.

---

## 7. Files

| File | What |
|---|---|
| `mace_sog_deployment.py` | Runnable MACE-shaped reference: implements all three protocols with closed-form physics and runs trace → export → metadata (optional AOTI compile). |
| `README.md` | This document. |

Run it from the repo root:

```bash
python example/mace/mace_sog_deployment.py            # trace + export + metadata
python example/mace/mace_sog_deployment.py --compile  # also AOTI-compile (torch>=2.6)
```

The script is a **MACE-shaped stand-in**, not the real MACE: a per-node feature
`f_i = W·coord_i`, an SR readout `e_i = ½|coord_i|²`, and a charge readout
`q_i = a·f_i` (the `sog_readouts`).  It preserves the exact coupling points a
real MACE port must hit while staying runnable without `e3nn`/MACE.
