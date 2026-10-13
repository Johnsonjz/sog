# 06 — Port the MD side (write your C++ adapters)

You have a trained latent-charge model and `libsog_lmp.so`. Two things are
missing to run MD with the *conservative* forces: something that gives SOG your
charges, and something that gives SOG `∂q/∂r` — both are C++ adapters you write
against two abstract classes. SOG itself is already model-agnostic; there is no
SOG code to modify.

> **Snippet status.** The C++ below is illustrative skeleton code (compiled
> against your framework, not this repo). The *working* reference is
> `lmp/adapters/deepmd_charge_response.{h,cpp}` + `lmp/fix_sog_response.cpp`,
> which compile in both build modes here — copy from those, and use the
> checklist at the bottom, not the snippets, as acceptance criteria.

```
   pair style (your model)                 SOG kspace                fix sog/response
   ────────────────────────                ──────────                ────────────────
   writes latent charges  ──(#2)──►  computes v_i = ∂E_k/∂q_i  ──►  asks your adapter:
   SogChargeSource                       (set_want_potential)        S = Σ v_i q_i
                                                                     F = −∂S/∂r   (#3)
                                                                     W = F ⊗ r
                                                          ◄─────────  adds F to atoms,
                                                                     tallies W
```

## 1. `SogChargeSource` — what your model gives SOG

File: `sog/lmp/sog_charge_source.h` (read it — 40 lines of contract). You
implement:

```cpp
class MyChargeSource : public SogChargeSource {
 public:
  int nchannels() const override { return model_channels_; }   // <=1 → atom->q path
  const double *charges() const override { return q_buffer_.data(); }
};
```

Guarantees you must provide:

- `charges()[i * nchannels() + ch]` for `i ∈ [0, nlocal)` — flat,
  channel-major, **LAMMPS units**, valid for the whole `compute()` call.
- Return `nullptr` only when `nchannels() <= 1` (the solver then reads
  `atom->q`).
- Ghost atoms: the solver spreads from `nlocal`; you only need local entries.

Resolution order in `SOGKSpace::compute()`: (1) a source registered via
`SOGKSpace::set_charge_source()`; (2) the compiled-in deepmd adapter;
(3) `atom->q`. **If your pair style already writes the charges into `atom->q`
(single channel), you need no source at all.** Copy
`lmp/adapters/deepmd_charge_source.h` for the multi-channel case — note its
bounds guard: it returns channel 1 when `dcharge_multi` is empty (a model with
latent charges off), instead of reading out of bounds.

## 2. `SogChargeResponse` — what SOG asks back

File: `sog/lmp/sog_charge_response.h`. The fix hands you:

- `v_per_atom[nlocal * nchannels]` — `v_i = ∂E_k/∂q_i` replicated per channel,
  **in your model's energy units** (the fix converted them; you convert nothing
  on input).
- You return `f_ext[nall * 3]` (ghost-inclusive; the fix reverse-comms it) and
  `virial9[9]` — both **in your model's units** (`energy_unit_factor()` /
  `force_unit_factor()` describe the conversion to LAMMPS units).

### The math (this is the whole job)

For PBC the instantaneous pressure tensor has **two terms** (`h` the box,
`s_i` fractional coordinates, `r_i = h s_i`):

```
P = −(1/det h) · [∂U({h s_i}; h)/∂h] · hᵀ
  = (1/det h) · [ Σ_i F_i ⊗ r_i  −  (∂U({r_i}; h)/∂h)|_r · hᵀ ]
```

1. `Σ F⊗r` — from the forces.
2. `(∂U/∂h)|_r · hᵀ` — the **box-deformation term**: the energy derivative at
   fixed Cartesian positions. For the k-space energy this term is the whole
   game: `lmp/sog.cpp` computes it analytically from `d(1/V)/dε`, `d(k²)/dε`
   and `dG/d(k²)` (comments "Analytic Fourier virial: the COMPLETE
   strain-derivative of E_k" in `compute_single`, and the Fourier form in
   `compute_direct`), and states the rule explicitly: *"A Σ r⊗F formulation is
   therefore INCOMPLETE — it omits the box-explicit strain term"*. **Never
   rebuild the k-space virial from k-space forces.**

Your adapter supplies the **response** contribution. With `S = Σ_i v_i · q_i`:

```
f_ext   = −∂S/∂r          ghost-inclusive, [nall, 3]
virial9 = Σ_a f_a ⊗ r_a   over the EXTENDED atoms (locals + ghosts),
                          row-major:  virial9[3*k + j] = Σ_a f_a,k · r_a,j
```

This `F⊗r` form is the *complete* strain derivative of `S` under two
conditions — verify both for your framework:

- the affine strain scales **every** extended coordinate identically
  (`r_i = h s_i`; ghosts `r_i + n·h`), so ghost entries are not optional —
  they carry the box dependence of `q`;
- the charge graph has **no explicit box input**: `q` depends on the box only
  through those coordinates. The DeepMD pattern does this —
  `charge_response_lower` takes no box argument. If your model *does* consume
  the box explicitly (`∂q/∂h|_r ≠ 0`), term 2 does not vanish for the
  response either: one extra VJP against a box leaf gives
  `−(∂S/∂h)|_r·hᵀ`; take the full derivative of `S` w.r.t. the box at fixed
  Cartesian coordinates instead of the `F⊗r`-only form.

Reference implementation: `deepmd/pt/model/model/sog_model.py::charge_response_lower`
(`virial = einsum("fak,faj->fkj", force_ext, cc_ext)`) — the box-free case.
Validate against the **full** `∂U/∂h` at fixed scaled coordinates (08 §2): an
`F⊗r`-only reference would confirm the same mistake twice.

### Getting the VJP from your framework

If your model runs in-process as a TorchScript module (the pattern
`pair_deepmd.cpp` uses: `torch::jit::load(model_path)`, inputs as
`torch::Tensor`s from raw pointers, `InputNlist` for the neighbor list), the
response is a **vector-Jacobian product** — one backward with an external
gradient, no second forward needed if the graph still exists:

```cpp
// 1. Forward (once per step, in your pair style), KEEPING the graph:
coord_t.requires_grad_(true);                     // [1, nall, 3]
auto out = module.forward({coord_t, atype_t, box_t, nlist_t});
auto q = out.toGenericDict().at("latent_charge").toTensor();   // [1, nlocal, nq]

// 2. In response(): S = Σ v_i q_i, then VJP:
torch::Tensor S = (v_t * q).sum();                // v_t built from v_per_atom
auto grad = torch::autograd::grad({S}, {coord_t},
                                  /*grad_outputs=*/{torch::ones_like(S)},
                                  /*retain_graph=*/true)[0];
// f_ext = -grad;  virial9 = einsum("fak,faj->fkj", f_ext, coord_t)
//   (valid only for box-free charge graphs — see "The math" above.  If the
//    module takes `box_t` explicitly, also VJP S against box_t and build the
//    full strain derivative of S at fixed Cartesian coordinates.)
```

Notes:

- **Unit conversion happens in `response()`**, not in your model: build `v_t`
  from the doubles the fix passes, compute, then scale the outputs by your
  factors when writing the `std::vector<double>`s.
- The deepmd adapter caches the graph across pair-forward and fix-backward
  (`compute_charge_response_cached`) and has a standalone
  forward+backward fallback (`DeepPot::compute_charge_response`) — copy that
  structure if your pair style can cooperate; otherwise start with the
  standalone fallback (correct, just slower: one extra forward per step).
- Message passing needs the comm-brick layout (`nswap`, `sendlist`, …) for
  `InputNlist`. In deepmd-kit that layout is friendship-gated to
  `PairDeepMD`/`FixSOGResponse`; if your LAMMPS tree is more permissive, use
  it directly — otherwise do what this repo does and assemble it inside a
  member of the fix (see `SYNC_NOTE.md`, "build_input_nlist").
- `f_ext` is **ghost-inclusive** because `∂q` involves neighbor atoms; write
  zeros for ghosts you don't need and let the fix's reverse-comm accumulate.

### No differentiable graph? (`fused` semantics and the no-op)

`fused_short_range()` answers one question: did the pair's forward already drop
the short-range force? If `false` (normal), your output is the *correction*
added on top of the pair's force. If `true`, your output must be the **total**
force and virial — a missing retained graph is a hard error, by design: in
fused mode silently dropping the short-range force would be a wrong-physics
bug, not a performance issue.

For fixed-charge models there is nothing to implement: the fix's factory
returns `NativeChargeResponse`, which warns once and returns zeros. That is the
documented degraded mode, not a silent one.

## 3. Registration (only needed for a standalone plugin)

Your adapter is compiled into the same shared library. Registration is the
LAMMPS plugin ABI:

```cpp
// sogplugin.cpp
extern "C" void lammpsplugin_init(void *lmp, void *handle, void *regfunc) {
  lammpsplugin_t plugin;
  plugin.version  = LAMMPS_VERSION;
  plugin.style    = "kspace";  plugin.name = "sog";
  plugin.creator.v1 = (lammpsplugin_factory1 *)&SOGKSpace::creator;
  ((lammpsplugin_regfunc)regfunc)(&plugin, handle);
  // ... second entry: style "fix", name "sog/response", FixSOGResponse::creator
}
```

Two conventions worth knowing (v30Mar2026):

- Fix constructors receive `narg` **excluding** the `fix` keyword:
  `arg[0]=ID, arg[1]=group, arg[2]=style`, keywords from `arg[3]`. A bare
  `fix ID group style` arrives as `narg == 3`.
- `comm->reverse_comm(this, 3)` (fix reverse-comm) and `v_tally` are the only
  LAMMPS-side duties of the fix; keep them.

## 4. Minimal end-to-end skeleton

Start from these files, in this order:

1. `lmp/adapters/native_charge_response.h` — the no-op, for the interface shape.
2. `lmp/adapters/deepmd_charge_response.h/.cpp` — a full adapter: constructor
   (model load + pair lookup + unit factors), `response()` with the VJP.
3. `lmp/fix_sog_response.cpp` — how the factory picks the adapter and how
   `vpot → v_per_atom → response → atom->f/v_tally` flows.

Then:

```cpp
// your_adapter.h  (compile behind your own SOG_WITH_MYMODEL guard)
class MyChargeResponse : public SogChargeResponse {
 public:
  double energy_unit_factor() const override { return 1.0; }  // if already eV
  double force_unit_factor() const override { return 1.0; }
  int nchannels() const override { return 1; }
  bool fused_short_range() const override { return false; }
  void response(std::vector<double> &f_ext, std::vector<double> &virial9,
                const double *v_per_atom, int nlocal, int nall,
                int nchannels) override;
};
```

Wire it in `fix_sog_response.cpp`'s `create_charge_response()` factory and add
your model's sources in `lmp/CMakeLists.txt` under a new option.

## Checklist (acceptance for an MD port)

- [ ] Charges reach the kspace: with your pair style active and
      `kspace_style sog`, `elong` changes when the charges change sign.
- [ ] `fix ... sog/response` runs with a nonzero correction
      (`dfcorr` not all zero) when the model has a differentiable graph.
- [ ] **FD test**: displace one atom, finite-difference the total `pe`, compare
      to the total force — this is the only test that validates the whole
      `vpot → VJP → f_ext → v_tally` chain, including signs.
- [ ] Virial: compare the 6-component `compute p` against the Python autograd
      virial at fixed scaled coordinates (see 08; mind the tree's component
      order).
- [ ] `fused` mode either works (retained graph present) or errors loudly —
      never a silent force drop.
- [ ] Ghost forces: a two-rank-decomposed run gives the same forces as
      single-rank (the fix reverse-comms; verify it).
- [ ] No-op degradation: building without your model's flag must warn and
      contribute zero, not crash.
