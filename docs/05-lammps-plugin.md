# 05 — The LAMMPS plugin

`libsog_lmp.so` registers two styles via `plugin load`:

- `kspace_style sog` — the solver (all model-agnostic; feeds on `atom->q` or a
  registered `SogChargeSource`).
- `fix ID group sog/response` — the charge-response correction for latent-charge
  models (feeds on a `SogChargeResponse`).

The exhaustive keyword reference and the file inventory live in
[`../lmp/README.md`](../lmp/README.md) — read that for parameter details. This
page is the integration view: what a port must know and in what order.

## Build

Build the plugin from `lmp/`; every knob (ABI, FFT, DeepMD/GPU), the
`undefined symbol ...Ss...` mismatch symptom and the MPI headers-only note are
in [`../lmp/README.md`](../lmp/README.md) §Build.  One fact specific to the
integration view: the standalone **in-graph op** (`op/`) links libtorch and must
**not** set `SOG_GLIBCXX_OLD_ABI` — that flag is for the LAMMPS plugin only.

## Minimal input (fixed charges, no latent charges)

```lammps
plugin load /path/to/libsog_lmp.so
pair_style lj/cut/coul/long 6.5      # any pair style that writes atom->q
pair_coeff * * 0.1 2.4
kspace_style sog amp <M> bandwidth <M²> spline cubes2_4 \
    remove_self_interaction yes phi_max 0.23 m 12
run 0
```

`fix sog/response` is **not** needed here: with fixed charges `∂q/∂r = 0`, the
response term vanishes. (If you add the fix anyway in a native build, the
`NativeChargeResponse` no-op warns once and contributes zero.)

## With a latent-charge model (DeepMD example)

```lammps
plugin load /path/to/libsog_lmp.so
pair_style deepmd model.pth latent_charge_to_q yes
kspace_style sog amp <M> bandwidth <M²> spline cubes2_4 \
    remove_self_interaction yes phi_max 0.23 m 12
fix sresp all sog/response model model.pth          # coupling #3
```

For a different framework, `fix sog/response` compiles to
`NativeChargeResponse` until you add a `SogChargeResponse` adapter — see 06.
The fix's job, concretely:

1. `pre_force`: tells the kspace to compute `v_i = ∂E_k/∂q_i`
   (`set_want_potential`).
2. `post_force`: hands `v_i` (in your units) to the response adapter, receives
   the correction force `[nall·3]` + 9 virial components, reverse-comms ghosts,
   adds into `atom->f`, tallies the virial.
3. With `fused yes`, by the time the fix runs the pair has already dropped the
   short-range force (`enable_charge_only_forward`), so the returned correction
   is the **total** force; a missing retained graph is a hard error by design.

## Operational notes

- **Restart**: like all plugin-provided fixes, `read_restart` does not restore
  the fix (or its kspace); re-issue `plugin load`, `kspace_style`, and the fix
  after restarting, then `pair_coeff` as usual.
- **`plugin load` ordering**: load the plugin before any style name that needs
  it appears.
- **Triclinic**: rejected at `init()` — orthorhombic only.
- **Multi-rank**: the mesh/FFT kspace is correct single-rank only (the charge
  mesh spreads from `nlocal`; no reverse-comm). The `direct yes` path reduces
  its structure factors across ranks — use it for MPI validation, not
  production.
- **`compute p` convention**: this tree's `compute_pressure` reports the
  kspace virial in raw LAMMPS order `[xx yy zz xy xz yz]`; `thermo press` (the
  scalar) is unaffected. Label tensor components accordingly when comparing
  against scripts.

## Verifying a build (30 seconds)

```bash
nm -D --undefined-only build/libsog_lmp.so | grep sog_gpu_   # 0 expected (stub build)
cd lmp/tests
lmp -in native_smoke.in -log native_smoke.log          # exit 0 + one-time no-op warning
lmp -in parity_direct.in -log parity_direct.log        # writes parity.dump
python parity_direct.py                                # PASS energy/force/pressure, exit 0
```

Or the same checks through pytest (auto-skips without a configured binary):

```bash
SOG_LMP_BINARY=/path/to/lmp pytest test/test_lammps_plugin.py
```

`parity_direct.py` is the cross-language acceptance test (thresholds 1e-8
relative; measured ~1e-15); the numbers it must reproduce and the unit-constant
traps are in [08-validation.md](08-validation.md). The inputs default the
plugin path to `../build/libsog_lmp.so`; override with
`lmp -var plugin /abs/path -in …`.

## Relationship to the PyTorch deployment layer

This plugin implements the **old chain** (`kspace_style sog` + `fix
sog/response`).  The in-graph operator and the `.pt2` artifacts used by the
1F+1B chain are built separately from `op/` — see
[09-in-graph-kspace.md](09-in-graph-kspace.md) and
[10-export-and-artifacts.md](10-export-and-artifacts.md).  A framework that
adopts the single-graph chain runs with `kspace_style none` and no fix; the
plugin is then not loaded.

The plugin deliberately trails the active tree on a few features (documented
in [11-lr-sync.md](11-lr-sync.md)): `box_tol`, the distributed mesh,
`kspace sog/kk`, and the device response branch.  The old chain works without
them; the in-graph op already has `box_tol`.
