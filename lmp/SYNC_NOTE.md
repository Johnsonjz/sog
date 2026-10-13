# Sync note: `sog/lmp` ↔ `deepmd-kit/source/lmp`

The plugin here tracks the canonical sources in
`/data/home/public/jiangzhen/dp/deepmd-kit/source/lmp/` with one deliberate
difference: **every model-specific piece was moved behind two abstract classes**
so SOG can serve any MLIP, not just DeepMD.

The exact interface diff is `patches/sog-model-agnostic.diff` — **verified to
apply cleanly** to an unmodified deepmd-kit tree (`git apply --check` from the
deepmd-kit root, then reproduces these files byte-for-byte). It is **not
applied** to deepmd-kit; the deepmd-kit repo keeps working through its own
copies. Apply it only when the abstraction should flow upstream:

```bash
cd /path/to/deepmd-kit
git apply --check /data/home/public/jiangzhen/sog/lmp/patches/sog-model-agnostic.diff
git apply         /data/home/public/jiangzhen/sog/lmp/patches/sog-model-agnostic.diff
```

## What the patch changes (10 files, ~4 modified + 6 added)

| File | Δ | What |
|---|---|---|
| `sog.cpp` | +31/−7 | Reads charges through a `SogChargeSource` (resolution: registered source → deepmd adapter → `atom->q`) instead of `dynamic_cast<PairDeepMD*>`; conditional include of the adapter; destructor deletes the default source. |
| `sog.h` | +11 | `set_charge_source()` / `get_charge_source()`; `charge_source` + owned `default_charge_source` members. |
| `fix_sog_response.cpp` | rewritten (+210) | Talks only to `SogChargeResponse`. Adds the `model`/`fused`/`pair_deepmd_index` keyword parsing, the adapter factory, and `FixSOGResponse::build_input_nlist()` (see below). All deepmd members, the pair match by name and the standalone `DeepPot` moved to the adapter. |
| `fix_sog_response.h` | rewritten (+56) | Holds `SogChargeResponse*` + `SOGKSpace*` instead of `PairDeepMD`/`DeepPot`/`type_idx_map`. |
| `sog_charge_source.h` (new) | — | Coupling #2 contract. |
| `sog_charge_response.h` (new) | — | Coupling #3 contract. |
| `adapters/deepmd_charge_source.h` (new) | — | `PairDeepMD` → `SogChargeSource` (~15 lines); guards an empty `dcharge_multi` (model with `latent_charge_to_q` off previously read out of bounds). |
| `adapters/deepmd_charge_response.cpp/.h` (new) | — | The retained-graph VJP (`enable_charge_only_forward`, `enable_charge_response_graph`, `compute_*_response_cached`, standalone `compute_charge_response` fallback). |
| `adapters/native_charge_response.h` (new) | — | Documented no-op for fixed-charge frameworks. |

Unchanged and byte-identical: `sog_spline.h`, `sog_gpu.cu`, `sog_gpu.cuh`.

**Deliberately dropped:** `type_idx_map` in the canonical `fix_sog_response.cpp`
was built every step and never read (dead code); the rewrite omits it.

**Not in this patch** (they exist for the *standalone plugin* build only, and
deepmd-kit has its own plugin CMakeLists): `sogplugin.cpp`,
`sog_gpu_stub.cpp`, `CMakeLists.txt`, `tests/`.

## The one subtlety: `build_input_nlist`

`deepmd::InputNlist` needs the comm-brick layout (`nswap`, `sendnum`,
`sendlist`, …), which `CommBrickDeepMD` exposes only to
`friend class PairDeepMD` and `friend class FixSOGResponse`
(`deepmd-kit/source/lmp/pair_deepmd.h:41-44`). The deepmd adapter class is not
a friend, so the fallback's neighbor list cannot be assembled there **without
editing deepmd-kit to widen the friendship**.

Instead, the access lives in
`FixSOGResponse::build_input_nlist()` — a member of the one class that *is* a
friend — and the adapter calls it through a back-pointer, which the fix hands
over at construction (`create_charge_response(lmp, this, ...)`).

Consequence for any upstream port: keep the free function/member split as is,
or widen the friendship in pair_deepmd.h (a deepmd-kit change).
