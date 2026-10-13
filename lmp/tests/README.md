# `lmp/tests` — LAMMPS plugin tests

Portable inputs + checker for `libsog_lmp.so`. Runs standalone here, or driven
by pytest from the repo root (`test/test_lammps_plugin.py`, auto-skips unless
`SOG_LMP_BINARY` is set).

All inputs take the plugin path as a LAMMPS variable, defaulting to
`../build/libsog_lmp.so` relative to the working directory, so run them from
**this directory**:

| Check | Command | Pass criterion |
|---|---|---|
| Native smoke | `lmp -in native_smoke.in -log native_smoke.log` | exit 0, no `ERROR`, the one-time `charge response is a no-op` warning |
| Direct k-sum parity | `lmp -in parity_direct.in -log parity_direct.log` then `python parity_direct.py` | checker prints `PASS energy/force/pressure` (rel < 1e-8; measured ~1e-15) and exits 0 |

Override paths with `-var`:

```
lmp -var plugin /abs/libsog_lmp.so -in native_smoke.in -log /tmp/smoke.log
lmp -var plugin /abs/libsog_lmp.so -var dumpfile /tmp/p.dump \
    -in parity_direct.in -log /tmp/p.log
python parity_direct.py /tmp/p.dump --log /tmp/p.log
```

## Notes

- `parity_direct.in` pins the k-set to `direct yes 1.5708`, identical to the
  Python side's `n_dl = 4.0`; the geometry is randomly displaced (rocksalt
  sites are inversion centers, so an undisplaced cell has exactly zero
  forces and the force check would be vacuous).
- The checker pins the Python unit constants to this LAMMPS tree's metal
  values (`qqr2e = 14.399645`, `nktv2p = 1.6021765e6`); a different binary
  needs `SOG_QQR2E` / `SOG_NKTV2P` — see the module docstring.
- `compare_fastsog_sog.cpp` is a standalone C++ dev harness (influence
  function / Green functions / spread+FFT vs the fastsog core); build and run
  per its header comment. It is not part of the automated checks.
