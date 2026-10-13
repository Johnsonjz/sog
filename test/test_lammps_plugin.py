"""LAMMPS plugin integration tests (ported inputs live in ``lmp/tests/``).

Runs the real ``lmp`` binary against the portable input files:

- ``native_smoke.in`` — plugin load + native charge path + the documented
  one-time no-op warning from ``fix sog/response``;
- ``parity_direct.in`` + ``parity_direct.py`` — cross-language numeric parity
  on the exact direct k-sum (energy/forces/pressure to ~1e-15).

Skipped unless a LAMMPS binary is configured::

    export SOG_LMP_BINARY=/path/to/lmp              # required
    export SOG_LMP_PLUGIN=/path/to/libsog_lmp.so    # default: lmp/build/libsog_lmp.so

    pytest test/test_lammps_plugin.py
"""

import importlib.util
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = ROOT / "lmp" / "tests"
LMP = os.environ.get("SOG_LMP_BINARY", "")
PLUGIN = Path(
    os.environ.get("SOG_LMP_PLUGIN", str(ROOT / "lmp" / "build" / "libsog_lmp.so"))
)

pytestmark = [
    pytest.mark.lammps,
    pytest.mark.skipif(
        not (LMP and Path(LMP).is_file() and PLUGIN.is_file()),
        reason="set SOG_LMP_BINARY (LAMMPS executable the plugin can load) and "
        "build the plugin at lmp/build/libsog_lmp.so (or set SOG_LMP_PLUGIN)",
    ),
]


def _run_lmp(input_name, cwd, extra=()):
    cmd = [LMP, "-var", "plugin", str(PLUGIN), *map(str, extra),
           "-in", str(TESTS_DIR / input_name)]
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                          timeout=600)


def _load_parity_module():
    spec = importlib.util.spec_from_file_location(
        "sog_parity_direct", TESTS_DIR / "parity_direct.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_native_plugin_smoke(tmp_path):
    """Plugin loads, native charge path feeds the kspace, no-op adapter warns."""
    log = tmp_path / "native_smoke.log"
    r = _run_lmp("native_smoke.in", tmp_path, extra=("-log", log))
    out = r.stdout + r.stderr
    assert r.returncode == 0, f"lmp exited {r.returncode}:\n{out[-2000:]}"
    assert "ERROR" not in out, f"lmp reported an error:\n{out[-2000:]}"
    assert "charge response is a no-op" in out, (
        "fix sog/response must emit the one-time native no-op warning"
    )


def test_direct_k_sum_parity(tmp_path):
    """LAMMPS `direct yes` vs the Python lib on the identical mode set."""
    dump = tmp_path / "parity.dump"
    log = tmp_path / "parity_direct.log"
    r = _run_lmp(
        "parity_direct.in",
        tmp_path,
        extra=("-var", "dumpfile", dump, "-log", log),
    )
    out = r.stdout + r.stderr
    assert r.returncode == 0, f"lmp exited {r.returncode}:\n{out[-2000:]}"
    assert "ERROR" not in out, f"lmp reported an error:\n{out[-2000:]}"

    parity = _load_parity_module()
    res = parity.compare(dump, log)

    tol = parity.TOL_REL
    assert res["energy_rel"] < tol, (
        f"energy rel {res['energy_rel']:.3e} — lmp {res['E_lmp']!r} "
        f"py {res['E_py']!r} ref {res['E_ref']!r}"
    )
    assert res["force_rel"] < tol, (
        f"force rel {res['force_rel']:.3e} (max |dF| {res['force_abs']:.3e} "
        f"on |F|max {res['force_scale']:.3e})"
    )
    assert res["press_rel"] < tol, (
        f"pressure rel {res['press_rel']:.3e}; diffs {res['P_diff']}"
    )
