"""``sog.v2`` must be an isolated, import-safe subpackage."""

import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.v2


def _run(code: str, env: dict | None = None):
    e = dict(os.environ)
    e.pop("SOG_KS_OP_LIB", None)
    if env:
        e.update(env)
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=e)


def test_import_sog_does_not_import_v2():
    r = _run("import sog, sys; assert 'sog.v2' not in sys.modules; print('ok')")
    assert r.returncode == 0, r.stderr


def test_legacy_api_still_available():
    r = _run(
        "from sog import Sog, SogKernel, build_sog, lr_energy, validate_batch; "
        "print('ok')"
    )
    assert r.returncode == 0, r.stderr


def test_v2_imports_without_op_library():
    code = "\n".join(
        [
            "import sog.v2 as v",
            "from sog.v2 import ExternalKSpace, TraceableSogKernel, KSpaceSettings",
            "import sog.v2.kspace_op as k",
            "assert k.op_available() is False, 'import must not load the op'",
            "try:",
            "    k.load_op_library('/nonexistent/libsog_kspace_op.so')",
            "except k.OpUnavailableError:",
            "    pass",
            "else:",
            "    raise AssertionError('an explicit missing path must raise')",
            "print('ok')",
        ]
    )
    r = _run(code, env={"SOG_KS_OP_LIB": "/nonexistent/libsog_kspace_op.so"})
    assert r.returncode == 0, r.stderr


def test_import_v2_does_not_need_cuda():
    r = _run(
        "import torch, sog.v2; "
        "from sog.v2 import TraceableSogKernel, ReferenceKSpace, ExternalKSpace; "
        "assert not torch.cuda.is_initialized(); print('ok')"
    )
    assert r.returncode == 0, r.stderr
