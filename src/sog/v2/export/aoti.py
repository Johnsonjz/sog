"""AOTInductor packaging for SOG artifacts (torch-version aware).

Two API generations exist:

* **torch >= 2.6/2.11** — ``torch._inductor.aoti_compile_and_package`` produces a
  ``.pt2`` package (a zip with the compiled ``.so`` + metadata).  This is the
  format the DeepMD C++ loader consumes.
* **torch 2.5** — ``torch._inductor.aot_compile`` produces a plain ``.so``.
  :func:`compile_artifact` writes those bytes to the requested path; load it via
  :func:`load_package` (which uses ``torch._export.aot_load``).

The Inductor lockdown mirrors the deployed DeepMD freeze:

* ``assert_indirect_indexing=False`` — avoids an undeclared-temporary codegen
  bug for data-dependent gathers;
* ``scalar_asserts=False`` / ``size_asserts=False`` — the dynamic edge/node
  axes otherwise emit ``Expected u2 <= 265`` style guards that kill CPU
  artifacts;
* ``realize_opcount_threshold=0`` on CUDA — prevents NaN in the backward
  fusion of attention descriptors.

Keys unknown to the installed torch are dropped (torch 2.5 has none of the
three), so the same config dict is portable.

Compiler constraints: ``CXX`` must be a plain ``g++`` (not ``mpiCC``), and a
compiled artifact must be produced with the same toolchain that will load it.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Optional, Sequence

import torch

DEFAULT_INDUCTOR_OPTIONS = {
    "assert_indirect_indexing": False,
    "scalar_asserts": False,
    "size_asserts": False,
}


def default_inductor_configs(*, cuda: bool = False) -> dict[str, Any]:
    """Return the Inductor options an SOG artifact should be compiled with."""
    cfg = dict(DEFAULT_INDUCTOR_OPTIONS)
    if cuda:
        cfg["realize_opcount_threshold"] = 0
    return cfg


def filter_inductor_configs(cfg: dict[str, Any]) -> dict[str, Any]:
    """Drop keys the installed torch's Inductor config does not know."""
    try:
        from torch._inductor import config as _cfg

        known = set(_cfg.get_config_copy().keys())
    except Exception:  # pragma: no cover - very old torch
        return dict(cfg)
    return {k: v for k, v in cfg.items() if k in known}


def _has_package_api() -> bool:
    try:
        from torch._inductor import aoti_compile_and_package  # noqa: F401

        return True
    except ImportError:
        return False


def compile_artifact(
    exported: torch.export.ExportedProgram,
    package_path: str | os.PathLike,
    *,
    sample_inputs: Optional[Sequence[torch.Tensor]] = None,
    inductor_configs: Optional[dict[str, Any]] = None,
    cuda: bool = False,
    source_fn=None,
) -> bytes:
    """Compile ``exported`` and write the artifact to ``package_path``.

    On torch >= 2.6 this is a ``.pt2`` package; on torch 2.5 it is a plain
    ``.so`` (``sample_inputs`` is then required).  Returns the file bytes.
    """
    path = Path(package_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg = (
        default_inductor_configs(cuda=cuda)
        if inductor_configs is None
        else dict(inductor_configs)
    )
    cfg = filter_inductor_configs(cfg)

    if _has_package_api():
        from torch._inductor import aoti_compile_and_package

        aoti_compile_and_package(exported, package_path=str(path), inductor_configs=cfg)
    else:  # torch 2.5: compile to a shared library
        if sample_inputs is None:
            raise ValueError(
                "sample_inputs is required to compile an artifact on torch < 2.6"
            )
        # ``torch._export.aot_compile`` (unlike the low-level
        # ``torch._inductor.aot_compile``) also records the pytree call spec
        # that ``aot_load`` needs, so use it on the 2.5 generation.
        import torch._export as _export

        # Prefer the original callable (module method): Dynamo re-exports it
        # with a proper call spec; a make_fx GraphModule with many inputs is
        # not re-exportable.
        fn = source_fn or getattr(exported, "_sog_source_fn", None)
        if fn is None:
            fn = exported.graph_module
        so_path = _export.aot_compile(fn, tuple(sample_inputs), options=cfg)
        shutil.copyfile(so_path, path)
    return path.read_bytes()


# Backwards-compatible alias.
compile_and_package = compile_artifact


def load_package(path: str | os.PathLike, *, device: str = "cpu"):
    """Load a compiled artifact (``.pt2`` package or torch-2.5 ``.so``).

    ``device`` is only used by the torch-2.5 ``aot_load`` path.
    """
    try:
        from torch._inductor import aoti_load_package

        return aoti_load_package(str(path))
    except ImportError:
        import torch._export as _export

        return _export.aot_load(str(path), device)


def check_toolchain() -> list[str]:
    """Return human-readable warnings about the compile environment."""
    problems: list[str] = []
    cxx = os.environ.get("CXX", "")
    if cxx and any(tag in Path(cxx).name for tag in ("mpiCC", "mpicxx", "mpic++")):
        problems.append(
            f"CXX={cxx} is an MPI compiler wrapper; Inductor's PCH handling and "
            "torch's ABI probing break under it — use g++."
        )
    gcc = shutil.which("g++")
    if gcc:
        try:
            out = subprocess.run([gcc, "-dumpversion"], capture_output=True, text=True)
            major = int((out.stdout.strip() or "0").split(".")[0])
            if major < 9:
                problems.append(
                    f"g++ {major} is too old for libtorch (needs >= 9); load a newer "
                    "toolchain (e.g. gcc-toolset-13) before freezing."
                )
        except Exception:  # pragma: no cover - best effort
            pass
    return problems
