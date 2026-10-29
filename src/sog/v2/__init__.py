"""``sog.v2`` — PyTorch 2.x deployment layer (isolated from legacy ``sog``).

This subpackage carries the *new* calling patterns (in-graph custom op,
``torch.export`` / AOTInductor artifacts, exact neutralization VJP).  It is
deliberately isolated:

* the legacy ``sog`` package never imports ``sog.v2``;
* ``sog.v2`` imports only the kernel-parameter contract
  (:class:`sog.kernel.SogKernel`) from the legacy package — never the legacy
  solver (``sog.Gaussian`` / ``sog.Sog``);
* importing ``sog.v2`` does not load CUDA or the op library.  The op is
  loaded on first use (:func:`sog.v2.kspace_op.load_op_library`), and every
  feature degrades to the pure-PyTorch reference backend when it is absent.

Modules
-------
``kspace_op``       loader + typed wrappers + exact neutralization VJP
``external_kspace`` ``E_k = 1/2 sum v q`` algebra, seed, share semantics
``exportable``      trace-safe static-grid reference kernel (CPU fallback)
``export``          ``torch.export`` / AOTI artifact helpers (two chains)
"""

from __future__ import annotations

from typing import Any

_LAZY = {
    "kspace_op": ".kspace_op",
    "external_kspace": ".external_kspace",
    "exportable": ".exportable",
    "export": ".export",
    "types": ".types",
    "SogSolution": ".types",
    "vir6_to_deepmd": ".types",
    "vir6_to_lammps6": ".types",
    "ExternalKSpace": ".external_kspace",
    "KSpaceSettings": ".external_kspace",
    "TraceableSogKernel": ".exportable",
    "ReferenceKSpace": ".exportable",
    "OpUnavailableError": ".kspace_op",
    "load_op_library": ".kspace_op",
    "op_available": ".kspace_op",
    "register_neutralize_autograd": ".kspace_op",
}

__all__ = list(_LAZY)


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        import importlib

        module = importlib.import_module(_LAZY[name], __name__)
        if _LAZY[name].rsplit(".", 1)[-1] == name:
            return module
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
