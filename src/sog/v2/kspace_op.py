"""Loader and typed wrappers for the ``sogkspace::*`` PyTorch custom ops.

The op library is built from ``op/`` (see ``op/CMakeLists.txt``) and provides:

* ``sogkspace::solve_auto`` — in-graph CubeS₂ mesh solve:
  ``(q, pos, box, amp, bandwidth, rcut, spline, phi_accuracy, box_tol,
    qscale, self_coeff, remove_self) -> (v, F_dir, vir6)``;
* ``sogkspace::neutralize`` — global owned-mean gauge projection (ghosts zeroed);
* ``sogkspace::qmean`` — value-only owned-mean reduction used by the exact VJP;
* ``sogkspace::setup`` / ``destroy`` / ``last_grid`` / ``set_comm``.

Everything here is import-safe without CUDA or the op library: nothing is
loaded until :func:`load_op_library` (or :class:`sog.v2.external_kspace.ExternalKSpace`
with the op backend) is called.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch

OP_NAMESPACE = "sogkspace"
DEFAULT_LIB_ENV = "SOG_KS_OP_LIB"

#: The op names this module depends on.
REQUIRED_OPS = ("solve_auto", "neutralize", "qmean")

_NEUTRALIZE_AUTGRAD_REGISTERED = False


class OpUnavailableError(RuntimeError):
    """Raised when the ``sogkspace`` op library is required but not available."""


# ── discovery / loading ──────────────────────────────────────────────────


def _namespace():
    return getattr(torch.ops, OP_NAMESPACE, None)


def _op_loaded() -> bool:
    ns = _namespace()
    return ns is not None and all(hasattr(ns, name) for name in REQUIRED_OPS)


def _candidate_paths(explicit: Optional[Path]) -> list[Path]:
    paths: list[Path] = []
    if explicit is not None:
        paths.append(Path(explicit))
    env = os.environ.get(DEFAULT_LIB_ENV)
    if env:
        paths.append(Path(env))
    here = Path(__file__).resolve()
    # src/sog/v2/kspace_op.py -> repo root
    root = here.parents[3]
    paths.extend(
        [
            here.parent / "_lib" / "libsog_kspace_op.so",
            root / "op" / "build" / "libsog_kspace_op.so",
            root / "build_op" / "libsog_kspace_op.so",
            root / "op" / "build" / "lib" / "libsog_kspace_op.so",
        ]
    )
    return paths


def resolve_library(path=None) -> Optional[Path]:
    """Return the first existing candidate, or ``None``.

    Resolution order: explicit ``path`` -> ``$SOG_KS_OP_LIB`` -> packaged
    ``sog/v2/_lib/`` -> in-repo build directories (``op/build``, ``build_op``).
    """
    explicit = Path(path) if path is not None else None
    if explicit is not None and not explicit.is_file():
        raise OpUnavailableError(f"sogkspace op library not found at {explicit}")
    for candidate in _candidate_paths(explicit):
        if candidate.is_file():
            return candidate
    return None


def load_op_library(path=None) -> bool:
    """Load (idempotently) the ``sogkspace`` op library.

    Returns ``True`` when the ops are available, ``False`` when no library
    could be found.  A library that was found but fails to load raises
    :class:`OpUnavailableError` (a broken library is never silently ignored).
    """
    if _op_loaded():
        return True
    lib = resolve_library(path)
    if lib is None:
        return False
    try:
        torch.ops.load_library(str(lib))
    except Exception as exc:  # pragma: no cover - platform dependent
        raise OpUnavailableError(
            f"failed to load the sogkspace op library {lib}: {exc}"
        ) from exc
    if not _op_loaded():
        raise OpUnavailableError(
            f"{lib} loaded but does not register the expected ops "
            f"{OP_NAMESPACE}::{', '.join(REQUIRED_OPS)}"
        )
    return True


def op_available() -> bool:
    """True when the ``sogkspace`` ops are already registered in this process."""
    return _op_loaded()


def require_op() -> None:
    """Load the library or raise a precise :class:`OpUnavailableError`."""
    if not load_op_library():
        raise OpUnavailableError(
            "the sogkspace op library was not found. Build it with "
            "`cmake -S op -B build_op -DTORCH_DIR=<torch cmake dir>` and set "
            f"${DEFAULT_LIB_ENV} to the resulting libsog_kspace_op.so, or use the "
            "pure-PyTorch reference backend (sog.v2.external_kspace backend='reference')."
        )


# ── validation helpers ───────────────────────────────────────────────────


def _check_tensor(name: str, t: torch.Tensor, *, dtype=torch.float64) -> None:
    if not torch.is_tensor(t):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(t)!r}")
    if t.dtype != dtype:
        raise TypeError(f"{name} must be {dtype}, got {t.dtype}")


def as_box9(box: torch.Tensor) -> torch.Tensor:
    """Return the box as a ``(9,)`` row-major tensor, validating orthogonality."""
    _check_tensor("box", box)
    if box.shape == (3, 3):
        box9 = box.reshape(9)
    elif box.shape == (9,):
        box9 = box
    else:
        raise ValueError(f"box must be (9,) or (3,3), got {tuple(box.shape)}")
    off_diag = torch.stack([box9[1], box9[2], box9[3], box9[5], box9[6], box9[7]])
    if bool((off_diag != 0).any()):
        raise ValueError(
            "sogkspace::solve_auto supports orthorhombic boxes only "
            "(the deployed solver rejects tilted cells too)"
        )
    return box9


def _check_q_pos(q: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    _check_tensor("q", q)
    _check_tensor("pos", pos)
    if q.dim() == 2 and q.shape[1] == 1:
        q = q.reshape(-1)
    if q.dim() != 1:
        raise ValueError(f"q must be 1-D (or (N,1)), got {tuple(q.shape)}")
    if pos.dim() != 2 or pos.shape[1] != 3:
        raise ValueError(f"pos must be (N,3), got {tuple(pos.shape)}")
    if pos.shape[0] != q.shape[0]:
        raise ValueError(
            f"pos has {pos.shape[0]} atoms but q has {q.shape[0]}"
        )
    return q


def _n_local_tensor(n_local, n: int, *, dtype, device) -> torch.Tensor:
    if n_local is None:
        return torch.tensor(float(n), dtype=dtype, device=device)
    if torch.is_tensor(n_local):
        return n_local.to(dtype=dtype, device=device).reshape(-1)[0]
    return torch.tensor(float(n_local), dtype=dtype, device=device)


# ── op wrappers ──────────────────────────────────────────────────────────


def solve_auto(
    q: torch.Tensor,
    pos: torch.Tensor,
    box: torch.Tensor,
    amp: torch.Tensor,
    bandwidth: torch.Tensor,
    *,
    rcut: float,
    spline: int = 4,
    phi_accuracy: float = 1e-4,
    box_tol: float = 1e-3,
    qscale: float = 14.3996454784255,
    self_coeff: float = 0.0,
    remove_self: bool = True,
    handle: int = 1,
):
    """Run the in-graph SOG solve; returns ``(v, F_dir, vir6)``.

    ``bandwidth`` must be the **squared** bandwidths (the C++ kernel consumes
    ``exp(-0.5 * bandwidth * k^2)``); ``amp`` is the internal amplitude.
    The op is CUDA-only: the caller must supply CUDA tensors.
    """
    require_op()
    q = _check_q_pos(q, pos)
    box9 = as_box9(box)
    if not q.is_cuda:
        raise OpUnavailableError(
            "sogkspace::solve_auto is CUDA-only; move the tensors to CUDA or "
            "use the pure-PyTorch reference backend"
        )
    for name, t in (("amp", amp), ("bandwidth", bandwidth)):
        if t.dim() != 1 or t.numel() == 0:
            raise ValueError(f"{name} must be a non-empty 1-D tensor")
    if amp.numel() != bandwidth.numel():
        raise ValueError(
            f"amp and bandwidth must have the same length "
            f"({amp.numel()} vs {bandwidth.numel()})"
        )
    v, f_dir, vir6 = torch.ops.sogkspace.solve_auto(
        int(handle), q.contiguous(), pos.contiguous(), box9.contiguous(),
        amp.to(device=q.device, dtype=torch.float64).contiguous(),
        bandwidth.to(device=q.device, dtype=torch.float64).contiguous(),
        float(rcut), int(spline), float(phi_accuracy), float(box_tol),
        float(qscale), float(self_coeff), bool(remove_self),
    )
    return v, f_dir, vir6


def neutralize(q: torch.Tensor, n_local=None) -> torch.Tensor:
    """Gauge projection: global owned mean removed, ghost rows zeroed."""
    require_op()
    if not torch.is_tensor(q):
        raise TypeError("q must be a torch.Tensor")
    n = q.shape[0]
    nl = _n_local_tensor(n_local, n, dtype=q.dtype, device=q.device)
    return torch.ops.sogkspace.neutralize(q, nl)


def qmean(q: torch.Tensor, n_local=None) -> torch.Tensor:
    """Global owned-prefix mean per charge channel (value-only)."""
    require_op()
    n = q.shape[0]
    nl = _n_local_tensor(n_local, n, dtype=q.dtype, device=q.device)
    return torch.ops.sogkspace.qmean(q, nl)


def set_comm(comm_ptr: int) -> None:
    """Register the framework's MPI communicator (LAMMPS ``world``)."""
    require_op()
    torch.ops.sogkspace.set_comm(int(comm_ptr))


def last_grid(handle: int = 1) -> torch.Tensor:
    """Return the ``(nx, ny, nz)`` mesh the last ``solve_auto`` used."""
    require_op()
    return torch.ops.sogkspace.last_grid(int(handle))


def destroy(handle: int = 1) -> None:
    """Release the solver state attached to ``handle``."""
    if not _op_loaded():
        return
    torch.ops.sogkspace.destroy(int(handle))


# ── exact autograd rule for neutralize ───────────────────────────────────


def _neutralize_setup_context(ctx, inputs, output):
    q, n_local = inputs
    ctx.save_for_backward(n_local)
    ctx.n_total = q.shape[0]
    ctx.q_ndim = q.dim()
    ctx.dtype = q.dtype


def _neutralize_backward(ctx, grad_out):
    (n_local,) = ctx.saved_tensors
    # Owned rows are the [0, n_local) prefix; the forward zeroes the rest, so
    # their VJP is exactly zero (a constant output row).
    node_index = torch.arange(ctx.n_total, dtype=torch.int64, device=grad_out.device)
    owned = (node_index < n_local.reshape(-1)[0]).to(grad_out.dtype)
    if ctx.q_ndim > 1:
        owned = owned.reshape((-1,) + (1,) * (ctx.q_ndim - 1))
    # Global (all-rank) mean of the cotangent over the owned rows.
    mean = torch.ops.sogkspace.qmean(grad_out, n_local)
    return owned * (grad_out - mean), None


def register_neutralize_autograd() -> None:
    """Register the exact VJP on ``sogkspace::neutralize`` (idempotent).

    The forward is a value-only CUDA kernel; without this rule ``sum(v*qn)``
    does not backpropagate through the projection and the charge-response
    force silently loses the ``-mean`` term.  The backward calls only
    ``qmean`` and elementwise ops, so make_fx / ``torch.export`` / AOTI can
    trace it (never a C++ ``autograd::Function``).
    """
    global _NEUTRALIZE_AUTGRAD_REGISTERED
    if _NEUTRALIZE_AUTGRAD_REGISTERED:
        return
    require_op()
    torch.library.register_autograd(
        f"{OP_NAMESPACE}::neutralize",
        _neutralize_backward,
        setup_context=_neutralize_setup_context,
    )
    _NEUTRALIZE_AUTGRAD_REGISTERED = True
