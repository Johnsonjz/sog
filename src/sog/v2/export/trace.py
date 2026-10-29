"""``make_fx`` trace + ``torch.export`` for SOG artifacts.

Mirrors the DeepMD freeze pipeline:

    make_fx(fn, tracing_mode="symbolic", _allow_non_fake_inputs=True)
      -> torch.export.export(..., strict=False,
                             prefer_deferred_runtime_asserts_over_guards=True)
      -> strip deferred shape assertions
      -> optional move_to_device_pass

The traced function must be free of host syncs (``.item()``/``.tolist()``) and
data-dependent Python control flow; this is why the solver used inside an
exported graph is either the ``sogkspace`` op (opaque) or
:class:`sog.v2.exportable.TraceableSogKernel` (static grid).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import torch


@dataclass
class ExportedArtifact:
    """Result of tracing + exporting one artifact."""

    exported: torch.export.ExportedProgram
    output_keys: tuple[str, ...] = field(default_factory=tuple)
    payload: Optional[bytes] = None


def strip_shape_assertions(graph_module: torch.nn.Module) -> None:
    """Neutralise ``aten._assert_scalar`` nodes (deferred shape guards).

    ``torch.export(..., prefer_deferred_runtime_asserts_over_guards=True)``
    inserts assertions with opaque symbolic names; the SOG ABI's inputs are
    always well-formed, so replace each condition with ``True`` (rewriting the
    node, not erasing it — erasing can yield NaN gradients on some versions).
    """
    graph = graph_module.graph
    for node in list(graph.nodes):
        if (
            node.op == "call_function"
            and node.target is torch.ops.aten._assert_scalar.default
        ):
            node.args = (True, node.args[1])
    graph_module.recompile()


def trace_with_make_fx(
    fn, sample_inputs: Sequence[torch.Tensor], *, tracing_mode: str = "symbolic"
) -> torch.nn.Module:
    """Trace ``fn`` with ``make_fx`` (symbolic shapes)."""
    from torch.fx.experimental.proxy_tensor import make_fx

    traced = make_fx(fn, tracing_mode=tracing_mode, _allow_non_fake_inputs=True)(
        *sample_inputs
    )
    # Keep the original callable: torch 2.5's AOTI path re-exports a callable
    # (a make_fx GraphModule with many inputs is not re-exportable by Dynamo).
    traced._sog_source_fn = fn
    return traced


def export_program(
    traced: torch.nn.Module,
    sample_inputs: Sequence[torch.Tensor],
    dynamic_shapes: Any = None,
    *,
    device: Optional[torch.device] = None,
    strict: bool = False,
) -> torch.export.ExportedProgram:
    """Run ``torch.export.export`` on a make_fx trace and post-process it."""
    # ``prefer_deferred_runtime_asserts_over_guards`` landed after torch 2.5;
    # pass it only when the installed torch exposes it.
    import inspect

    kwargs = {"dynamic_shapes": dynamic_shapes}
    params = inspect.signature(torch.export.export).parameters
    if "strict" in params:
        kwargs["strict"] = strict
    if "prefer_deferred_runtime_asserts_over_guards" in params:
        kwargs["prefer_deferred_runtime_asserts_over_guards"] = True
    exported = torch.export.export(traced, tuple(sample_inputs), **kwargs)
    try:
        exported._sog_source_fn = getattr(traced, "_sog_source_fn", None)
    except Exception:  # pragma: no cover - ExportedProgram may be frozen
        pass
    strip_shape_assertions(exported.graph_module)
    if device is not None and torch.device(device).type != "cpu":
        try:
            from torch.export.passes import move_to_device_pass
        except ImportError:  # pragma: no cover - older torch
            raise RuntimeError(
                "this torch version has no torch.export.passes.move_to_device_pass; "
                "trace/export on CPU and compile for the target device instead"
            )
        exported = move_to_device_pass(exported, torch.device(device))
    return exported


def trace_and_export(
    fn,
    sample_inputs: Sequence[torch.Tensor],
    dynamic_shapes: Any = None,
    *,
    device: Optional[torch.device] = None,
    tracing_mode: str = "symbolic",
) -> tuple[torch.nn.Module, torch.export.ExportedProgram]:
    """Convenience wrapper: trace then export."""
    traced = trace_with_make_fx(fn, sample_inputs, tracing_mode=tracing_mode)
    exported = export_program(traced, sample_inputs, dynamic_shapes, device=device)
    return traced, exported


def output_keys_of(
    traced: torch.nn.Module, sample_inputs: Sequence[torch.Tensor]
) -> tuple[str, ...]:
    """Run the traced module on the sample inputs and return its output keys."""
    with torch.no_grad():
        out = traced(*sample_inputs)
    if not isinstance(out, dict):
        raise TypeError("the traced module must return a dict of tensors")
    return tuple(out.keys())
