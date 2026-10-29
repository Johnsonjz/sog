"""Artifact ABIs and the protocols a framework implements.

The SOG library standardises the *SOG-specific* parts of an exported artifact
(the solver call, the neutralized seed, the output keys and the metadata), but
it cannot know a framework's graph representation.  A framework therefore
implements the protocols below; the library provides tracing/export/packaging
around them.

The canonical DeepMD full-LR ABIs are recorded here as constants so adapters
can mirror them exactly:

* single rank: 12 inputs = 10 graph tensors + ``coord (nloc,3)`` + ``box (9,)``;
* with comm:   20 inputs = 10 graph tensors + 8 comm tensors + ``coord`` + ``box``
  (the DeepMD export also carries three ``None`` conditioning slots, making its
  on-disk ABI 23 inputs; they are not part of the SOG contract).
"""

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

import torch

GRAPH_TENSOR_NAMES = (
    "atype",
    "n_node",
    "n_local",
    "edge_index",
    "edge_vec",
    "edge_mask",
    "destination_order",
    "destination_row_ptr",
    "source_order",
    "source_row_ptr",
)

COMM_TENSOR_NAMES = (
    "send_list",
    "send_proc",
    "recv_proc",
    "send_num",
    "recv_num",
    "communicator",
    "nlocal",
    "nghost",
)

SINGLE_RANK_FULL_LR_INPUTS = GRAPH_TENSOR_NAMES + ("coord", "box")
WITH_COMM_FULL_LR_INPUTS = GRAPH_TENSOR_NAMES + COMM_TENSOR_NAMES + ("coord", "box")

FULL_LR_OUTPUT_KEYS = ("energy", "force", "virial")
ENERGY_ONLY_OUTPUT_KEYS = ("energy", "latent_charge")
RESPONSE_OUTPUT_KEYS = ("latent_charge", "charge_response", "charge_response_virial")


@runtime_checkable
class FullLRTraceable(Protocol):
    """Single-graph SR+LR trace entry point (1F+1B)."""

    def full_lr_sample_inputs(self, *, n_node: int, device: torch.device = torch.device("cpu"),
                              dtype: torch.dtype = torch.float64) -> tuple: ...

    def full_lr_dynamic_shapes(self, *sample_inputs) -> tuple: ...

    def trace_full_lr(self, *inputs, tracing_mode: str = "symbolic", **kwargs) -> torch.nn.Module: ...


@runtime_checkable
class EnergyOnlyTraceable(Protocol):
    """Forward-only artifact: returns ``(energy, latent_charge)`` (Mode A pair)."""

    def energy_only_sample_inputs(self, *, n_node: int, device=torch.device("cpu"),
                                  dtype=torch.float64) -> tuple: ...

    def trace_energy_only(self, *inputs, tracing_mode: str = "symbolic", **kwargs) -> torch.nn.Module: ...


@runtime_checkable
class ResponseTraceable(Protocol):
    """Response VJP artifact: takes the graph + ``v_charge`` and returns the
    charge-response force/virial (Mode A fix)."""

    def response_sample_inputs(self, *, n_node: int, device=torch.device("cpu"),
                               dtype=torch.float64) -> tuple: ...

    def trace_response(self, *inputs, tracing_mode: str = "symbolic", **kwargs) -> torch.nn.Module: ...


def validate_output_keys(outputs: Any, expected: Sequence[str], *, what: str = "artifact") -> None:
    """Fail loudly when a traced module's output keys diverge from the ABI.

    The C++ consumers index outputs positionally, so a silently reordered dict
    corrupts force/virial.
    """
    if isinstance(outputs, dict):
        got = tuple(outputs.keys())
    elif isinstance(outputs, (tuple, list)):
        got = tuple(str(i) for i in range(len(outputs)))
    else:
        raise TypeError(f"{what}: traced module returned {type(outputs)!r}")
    if got != tuple(expected):
        raise RuntimeError(
            f"{what} output keys diverge from the ABI: got {got}, expected {tuple(expected)}"
        )


def make_coord_box_samples(
    nloc: int,
    box_length: float,
    *,
    n_frames: int = 1,
    dtype: torch.dtype = torch.float64,
    device: torch.device = torch.device("cpu"),
):
    """Deterministic ``(coord, box)`` samples for tracing a full-LR artifact."""
    g = torch.Generator(device="cpu").manual_seed(1234)
    coord = (torch.rand(n_frames * nloc, 3, generator=g, dtype=torch.float64) * box_length)
    coord = coord.view(n_frames, nloc, 3).to(dtype=dtype, device=device)
    box = torch.diag(torch.tensor([box_length] * 3, dtype=torch.float64)).reshape(9)
    box = box.to(dtype=dtype, device=device)
    return coord, box
