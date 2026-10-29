"""A framework-neutral reference model for the export-pipeline tests.

It implements the three traceable protocols from :mod:`sog.v2.export.abi`:

* ``FullLRTraceable``     — SR + LR in one graph (1F+1B, Mode B);
* ``EnergyOnlyTraceable`` — the Mode-A pair forward (``energy`` + ``latent_charge``);
* ``ResponseTraceable``   — the Mode-A response VJP (graph + ``v_charge``).

The physics is deliberately trivial and closed-form:

* charge head: ``q_raw_i = w · coord_i`` (so ``dq/dx = w``);
* short-range energy: ``0.5 * |coord|^2`` (force ``-coord``);
* long range: neutralization + the static-grid reference solver.

It exists to exercise the *artifact mechanics* (trace, export, packaging,
metadata), not to be a realistic model.  The class is an ``nn.Module`` with
``w`` registered as a buffer so the torch-2.5 AOTI path (which re-exports a
callable via Dynamo) can capture it.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from sog.v2.external_kspace import neutralize_reference
from sog.v2.export.abi import make_coord_box_samples
from sog.v2.export.trace import trace_with_make_fx
from sog.v2.exportable import ReferenceKSpace, TraceableSogKernel
from sog.v2.types import vir6_to_3x3


class TinyChargeModel(nn.Module):
    def __init__(self, kernel, *, nk: int = 24, n_dl: float = 0.5,
                 box_length: float = 10.0) -> None:
        super().__init__()
        self.box_length = float(box_length)
        self.register_buffer("w", torch.tensor([0.3, -0.2, 0.1], dtype=torch.float64))
        self.ref = ReferenceKSpace(
            TraceableSogKernel.from_kernel(kernel, n_dl=float(n_dl), nk=int(nk))
        )

    # ── graph tensors ────────────────────────────────────────────────────
    def _graph_inputs(self, n_node: int = 8, n_edge: int = 12,
                      dtype=torch.float64, device=torch.device("cpu")):
        N, E = int(n_node), int(n_edge)
        atype = torch.zeros(N, dtype=torch.int64, device=device)
        n_node_t = torch.tensor([N], dtype=torch.int64, device=device)
        n_local_t = torch.tensor([N], dtype=torch.int64, device=device)
        src = torch.arange(E, device=device) % N
        dst = (src + 1) % N
        edge_index = torch.stack([src, dst]).to(torch.int64)
        edge_vec = torch.zeros(E, 3, dtype=dtype, device=device)
        edge_mask = torch.ones(E, dtype=dtype, device=device)
        dest_order = torch.argsort(dst)
        dest_row = torch.searchsorted(dst[dest_order], torch.arange(N + 1, device=device))
        src_order = torch.argsort(src)
        src_row = torch.searchsorted(src[src_order], torch.arange(N + 1, device=device))
        coord, box = make_coord_box_samples(
            N, self.box_length, dtype=dtype, device=device
        )
        coord = coord.reshape(-1, 3)
        return (
            atype, n_node_t, n_local_t, edge_index, edge_vec, edge_mask,
            dest_order.to(torch.int64), dest_row.to(torch.int64),
            src_order.to(torch.int64), src_row.to(torch.int64), coord, box,
        )

    # ── traceable callables (nn.Module methods: AOTI can re-export them) ──
    def full_lr(self, atype, n_node, n_local, edge_index, edge_vec, edge_mask,
                destination_order, destination_row_ptr, source_order,
                source_row_ptr, coord, box):
        w = self.w
        q_raw = (coord * w).sum(-1)
        qn = neutralize_reference(q_raw, n_local)
        v, f_dir, vir6 = self.ref.solve(qn, coord, box)
        seed = (v.reshape(-1) * qn.reshape(-1)).sum()
        e_sr = 0.5 * (coord * coord).sum()
        f_sr = -coord
        f_resp = -v.unsqueeze(-1) * w
        w_tot = (
            f_sr.transpose(0, 1) @ coord
            + f_resp.transpose(0, 1) @ coord
            + vir6_to_3x3(vir6)
        )
        return {
            "energy": (e_sr.detach() + 0.5 * seed.detach()).reshape(1),
            "force": f_sr + f_resp + f_dir,
            "virial": w_tot.reshape(1, 9),
        }

    def energy_only(self, atype, n_node, n_local, edge_index, edge_vec,
                    edge_mask, destination_order, destination_row_ptr,
                    source_order, source_row_ptr, coord, box):
        e_sr = 0.5 * (coord * coord).sum().reshape(1)
        q_raw = ((coord * self.w).sum(-1)).reshape(-1, 1)
        return {"energy": e_sr, "latent_charge": q_raw}

    def response(self, atype, n_node, n_local, edge_index, edge_vec, edge_mask,
                 destination_order, destination_row_ptr, source_order,
                 source_row_ptr, coord, box, v_charge):
        q_raw = (coord * self.w).sum(-1)
        qn = neutralize_reference(q_raw, n_local)
        v = v_charge.reshape(-1)
        f_resp = -v.unsqueeze(-1) * self.w
        w_resp = f_resp.transpose(0, 1) @ coord
        return {
            "latent_charge": qn.reshape(-1, 1),
            "charge_response": f_resp,
            "charge_response_virial": w_resp.reshape(9),
        }

    # ── protocols ────────────────────────────────────────────────────────
    def full_lr_sample_inputs(self, *, n_node: int = 8, device=torch.device("cpu"),
                              dtype=torch.float64):
        return self._graph_inputs(n_node=n_node, device=device, dtype=dtype)

    def full_lr_dynamic_shapes(self, *sample_inputs):
        return None  # the fixture keeps static shapes

    def trace_full_lr(self, *inputs, tracing_mode: str = "symbolic", **kwargs):
        return trace_with_make_fx(_FullLR(self), inputs, tracing_mode=tracing_mode)

    def energy_only_sample_inputs(self, *, n_node: int = 8,
                                  device=torch.device("cpu"), dtype=torch.float64):
        return self._graph_inputs(n_node=n_node, device=device, dtype=dtype)

    def trace_energy_only(self, *inputs, tracing_mode: str = "symbolic", **kwargs):
        return trace_with_make_fx(_EnergyOnly(self), inputs, tracing_mode=tracing_mode)

    def response_sample_inputs(self, *, n_node: int = 8,
                               device=torch.device("cpu"), dtype=torch.float64):
        inputs = self._graph_inputs(n_node=n_node, device=device, dtype=dtype)
        v_charge = torch.rand(
            (n_node, 1), generator=torch.Generator().manual_seed(11),
            dtype=torch.float64,
        ).to(dtype=dtype, device=device)
        return tuple(inputs) + (v_charge,)

    def trace_response(self, *inputs, tracing_mode: str = "symbolic", **kwargs):
        return trace_with_make_fx(_Response(self), inputs, tracing_mode=tracing_mode)

    # AOTI (torch 2.5) re-exports a callable: hand over the wrapper modules.
    @property
    def full_lr_callable(self):
        return _FullLR(self)

    @property
    def response_callable(self):
        return _Response(self)


class _FullLR(nn.Module):
    def __init__(self, model: "TinyChargeModel") -> None:
        super().__init__()
        self.model = model

    def forward(self, atype, n_node, n_local, edge_index, edge_vec, edge_mask,
                destination_order, destination_row_ptr, source_order,
                source_row_ptr, coord, box):
        return self.model.full_lr(
            atype, n_node, n_local, edge_index, edge_vec, edge_mask,
            destination_order, destination_row_ptr, source_order,
            source_row_ptr, coord, box,
        )


class _EnergyOnly(nn.Module):
    def __init__(self, model: "TinyChargeModel") -> None:
        super().__init__()
        self.model = model

    def forward(self, atype, n_node, n_local, edge_index, edge_vec, edge_mask,
                destination_order, destination_row_ptr, source_order,
                source_row_ptr, coord, box):
        return self.model.energy_only(
            atype, n_node, n_local, edge_index, edge_vec, edge_mask,
            destination_order, destination_row_ptr, source_order,
            source_row_ptr, coord, box,
        )


class _Response(nn.Module):
    def __init__(self, model: "TinyChargeModel") -> None:
        super().__init__()
        self.model = model

    def forward(self, atype, n_node, n_local, edge_index, edge_vec, edge_mask,
                destination_order, destination_row_ptr, source_order,
                source_row_ptr, coord, box, v_charge):
        return self.model.response(
            atype, n_node, n_local, edge_index, edge_vec, edge_mask,
            destination_order, destination_row_ptr, source_order,
            source_row_ptr, coord, box, v_charge,
        )
