"""MACE -> ``sog.v2`` deployment migration: a runnable reference.

This is the concrete MACE instance of the procedure in the README port section
("Port SOG into your framework (for agents)") and
``docs/10-export-and-artifacts.md``.  It shows how to take
the training-side ``MACESOG`` (``mace/mace/modules/extensions.py``) and expose the
same computation as the three ``sog.v2`` traceable protocols, so the model can be
exported to Mode-A (pair+kspace+fix, 2F+1B) and Mode-B (full_lr, 1F+1B) artifacts.

The model is a MACE-*shaped stand-in*, not the real MACE -- MACE/e3nn are not
imported here.  It preserves the exact coupling points a real port must hit while
staying runnable from a fresh checkout:

* node features        ``f_i = W @ coord_i``        (MACE ``node_feats``)
* short-range readout  ``e_i = 0.5 |coord_i|^2``    (MACE ``readouts``)
* charge readout       ``q_i = a . f_i``            (MACE ``sog_readouts``)
* long range           ``E_k`` from the reference solver (MACE ``self.sog``)

Run from the repo root::

    python example/mace/mace_sog_deployment.py            # trace + export + metadata
    python example/mace/mace_sog_deployment.py --compile  # also AOTI-compile (torch>=2.6)

The ``--compile`` step needs the freeze toolchain to match the load toolchain
(``CXX=g++``, gcc >= 9); :func:`sog.v2.export.check_toolchain` reports problems.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import tempfile

# Run from a fresh checkout without ``pip install -e .``: put ``src/`` on the path.
_SRC = pathlib.Path(__file__).resolve().parents[2] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch
import torch.nn as nn

from sog import SogKernel
from sog.v2.external_kspace import neutralize_reference
from sog.v2.export import (
    ENERGY_ONLY_OUTPUT_KEYS,
    FULL_LR_OUTPUT_KEYS,
    RESPONSE_OUTPUT_KEYS,
    build_metadata,
    check_toolchain,
    compile_artifact,
    export_program,
    make_coord_box_samples,
    trace_with_make_fx,
    validate_output_keys,
)
from sog.v2.exportable import ReferenceKSpace, TraceableSogKernel
from sog.v2.types import vir6_to_3x3


class MACESOGDeployable(nn.Module):
    """MACE-shaped stand-in implementing the three ``sog.v2`` traceable protocols.

    The physics is deliberately closed-form so the mechanics (trace, export,
    metadata) are visible without a real MACE backbone.  The three coupling
    points are the same ones ``MACESOG.forward`` hits:

    * ``node_feats`` -> charge head (``sog_readouts``);
    * ``node_feats`` -> SR energy (``readouts``);
    * ``q`` -> ``self.sog`` -> ``E_lr``.

    Weights are registered as **buffers** (not plain attributes) so the
    torch-2.5 AOTI path -- which re-exports a callable via Dynamo -- can capture
    them.
    """

    def __init__(
        self,
        kernel: SogKernel,
        *,
        n_dl: float = 0.5,
        nk: int = 24,
        box_length: float = 10.0,
        feat_dim: int = 4,
    ) -> None:
        super().__init__()
        self.box_length = float(box_length)
        self.register_buffer(
            "W", torch.randn(feat_dim, 3, dtype=torch.float64) * 0.3
        )
        self.register_buffer(
            "a", torch.randn(feat_dim, dtype=torch.float64) * 0.2
        )
        self.ref = ReferenceKSpace(
            TraceableSogKernel.from_kernel(kernel, n_dl=float(n_dl), nk=int(nk))
        )

    # ── the graph input sample (10 graph tensors + coord + box) ──────────
    def _graph_inputs(
        self,
        n_node: int = 8,
        n_edge: int | None = None,
        dtype: torch.dtype = torch.float64,
        device: torch.device = torch.device("cpu"),
    ):
        N = int(n_node)
        E = int(n_edge if n_edge is not None else max(1, (3 * N) // 2))
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

    # ── traceable callables (nn.Module methods; AOTI can re-export them) ──
    def _node_features(self, coord: torch.Tensor) -> torch.Tensor:
        return coord @ self.W.t()                      # (N, feat_dim)

    def _short_range(self, coord: torch.Tensor):
        e_sr = 0.5 * (coord * coord).sum()             # scalar
        f_sr = -coord                                  # -dE_sr/dr
        return e_sr, f_sr

    def _charge_head(self, coord: torch.Tensor) -> torch.Tensor:
        return (self._node_features(coord) * self.a).sum(-1)   # (N,)

    def full_lr(
        self, atype, n_node, n_local, edge_index, edge_vec, edge_mask,
        destination_order, destination_row_ptr, source_order, source_row_ptr,
        coord, box,
    ):
        """Mode B (1F+1B): SR + LR in one graph.  Mirrors the DeepMD full-LR body."""
        e_sr, f_sr = self._short_range(coord)
        q_raw = self._charge_head(coord)
        qn = neutralize_reference(q_raw, n_local)       # global owned mean, ghosts -> 0
        v, f_dir, vir6 = self.ref.solve(qn, coord, box)
        seed = (v.reshape(-1) * qn.reshape(-1)).sum()   # NOT 0.5 * seed
        # dq_i/dr_i = a @ W (constant) -> F_resp = -v_i * (a @ W)
        dq_dr = self.a @ self.W
        f_resp = -v.unsqueeze(-1) * dq_dr
        w_tot = (
            f_sr.t() @ coord
            + f_resp.t() @ coord
            + vir6_to_3x3(vir6)                         # complete strain form
        )
        return {
            "energy": (e_sr.detach() + 0.5 * seed.detach()).reshape(1),
            "force": f_sr + f_resp + f_dir,
            "virial": w_tot.reshape(1, 9),
        }

    def energy_only(
        self, atype, n_node, n_local, edge_index, edge_vec, edge_mask,
        destination_order, destination_row_ptr, source_order, source_row_ptr,
        coord, box,
    ):
        """Mode A pair forward: short-range energy + raw latent charge."""
        e_sr, _ = self._short_range(coord)
        q_raw = self._charge_head(coord).reshape(-1, 1)
        return {"energy": e_sr.reshape(1), "latent_charge": q_raw}

    def response(
        self, atype, n_node, n_local, edge_index, edge_vec, edge_mask,
        destination_order, destination_row_ptr, source_order, source_row_ptr,
        coord, box, v_charge,
    ):
        """Mode A fix VJP: graph + v_charge -> charge-response force/virial."""
        q_raw = self._charge_head(coord)
        qn = neutralize_reference(q_raw, n_local)
        v = v_charge.reshape(-1)
        dq_dr = self.a @ self.W
        f_resp = -v.unsqueeze(-1) * dq_dr
        w_resp = f_resp.t() @ coord
        return {
            "latent_charge": qn.reshape(-1, 1),
            "charge_response": f_resp,
            "charge_response_virial": w_resp.reshape(9),
        }

    # ── protocols ────────────────────────────────────────────────────────
    def full_lr_sample_inputs(self, *, n_node=8, device=torch.device("cpu"),
                              dtype=torch.float64):
        return self._graph_inputs(n_node=n_node, device=device, dtype=dtype)

    def full_lr_dynamic_shapes(self, *sample_inputs):
        return None  # the stand-in keeps static shapes

    def trace_full_lr(self, *inputs, tracing_mode: str = "symbolic", **kwargs):
        return trace_with_make_fx(_FullLR(self), inputs, tracing_mode=tracing_mode)

    def energy_only_sample_inputs(self, *, n_node=8, device=torch.device("cpu"),
                                  dtype=torch.float64):
        return self._graph_inputs(n_node=n_node, device=device, dtype=dtype)

    def trace_energy_only(self, *inputs, tracing_mode: str = "symbolic", **kwargs):
        return trace_with_make_fx(_EnergyOnly(self), inputs, tracing_mode=tracing_mode)

    def response_sample_inputs(self, *, n_node=8, device=torch.device("cpu"),
                               dtype=torch.float64):
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
    def energy_only_callable(self):
        return _EnergyOnly(self)

    @property
    def response_callable(self):
        return _Response(self)


class _FullLR(nn.Module):
    def __init__(self, model: MACESOGDeployable) -> None:
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
    def __init__(self, model: MACESOGDeployable) -> None:
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
    def __init__(self, model: MACESOGDeployable) -> None:
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


def _torch_has_package_api() -> bool:
    try:
        from torch._inductor import aoti_compile_and_package  # noqa: F401

        return True
    except ImportError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MACE -> sog.v2 deployment migration (runnable reference)."
    )
    parser.add_argument("--n-node", type=int, default=8)
    parser.add_argument("--box-length", type=float, default=10.0)
    parser.add_argument(
        "--compile", action="store_true",
        help="AOTI-compile the artifacts (torch>=2.6; needs a matching toolchain)",
    )
    parser.add_argument(
        "--out-dir", type=str, default=None,
        help="directory for compiled artifacts (default: a temp dir)",
    )
    args = parser.parse_args()

    kernel = SogKernel(r_cut=5.0, b=1.6297670882677646, m=12)
    model = MACESOGDeployable(kernel, box_length=args.box_length)

    print("== 1. trace the three protocols ==")
    flr_in = model.full_lr_sample_inputs(n_node=args.n_node)
    flr_traced = model.trace_full_lr(*flr_in)
    eo_in = model.energy_only_sample_inputs(n_node=args.n_node)
    eo_traced = model.trace_energy_only(*eo_in)
    resp_in = model.response_sample_inputs(n_node=args.n_node)
    resp_traced = model.trace_response(*resp_in)

    with torch.no_grad():
        flr_out = flr_traced(*flr_in)
        eo_out = eo_traced(*eo_in)
        resp_out = resp_traced(*resp_in)
    validate_output_keys(flr_out, FULL_LR_OUTPUT_KEYS, what="full_lr")
    validate_output_keys(eo_out, ENERGY_ONLY_OUTPUT_KEYS, what="energy_only")
    validate_output_keys(resp_out, RESPONSE_OUTPUT_KEYS, what="response")
    print(f"  full_lr      keys {tuple(flr_out)}")
    print(f"  energy_only  keys {tuple(eo_out)}")
    print(f"  response     keys {tuple(resp_out)}")

    print("== 2. trace fidelity (eager vs make_fx) ==")
    with torch.no_grad():
        eager = model.full_lr(*flr_in)
    for k in FULL_LR_OUTPUT_KEYS:
        diff = (eager[k] - flr_out[k]).abs().max().item()
        print(f"  {k:8s} max|eager - traced| = {diff:.2e}")

    print("== 3. artifact metadata ==")
    meta = build_metadata({
        "full_lr": FULL_LR_OUTPUT_KEYS,
        "energy_only": ENERGY_ONLY_OUTPUT_KEYS,
        "response": RESPONSE_OUTPUT_KEYS,
        "response_combined": ("energy", "force", "virial"),
    })
    for key in sorted(meta):
        print(f"  {key} = {meta[key]}")

    print("== 4. export programs ==")
    flr_exported = export_program(
        flr_traced, flr_in, dynamic_shapes=model.full_lr_dynamic_shapes(*flr_in)
    )
    eo_exported = export_program(eo_traced, eo_in, dynamic_shapes=None)
    resp_exported = export_program(resp_traced, resp_in, dynamic_shapes=None)
    print("  full_lr / energy_only / response exported")

    if not args.compile:
        print("== done (pass --compile to AOTI-compile the artifacts) ==")
        return

    print("== 5. AOTI compile ==")
    for warning in check_toolchain():
        print(f"  toolchain warning: {warning}")
    out_dir = pathlib.Path(args.out_dir or tempfile.mkdtemp(prefix="sog_mace_"))
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = ".pt2" if _torch_has_package_api() else ".so"
    jobs = (
        ("full_lr", flr_exported, flr_in, model.full_lr_callable),
        ("energy_only", eo_exported, eo_in, model.energy_only_callable),
        ("response", resp_exported, resp_in, model.response_callable),
    )
    for name, exported, sample, src in jobs:
        path = out_dir / f"sog_{name}{ext}"
        try:
            compile_artifact(
                exported, path, sample_inputs=sample, cuda=False, source_fn=src
            )
            print(f"  compiled {name:12s} -> {path}")
        except Exception as exc:  # pragma: no cover - toolchain dependent
            print(f"  compile {name} failed: {exc}")


if __name__ == "__main__":
    main()
