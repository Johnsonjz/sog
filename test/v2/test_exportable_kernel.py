"""The pure-PyTorch reference kernel: analytic derivatives and traceability."""

import math

import pytest
import torch

from sog.v2.exportable import ReferenceKSpace, TraceableSogKernel

from _helpers import delta_for, make_kernel, make_system

pytestmark = pytest.mark.v2


def _kernel(n=40, L=12.0, seed=0):
    k = make_kernel()
    delta = delta_for(k)
    nk = int(math.ceil(L / delta)) + 1
    return TraceableSogKernel.from_kernel(k, n_dl=delta, nk=nk)


def test_analytic_solution_matches_autograd():
    torch.manual_seed(0)
    q, pos, box = make_system(n=40, L=12.0, seed=0)
    L = 12.0
    cell = box.reshape(1, 3, 3)
    tk = _kernel()

    v, f_dir, vir6 = tk.solution(q, pos, cell)
    v, f_dir, vir6 = v[0], f_dir[0], vir6[0]

    q1 = q.clone().requires_grad_(True)
    e = tk.energy(q1, pos, cell).sum()
    (gq,) = torch.autograd.grad(e, q1)
    p1 = pos.clone().requires_grad_(True)
    e2 = tk.energy(q, p1, cell).sum()
    (gp,) = torch.autograd.grad(e2, p1)
    s = (pos / L).clone().requires_grad_(True)
    h = (torch.eye(3, dtype=torch.float64) * L).unsqueeze(0).requires_grad_(True)
    e3 = tk.energy(q, s @ h[0], h).sum()
    (gh,) = torch.autograd.grad(e3, h)
    w = -(gh[0] @ h[0].transpose(0, 1))
    vir_ref = torch.stack([w[0, 0], w[1, 1], w[2, 2], w[0, 1], w[0, 2], w[1, 2]])

    assert torch.allclose(v, gq, atol=1e-10, rtol=0), f"v diff {(v-gq).abs().max()}"
    assert torch.allclose(f_dir, -gp, atol=1e-10, rtol=0), f"F diff {(f_dir+gp).abs().max()}"
    assert torch.allclose(vir6, vir_ref, atol=1e-10, rtol=0), f"vir6 diff {(vir6-vir_ref).abs().max()}"


def test_energy_equals_half_sum_vq():
    q, pos, box = make_system(n=64, L=12.0, seed=1)
    cell = box.reshape(1, 3, 3)
    tk = _kernel()
    e = tk.energy(q, pos, cell).sum()
    v, _f, _vir = tk.solution(q, pos, cell)
    e_identity = 0.5 * (v.reshape(-1) * q).sum()
    assert abs(e.item() - e_identity.item()) <= 1e-10 * max(abs(e.item()), 1.0)


def test_frames_are_independent():
    tk = _kernel()
    q1, p1, box = make_system(n=32, L=12.0, seed=2)
    q2, p2, _ = make_system(n=32, L=12.0, seed=3)
    cell = box.reshape(1, 3, 3).repeat(2, 1, 1)
    e_batch = tk.energy(torch.stack([q1, q2]), torch.stack([p1, p2]), cell)
    e1 = tk.energy(q1, p1, cell[:1])
    e2 = tk.energy(q2, p2, cell[:1])
    assert torch.allclose(e_batch, torch.stack([e1[0], e2[0]]), atol=1e-12, rtol=0)


def test_make_fx_traceable():
    from torch.fx.experimental.proxy_tensor import make_fx

    tk = _kernel()
    q, pos, box = make_system(n=16, L=12.0, seed=4)
    cell = box.reshape(1, 3, 3)
    traced = make_fx(lambda qq, pp, cc: tk.energy(qq, pp, cc), tracing_mode="symbolic")(
        q.unsqueeze(0), pos.unsqueeze(0), cell
    )
    with torch.no_grad():
        e_traced = traced(q.unsqueeze(0), pos.unsqueeze(0), cell)
        e_eager = tk.energy(q, pos, cell)
    assert torch.allclose(e_traced, e_eager, atol=1e-10, rtol=0)


def test_reference_solution_is_detached():
    k = make_kernel()
    from sog.v2.exportable import TraceableSogKernel as TK

    delta = delta_for(k)
    tk = TK.from_kernel(k, n_dl=delta, nk=32)
    ref = ReferenceKSpace(tk)
    q, pos, box = make_system(n=32, L=12.0, seed=5)
    q = q.clone().requires_grad_(True)
    v, f, vir = ref.solve(q, pos, box)
    assert v.grad_fn is None and f.grad_fn is None and vir.grad_fn is None
