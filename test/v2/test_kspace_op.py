"""The CUDA ``sogkspace::*`` custom op (in-graph solve + neutralization)."""

import pytest
import torch

from sog import SogKernel
from sog.module.gaussian import Gaussian
from sog.v2 import kspace_op

from _helpers import expected_grid_axis, make_kernel, make_system, op_ready

pytestmark = [
    pytest.mark.v2,
    pytest.mark.torch_op,
    pytest.mark.skipif(not op_ready(), reason="sogkspace op library / CUDA not available"),
]


def _amp_bw(k: SogKernel):
    return (
        k.amp_internal().detach().to(torch.float64),
        k.bandwidth_sq().detach().to(torch.float64),
    )


def test_schema_registered():
    for name in ("solve_auto", "neutralize", "qmean", "last_grid", "set_comm", "destroy"):
        assert hasattr(torch.ops.sogkspace, name), f"missing op {name}"
    schema = str(torch.ops.sogkspace.solve_auto.default._schema)
    assert "solve_auto" in schema and "Tensor" in schema


def test_grid_matches_phi_max_rule():
    q, pos, box = make_system(n=64, L=12.0, seed=10)
    k = make_kernel()
    amp, bw2 = _amp_bw(k)
    handle = 7
    kspace_op.solve_auto(q.cuda(), pos.cuda(), box.cuda(), amp, bw2, rcut=5.0, handle=handle)
    grid = kspace_op.last_grid(handle).tolist()
    want = expected_grid_axis(12.0, k)
    assert grid == [want, want, want], f"grid {grid} != expected {want}^3"


def test_solve_matches_python_cubes2():
    """Energy/force against the Python CubeS₂ mesh; virial against the direct
    reference.

    NOTE: the legacy ``Gaussian`` CubeS₂ autograd virial does not contain the
    self-interaction strain term (toggling ``remove_self_interaction`` leaves it
    unchanged), while the deployed C++ mesh — and therefore the op — uses the
    complete form ``W_ks + W_self``.  The direct-k-sum reference implements the
    same complete form and was verified against autograd to 1e-12, so it is the
    right virial reference here.
    """
    import math

    from sog.v2.exportable import TraceableSogKernel
    from sog.tools.phi_max_rule import phi_max, sigma_min_of

    q, pos, box = make_system(n=64, L=12.0, seed=11)
    k = make_kernel()
    amp, bw2 = _amp_bw(k)
    v, f_dir, vir6 = kspace_op.solve_auto(
        q.cuda(), pos.cuda(), box.cuda(), amp, bw2, rcut=5.0, handle=8
    )
    e_op = 0.5 * (v.reshape(-1) * q.cuda()).sum()

    phi = float(phi_max(sigma_min_of(bw2), 5.0, 4, 1e-4))
    g = Gaussian(
        amp=amp.clone(), bandwidth=bw2.clone(),
        kernel_param_mode="internal", kernel_tensor_mode="external",
        rcut=5.0, b=1.6297670882677646, remove_self_interaction=True,
        use_cubes2_fft=True, use_nufft=False, cubes2_phi_max=phi, trainable=False,
    )
    r = pos.clone().requires_grad_(True)
    cell = box.reshape(1, 3, 3).clone()
    e_ref = g(q=q, r=r, cell=cell).sum()
    (g_r,) = torch.autograd.grad(e_ref, r)
    f_ref = -g_r

    e_scale = max(abs(e_ref.item()), 1e-6)
    assert abs(e_op.item() - e_ref.item()) / e_scale < 2e-2, (
        f"E op {e_op.item()} vs py {e_ref.item()}"
    )
    df = (f_dir.cpu() - f_ref).abs().max().item()
    assert df / f_ref.abs().max().item() < 2e-2

    # Complete-virial reference: the static direct k-sum (analytic derivatives).
    delta = phi * 5.0
    tk = TraceableSogKernel.from_kernel(
        k, n_dl=delta, nk=int(math.ceil(12.0 / delta)) + 1
    )
    _v, _f, vir_ref = tk.solution(q, pos, torch.eye(3, dtype=torch.float64).unsqueeze(0) * 12.0)
    dv = (vir6.cpu() - vir_ref[0]).abs().max().item()
    assert dv / max(vir_ref[0].abs().max().item(), 1e-6) < 2e-2, (
        f"vir6 op {vir6.tolist()} vs direct reference {vir_ref[0].tolist()}"
    )


def test_rejects_fp32_and_triclinic():
    q, pos, box = make_system(n=16, L=10.0, seed=12)
    k = make_kernel()
    amp, bw2 = _amp_bw(k)
    with pytest.raises(TypeError):
        kspace_op.solve_auto(q.float().cuda(), pos.cuda(), box.cuda(), amp, bw2, rcut=5.0)
    with pytest.raises(ValueError):
        bad = torch.tensor([10.0, 0.0, 0.0, 0.7, 10.0, 0.0, 0.0, 0.0, 10.0], dtype=torch.float64)
        kspace_op.solve_auto(q.cuda(), pos.cuda(), bad.cuda(), amp, bw2, rcut=5.0)
