"""The framework-neutral long-range algebra (``E=1/2 sum v q``, seed, virial)."""

import pytest
import torch

from sog.v2.external_kspace import ExternalKSpace
from sog.v2.types import vir6_to_deepmd, vir6_to_lammps6

from _helpers import make_kernel, make_system, op_ready

pytestmark = pytest.mark.v2


def _reference():
    return ExternalKSpace(make_kernel(), backend="reference")


def test_energy_identity_and_fused_seed():
    q, pos, box = make_system(n=64, L=12.0, seed=1)
    ek = _reference()
    sol = ek.solve(q, pos, box)
    assert torch.allclose(sol.energy(), 0.5 * (sol.v.reshape(-1) * sol.qn.reshape(-1)).sum())
    seed = ek.fused_seed(sol)
    assert torch.allclose(seed, (sol.v.reshape(-1) * sol.qn.reshape(-1)).sum() / q.numel())
    # n_owned override (multi-rank share semantics)
    n_owned = torch.tensor(48.0)
    assert torch.allclose(ek.fused_seed(sol, n_owned), seed * q.numel() / 48.0)


def test_vir6_mappings():
    m = torch.arange(1.0, 7.0)
    d = vir6_to_deepmd(m)
    assert d.shape == (1, 9)
    # (xx, xy, xz, yx, yy, yz, zx, zy, zz)
    assert d.reshape(-1).tolist() == [1.0, 4.0, 5.0, 4.0, 2.0, 6.0, 5.0, 6.0, 3.0]
    assert vir6_to_lammps6(m).tolist() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


def test_neutralization_global_mean_and_ghost_zeroing():
    q, pos, box = make_system(n=40, L=12.0, seed=2, neutral=False)
    q[:32] = q[:32] + 0.7  # arbitrary offset -> nonzero global mean
    q[32:] = torch.linspace(-3.0, 3.0, 8)  # ghosts carry junk
    ek = _reference()
    sol = ek.solve(q, pos, box, n_local=32)
    assert sol.qn[32:].abs().max().item() == 0.0, "ghost rows must be zeroed"
    assert abs(sol.qn[:32].sum().item()) < 1e-12, "owned rows must be zero-mean"


def test_eager_efv_force_matches_finite_difference():
    L, n = 12.0, 64
    q0, pos0, box = make_system(n=n, L=L, seed=3)
    ek = _reference()

    g = torch.Generator().manual_seed(7)
    W = torch.randn(3, 1, generator=g, dtype=torch.float64)
    p = pos0.clone().requires_grad_(True)

    def charges(pos):
        q = (pos @ W).squeeze(-1)
        return q - q.mean()

    out = ek.eager_efv(charges(p), p, box)

    def e_at(pos):
        s = ek.solve(charges(pos), pos, box)
        return 0.5 * (s.v * s.qn).sum()

    eps = 1e-5
    for i in (0, 5, 17):
        dp = torch.zeros_like(pos0)
        dp[i, 0] = eps
        fd = -(e_at(pos0 + dp) - e_at(pos0 - dp)) / (2 * eps)
        assert abs(fd.item() - out["force"][i, 0].item()) < 1e-5, (
            f"atom {i}: FD {fd.item():.8f} vs force {out['force'][i, 0].item():.8f}"
        )


def test_recorded_backends_agree():
    if not op_ready():
        pytest.skip("sogkspace op library / CUDA not available")
    q, pos, box = make_system(n=64, L=12.0, seed=4)
    ref = ExternalKSpace(make_kernel(), backend="reference").solve(q, pos, box)
    op = ExternalKSpace(make_kernel(), backend="op").solve(q.cuda(), pos.cuda(), box.cuda())
    e_ref, e_op = ref.energy().item(), op.energy().item()
    assert abs(e_op - e_ref) / max(abs(e_ref), 1e-6) < 2e-2
    df = (op.F_dir.cpu() - ref.F_dir).abs().max().item()
    scale = ref.F_dir.abs().max().item()
    assert df / scale < 2e-2
    dv = (op.vir6.cpu() - ref.vir6).abs().max().item()
    vscale = ref.vir6.abs().max().item()
    assert dv / max(vscale, 1e-6) < 2e-2
