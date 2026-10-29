"""Exact autograd rule for ``sogkspace::neutralize`` (global owned-mean gauge)."""

import pytest
import torch

from sog.v2 import kspace_op

from _helpers import op_ready

pytestmark = [
    pytest.mark.v2,
    pytest.mark.torch_op,
    pytest.mark.skipif(not op_ready(), reason="sogkspace op library / CUDA not available"),
]


def _setup(n=32, n_local=24, channels=None, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn((n,) if channels is None else (n, channels),
                    generator=g, dtype=torch.float64)
    return q, torch.tensor(float(n_local), dtype=torch.float64)


def test_forward_zeroes_ghosts_and_global_mean():
    q, nl = _setup(n=32, n_local=24)
    qn = kspace_op.neutralize(q.cuda(), nl.cuda()).cpu()
    assert qn[24:].abs().max().item() == 0.0
    assert abs(qn[:24].sum().item()) < 1e-12


def test_vjp_matches_explicit_autograd():
    q, nl = _setup(n=32, n_local=24, seed=1)
    kspace_op.register_neutralize_autograd()

    qc = q.clone().cuda().requires_grad_(True)
    w = torch.randn(32, generator=torch.Generator().manual_seed(2),
                    dtype=torch.float64).cuda()
    qn = kspace_op.neutralize(qc, nl.cuda())
    (qn * w).sum().backward()

    # Reference: qn = owned * (q - mean_owned(q))
    q_ref = q.clone().requires_grad_(True)
    owned = (torch.arange(32) < 24).to(torch.float64)
    mean = (q_ref * owned).sum() / 24.0
    (owned * (q_ref - mean) * w.cpu()).sum().backward()

    assert torch.allclose(qc.grad.cpu(), q_ref.grad, atol=1e-12, rtol=0), (
        f"VJP diff {(qc.grad.cpu() - q_ref.grad).abs().max():.3e}"
    )


def test_registration_is_idempotent():
    kspace_op.register_neutralize_autograd()
    kspace_op.register_neutralize_autograd()  # must not raise
