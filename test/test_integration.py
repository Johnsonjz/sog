"""The framework-facing integration contract: guards, modes, and conventions."""

import torch

from sog import Sog
from sog.integration import (
    AtomwiseChargeProvider,
    ChargeNeutralityPolicy,
    build_sog,
    lr_energy,
    validate_batch,
    validate_charges,
)


def test_validate_batch_accepts_canonical_layout():
    b = torch.tensor([0, 0, 0, 1, 1])
    assert validate_batch(b, 5) is b
    # None means a single frame.
    z = validate_batch(None, 3)
    assert z.shape == (3,) and z.sum() == 0


def test_validate_batch_rejects_non_canonical_layout():
    """Interleaved, reversed or gapped frame ids silently corrupt energies."""
    for bad in ([0, 1, 0, 1], [1, 1, 0, 0], [0, 0, 2, 2]):
        try:
            validate_batch(torch.tensor(bad), 4)
        except ValueError:
            pass
        else:
            raise AssertionError(f"batch {bad} should have been rejected")


def test_validate_batch_rejects_shape_mismatch():
    for bad, n in ((torch.tensor([0, 0]), 3), (torch.tensor([[0], [0]]), 2)):
        try:
            validate_batch(bad, n)
        except ValueError:
            pass
        else:
            raise AssertionError(f"batch {bad} with n_atoms={n} should be rejected")


def test_validate_charges_enforces_shape_and_channels():
    q = torch.rand(4, 2)
    assert validate_charges(q, 4, 2) is q
    # A 1-D input is promoted to a single channel.
    assert validate_charges(torch.rand(4), 4, 1).shape == (4, 1)
    for bad, n, c in (
        (torch.rand(5, 2), 4, 2),   # wrong atom count
        (torch.rand(4, 3), 4, 2),   # wrong channel count
        (torch.rand(4, 2) * float("nan"), 4, 2),  # non-finite
    ):
        try:
            validate_charges(bad, n, c)
        except ValueError:
            pass
        else:
            raise AssertionError(f"charges with n={n} c={c} should be rejected")


def test_build_sog_owned_mode_accepts_kwargs():
    model = build_sog(mode="owned", r_cut=6.0, n_channels=2)
    assert isinstance(model, Sog)
    assert model.n_out == 2


def test_build_sog_external_mode_requires_tensors():
    try:
        build_sog(mode="external", r_cut=6.0)
    except ValueError:
        pass
    else:
        raise AssertionError("external mode without amp/bandwidth should be rejected")


def test_lr_energy_matches_direct_call():
    torch.manual_seed(7)
    r = torch.rand(12, 3, dtype=torch.float64)
    q = torch.rand(12, dtype=torch.float64) - 0.5
    q = q - q.mean()
    cell = torch.eye(3, dtype=torch.float64).unsqueeze(0) * 10.0

    model = build_sog(mode="owned", r_cut=6.0)
    e = lr_energy(model, r, cell, q)
    e2 = model(positions=r, cell=cell, latent_charges=q)["E_lr"]
    assert torch.equal(e, e2)


def test_neutrality_policy_hard_subtracts_frame_means():
    torch.manual_seed(9)
    q = torch.rand(6, 2, dtype=torch.float64)
    batch = torch.tensor([0, 0, 0, 1, 1, 1])

    plain = ChargeNeutralityPolicy()
    assert torch.equal(plain.apply(q, batch), q), "default policy must not touch charges"

    hard = ChargeNeutralityPolicy(hard_in_model=True)
    out = hard.apply(q, batch)
    for f in (0, 1):
        assert out[batch == f].sum(dim=0).abs().max() < 1e-12


def test_neutrality_policy_describes_itself():
    p = ChargeNeutralityPolicy(hard_in_model=True, soft_lambda=0.5)
    s = p.describe()
    assert "k=0" in s and "hard" in s and "0.5" in s


def test_atomwise_charge_provider_shape():
    torch.manual_seed(11)
    # build_mlp requires len(n_hidden) == n_layers - 1 when n_hidden is a list.
    provider = AtomwiseChargeProvider(n_in=8, n_out=3, n_layers=2, n_hidden=[32])
    desc = torch.rand(5, 8)
    out = provider(desc)
    assert out.shape == (5, 3)
    assert provider.n_channels == 3


def test_charges_must_not_be_detached_for_a_conservative_force():
    """Detaching the charges silently drops the charge-response force term."""
    torch.manual_seed(13)
    r = torch.rand(10, 3, dtype=torch.float64)
    q0 = torch.rand(10, dtype=torch.float64) - 0.5
    q0 = q0 - q0.mean()
    cell = torch.eye(3, dtype=torch.float64).unsqueeze(0) * 9.0

    model = build_sog(mode="owned", r_cut=6.0)
    q = q0.clone().requires_grad_(True)
    e = lr_energy(model, r, cell, q)
    (grad_q,) = torch.autograd.grad(e.sum(), q, retain_graph=True)
    # dE/dq must be non-zero, otherwise the response term would vanish.
    assert grad_q is not None and grad_q.abs().sum() > 0
