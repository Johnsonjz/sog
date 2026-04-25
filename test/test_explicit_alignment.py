import torch

from sog import Sog
from sog.module.gaussian import HAS_PYTORCH_FINUFFT


def test_explicit_force_virial_align_autograd_same_input():
    if not HAS_PYTORCH_FINUFFT:
        import pytest

        pytest.skip("pytorch_finufft is not available")

    torch.manual_seed(19)

    model = Sog(
        {
            "use_atomwise": False,
            "use_nufft": True,
            "nufft_eps": 1e-4,
            "remove_self_interaction": True,
            "trainable_kernel": False,
        }
    )

    r = torch.rand(20, 3, dtype=torch.float64)
    q = torch.rand(20, 2, dtype=torch.float64) - 0.5
    q = q - q.mean(dim=0, keepdim=True)

    cell = torch.tensor(
        [[9.0, 0.0, 0.0], [0.6, 8.7, 0.0], [0.3, 0.5, 8.1]],
        dtype=torch.float64,
    ).unsqueeze(0)

    out_exp = model(
        positions=r.clone(),
        cell=cell,
        latent_charges=q,
        compute_energy=True,
        compute_force=True,
        compute_virial=True,
        use_explicit_derivatives=True,
    )
    out_auto = model(
        positions=r.clone(),
        cell=cell,
        latent_charges=q,
        compute_energy=True,
        compute_force=True,
        compute_virial=True,
        use_explicit_derivatives=False,
    )

    assert out_exp["used_explicit_derivatives"] is True

    e_exp = out_exp["E_lr"]
    e_auto = out_auto["E_lr"]
    f_exp = out_exp["forces"]
    f_auto = out_auto["forces"]
    v_exp = out_exp["virial"]
    v_auto = out_auto["virial"]

    assert e_exp is not None and e_auto is not None
    assert f_exp is not None and f_auto is not None
    assert v_exp is not None and v_auto is not None

    assert torch.allclose(e_exp, e_auto, rtol=3e-4, atol=3e-5)
    assert torch.allclose(f_exp, f_auto, rtol=5e-3, atol=3e-4)
    assert torch.allclose(v_exp, v_auto, rtol=6e-3, atol=6e-4)
