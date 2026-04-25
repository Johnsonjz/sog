import torch

from sog import Sog
from sog.util.deepmd_ref import DeepMDSOGReference


def test_sog_aligns_with_deepmd_reference_same_input():
    try:
        ref_probe = DeepMDSOGReference(
            q_dim=1,
            amp=1.0,
            bandwidth=[1.0],
            n_dl=1.0,
            remove_self_interaction=True,
        )
        del ref_probe
    except Exception as exc:
        import pytest

        pytest.skip(f"deepmd reference is not available: {exc}")

    torch.manual_seed(23)

    model = Sog(
        {
            "use_atomwise": False,
            "use_nufft": True,
            "nufft_eps": 1e-4,
            "remove_self_interaction": True,
            "trainable_kernel": False,
        }
    )

    r = torch.rand(18, 3, dtype=torch.float64)
    q = torch.rand(18, 2, dtype=torch.float64) - 0.5
    q = q - q.mean(dim=0, keepdim=True)

    cell = torch.tensor(
        [[9.2, 0.0, 0.0], [0.7, 8.6, 0.0], [0.2, 0.4, 8.0]],
        dtype=torch.float64,
    ).unsqueeze(0)

    out_sog = model(
        positions=r.clone(),
        cell=cell,
        latent_charges=q,
        compute_energy=True,
        compute_force=True,
        compute_virial=True,
        use_explicit_derivatives=True,
    )

    assert out_sog["used_explicit_derivatives"] is True

    ref = DeepMDSOGReference.from_gaussian(model.gaussian, q_dim=q.shape[1])
    out_dp = ref.compute_bundle(
        coord=r.unsqueeze(0),
        latent_charge=q.unsqueeze(0),
        box=cell,
        compute_force=True,
        compute_virial=True,
    )

    e_sog = out_sog["E_lr"]
    f_sog = out_sog["forces"]
    v_sog = out_sog["virial"]

    e_dp = out_dp["energy"]
    f_dp = out_dp["forces"]
    v_dp = out_dp["virial"]

    assert e_sog is not None and f_sog is not None and v_sog is not None
    assert f_dp is not None and v_dp is not None

    assert torch.allclose(e_sog, e_dp, rtol=3e-4, atol=3e-5)
    assert torch.allclose(f_sog, f_dp[0], rtol=5e-3, atol=3e-4)
    assert torch.allclose(v_sog, v_dp, rtol=6e-3, atol=6e-4)
