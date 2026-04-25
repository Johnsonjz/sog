import torch

from sog import Sog


def test_sog_forward_latent_charge_path():
    torch.manual_seed(3)

    model = Sog(
        {
            "use_atomwise": False,
            "use_nufft": False,
            "remove_self_interaction": True,
            "trainable_kernel": False,
        }
    )

    r = torch.rand(12, 3, dtype=torch.float64)
    q = torch.rand(12, dtype=torch.float64) - 0.5
    q = q - q.mean()

    cell = torch.eye(3, dtype=torch.float64).unsqueeze(0) * 10.0

    out = model(
        positions=r,
        cell=cell,
        latent_charges=q,
        compute_energy=True,
        compute_force=True,
        compute_virial=True,
        compute_bec=False,
    )

    assert out["E_lr"] is not None
    assert out["latent_charges"] is not None
    assert out["forces"] is not None
    assert out["virial"] is not None

    assert out["E_lr"].shape == (1,)
    assert out["latent_charges"].shape == (12, 1)
    assert out["forces"].shape == (12, 3)
    assert out["virial"].shape == (1, 3, 3)
