import torch

from sog import Sog


def test_force_matches_fd_in_realspace():
    torch.manual_seed(11)

    model = Sog(
        {
            "use_atomwise": False,
            "use_nufft": False,
            "remove_self_interaction": True,
            "trainable_kernel": False,
        }
    )

    r = torch.rand(6, 3, dtype=torch.float64)
    q = torch.rand(6, dtype=torch.float64) - 0.5
    q = q - q.mean()

    out = model(
        positions=r.clone(),
        cell=None,
        latent_charges=q,
        compute_energy=True,
        compute_force=True,
    )
    force = out["forces"]
    assert force is not None

    eps = 1e-6
    r_p = r.clone()
    r_m = r.clone()
    r_p[0, 0] += eps
    r_m[0, 0] -= eps

    e_p = model(
        positions=r_p,
        cell=None,
        latent_charges=q,
        compute_energy=True,
        compute_force=False,
    )["E_lr"]
    e_m = model(
        positions=r_m,
        cell=None,
        latent_charges=q,
        compute_energy=True,
        compute_force=False,
    )["E_lr"]

    assert e_p is not None and e_m is not None
    fd = -((e_p[0] - e_m[0]) / (2.0 * eps))

    assert torch.allclose(force[0, 0], fd, rtol=2e-2, atol=2e-3)
