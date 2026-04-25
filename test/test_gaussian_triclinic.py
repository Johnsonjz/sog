import torch

from sog.module import Gaussian


def replicate_cell(
    r: torch.Tensor,
    q: torch.Tensor,
    cell: torch.Tensor,
    nx: int,
    ny: int,
    nz: int,
):
    shifts = []
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                coeff = torch.tensor([ix, iy, iz], dtype=r.dtype, device=r.device)
                shift = coeff @ cell
                shifts.append(shift)

    all_r = [r + s for s in shifts]
    all_q = [q for _ in shifts]
    return torch.cat(all_r, dim=0), torch.cat(all_q, dim=0)


def test_triclinic_replication_energy_density():
    torch.manual_seed(7)

    core_remove_si = Gaussian(
        n_dl=2.0,
        use_nufft=False,
        remove_self_interaction=True,
        trainable=False,
    )
    core_keep_si = Gaussian(
        n_dl=2.0,
        use_nufft=False,
        remove_self_interaction=False,
        trainable=False,
    )

    r = torch.rand(24, 3, dtype=torch.float64) * 9.0
    q = torch.rand(24, 2, dtype=torch.float64) - 0.5
    q = q - q.mean(dim=0, keepdim=True)

    # Two identical graphs in one batch should produce identical energies.
    r_b = torch.cat([r, r], dim=0)
    q_b = torch.cat([q, q], dim=0)
    batch = torch.cat(
        [
            torch.zeros(r.shape[0], dtype=torch.int64),
            torch.ones(r.shape[0], dtype=torch.int64),
        ],
        dim=0,
    )

    cell = torch.tensor(
        [[9.0, 0.0, 0.0], [0.8, 8.5, 0.0], [0.4, 0.6, 8.2]],
        dtype=torch.float64,
    )
    cell_b = torch.stack([cell, cell], dim=0)

    e_b = core_remove_si(q=q_b, r=r_b, cell=cell_b, batch=batch)
    assert e_b.shape == (2,)
    assert torch.allclose(e_b[0], e_b[1], rtol=1e-10, atol=1e-10)

    # Self-interaction switch should change energy in general.
    e_rm = core_remove_si(q=q, r=r, cell=cell.unsqueeze(0), batch=None)[0]
    e_keep = core_keep_si(q=q, r=r, cell=cell.unsqueeze(0), batch=None)[0]
    assert torch.abs(e_rm - e_keep) > 1e-8
