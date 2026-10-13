"""The NUFFT branch's explicit forces/virial must match autograd on its own energy.

The NUFFT branch supplies *explicit* (analytic) forces while its energy is itself
built from differentiable operations.  That makes it the one place where the two
routes to a derivative can be compared on the same energy, so it is the natural
regression test for both.

The virial comparison matters in particular: the affine virial is

    W = Σ_i F_i ⊗ r_i  -  (∂E/∂h)ᵀ h

and an implementation that keeps only the first term still looks plausible on a
cubic cell (where the second term is diagonal) while being wrong off-diagonal.
"""

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

    # A strained (triclinic) cell, so the off-diagonal virial is exercised.
    cell = torch.tensor(
        [[9.0, 0.0, 0.0], [0.6, 8.7, 0.0], [0.3, 0.5, 8.1]],
        dtype=torch.float64,
    ).unsqueeze(0)

    # ── explicit route: the solver reports forces (and no explicit virial) ──
    out_exp = model(
        positions=r.clone().requires_grad_(True),
        cell=cell.clone().requires_grad_(True),
        latent_charges=q,
        compute_energy=True,
        compute_force=True,
        compute_virial=True,
    )

    e_exp = out_exp["E_lr"]
    f_exp = out_exp["forces"]
    v_exp = out_exp["virial"]
    assert e_exp is not None and f_exp is not None and v_exp is not None

    # ── autograd route: differentiate the same energy directly ──
    r_ag = r.clone().requires_grad_(True)
    cell_ag = cell.clone().requires_grad_(True)
    out_ag = model(
        positions=r_ag,
        cell=cell_ag,
        latent_charges=q,
        compute_energy=True,
    )
    e_ag = out_ag["E_lr"]
    assert e_ag is not None

    assert torch.allclose(e_exp, e_ag, rtol=1e-10, atol=1e-12)

    grad_r, grad_h = torch.autograd.grad(e_ag.sum(), [r_ag, cell_ag])
    f_auto = -grad_r
    pos_vir = torch.einsum("ic,id->cd", f_auto, r).unsqueeze(0)
    box_vir = -torch.einsum("fga,fgb->fab", grad_h, cell)
    v_auto = pos_vir + box_vir

    assert torch.allclose(f_exp, f_auto, rtol=1e-8, atol=1e-10), (
        f"explicit force != autograd force, max abs diff "
        f"{(f_exp - f_auto).abs().max().item():.3e}"
    )
    assert torch.allclose(v_exp, v_auto, rtol=1e-6, atol=1e-8), (
        f"explicit virial != autograd affine virial, max abs diff "
        f"{(v_exp - v_auto).abs().max().item():.3e}"
    )
