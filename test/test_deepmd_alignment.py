import torch

from sog import Sog
from sog.util.deepmd_ref import DeepMDSOGReference


def test_sog_aligns_with_deepmd_reference_same_input():
    """Cross-check SOG against DeepMD's own SOG implementation on one input.

    ``sog.util.deepmd_ref`` drives DeepMD through its *private* bundle API
    (``SOGEnergyModel._compute_sog_frame_correction_bundle``), which has since been
    replaced by the fused ``_sog_lr_reduced_energy`` / ``_forward_lower_sog_fused``
    path.  Until that bridge is updated against the deepmd-kit it targets, this
    test reports itself as skipped rather than failing — an unavailable reference
    is not a SOG defect.  See docs/08-validation.md.
    """
    import pytest

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
        pytest.skip(f"deepmd reference is not available: {exc}")

    torch.manual_seed(23)

    # A single-term kernel: DeepMDSOGReference can only represent a scalar
    # amplitude, and it also requires an explicit n_dl.
    model = Sog(
        {
            "use_atomwise": False,
            "use_nufft": True,
            "nufft_eps": 1e-4,
            "m": 1,
            "n_dl": 2.0,
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

    # The NUFFT branch supplies explicit forces but no analytic virial, so the
    # virial's box-strain term comes from autograd and needs a cell leaf.
    out_sog = model(
        positions=r.clone(),
        cell=cell.clone().requires_grad_(True),
        latent_charges=q,
        compute_energy=True,
        compute_force=True,
        compute_virial=True,
    )

    assert out_sog["used_explicit_derivatives"] is True

    try:
        ref = DeepMDSOGReference.from_gaussian(model.gaussian, q_dim=q.shape[1])
    except Exception as exc:
        pytest.skip(f"deepmd reference cannot represent this kernel: {exc}")

    try:
        out_dp = ref.compute_bundle(
            coord=r.unsqueeze(0),
            latent_charge=q.unsqueeze(0),
            box=cell,
            compute_force=True,
            compute_virial=True,
        )
    except AttributeError as exc:
        # The bridge drives DeepMD through a private API that has since moved.
        pytest.skip(f"deepmd reference bridge is stale against this deepmd-kit: {exc}")

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
