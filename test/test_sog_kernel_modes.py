import pytest
import torch

from sog import Sog
from sog.module import Gaussian


def test_gaussian_auto_tensor_mode_uses_external_for_differentiable_inputs():
    bw2 = torch.tensor([1.2, 1.8], dtype=torch.float64, requires_grad=True)
    amp_internal = torch.tensor([0.6, 0.9], dtype=torch.float64, requires_grad=True)

    core = Gaussian(
        n_dl=2.0,
        amp=amp_internal,
        bandwidth=bw2,
        kernel_param_mode="internal",
        kernel_tensor_mode="auto",
        use_nufft=False,
        trainable=False,
    )

    assert "amp" not in core._parameters
    assert "bandwidth" not in core._parameters


def test_sog_legacy_bool_keys_map_to_new_modes():
    model = Sog(
        {
            "use_atomwise": False,
            "amp_is_internal": True,
            "bandwidth_is_squared": True,
            "use_external_kernel_tensors": True,
        }
    )

    assert model.kernel_param_mode == "internal"
    assert model.kernel_tensor_mode == "external"


def test_sog_conflicting_new_and_legacy_mode_keys_raise():
    with pytest.raises(ValueError):
        Sog(
            {
                "use_atomwise": False,
                "kernel_param_mode": "raw",
                "amp_is_internal": True,
                "bandwidth_is_squared": True,
            }
        )
