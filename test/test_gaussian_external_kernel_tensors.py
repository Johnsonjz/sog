import torch

from sog.module import Gaussian


def test_gaussian_external_internal_kernel_tensors_matches_standard_semantics():
    torch.manual_seed(123)

    bw2 = torch.tensor([1.3, 2.1, 3.4], dtype=torch.float64)
    amp_internal = torch.tensor([0.7, 1.1, 1.4], dtype=torch.float64)

    # New external/internal path: inputs are already (amp_internal, bw2).
    core_ext = Gaussian(
        n_dl=2.0,
        amp=amp_internal.clone().requires_grad_(True),
        bandwidth=bw2.clone().requires_grad_(True),
        kernel_param_mode="internal",
        kernel_tensor_mode="external",
        use_nufft=False,
        trainable=False,
    )

    # Legacy path expects raw amp and raw bandwidth.
    amp_raw = amp_internal / bw2
    bw_raw = torch.sqrt(bw2)
    core_std = Gaussian(
        n_dl=2.0,
        amp=amp_raw,
        bandwidth=bw_raw,
        use_nufft=False,
        trainable=False,
    )

    r = torch.rand(10, 3, dtype=torch.float64)
    q = torch.rand(10, dtype=torch.float64) - 0.5
    q = q - q.mean()

    e_ext = core_ext(q=q, r=r, cell=None, batch=None)
    e_std = core_std(q=q, r=r, cell=None, batch=None)

    assert torch.allclose(e_ext, e_std, rtol=2e-7, atol=2e-6)


def test_gaussian_external_internal_kernel_tensors_keeps_grad_path():
    torch.manual_seed(321)

    bw2 = torch.tensor([1.1, 1.9], dtype=torch.float64, requires_grad=True)
    amp_internal = torch.tensor([0.5, 0.8], dtype=torch.float64, requires_grad=True)

    core = Gaussian(
        n_dl=2.0,
        amp=amp_internal,
        bandwidth=bw2,
        kernel_param_mode="internal",
        kernel_tensor_mode="external",
        use_nufft=False,
        trainable=False,
    )

    # In external tensor mode, Gaussian should not own these as nn.Parameter.
    assert "amp" not in core._parameters
    assert "bandwidth" not in core._parameters

    r = torch.rand(9, 3, dtype=torch.float64)
    q = torch.rand(9, dtype=torch.float64) - 0.5
    q = q - q.mean()

    e = core(q=q, r=r, cell=None, batch=None).sum()
    e.backward()

    assert amp_internal.grad is not None
    assert bw2.grad is not None
    assert torch.linalg.norm(amp_internal.grad) > 0
    assert torch.linalg.norm(bw2.grad) > 0
