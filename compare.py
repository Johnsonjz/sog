import torch
from sog.module import Gaussian

torch.manual_seed(123)
bw2 = torch.tensor([1.3, 2.1, 3.4], dtype=torch.float64)
amp_internal = torch.tensor([0.7, 1.1, 1.4], dtype=torch.float64)

core_ext = Gaussian(
    n_dl=2.0,
    amp=amp_internal.clone().requires_grad_(True),
    bandwidth=bw2.clone().requires_grad_(True),
    amp_is_internal=True,
    bandwidth_is_squared=True,
    use_external_kernel_tensors=True,
    use_nufft=False,
    trainable=False,
)

amp_raw = amp_internal / bw2
bw_raw = torch.sqrt(bw2)
core_std = Gaussian(
    n_dl=2.0,
    amp=amp_raw,
    bandwidth=bw_raw,
    use_nufft=False,
    trainable=False,
)

print('ext amp', core_ext.amp.detach().cpu().numpy())
print('ext bw', core_ext.bandwidth.detach().cpu().numpy())
print('std amp', core_std.amp.detach().cpu().numpy())
print('std bw', core_std.bandwidth.detach().cpu().numpy())

r = torch.rand(10, 3, dtype=torch.float64)
q = torch.rand(10, dtype=torch.float64) - 0.5
q = q - q.mean()

e_ext = core_ext(q=q, r=r, cell=None, batch=None)
e_std = core_std(q=q, r=r, cell=None, batch=None)
print('e_ext', e_ext)
print('e_std', e_std)
print('abs diff', (e_ext - e_std).abs())
print('rel diff', (e_ext - e_std).abs() / e_std.abs().clamp_min(1e-30))
