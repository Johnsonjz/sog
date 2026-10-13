"""SogKernel: parameter generation, LAMMPS export, and format round-trips."""

import torch

from sog import Sog, SogKernel
from sog.module.gaussian import Gaussian


def _assert_bitexact(a, b, what):
    a = a.detach().reshape(-1).double()
    b = b.detach().reshape(-1).double()
    assert a.shape == b.shape, f"{what}: shape {tuple(a.shape)} != {tuple(b.shape)}"
    assert torch.equal(a, b), f"{what}: max abs diff {(a - b).abs().max().item():.3e}"


def test_kernel_matches_gaussian_rcut_convention():
    """r_cut-derived parameters must reproduce Gaussian's bit for bit."""
    for nlayers in (1, 2):
        g = Gaussian(rcut=6.0, b=2.0, m=12, nlayers=nlayers)
        k = SogKernel(r_cut=6.0, b=2.0, m=12, nlayers=nlayers)
        _assert_bitexact(g.amp, k.amp_internal(), f"amp (nlayers={nlayers})")
        _assert_bitexact(g._bw2(), k.bandwidth_sq(), f"bw2 (nlayers={nlayers})")
        assert abs(k.sigma - 6.0 * nlayers / 1.9892536839080267) < 1e-15


def test_kernel_matches_gaussian_b_sigma_convention():
    """The no-cutoff branch uses a different (equivalent) expression; match it too."""
    g = Gaussian(b=1.629767, sigma=2.18, m=8, n_dl=1.0)
    k = SogKernel(b=1.629767, sigma=2.18, m=8, n_dl=1.0)
    _assert_bitexact(g.amp, k.amp_internal(), "amp")
    _assert_bitexact(g._bw2(), k.bandwidth_sq(), "bw2")


def test_kernel_matches_gaussian_explicit_parameters():
    torch.manual_seed(0)
    bw = torch.rand(5) + 0.5
    am = torch.rand(5) + 0.1
    for mode in ("raw", "internal"):
        g = Gaussian(amp=am.clone(), bandwidth=bw.clone(), kernel_param_mode=mode)
        k = SogKernel(amp=am.clone(), bandwidth=bw.clone(), kernel_param_mode=mode)
        _assert_bitexact(g.amp, k.amp_internal(), f"amp ({mode})")
        _assert_bitexact(g._bw2(), k.bandwidth_sq(), f"bw2 ({mode})")


def test_init_from_rcut_reproduces_constructor():
    k = SogKernel(b=2.0, sigma=3.0, m=12)
    k.init_from_rcut(6.0, nlayers=2)
    g = Gaussian(rcut=6.0, nlayers=2, b=2.0, m=12)
    _assert_bitexact(g.amp, k.amp_internal(), "amp")
    _assert_bitexact(g._bw2(), k.bandwidth_sq(), "bw2")


def test_init_from_rcut_refuses_external_tensors():
    """Overwriting a framework's own parameters is never correct."""
    k = SogKernel(kernel_tensor_mode="external")
    amp = torch.ones(4)
    bw = torch.ones(4)
    k.bind_external(amp, bw, amp_is_internal=True, bandwidth_is_squared=False)
    try:
        k.init_from_rcut(6.0)
    except RuntimeError:
        pass
    else:
        raise AssertionError("init_from_rcut should refuse externally bound tensors")
    # The caller's tensors are untouched.
    assert torch.equal(bw, torch.ones(4))


def test_lammps_export_convention():
    """The C++ plugin reads amp as the INTERNAL amplitude and bandwidth as bw²."""
    k = SogKernel(r_cut=6.0, b=2.0, m=12)
    d = k.to_lammps_dict()

    assert len(d["amp"]) == len(d["bandwidth"]) == 12
    _assert_bitexact(
        torch.as_tensor(d["amp"]), k.amp_internal(), "exported amp"
    )
    _assert_bitexact(
        torch.as_tensor(d["bandwidth"]), k.bandwidth_sq(), "exported bandwidth (must be bw^2)"
    )

    # Grid keyword naming: the plugin parses `phi_max` (sog.cpp:540) and rejects
    # `cubes2_phi_max`; the term count is lowercase `m` (sog.cpp:604).
    k2 = SogKernel(r_cut=6.0, cubes2_phi_max=0.10, m=12)
    d2 = k2.to_lammps_dict()
    assert d2["phi_max"] == 0.10
    assert d2["cubes2_phi_max"] == 0.10  # deprecated alias, same value
    assert isinstance(d2["m"], int) and d2["m"] == 12

    toks = k2.to_lammps_tokens()
    assert "phi_max" in toks, "must emit the keyword the plugin actually accepts"
    assert "cubes2_phi_max" not in toks, "the plugin rejects this keyword"
    assert "m" in toks

    # The generated input must be the pure keyword form: a numeric arg[0] would
    # be parsed by the plugin as the legacy 'accuracy b sigma M [n_dl]' header.
    inp = k2.to_lammps_input()
    first = inp.split("kspace_style sog")[1].split()[0]
    assert first == "amp", f"input line must start with a keyword, got '{first}'"


def test_state_dict_roundtrip_is_bitexact():
    k = SogKernel(r_cut=6.0, b=2.0, m=12, cubes2_phi_max=0.10)
    k2 = SogKernel.from_state_dict(k.to_state_dict())
    _assert_bitexact(k.amp, k2.amp, "amp round-trip")
    _assert_bitexact(k.bandwidth_sq(), k2.bandwidth_sq(), "bw2 round-trip")
    assert k2.M == k.M
    assert abs(k2.sigma - k.sigma) < 1e-15


def test_v1_payload_is_upconverted():
    """A payload without the version marker stores bw² and must be sqrt-recovered."""
    k = SogKernel(r_cut=6.0, b=2.0, m=12)
    legacy = {
        "format": "sog-kernel",
        "b": 2.0,
        "sigma": k.sigma,
        "m": 12,
        "amp": k.amp_internal().detach().clone(),
        "bandwidth": k.bandwidth_sq().detach().clone(),  # SQUARED, unlabelled
    }
    k2 = SogKernel.from_state_dict(legacy)
    _assert_bitexact(k.amp_internal(), k2.amp_internal(), "v1 amp")
    # sqrt-then-square cannot be bit-exact in general float32; require closeness.
    a = k.bandwidth_sq().detach().double()
    b = k2.bandwidth_sq().detach().double()
    assert torch.allclose(a, b, rtol=1e-6), f"v1 bw2 max rel {(a - b).abs().div(a).max().item():.3e}"


def test_bind_external_drops_owned_parameters():
    """Switching to external mode must not leave stale parameters registered."""
    k = SogKernel(r_cut=6.0, b=2.0, m=12)
    assert "amp" in dict(k.named_parameters())
    k.bind_external(torch.ones(12), torch.ones(12))
    assert "amp" not in dict(k.named_parameters())
    assert "bandwidth" not in dict(k.named_parameters())
    # The _bw_format_version buffer legitimately remains; only the parameters
    # must disappear so an optimizer cannot touch the caller's tensors.
    assert "amp" not in k.state_dict() and "bandwidth" not in k.state_dict()


def test_external_binding_reaches_framework_tensors():
    """Mode B: gradients must flow to the caller's own parameters."""
    torch.manual_seed(5)
    amp = torch.nn.Parameter(torch.rand(12, dtype=torch.float64))
    bw = torch.nn.Parameter(torch.rand(12, dtype=torch.float64) + 0.5)
    k = SogKernel(r_cut=6.0, b=2.0, m=12)
    k.bind_external(amp, bw, amp_is_internal=True, bandwidth_is_squared=False)

    r = torch.rand(10, 3, dtype=torch.float64)
    q = torch.rand(10, dtype=torch.float64) - 0.5
    q = q - q.mean()
    cell = torch.eye(3, dtype=torch.float64).unsqueeze(0) * 8.0

    model = Sog({"use_atomwise": False}, kernel=k)
    model(positions=r, cell=cell, latent_charges=q, compute_energy=True)["E_lr"].sum().backward()

    assert amp.grad is not None and amp.grad.abs().sum() > 0
    assert bw.grad is not None and bw.grad.abs().sum() > 0


def test_sog_adopt_and_as_kernel():
    """A plain Sog exposes a read-only kernel view with no duplicated parameters."""
    m = Sog({"use_atomwise": False, "r_cut": 6.0, "b": 2.0, "m": 12,
             "trainable_kernel": False})
    k = m.as_kernel()
    _assert_bitexact(m.gaussian.amp, k.amp_internal(), "adopted amp")
    _assert_bitexact(m.gaussian._bw2(), k.bandwidth_sq(), "adopted bw2")
    # Adopted kernels own nothing, so they do not duplicate parameters.
    assert k.state_dict() == {}


def test_kernel_solves_identically_to_legacy_construction():
    """Wiring a kernel through Sog must not change a single number."""
    torch.manual_seed(3)
    r = torch.rand(12, 3, dtype=torch.float64)
    q = torch.rand(12, dtype=torch.float64) - 0.5
    q = q - q.mean()
    cell = torch.eye(3, dtype=torch.float64).unsqueeze(0) * 10.0

    legacy = Sog({"use_atomwise": False, "trainable_kernel": False})
    e_legacy = legacy(positions=r, cell=cell, latent_charges=q)["E_lr"]

    driven = Sog({"use_atomwise": False}, kernel=legacy.as_kernel())
    e_driven = driven(positions=r, cell=cell, latent_charges=q)["E_lr"]

    assert torch.equal(e_legacy, e_driven), (
        f"kernel wiring changed the energy: {e_legacy.item()} vs {e_driven.item()}"
    )
