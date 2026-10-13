"""Small-scale tests: SOG (CubeS₂ FFT) vs direct k-sum at training-relevant
scales — accuracy + (informational) speed.  No GPU required.

1. ``test_water_192`` — 192 atoms, 12.44 Å box, b=1.63, r_c=6.0 Å.
2. ``test_nacl_512`` — 512 ions, 24 Å box, b=2.0, r_c=10.0 Å.
3. ``test_accuracy_vs_n`` — force agreement for N=100…5000 in a 30 Å box
   (timing printed for information only — machine-dependent, never asserted).

Kernel-construction note (was the original script's bug): giving ``rcut``
OVERRIDES an explicit ``sigma`` (σ = r_c·nlayers/RCUT_TO_SIGMA, gaussian.py
:137); both sides of a comparison must therefore be built with the same
``rcut`` — mixing ``sigma=3.63`` with ``rcut=6.0`` silently compares two
different kernels and no grid refinement can reconcile them.

Energy-accuracy note: for random ±q salts the total energy is a large
cancellation, so its *relative* error overstates the mesh error; the
assertions below are therefore force-based (φ_max is calibrated on force
accuracy), with energy checks only where they are meaningful.
"""

import time
import warnings

import torch

torch.set_default_dtype(torch.float64)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from sog.module.gaussian import Gaussian  # noqa: E402
from sog.module.cubes2_fft import (  # noqa: E402
    _INFLUENCE_CACHE,
    _compute_phi_max,
    _compute_grid_from_phi,
)


def _ef(g, q, r, cell):
    r2 = r.clone().requires_grad_(True)
    e = g(q, r2, cell)
    f = -torch.autograd.grad(e, r2, grad_outputs=torch.ones_like(e))[0]
    return e.detach(), f.detach()


def _direct(q, r, cell, b, rcut, n_dl=2.0, nlayers=1):
    g = Gaussian(n_dl=n_dl, b=b, rcut=rcut, nlayers=nlayers,
                 use_cubes2_fft=False, use_nufft=False,
                 remove_self_interaction=True)
    return _ef(g, q, r, cell)


def _fft(q, r, cell, phi, b, rcut, nlayers=1):
    _INFLUENCE_CACHE.clear()
    g = Gaussian(cubes2_phi_max=phi, b=b, rcut=rcut, nlayers=nlayers,
                 use_cubes2_fft=True, remove_self_interaction=True)
    return _ef(g, q, r, cell)


def test_water_192():
    """Water 64H₂O scale: 192 atoms, ~12.44 Å box, b=1.63, r_c=6.0 Å."""
    torch.manual_seed(42)
    L, r_c, b_val = 12.44, 6.0, 1.63
    r = torch.rand(192, 3) * L
    q = torch.randn(96) * 0.5
    q = torch.cat([q, -q])
    cell = torch.eye(3).unsqueeze(0) * L

    e_dir, f_dir = _direct(q, r, cell, b_val, r_c)

    phi_auto = _compute_phi_max(b_val, 4)
    e_fft, f_fft = _fft(q, r, cell, phi_auto, b_val, r_c)

    cos = torch.cosine_similarity(f_dir.flatten(), f_fft.flatten(), dim=0).item()
    e_rel = abs(e_fft.item() - e_dir.item()) / max(abs(e_dir.item()), 1e-30)

    assert cos > 0.99999, f"force cosine {cos:.8f} at φ={phi_auto:.4f} (measured ≈1.0)"
    assert e_rel < 1e-3, (
        f"energy rel {e_rel:.2e} at φ_auto={phi_auto:.4f} (measured 2.8e-4)"
    )

    # a finer grid must not be worse
    e_finer, _ = _fft(q, r, cell, 0.03, b_val, r_c)
    e_rel_finer = abs(e_finer.item() - e_dir.item()) / max(abs(e_dir.item()), 1e-30)
    assert e_rel_finer < e_rel, f"refinement worsened the energy: {e_rel:.2e} -> {e_rel_finer:.2e}"


def test_nacl_512():
    """NaCl 512 ions (256 Na + 256 Cl), ~24 Å box, b=2.0, r_c=10.0 Å."""
    torch.manual_seed(42)
    L, r_c, b_val, n = 24.0, 10.0, 2.0, 512
    r = torch.rand(n, 3) * L
    q = torch.ones(n)
    q[n // 2:] = -1.0  # half Na+, half Cl-
    q[: n // 2] = 1.0
    cell = torch.eye(3).unsqueeze(0) * L

    e_dir, f_dir = _direct(q, r, cell, b_val, r_c)

    phi_auto = _compute_phi_max(b_val, 4)
    e_fft, f_fft = _fft(q, r, cell, phi_auto, b_val, r_c)
    cos = torch.cosine_similarity(f_dir.flatten(), f_fft.flatten(), dim=0).item()
    assert cos > 0.9999, (
        f"force cosine {cos:.8f} at φ={phi_auto:.4f} (measured 0.9999990)"
    )

    # refinement must improve FORCE agreement, and bring the (cancellation-
    # amplified) energy error down as well
    e_fine, f_fine = _fft(q, r, cell, 0.05, b_val, r_c)
    cos_fine = torch.cosine_similarity(f_dir.flatten(), f_fine.flatten(), dim=0).item()
    assert cos_fine >= cos, f"refinement worsened forces: {cos:.9f} -> {cos_fine:.9f}"
    dE_fine = abs(e_fine.item() - e_dir.item())
    assert dE_fine < 0.05, (
        f"|ΔE|={dE_fine:.3e} eV at φ=0.05 on the 512-ion salt (measured 3.1e-3; "
        f"note E≈7 eV is a large cancellation)"
    )


def test_accuracy_vs_n():
    """Force agreement (and informational timing) for N = 100…5000."""
    L, r_c, b_val = 30.0, 10.0, 2.0
    for n_pts in (100, 500, 2000, 5000):
        torch.manual_seed(42)
        r = torch.rand(n_pts, 3) * L
        q = torch.randn(n_pts // 2) * 0.5
        q = torch.cat([q, -q])
        if n_pts % 2 == 1:
            q = torch.cat([q, torch.zeros(1)])
        perm = torch.randperm(n_pts)
        r, q = r[perm], q[perm]
        cell = torch.eye(3).unsqueeze(0) * L

        t0 = time.perf_counter()
        _e_dir, f_dir = _direct(q, r, cell, b_val, r_c)
        t_dir = (time.perf_counter() - t0) * 1e3

        t0 = time.perf_counter()
        _e_fft, f_fft = _fft(q, r, cell, 0.15, b_val, r_c)
        t_fft = (time.perf_counter() - t0) * 1e3

        cos = torch.cosine_similarity(f_dir.flatten(), f_fft.flatten(), dim=0).item()
        print(f"  N={n_pts:5d}: cos={cos:.8f}  dir={t_dir:7.1f}ms  "
              f"fft={t_fft:7.1f}ms  speedup={t_dir / t_fft:5.2f}x")
        assert cos > 0.9999, f"N={n_pts}: force cosine {cos:.8f} at φ=0.15"
