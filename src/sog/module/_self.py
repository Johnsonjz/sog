"""Closed-form SOG self-energy coefficient.

The SOG energy is the k-space form of a *pair* potential.  Summing over the
**complete** reciprocal lattice turns the structure-factor expression back into
that pair sum exactly, whose ``i == j, R == 0`` term is ``½·Σq²·g(0)`` with

    g(0) = Σ_m amp_m (2π bw²_m)^(-3/2)

(``amp`` = internal amplitude, ``bw²`` = squared bandwidth).  It is a real-space
quantity: exactly ``∝ q²`` (hence exactly volume-extensive) and with **zero**
strain derivative (no self virial).

This replaces the old FFT-mesh Riemann sum ``Σ_k K(k²) / (2V)``, which was
neither volume-extensive (≈9× for a 2× cell instead of 8×) nor isotropic (its
mesh error injected a spurious shear).  The two agree only in the
infinite-mesh limit.
"""

from __future__ import annotations

import math

import torch

__all__ = ["self_kernel_sum", "self_diag"]


def self_kernel_sum(amp, bw2):
    """``g(0) = Σ_m amp_m (2π bw²_m)^(-3/2)`` — the kernel at zero separation."""
    amp = torch.as_tensor(amp)
    bw2 = torch.as_tensor(bw2)
    return (amp * (2.0 * math.pi * bw2) ** (-1.5)).sum()


def self_diag(amp, bw2):
    """``½·g(0)`` — the coefficient that multiplies ``Σq²`` in ``E_self``.

    This is exactly the ``diag_sum`` the solvers subtract
    (``E -= Σq² · diag_sum``), so it drops into every place the old mesh sum
    was used.
    """
    return 0.5 * self_kernel_sum(amp, bw2)
