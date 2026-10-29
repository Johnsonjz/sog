"""Shared helpers for the ``sog.v2`` test modules (not collected as tests)."""

from __future__ import annotations

import math

import torch

from sog import SogKernel
from sog.tools.phi_max_rule import phi_max, sigma_min_of
from sog.v2 import kspace_op


def make_kernel(r_cut: float = 5.0) -> SogKernel:
    return SogKernel(r_cut=r_cut, b=1.6297670882677646, m=12)


def delta_for(kernel: SogKernel, eps: float = 1e-4, spline: int = 4) -> float:
    """Grid spacing ``Delta = phi_max(sigma_min) * r_cut`` (op's rule)."""
    bw2 = kernel.bandwidth_sq().detach().to(torch.float64)
    return float(phi_max(sigma_min_of(bw2), float(kernel.r_cut), spline, eps)) * float(
        kernel.r_cut
    )


def smooth235(n: int) -> int:
    while n > 1:
        if n % 2 == 0:
            n //= 2
        elif n % 3 == 0:
            n //= 3
        elif n % 5 == 0:
            n //= 5
        else:
            return n  # not factorable -> caller advances
    return n


def next_smooth235(n: int) -> int:
    while True:
        m = n
        while m > 1:
            if m % 2 == 0:
                m //= 2
            elif m % 3 == 0:
                m //= 3
            elif m % 5 == 0:
                m //= 5
            else:
                break
        if m == 1:
            return n
        n += 1


def expected_grid_axis(L: float, kernel: SogKernel, *, eps: float = 1e-4, spline: int = 4) -> int:
    """Replicate ``sog_green::grid_axis`` (min 8, 2/3/5-smooth)."""
    delta = delta_for(kernel, eps=eps, spline=spline)
    return next_smooth235(max(8, int(math.ceil(L / delta))))


def make_system(n: int = 64, L: float = 12.0, seed: int = 0, neutral: bool = True):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(n, generator=g, dtype=torch.float64)
    if neutral:
        q = q - q.mean()
    pos = torch.rand(n, 3, generator=g, dtype=torch.float64) * L
    box = torch.tensor([L, 0.0, 0.0, 0.0, L, 0.0, 0.0, 0.0, L], dtype=torch.float64)
    return q, pos, box


def try_load_op() -> bool:
    try:
        return kspace_op.load_op_library()
    except Exception:
        return False


def op_ready() -> bool:
    return try_load_op() and torch.cuda.is_available()
