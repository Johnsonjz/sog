"""Shared value types for :mod:`sog.v2`.

Deliberately dependency-free (only ``torch``) so every v2 module can import it
without creating cycles.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SogSolution:
    """One long-range solve.

    ``E_k = 1/2 * sum_i v_i * qn_i`` exactly, because the mesh operator is
    symmetric (see the algorithm design doc, section 2.2).  ``v`` and ``qn``
    are detached values coming out of the solver; the *charge-response*
    gradient is recovered by seeding a single backward with ``sum(v*qn)``.
    """

    qn: torch.Tensor       # neutralized charges, [N] or [N, nch]
    v: torch.Tensor        # dE_k/dqn, [N]
    F_dir: torch.Tensor    # -dE_k/dr at fixed q, [N, 3]
    vir6: torch.Tensor     # (xx, yy, zz, xy, xz, yz), LAMMPS order

    def energy(self) -> torch.Tensor:
        """``E_k`` as a scalar tensor (exact quadratic-form identity)."""
        return 0.5 * (self.v.reshape(-1) * self.qn.reshape(-1)).sum()


def vir6_to_deepmd(vir6: torch.Tensor) -> torch.Tensor:
    """(xx,yy,zz,xy,xz,yz) -> deepmd row-major symmetric 3x3, shape ``(1, 9)``."""
    m = vir6.reshape(-1)
    return torch.stack([m[0], m[3], m[4], m[3], m[1], m[5], m[4], m[5], m[2]]).reshape(1, 9)


def vir6_to_lammps6(vir6: torch.Tensor) -> torch.Tensor:
    """Return the LAMMPS 6-vector order (the op already uses it)."""
    return vir6.reshape(-1)[:6]


def vir6_to_3x3(vir6: torch.Tensor) -> torch.Tensor:
    """(xx,yy,zz,xy,xz,yz) -> symmetric 3x3 tensor."""
    m = vir6.reshape(-1)
    row0 = torch.stack([m[0], m[3], m[4]])
    row1 = torch.stack([m[3], m[1], m[5]])
    row2 = torch.stack([m[4], m[5], m[2]])
    return torch.stack([row0, row1, row2])
