"""Framework-neutral long-range algebra for the deployment (inference/export) path.

This module turns a trained kernel + latent charges into the three linear
quantities the 1F+1B composition needs:

* ``v = dE_k/dq`` (per-atom potential),
* ``F_dir = -dE_k/dr`` at fixed charges,
* the complete box-strain virial ``vir6`` in LAMMPS order
  ``(xx, yy, zz, xy, xz, yz)``,

plus the fused seed ``sum(v * qn) / n_owned`` whose backward produces the
charge-response force.  Two backends are available:

* ``"op"`` — the CUDA ``sogkspace::solve_auto`` custom op (deployment path);
* ``"reference"`` — the pure-PyTorch static-grid direct sum in
  :mod:`sog.v2.exportable` (CPU, exportable, slower);
* ``"auto"`` — op when CUDA + the library are available, else reference.

The share semantics follow the deployed chain: ``E_k``/seed are this rank's
share, ``F_dir`` is per local node, and ``vir6`` is the global value divided
by the rank count (the op does the division).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch

from sog.kernel import E2_PER_ANGSTROM_TO_EV, SogKernel

from . import kspace_op
from .types import SogSolution, vir6_to_deepmd, vir6_to_lammps6

__all__ = [
    "KSpaceSettings",
    "ExternalKSpace",
    "SogSolution",
    "vir6_to_deepmd",
    "vir6_to_lammps6",
]


@dataclass(frozen=True)
class KSpaceSettings:
    """Solver settings forwarded to the op / reference backend."""

    spline: int = 4
    phi_accuracy: float = 1e-4
    box_tol: float = 1e-3
    qscale: float = E2_PER_ANGSTROM_TO_EV
    self_coeff: float = 0.0
    remove_self: bool = True
    #: Real-space cutoff used for grid sizing; defaults to ``SogKernel.r_cut``.
    rcut: Optional[float] = None
    #: Direct k-sum cutoff for the reference backend (Angstrom); derived from
    #: the phi_max rule when omitted.
    n_dl: Optional[float] = None


def neutralize_reference(q: torch.Tensor, n_local=None) -> torch.Tensor:
    """Pure-PyTorch gauge projection (traceable, exact autograd).

    ``qn = owned * (q - mean_owned(q))`` with ghost rows zeroed.  This is the
    reference for the op's ``neutralize`` and the export-safe fallback.
    """
    n = q.shape[0]
    if n_local is None:
        nl = torch.tensor(float(n), dtype=q.dtype, device=q.device)
    elif torch.is_tensor(n_local):
        nl = n_local.to(dtype=q.dtype, device=q.device).reshape(-1)[0]
    else:
        nl = torch.tensor(float(n_local), dtype=q.dtype, device=q.device)
    shape = (-1,) + (1,) * (q.dim() - 1)
    owned = (torch.arange(n, device=q.device) < nl).to(q.dtype).reshape(shape)
    mean = (q * owned).sum(0) / nl
    return (q - mean) * owned


class ExternalKSpace:
    """One charge set -> ``(vn, v, F_dir, vir6)`` through op or reference."""

    def __init__(
        self,
        kernel: SogKernel,
        *,
        settings: Optional[KSpaceSettings] = None,
        handle: int = 1,
        backend: str = "auto",
        nk: int = 32,
        reference=None,
    ) -> None:
        if backend not in ("auto", "op", "reference"):
            raise ValueError("backend must be 'auto', 'op' or 'reference'")
        self.kernel = kernel
        self.settings = settings or KSpaceSettings()
        self.handle = int(handle)
        self.backend = backend
        self.nk = int(nk)
        # A pre-built ``ReferenceKSpace`` (fixed grid) is REQUIRED for export:
        # the lazy builder below sizes the grid from the runtime box, which is
        # a host sync and therefore not traceable.
        self._ref = reference
        self._ref_nk = None if reference is None else -1

    # ── configuration ────────────────────────────────────────────────────

    @property
    def rcut(self) -> float:
        if self.settings.rcut is not None:
            return float(self.settings.rcut)
        if self.kernel.r_cut is not None:
            return float(self.kernel.r_cut)
        raise ValueError(
            "KSpaceSettings.rcut (or SogKernel.r_cut) is required for the "
            "periodic solve; pass the real-space cutoff the charges were "
            "trained with"
        )

    def _amp_bw(self, dtype, device):
        amp = self.kernel.amp_internal().detach().to(dtype=dtype, device=device)
        bw2 = self.kernel.bandwidth_sq().detach().to(dtype=dtype, device=device)
        return amp.reshape(-1), bw2.reshape(-1)

    def _use_op(self, positions: torch.Tensor) -> bool:
        if self.backend == "reference":
            return False
        if self.backend == "op":
            kspace_op.require_op()
            return True
        return positions.is_cuda and kspace_op.load_op_library()

    def _reference(self, box: torch.Tensor):
        """Return the reference backend; lazily build it sized for ``box``."""
        if self._ref is not None:
            return self._ref
        from .exportable import ReferenceKSpace, TraceableSogKernel

        cell = box.reshape(3, 3) if box.shape in ((9,), (3, 3)) else box.reshape(3, 3)
        Lmax = float(torch.abs(torch.diagonal(cell)).max())
        n_dl = self.settings.n_dl
        if n_dl is None:
            n_dl = self._reference_n_dl()
        want_nk = int(math.ceil(Lmax / float(n_dl))) + 1
        if self._ref is None or want_nk > self._ref_nk:
            tk = TraceableSogKernel(
                self.kernel.amp_internal().detach().to(torch.float64),
                self.kernel.bandwidth_sq().detach().to(torch.float64),
                n_dl=float(n_dl),
                nk=want_nk,
                remove_self_interaction=bool(self.settings.remove_self),
                norm_factor=float(self.settings.qscale),
            )
            self._ref = ReferenceKSpace(tk)
            self._ref_nk = want_nk
        return self._ref

    def _reference_n_dl(self) -> float:
        """Grid spacing from the phi_max rule (matches the op's sizing)."""
        bw2 = self.kernel.bandwidth_sq().detach().to(torch.float64)
        sigma_min = float(torch.sqrt(bw2.min()))
        try:
            from sog.tools.phi_max_rule import phi_max

            delta = phi_max(sigma_min, self.rcut, int(self.settings.spline),
                            float(self.settings.phi_accuracy)) * self.rcut
            return float(delta)
        except Exception:  # pragma: no cover - tools always ship with the package
            return float(
                math.pi * math.sqrt(2.0) * sigma_min
                / math.sqrt(math.log(1.0 / self.settings.phi_accuracy))
            )

    # ── neutralization ───────────────────────────────────────────────────

    def _neutralize(self, q: torch.Tensor, n_local, *, use_op: bool) -> torch.Tensor:
        n = q.shape[0]
        nl = (
            torch.tensor(float(n), dtype=q.dtype, device=q.device)
            if n_local is None
            else (n_local.to(dtype=q.dtype, device=q.device).reshape(-1)[0]
                  if torch.is_tensor(n_local)
                  else torch.tensor(float(n_local), dtype=q.dtype, device=q.device))
        )
        if use_op:
            kspace_op.register_neutralize_autograd()
            return kspace_op.neutralize(q, nl)
        return neutralize_reference(q, nl)

    # ── public API ───────────────────────────────────────────────────────

    def solve(self, q_raw: torch.Tensor, positions: torch.Tensor,
              box: torch.Tensor, n_local=None) -> SogSolution:
        """Neutralize and solve; returns a :class:`SogSolution`."""
        q = q_raw
        if q.dim() == 2 and q.shape[-1] == 1:
            q = q.reshape(-1)
        if q.dim() != 1:
            raise ValueError(
                "ExternalKSpace supports a single latent-charge channel "
                f"(q must be 1-D), got {tuple(q.shape)}"
            )
        if positions.dim() != 2 or positions.shape[-1] != 3:
            raise ValueError(f"positions must be (N,3), got {tuple(positions.shape)}")
        use_op = self._use_op(positions)
        q = q.to(device=positions.device, dtype=positions.dtype)
        qn = self._neutralize(q, n_local, use_op=use_op)
        if use_op:
            amp, bw2 = self._amp_bw(positions.dtype, positions.device)
            v, f_dir, vir6 = kspace_op.solve_auto(
                qn,
                positions,
                box,
                amp,
                bw2,
                rcut=self.rcut,
                spline=int(self.settings.spline),
                phi_accuracy=float(self.settings.phi_accuracy),
                box_tol=float(self.settings.box_tol),
                qscale=float(self.settings.qscale),
                self_coeff=float(self.settings.self_coeff),
                remove_self=bool(self.settings.remove_self),
                handle=self.handle,
            )
        else:
            v, f_dir, vir6 = self._reference(box).solve(qn, positions, box)
        return SogSolution(qn=qn, v=v, F_dir=f_dir, vir6=vir6)

    def fused_seed(self, sol: SogSolution, n_owned=None) -> torch.Tensor:
        """``sum(v * qn) / n_owned`` — inject into the differentiated energy.

        Do **not** multiply by ``1/2`` here: ``v`` is frozen, so ``1/2`` would
        halve the physical charge-response force.  The reported energy uses
        ``energy_from_solution`` (which does apply ``1/2``).
        """
        seed = (sol.v.reshape(-1) * sol.qn.reshape(-1)).sum()
        if n_owned is None:
            return seed / sol.qn.shape[0]
        if torch.is_tensor(n_owned):
            return seed / n_owned.to(seed.dtype).reshape(-1)[0]
        return seed / float(n_owned)

    def energy_from_solution(self, sol: SogSolution) -> torch.Tensor:
        """``E_k = 1/2 sum_i v_i qn_i`` (this rank's share)."""
        return sol.energy()

    def eager_efv(self, q_raw: torch.Tensor, positions: torch.Tensor,
                  box: torch.Tensor, n_local=None,
                  e_sr: Optional[torch.Tensor] = None) -> dict:
        """Non-export reference composition (autograd; slow but exact).

        ``positions`` must be the tensor the charge model was evaluated on
        (i.e. ``q_raw`` must be connected to it in the autograd graph), and it
        must require grad for a response force to exist.  The returned force
        and virial are the **long-range contributions only**; the caller adds
        its own short-range terms.  ``e_sr`` (if given) is only added to the
        returned energy scalar.

        Virial convention: the response term is ``F_resp (x) r`` — exact for
        charge graphs whose box dependence enters only through the
        coordinates (the standard pattern).  A graph that consumes the box
        explicitly must differentiate its own box leaf; see the design notes
        in ``docs/``.
        """
        sol = self.solve(q_raw, positions, box, n_local)
        # Differentiate the FULL seed sum(v*q), not 0.5*sum(v*q): v is frozen,
        # so the 1/2 belongs to the reported energy only (injecting 0.5*S would
        # halve the physical response force).
        seed = (sol.v.reshape(-1) * sol.qn.reshape(-1)).sum()
        if positions.requires_grad:
            (g,) = torch.autograd.grad(
                seed, positions, retain_graph=False, allow_unused=True
            )
            f_resp = torch.zeros_like(positions) if g is None else -g
        else:
            f_resp = torch.zeros_like(positions)
        w = _force_outer(f_resp, positions) + _vir6_to_3x3(sol.vir6)
        energy_lr = 0.5 * seed
        total = energy_lr if e_sr is None else e_sr + energy_lr
        return {
            "energy": total.detach(),
            "force": (f_resp + sol.F_dir).detach(),
            "virial": w.detach().reshape(1, 9),
            "solution": sol,
        }


def _force_outer(force: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """``F (x) r`` as a 3x3 tensor (response virial for box-free charge graphs)."""
    return force.transpose(0, 1) @ positions


def _vir6_to_3x3(vir6: torch.Tensor) -> torch.Tensor:
    m = vir6.reshape(-1)
    eye = torch.zeros(3, 3, dtype=vir6.dtype, device=vir6.device)
    eye[0, 0], eye[1, 1], eye[2, 2] = m[0], m[1], m[2]
    eye[0, 1] = eye[1, 0] = m[3]
    eye[0, 2] = eye[2, 0] = m[4]
    eye[1, 2] = eye[2, 1] = m[5]
    return eye
