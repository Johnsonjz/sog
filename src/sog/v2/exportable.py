"""Trace-safe pure-PyTorch SOG kernel (the no-CUDA fallback).

The legacy :class:`sog.Gaussian` solver is written for eager speed: it syncs
the host (``.item()``/``.tolist()``, data-dependent branches) and therefore
cannot be traced by ``make_fx`` / ``torch.export``.  This module is a separate,
export-safe reference:

* **static reciprocal grid** — ``nk0/nk1/nk2`` are fixed at construction; the
  spherical cutoff ``|k| <= 2*pi/n_dl`` is applied as an elementwise mask, so
  the graph contains no ``.item()``, no Python loop over frames and no
  ``nonzero``/``index_select``;
* **analytic derivatives** — ``v = dE/dq``, ``F_dir = -dE/dr`` and the complete
  box-strain virial are computed from the same structure factors (no internal
  ``torch.autograd.grad``), which is what lets a single backward through
  ``sum(v*q)`` produce the charge-response force inside an exported graph.

The discretization is a direct k-sum (not the CubeS₂ mesh).  That is
intentional: the algorithm design allows training/inference solvers to differ,
and the direct sum gives a clean, self-contained reference for validating the
op path.  ``n_dl`` is the spherical cutoff parameter, identical in meaning to
the legacy direct path.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from sog.kernel import E2_PER_ANGSTROM_TO_EV, SogKernel

_TWO_PI = 2.0 * math.pi


class TraceableSogKernel(nn.Module):
    """Static-grid, single-channel direct k-sum with analytic v/F/virial.

    Parameters
    ----------
    amp:
        Internal amplitudes ``A_m`` (``SogKernel.amp_internal()``).
    bandwidth_sq:
        Squared bandwidths ``beta_m`` (``SogKernel.bandwidth_sq()``).
    n_dl:
        Spherical k-space cutoff ``|k| <= 2*pi/n_dl`` (Angstrom).
    nk:
        Static grid extent per axis (``2*nk+1`` modes per axis).  Must cover
        the largest cell: ``nk >= ceil(L_i/n_dl)`` for every runtime box.
    remove_self_interaction:
        Subtract the k-space self term (the default, as in ``SogKernel``).
    norm_factor:
        Coulomb constant (eV Angstrom / e^2).
    """

    def __init__(
        self,
        amp: torch.Tensor,
        bandwidth_sq: torch.Tensor,
        *,
        n_dl: float,
        nk: int = 32,
        nk0: Optional[int] = None,
        nk1: Optional[int] = None,
        nk2: Optional[int] = None,
        remove_self_interaction: bool = True,
        norm_factor: float = E2_PER_ANGSTROM_TO_EV,
    ) -> None:
        super().__init__()
        if n_dl is None or not (float(n_dl) > 0.0):
            raise ValueError("n_dl must be a positive number")
        amp = torch.as_tensor(amp, dtype=torch.float64).reshape(-1)
        bw2 = torch.as_tensor(bandwidth_sq, dtype=torch.float64).reshape(-1)
        if amp.numel() == 0 or amp.numel() != bw2.numel():
            raise ValueError("amp and bandwidth_sq must be non-empty and equal length")
        # Stored as Python constants, not buffers: an exported artifact bakes
        # the frozen kernel into the graph, and constant tensors must not be
        # captured as real objects (that breaks symbolic/FakeTensor tracing).
        self._amp = tuple(float(x) for x in amp.tolist())
        self._bw2 = tuple(float(x) for x in bw2.tolist())
        self.n_dl = float(n_dl)
        self.nk0 = int(nk if nk0 is None else nk0)
        self.nk1 = int(nk if nk1 is None else nk1)
        self.nk2 = int(nk if nk2 is None else nk2)
        self.remove_self_interaction = bool(remove_self_interaction)
        self.norm_factor = float(norm_factor)

    @classmethod
    def from_kernel(
        cls,
        kernel: SogKernel,
        *,
        n_dl: Optional[float] = None,
        nk: int = 32,
        box_size_max: Optional[float] = None,
        **kwargs,
    ) -> "TraceableSogKernel":
        """Build from a ``SogKernel``; ``n_dl`` defaults to ``box/nk`` sizing."""
        if n_dl is None:
            if box_size_max is None:
                raise ValueError(
                    "pass n_dl explicitly, or box_size_max to derive nk from it"
                )
            n_dl = float(box_size_max) / float(nk - 1)
        want = None
        if box_size_max is not None:
            want = int(math.ceil(float(box_size_max) / float(n_dl))) + 1
        nk = want if want is not None else nk
        return cls(
            kernel.amp_internal().detach().to(torch.float64),
            kernel.bandwidth_sq().detach().to(torch.float64),
            n_dl=float(n_dl),
            nk=nk,
            remove_self_interaction=bool(kernel.remove_self_interaction),
            norm_factor=float(kernel.norm_factor),
            **kwargs,
        )

    # ── grid / spectral tables ───────────────────────────────────────────

    def _spectral(self, cell: torch.Tensor, dtype, device):
        """Return ``(g_cart, ksq, kfac, kfac_v)`` for a ``(nf,3,3)`` cell."""
        nf = cell.shape[0]
        inv = torch.linalg.inv(cell)  # (nf,3,3)
        n1 = torch.arange(-self.nk0, self.nk0 + 1, dtype=dtype, device=device)
        n2 = torch.arange(-self.nk1, self.nk1 + 1, dtype=dtype, device=device)
        n3 = torch.arange(-self.nk2, self.nk2 + 1, dtype=dtype, device=device)
        s1, s2, s3 = n1.numel(), n2.numel(), n3.numel()
        k_grid = torch.stack(
            (
                n1.view(s1, 1, 1).expand(s1, s2, s3),
                n2.view(1, s2, 1).expand(s1, s2, s3),
                n3.view(1, 1, s3).expand(s1, s2, s3),
            ),
            dim=0,
        )  # (3,s1,s2,s3)
        g_cart = _TWO_PI * torch.einsum("fik,k...->fi...", inv, k_grid)
        ksq = torch.sum(g_cart * g_cart, dim=1)  # (nf,s1,s2,s3)

        zero = (k_grid[0] == 0) & (k_grid[1] == 0) & (k_grid[2] == 0)
        ksq_max = (_TWO_PI / self.n_dl) ** 2
        mask = (~zero) & (ksq <= ksq_max)

        # Materialize the frozen kernel inside the traced function so the graph
        # carries constants, never captured real tensors (FakeTensor-safe).
        amp = torch.tensor(self._amp, dtype=dtype, device=device)
        bw2 = torch.tensor(self._bw2, dtype=dtype, device=device)
        expo = torch.exp(-0.5 * bw2.view(1, 1, 1, 1, -1) * ksq.unsqueeze(-1))
        kfac = (amp.view(1, 1, 1, 1, -1) * expo).sum(-1)
        kfac_v = ((amp * bw2).view(1, 1, 1, 1, -1) * expo).sum(-1)
        mk = mask.to(dtype)
        return g_cart, ksq, kfac * mk, kfac_v * mk

    def _canon(self, positions: torch.Tensor, q: torch.Tensor, cell: Optional[torch.Tensor]):
        if positions.dim() == 2:
            positions = positions.unsqueeze(0)
        if positions.dim() != 3 or positions.shape[-1] != 3:
            raise ValueError(f"positions must be (nf,nloc,3) or (nloc,3), got {tuple(positions.shape)}")
        q = torch.as_tensor(q)
        if q.dim() == 1:
            q = q.unsqueeze(0)
        if q.dim() == 3 and q.shape[-1] == 1:
            q = q.squeeze(-1)
        if q.dim() != 2:
            raise ValueError(f"q must be (nf,nloc) or (nloc,), got {tuple(q.shape)}")
        if q.shape != positions.shape[:2]:
            raise ValueError(f"q shape {tuple(q.shape)} != positions[:2] {tuple(positions.shape[:2])}")
        if cell is None:
            raise ValueError("TraceableSogKernel requires a periodic cell (cell != None)")
        if cell.dim() == 2:
            cell = cell.unsqueeze(0)
        if cell.shape != (positions.shape[0], 3, 3):
            raise ValueError(f"cell must be (nf,3,3), got {tuple(cell.shape)}")
        return positions, q, cell

    # ── public API ───────────────────────────────────────────────────────

    def energy(self, q: torch.Tensor, positions: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
        """Return ``E_k`` with shape ``(nf,)``."""
        positions, q, cell = self._canon(positions, q, cell)
        g_cart, _ksq, kfac, _kv = self._spectral(cell, positions.dtype, positions.device)
        volume = torch.abs(torch.linalg.det(cell))  # (nf,)
        nf = positions.shape[0]
        kvec = g_cart.flatten(2).transpose(1, 2)          # (nf,Nk,3)
        k_dot_r = torch.matmul(positions, kvec.transpose(1, 2))  # (nf,nloc,Nk)
        cos_kr = torch.cos(k_dot_r)
        sin_kr = torch.sin(k_dot_r)
        s_re = torch.einsum("fn,fnk->fk", q, cos_kr)
        s_im = torch.einsum("fn,fnk->fk", q, sin_kr)
        s_sq = s_re * s_re + s_im * s_im                  # (nf,Nk)
        e = (kfac.flatten(1) * s_sq).sum(1) / (2.0 * volume)
        if self.remove_self_interaction:
            e = e - (q * q).sum(1) * kfac.flatten(1).sum(1) / (2.0 * volume)
        return e * self.norm_factor

    def solution(self, qn: torch.Tensor, positions: torch.Tensor, cell: torch.Tensor):
        """Return ``(v, F_dir, vir6)`` with shapes ``(nf,nloc)``, ``(nf,nloc,3)``, ``(nf,6)``."""
        positions, qn, cell = self._canon(positions, qn, cell)
        g_cart, ksq, kfac, kfac_v = self._spectral(cell, positions.dtype, positions.device)
        volume = torch.abs(torch.linalg.det(cell))        # (nf,)
        nf, nloc = positions.shape[:2]
        kvec = g_cart.flatten(2).transpose(1, 2)          # (nf,Nk,3)
        k_dot_r = torch.matmul(positions, kvec.transpose(1, 2))  # (nf,nloc,Nk)
        cos_kr = torch.cos(k_dot_r)
        sin_kr = torch.sin(k_dot_r)
        s_re = torch.einsum("fn,fnk->fk", qn, cos_kr)     # (nf,Nk)
        s_im = torch.einsum("fn,fnk->fk", qn, sin_kr)
        s_sq = s_re * s_re + s_im * s_im
        kf = kfac.flatten(1)                              # (nf,Nk)
        kv = kfac_v.flatten(1)                            # (nf,Nk)

        # v_i = (1/V) sum_k K (S_re cos + S_im sin) - (q_i/V) sum_k K
        v = torch.einsum("fk,fnk->fn", kf, s_re * cos_kr + s_im * sin_kr) / volume.unsqueeze(1)
        if self.remove_self_interaction:
            v = v - qn * kf.sum(1).unsqueeze(1) / volume.unsqueeze(1)

        # F_i = -(q_i/V) sum_k K k (S_im cos - S_re sin)
        f_weight = (s_im.unsqueeze(1) * cos_kr - s_re.unsqueeze(1) * sin_kr)  # (nf,nloc,Nk)
        f_dir = -qn.unsqueeze(-1) * torch.einsum("fk,fka,fnk->fna", kf, kvec, f_weight)
        f_dir = f_dir / volume.view(nf, 1, 1)

        # vir6 = k-space + self strain derivatives (both complete strain forms)
        #   W_ks   = (1/2V) sum_k [K d_ab - K_v k_a k_b] |S|^2
        #   W_self = (1/2V) q^2sum [sum_k K_v k_a k_b - d_ab sum_k K]
        eye = torch.eye(3, dtype=positions.dtype, device=positions.device).view(1, 1, 3, 3)
        kk = kvec.unsqueeze(-1) * kvec.unsqueeze(-2)      # (nf,Nk,3,3)
        s_sq_e = s_sq.unsqueeze(-1).unsqueeze(-1)
        w_ks = (kf.unsqueeze(-1).unsqueeze(-1) * eye
                - kv.unsqueeze(-1).unsqueeze(-1) * kk) * s_sq_e
        w_ks = w_ks.sum(1) / (2.0 * volume.view(nf, 1, 1))
        if self.remove_self_interaction:
            qsq = (qn * qn).sum(1).view(nf, 1, 1)
            w_self = qsq * (kv.unsqueeze(-1).unsqueeze(-1) * kk).sum(1)
            w_self = w_self - qsq * kf.sum(1).view(nf, 1, 1) * eye.view(1, 3, 3)
            w_self = w_self / (2.0 * volume.view(nf, 1, 1))
            w = w_ks + w_self
        else:
            w = w_ks
        vir6 = torch.stack(
            [w[:, 0, 0], w[:, 1, 1], w[:, 2, 2], w[:, 0, 1], w[:, 0, 2], w[:, 1, 2]], dim=1
        ) * self.norm_factor
        return v * self.norm_factor, f_dir * self.norm_factor, vir6


class ReferenceKSpace(nn.Module):
    """``ExternalKSpace``-compatible pure-PyTorch backend (CPU/CUDA, traceable)."""

    def __init__(self, kernel: TraceableSogKernel) -> None:
        super().__init__()
        self.kernel = kernel

    def solve(self, qn: torch.Tensor, positions: torch.Tensor, box: torch.Tensor):
        """Return ``(v, F_dir, vir6)`` as **values** (detached), like the op.

        The op is a value-only custom op: its outputs carry no autograd.  The
        reference must match that contract, otherwise the fused seed
        ``sum(v*q)`` would differentiate the solver's own dependence on the
        coordinates and double-count the direct force.  The response force is
        always recovered from the q-chain only.
        """
        cell = _box_to_cell(box, positions)
        v, f_dir, vir6 = self.kernel.solution(qn, positions, cell)
        v, f_dir, vir6 = v.detach(), f_dir.detach(), vir6.detach()
        nf = v.shape[0]
        if nf == 1:
            return v[0], f_dir[0], vir6[0]
        return v, f_dir, vir6


def _box_to_cell(box: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    if box.shape == (3, 3):
        cell = box.unsqueeze(0)
    elif box.shape == (9,):
        cell = box.reshape(1, 3, 3)
    else:
        raise ValueError(f"box must be (9,) or (3,3), got {tuple(box.shape)}")
    if positions.dim() == 3:
        cell = cell.expand(positions.shape[0], -1, -1)
    return cell
