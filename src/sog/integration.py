"""Plugging SOG into an MLIP framework.

SOG needs exactly one thing from the host potential: **per-atom latent charges**.
Everything else — the descriptor, the message passing, the short-range energy —
stays with the framework.  This module defines that contract and the two ways
real frameworks satisfy it.

The two integration modes
-------------------------

**Mode A — owned kernel, autograd append** (what MACE does)

    The framework owns the kernel parameters inside SOG and adds the returned
    long-range energy to its own total energy.  The framework's existing autograd
    then produces forces and the virial, including the charge-response term
    :math:`\\partial E_{lr}/\\partial q \\cdot \\partial q/\\partial r`::

        sog = build_sog(mode="owned", r_cut=6.0, n_channels=1)
        # ... in the framework's forward:
        q = my_readout(node_features)                   # [N, nq]
        e_lr = sog(positions=r, cell=cell, batch=batch,
                   latent_charges=q)["E_lr"]            # [nf]
        total_energy = e_sr + e_lr                      # one graph, one backward
        # no explicit force call: the framework differentiates total_energy

**Mode B — external kernel, fused backward** (what DeepMD does)

    The framework already owns ``amp``/``bandwidth`` as ``nn.Parameter``s (e.g.
    inside a fitting net).  SOG binds *those exact tensors* so gradients flow back
    to them, and the framework runs a single combined ``autograd.grad`` over
    ``E_sr + E_lr``.  Pass a box leaf to also obtain the reciprocal box-strain::

        sog = build_sog(mode="external", r_cut=6.0,
                        amp=fitting.amp,                       # internal amplitude
                        bandwidth=fitting.bandwidth,           # pre-square σ·b^m
                        bandwidth_is_squared=False)
        # ... in the framework's fused forward:
        box_leaf = box.detach().clone().requires_grad_(True)   # for the virial
        e_lr = sog(positions=r, cell=box_leaf, batch=batch,
                   latent_charges=q)["E_lr"]
        grads = torch.autograd.grad([e_sr + e_lr], [coords, box_leaf], ...)

Do not detach the charges in either mode — that is what makes the force
conservative rather than merely electrostatic.

What the framework must guarantee
---------------------------------

1. **Shape**: ``latent_charges`` is ``[N, nq]`` (or ``[N]`` for ``nq == 1``) aligned
   with the ``positions`` handed to SOG.
2. **Batch layout**: frames are contiguous blocks in canonical order, with ids
   ``0 .. nf-1``.  Use :func:`validate_batch` — the fast paths assume this and a
   violation silently produces wrong energies.
3. **Units**: positions in Å, charges in e, cell rows are lattice vectors in Å.
   Energy comes back in eV (``norm_factor`` = 14.3996454784255).
4. **Neutrality**: the k=0 mode is excluded unconditionally, so a non-neutral
   system silently loses that contribution.  See :class:`ChargeNeutralityPolicy`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, Tuple, runtime_checkable

import torch
import torch.nn as nn

from .kernel import E2_PER_ANGSTROM_TO_EV, SogKernel
from .module import Atomwise
from .sog import Sog


__all__ = [
    "LatentChargeProvider",
    "AtomwiseChargeProvider",
    "validate_batch",
    "validate_charges",
    "ChargeNeutralityPolicy",
    "build_sog",
    "lr_energy",
]


@runtime_checkable
class LatentChargeProvider(Protocol):
    """What a framework implements to supply SOG with charges.

    The framework closes over whatever state it needs (descriptors, node
    features, atom types); SOG never inspects it.  The only contract is the
    returned tensor: ``[N, n_channels]`` per-atom charges, ordered consistently
    with the ``positions`` passed to :class:`~sog.sog.Sog`.

    Example — a framework with a descriptor::

        class MyProvider:
            n_channels = 1
            def __init__(self, readout): self.readout = readout
            def __call__(self, desc):
                return self.readout(desc)        # [N, 1]
    """

    n_channels: int

    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor:  # pragma: no cover
        ...


class AtomwiseChargeProvider(nn.Module):
    """Reference :class:`LatentChargeProvider`: an MLP over per-atom descriptors.

    This is the simplest possible charge head — useful as a worked example, and as
    a drop-in when a framework already has per-atom descriptors in hand.
    """

    n_channels: int

    def __init__(
        self,
        n_in: Optional[int] = None,
        n_out: int = 1,
        n_hidden=None,
        n_layers: int = 3,
        add_linear_nn: bool = True,
        output_scaling_factor: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_channels = int(n_out)
        self.net = Atomwise(
            n_in=n_in,
            n_out=n_out,
            n_hidden=n_hidden if n_hidden is not None else [32, 16],
            n_layers=n_layers,
            add_linear_nn=add_linear_nn,
            output_scaling_factor=output_scaling_factor,
        )

    def forward(self, desc: torch.Tensor, batch: Optional[torch.Tensor] = None):
        return self.net(desc, batch)


# ── guards ──────────────────────────────────────────────────────────────────


def validate_batch(batch: Optional[torch.Tensor], n_atoms: int) -> torch.Tensor:
    """Return ``batch`` after checking the layout SOG's fast paths assume.

    The contiguous batching convention (frames in canonical order, ids
    ``0..nf-1``, each frame one contiguous block) is *assumed* but never checked
    inside :class:`~sog.module.gaussian.Gaussian`; a violation produces wrong
    energies rather than an error.  Calling this at the integration boundary turns
    that into a clear failure.

    ``batch=None`` returns a single-frame all-zero batch.
    """
    if batch is None:
        return torch.zeros(n_atoms, dtype=torch.int64)
    if batch.dim() != 1:
        raise ValueError(f"`batch` should be 1-D, got shape {tuple(batch.shape)}.")
    if batch.numel() != n_atoms:
        raise ValueError(
            f"`batch` has {batch.numel()} entries but there are {n_atoms} atoms."
        )
    if batch.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"`batch` should be an integer tensor, got {batch.dtype}.")
    if batch.numel() == 0:
        return batch
    b = batch.to(torch.int64)
    if int(b.min()) != 0:
        raise ValueError(f"`batch` frame ids should start at 0, got min {int(b.min())}.")
    n_frames = int(b.max()) + 1
    # The canonical layout is exactly "arange(n_frames) repeat_interleaved by the
    # per-frame atom counts"; anything else (interleaved frames, reversed order,
    # out-of-order blocks) is rejected.
    counts = torch.bincount(b, minlength=n_frames)
    if bool((counts == 0).any()):
        empty = torch.nonzero(counts == 0).reshape(-1).tolist()
        raise ValueError(
            f"frame ids {empty} have no atoms.  SOG's per-frame loop skips empty "
            "frames, which shortens the returned energy vector; drop unused frame "
            "ids or renumber them densely."
        )
    expected = torch.arange(n_frames, device=b.device).repeat_interleave(counts)
    if not bool(torch.equal(b, expected)):
        raise ValueError(
            "`batch` must list frames as contiguous blocks in canonical order "
            "(0,0,..,1,1,..).  SOG's batched fast paths assume this and a "
            "violation silently yields wrong energies.  Frames are currently "
            f"ordered {b.tolist()[:16]}{'...' if b.numel() > 16 else ''}."
        )
    return batch


def validate_charges(charges: torch.Tensor, n_atoms: int, n_channels: int = 1) -> torch.Tensor:
    """Check a provider's output: shape, finiteness, and channel count."""
    if charges.dim() == 1:
        charges = charges.unsqueeze(1)
    if charges.dim() != 2:
        raise ValueError(
            f"latent charges should be [N, nq], got shape {tuple(charges.shape)}."
        )
    if charges.shape[0] != n_atoms:
        raise ValueError(
            f"latent charges have {charges.shape[0]} rows but there are "
            f"{n_atoms} atoms."
        )
    if charges.shape[1] != n_channels:
        raise ValueError(
            f"provider declares n_channels={n_channels} but returned "
            f"{charges.shape[1]} channels."
        )
    if not torch.isfinite(charges).all():
        raise ValueError("latent charges contain non-finite values.")
    return charges


# ── charge neutrality ───────────────────────────────────────────────────────


@dataclass
class ChargeNeutralityPolicy:
    """How per-frame charge neutrality is enforced.

    SOG has three mechanisms, applied at different layers.  This records which
    are active so an integrator can see the decision in one place.

    ``k0_excluded``
        The k=0 mode is dropped by the solver unconditionally (there is no switch).
        A net-charged system therefore loses that term silently — worth checking
        when energies look wrong.
    ``hard_in_model``
        ``Sog(charge_neutral=True)``: subtract the per-frame, per-channel mean
        before solving.  Mirrors DeepMD's ``_corr_head``.
    ``soft_penalty``
        ``Gaussian(charge_neutral_lambda=λ)``: add ``λ·Σ_c mean(q_c)²`` to the
        energy, letting the network learn neutrality instead of being forced.
    """

    k0_excluded: bool = True
    hard_in_model: bool = False
    soft_lambda: Optional[float] = None

    def apply(self, charges: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """Apply the in-model corrections this policy describes.

        Only the hard correction acts on the charges; ``soft_lambda`` is consumed
        by the solver instead (it enters the energy), so it is a no-op here.
        """
        if not self.hard_in_model:
            return charges
        n_frames = int(batch.max().item()) + 1 if batch.numel() else 0
        if n_frames == 0:
            return charges
        sums = torch.zeros(
            n_frames, charges.shape[1], dtype=charges.dtype, device=charges.device
        )
        counts = torch.zeros(n_frames, dtype=charges.dtype, device=charges.device)
        sums.scatter_add_(0, batch.unsqueeze(-1).expand_as(charges), charges)
        counts.scatter_add_(0, batch, torch.ones_like(charges[:, 0]))
        means = sums / counts.clamp(min=1).unsqueeze(-1)
        return charges - means[batch]

    def describe(self) -> str:
        parts = ["k=0 mode excluded (unconditional)"]
        if self.hard_in_model:
            parts.append("hard per-frame mean subtraction")
        if self.soft_lambda is not None:
            parts.append(f"soft penalty lambda={self.soft_lambda}")
        return "; ".join(parts)


# ── construction ────────────────────────────────────────────────────────────


def build_sog(
    *,
    mode: str = "owned",
    r_cut: float,
    nlayers: int = 1,
    n_channels: int = 1,
    amp: Optional[torch.Tensor] = None,
    bandwidth: Optional[torch.Tensor] = None,
    bandwidth_is_squared: bool = False,
    amp_is_internal: bool = True,
    b: float = 2.0,
    m: int = 12,
    trainable_kernel: bool = True,
    cubes2_phi_max: Optional[float] = None,
    n_dl: Optional[float] = None,
    remove_self_interaction: bool = True,
    neutrality: Optional[ChargeNeutralityPolicy] = None,
    use_cubes2_fft: bool = False,
    norm_factor: float = E2_PER_ANGSTROM_TO_EV,
    extra_arguments: Optional[Dict[str, Any]] = None,
) -> Sog:
    """Construct a :class:`~sog.sog.Sog` for one of the two integration modes.

    Parameters
    ----------
    mode : {"owned", "external"}
        ``"owned"`` (Mode A) — SOG creates and owns the kernel parameters.
        ``"external"`` (Mode B) — ``amp``/``bandwidth`` must be supplied and are
        bound by reference, so gradients reach the framework's parameters.
    r_cut : float
        Real-space cutoff the charges' descriptor was trained with.  Sets
        ``sigma = r_cut * nlayers / RCUT_TO_SIGMA``.
    n_channels : int
        Number of latent-charge channels the provider emits.
    amp, bandwidth : torch.Tensor
        Required for ``mode="external"``.  ``bandwidth`` is interpreted per
        ``bandwidth_is_squared``.
    neutrality : ChargeNeutralityPolicy, optional
        Records/apply the neutrality decision.  Defaults to no in-model
        correction, matching the historical behaviour.
    extra_arguments : dict, optional
        Passed through to ``Sog``'s ``sog_arguments`` (e.g. ``use_nufft``).

    Returns
    -------
    Sog
        A model whose ``forward`` takes ``(positions, cell, batch,
        latent_charges)`` and returns ``{"E_lr": [nf], ...}``.
    """
    mode = str(mode).strip().lower()
    if mode not in ("owned", "external"):
        raise ValueError("`mode` should be 'owned' or 'external'.")
    if mode == "external" and (amp is None or bandwidth is None):
        raise ValueError(
            "mode='external' requires both `amp` and `bandwidth` (the framework's "
            "own tensors)."
        )

    args: Dict[str, Any] = {
        "use_atomwise": False,
        "n_out": int(n_channels),
        "r_cut": float(r_cut),
        "nlayers": int(nlayers),
        "b": float(b),
        "m": int(m),
        "remove_self_interaction": bool(remove_self_interaction),
        "norm_factor": float(norm_factor),
        "use_cubes2_fft": bool(use_cubes2_fft),
        "trainable_kernel": bool(trainable_kernel),
    }
    if cubes2_phi_max is not None:
        args["cubes2_phi_max"] = float(cubes2_phi_max)
    elif n_dl is not None:
        args["n_dl"] = float(n_dl)
    if neutrality is not None:
        args["charge_neutral"] = bool(neutrality.hard_in_model)
        if neutrality.soft_lambda is not None:
            args["charge_neutral_lambda"] = float(neutrality.soft_lambda)
    if extra_arguments:
        args.update(extra_arguments)

    if mode == "owned":
        return Sog(args)

    # External mode: bind the framework's tensors through a SogKernel.
    kernel = SogKernel(
        r_cut=float(r_cut),
        nlayers=int(nlayers),
        b=float(b),
        remove_self_interaction=bool(remove_self_interaction),
        norm_factor=float(norm_factor),
        cubes2_phi_max=cubes2_phi_max,
        n_dl=n_dl,
        use_cubes2_fft=bool(use_cubes2_fft),
    )
    kernel.bind_external(
        amp,
        bandwidth,
        amp_is_internal=bool(amp_is_internal),
        bandwidth_is_squared=bool(bandwidth_is_squared),
    )
    return Sog(args, kernel=kernel)


def lr_energy(
    sog: Sog,
    positions: torch.Tensor,
    cell: Optional[torch.Tensor],
    latent_charges: torch.Tensor,
    batch: Optional[torch.Tensor] = None,
    *,
    n_channels: Optional[int] = None,
    validate: bool = True,
) -> torch.Tensor:
    """The Mode-A helper: validated charges in, ``E_lr`` ``[nf]`` out.

    Add the result to the framework's total energy and let the framework's own
    autograd take it from there.  Do **not** detach or ``.data`` the charges.
    """
    batch = validate_batch(batch, positions.shape[0]) if validate else batch
    if validate:
        latent_charges = validate_charges(
            latent_charges,
            positions.shape[0],
            n_channels if n_channels is not None else sog.n_out,
        )
    return sog(
        positions=positions,
        cell=cell,
        latent_charges=latent_charges,
        batch=batch,
        compute_energy=True,
        compute_bec=False,
    )["E_lr"]
