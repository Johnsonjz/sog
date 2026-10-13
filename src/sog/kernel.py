"""SOG kernel parameters — the single source of truth for the Gaussian decomposition.

This module owns everything about *which* sum-of-Gaussians approximates the
Coulomb kernel.  It deliberately knows nothing about the electrostatics solver
(that is :class:`sog.module.gaussian.Gaussian`) nor about any MLIP framework.

The kernel is

.. math::

    K(k^2) = \\sum_m \\mathrm{amp}_m \\exp\\left(-\\tfrac{1}{2} bw^2_m k^2\\right)

Two parameterisations are accepted, matching the historical ``sog_arguments``
keys so a framework's config dict maps onto this class without translation:

``kernel_param_mode="raw"``
    ``amp`` is a raw coefficient and ``bandwidth`` is :math:`bw` (not squared).
    The stored internal amplitude becomes ``amp * bw**2``.
``kernel_param_mode="internal"``
    ``amp`` is already the internal amplitude and ``bandwidth`` is already
    :math:`bw^2`.

Two ownership modes, matching ``Gaussian``'s:

``kernel_tensor_mode="owned"``
    The kernel owns ``nn.Parameter`` tensors.  ``bandwidth`` is stored
    **pre-square** (:math:`\\sigma b^m`) so a parameter drifting negative can
    never turn a Gaussian into an exponential.
``kernel_tensor_mode="external"``
    The kernel binds caller-owned tensors; gradients flow to the caller.  The
    bound ``bandwidth`` is already squared.

This is a *leaf* module: it imports only ``math``, ``torch`` and ``torch.nn``.
``sog.module.gaussian`` imports its constants from here so there is exactly one
definition of :data:`RCUT_TO_SIGMA` and one implementation of
:func:`compute_w0`.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn


E2_PER_ANGSTROM_TO_EV = 14.3996454784255
SOG_DEFAULT_B = 2.0
SOG_DEFAULT_SIGMA = 2.180230445405648
SOG_DEFAULT_M = 12
RCUT_TO_SIGMA = 1.9892536839080267  # r_c/σ for b=2, C¹ continuity (midtown-sog.md)

#: Checkpoint format marker.  2 == ``bandwidth`` is stored PRE-SQUARE (σ·b^m).
BW_FORMAT_VERSION = 2


def compute_w0(r0: float, b: float, nterms: int = 500, tol: float = 1e-14) -> float:
    """Real-space correction factor ``w0`` of the fastsog decomposition.

    ``w0 = (1/G(1,r0)) * (1/(2*ln(b)*r0) - Σ_i b^{-i} * G(1, b^{-i}*r0))``
    where ``G(sigma, r) = exp(-r²/(2*sigma²))/sqrt(2*pi*sigma²)``.

    Adaptive convergence: stops when ``|term| < tol * |sum|``.
    """
    def G(s: float, r: float) -> float:
        return math.exp(-r * r / (2.0 * s * s)) / math.sqrt(2.0 * math.pi * s * s)

    s = 0.0
    abs_sum = 0.0
    for i in range(1, nterms + 1):
        bi = b ** (-i)
        term = bi * G(1.0, bi * r0)
        s += term
        abs_sum += abs(term)
        if i >= 10 and abs(term) < tol * max(abs_sum, 1e-30):
            break
    return (1.0 / G(1.0, r0)) * (1.0 / (2.0 * math.log(b) * r0) - s)


def _as_1d_tensor(value: object, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.reshape(-1)
    return torch.as_tensor(value, dtype=dtype).reshape(-1)


def _normalize_mode(value: str, valid: set, name: str) -> str:
    mode = str(value).strip().lower()
    if mode not in valid:
        raise ValueError(f"`{name}` should be one of {sorted(valid)}, got {value!r}.")
    return mode


def _is_differentiable_tensor(value: object) -> bool:
    return isinstance(value, torch.Tensor) and bool(value.requires_grad)


class SogKernel(nn.Module):
    """Kernel hyperparameters plus the Gaussian amplitudes/bandwidths.

    Responsibilities (the parts every MLIP framework would otherwise
    re-implement):

    * generate ``amp``/``bw²`` from ``r_cut`` (:meth:`init_from_rcut`);
    * own or bind the tensors (:meth:`bind_external`);
    * expose them in the exact form a solver consumes (:meth:`resolve`);
    * export them in the LAMMPS ``kspace_style sog`` convention
      (:meth:`to_lammps_dict` / :meth:`to_lammps_tokens`);
    * round-trip them across the pre-square/squared format change
      (:meth:`to_state_dict` / :meth:`from_state_dict`).

    Example — owned kernel, generated from a cutoff::

        k = SogKernel(r_cut=6.0, b=2.0, m=12)
        k.init_from_rcut(6.0)
        amp_int, bw2 = k.resolve()

    Example — a framework that already owns the parameters::

        k = SogKernel(kernel_tensor_mode="external")
        k.bind_external(fitting.amp, fitting.get_bandwidth_sq(),
                        amp_is_internal=True, bandwidth_is_squared=True)
    """

    def __init__(
        self,
        *,
        b: float = SOG_DEFAULT_B,
        sigma: float = SOG_DEFAULT_SIGMA,
        m: int = SOG_DEFAULT_M,
        nlayers: int = 1,
        r_cut: Optional[float] = None,
        n_dl: Optional[float] = None,
        cubes2_phi_max: Optional[float] = None,
        cubes2_order: int = 4,
        quads_order: int = 6,
        remove_self_interaction: bool = True,
        norm_factor: float = E2_PER_ANGSTROM_TO_EV,
        amp: Union[torch.Tensor, float, Sequence[float], None] = None,
        bandwidth: Union[torch.Tensor, Sequence[float], None] = None,
        kernel_param_mode: str = "raw",
        kernel_tensor_mode: str = "owned",
        trainable: bool = True,
        use_cubes2_fft: bool = False,
        use_quads_fft: bool = False,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()

        param_mode = _normalize_mode(
            kernel_param_mode, {"raw", "internal"}, "kernel_param_mode"
        )
        tensor_mode = _normalize_mode(
            kernel_tensor_mode, {"owned", "external", "auto"}, "kernel_tensor_mode"
        )
        if tensor_mode == "auto":
            tensor_mode = (
                "external"
                if (not trainable)
                and (
                    _is_differentiable_tensor(amp)
                    or _is_differentiable_tensor(bandwidth)
                )
                else "owned"
            )

        # ── hyperparameters ──
        self.b = float(b)
        self.M = max(1, int(m))
        self.nlayers = int(nlayers)
        self.n_dl = float(n_dl) if n_dl is not None else None
        self.cubes2_phi_max = (
            float(cubes2_phi_max) if cubes2_phi_max is not None else None
        )
        self.cubes2_order = int(cubes2_order)
        self.quads_order = int(quads_order)
        self.remove_self_interaction = bool(remove_self_interaction)
        self.norm_factor = float(norm_factor)
        self.trainable = bool(trainable)
        self.use_cubes2_fft = bool(use_cubes2_fft)
        self.use_quads_fft = bool(use_quads_fft)

        if self.n_dl is not None and (not math.isfinite(self.n_dl) or self.n_dl <= 0.0):
            raise ValueError("`n_dl` should be a positive finite number.")
        if self.cubes2_phi_max is not None and (
            not math.isfinite(self.cubes2_phi_max) or self.cubes2_phi_max <= 0.0
        ):
            raise ValueError("`cubes2_phi_max` should be a positive finite number.")

        # ``r_cut`` drives σ exactly as ``Gaussian`` does; an explicit ``sigma``
        # only applies when no cutoff is given.
        self.r_cut = float(r_cut) if r_cut is not None else None
        if self.r_cut is not None:
            if self.r_cut <= 0.0:
                raise ValueError("`r_cut` should be positive.")
            self.sigma = self.r_cut * self.nlayers / RCUT_TO_SIGMA
        else:
            self.sigma = float(sigma)
            if self.sigma <= 0.0:
                raise ValueError("`sigma` should be positive.")

        self._param_mode = param_mode
        self._tensor_mode = tensor_mode
        self._adopted = False
        self._version = 0

        amp_tensor, bw2 = self._generate(amp, bandwidth, dtype)

        if tensor_mode == "external":
            self.amp = amp_tensor
            self.bandwidth = bw2
            self._bandwidth_is_squared = True
        else:
            # Store the pre-square bandwidth (σ·b^m) so bw² can never go negative.
            self.amp = nn.Parameter(amp_tensor, requires_grad=self.trainable)
            self.bandwidth = nn.Parameter(bw2.sqrt(), requires_grad=self.trainable)
            self._bandwidth_is_squared = False
        # ``_generate`` always normalises the amplitude to the internal form.
        self._stored_amp_is_internal = True

        # Checkpoint format marker: v2 == bandwidth stored PRE-SQUARE.  Mirrors
        # Gaussian._bw_format_version, but namespaced under this module.
        self.register_buffer("_bw_format_version", torch.tensor(BW_FORMAT_VERSION))

    # ── construction helpers ────────────────────────────────────────────────

    def _generate(
        self,
        amp: Union[torch.Tensor, float, Sequence[float], None],
        bandwidth: Union[torch.Tensor, Sequence[float], None],
        dtype: Optional[torch.dtype],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(amp_internal, bw2)`` from the constructor's raw inputs.

        ``dtype=None`` falls back to :func:`torch.get_default_dtype`, matching
        ``Gaussian``.
        """
        dt = dtype if dtype is not None else torch.get_default_dtype()
        m_value = self.M

        if bandwidth is None:
            if self.r_cut is not None:
                # fastsog.cpp convention: recurrence bw2[m] = bw2[m-1]·b²
                sigma2 = self.sigma * self.sigma
                b2 = self.b * self.b
                bw2 = torch.empty(m_value, dtype=dt)
                bw2[0] = sigma2
                for mm in range(1, m_value):
                    bw2[mm] = bw2[mm - 1] * b2
            else:
                # Legacy branch: bw = σ·b^m, squared.  Mathematically the same
                # series as the recurrence above, but not bit-identical -- it
                # rounds differently, so the exact expression is kept per branch.
                bw = self.sigma * torch.pow(
                    torch.tensor(self.b, dtype=dt),
                    torch.arange(m_value, dtype=dt),
                )
                bw2 = bw.square()
        else:
            bw_in = _as_1d_tensor(bandwidth, dt)
            if bw_in.numel() == 0:
                raise ValueError("`bandwidth` should not be empty.")
            if not torch.isfinite(bw_in).all():
                raise ValueError("`bandwidth` should be finite.")
            if torch.any(bw_in <= 0.0):
                raise ValueError("`bandwidth` values should be positive.")
            bw2 = bw_in if self._param_mode == "internal" else bw_in.square()

        if amp is None:
            if self.b <= 0.0:
                raise ValueError("`b` should be positive when `amp` is not provided.")
            if self.r_cut is not None:
                # fastsog convention (fastsog.cpp:530-541):
                #   amp[0] = 4π·ln(b)·w0·σ²
                #   amp[1] = 4π·ln(b)·σ²·b²
                #   amp[m] = amp[m-1]·b²   (m >= 2)
                sigma2 = self.sigma * self.sigma
                b2 = self.b * self.b
                logb = math.log(self.b)
                w0 = compute_w0(self.r_cut / self.sigma, self.b)
                amp_factor = 4.0 * math.pi * logb
                amp_tensor = torch.empty(m_value, dtype=dt)
                amp_tensor[0] = amp_factor * w0 * sigma2
                if m_value > 1:
                    amp_tensor[1] = amp_factor * sigma2 * b2
                for mm in range(2, m_value):
                    amp_tensor[mm] = amp_tensor[mm - 1] * b2
                amp_tensor = amp_tensor.reshape(-1)
                # amp is already the internal amplitude in this branch.
                return amp_tensor, bw2
            amp_tensor = torch.full_like(bw2, fill_value=4.0 * math.pi * math.log(self.b))
        else:
            amp_tensor = _as_1d_tensor(amp, dt)
            if amp_tensor.numel() == 0:
                raise ValueError("`amp` should not be empty.")
            if not torch.isfinite(amp_tensor).all():
                raise ValueError("`amp` should be finite.")
            if amp_tensor.numel() == 1 and bw2.numel() > 1:
                amp_tensor = amp_tensor.expand_as(bw2).clone()
            elif amp_tensor.numel() != bw2.numel():
                raise ValueError(
                    "`amp` should be scalar or have the same length as `bandwidth`."
                )

        # Convert a raw amplitude into the internal one.  Mirrors Gaussian: a
        # *generated* amplitude is converted unconditionally (it is a raw
        # coefficient by construction), an explicit one follows the declared mode.
        # The fastsog branch above returns early -- its amplitude is already
        # internal, which is why it must not fall through to here.
        if (self._param_mode == "raw") or (amp is None):
            amp_tensor = amp_tensor * bw2

        return amp_tensor, bw2

    # ── accessors ───────────────────────────────────────────────────────────

    def bandwidth_sq(self) -> torch.Tensor:
        """``bw²``, guaranteed non-negative and graph-connected to the parameter."""
        if self._bandwidth_is_squared:
            return self.bandwidth
        return self.bandwidth.square()

    def amp_internal(self) -> torch.Tensor:
        """The amplitude exactly as ``K(k²) = Σ amp·exp(-½ bw² k²)`` consumes it.

        The constructor always normalises to this form, so the conversion only
        happens for tensors bound later via ``bind_external(amp_is_internal=False)``.
        """
        if self._stored_amp_is_internal:
            return self.amp
        return self.amp * self.bandwidth_sq()

    def resolve(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(amp_internal, bw²)`` — the pair to hand to ``Gaussian`` in
        ``kernel_param_mode="internal"``, ``kernel_tensor_mode="external"``."""
        return self.amp_internal(), self.bandwidth_sq()

    @property
    def n_terms(self) -> int:
        """Number of Gaussian terms actually stored."""
        return int(self.bandwidth.reshape(-1).numel())

    def __repr__(self) -> str:
        return (
            f"SogKernel(b={self.b}, sigma={self.sigma:.6g}, M={self.M}, "
            f"r_cut={self.r_cut}, param_mode={self._param_mode!r}, "
            f"tensor_mode={self._tensor_mode!r}, "
            f"trainable={self.trainable})"
        )

    # ── regeneration ────────────────────────────────────────────────────────

    def init_from_rcut(
        self,
        r_cut: float,
        nlayers: Optional[int] = None,
        *,
        b: Optional[float] = None,
        m: Optional[int] = None,
    ) -> "SogKernel":
        """Regenerate σ, ``bw²`` and ``amp`` from a real-space cutoff.

        Reproduces exactly what ``Gaussian`` does when constructed with ``rcut``::

            sigma  = r_cut * nlayers / RCUT_TO_SIGMA
            bw2[m] = sigma**2 * b**(2m)
            amp[0] = 4*pi*ln(b) * w0 * sigma**2
            amp[1] = 4*pi*ln(b) * sigma**2 * b**2
            amp[m] = amp[m-1] * b**2                (m >= 2)

        Values are written with ``copy_`` so optimizer state survives a mid-training
        re-initialisation.

        Raises
        ------
        RuntimeError
            If the kernel is in external mode.  Silently overwriting a tensor the
            caller owns (a framework's trainable parameter) is never correct;
            update that module's parameters instead.
        """
        if self._tensor_mode == "external" or self._adopted:
            raise RuntimeError(
                "init_from_rcut() cannot write into externally bound kernel tensors. "
                "Initialise the kernel before bind_external(), or update the owning "
                "module's parameters directly."
            )
        rcut = float(r_cut)
        if rcut <= 0.0:
            raise ValueError("`r_cut` should be positive.")
        if nlayers is not None:
            self.nlayers = int(nlayers)
        if b is not None:
            self.b = float(b)
            if self.b <= 0.0:
                raise ValueError("`b` should be positive.")
        if m is not None:
            self.M = max(1, int(m))
        if self.b <= 0.0:
            raise ValueError("`b` should be positive.")

        self.r_cut = rcut
        self.sigma = rcut * self.nlayers / RCUT_TO_SIGMA

        sigma2 = self.sigma * self.sigma
        b2 = self.b * self.b
        logb = math.log(self.b)
        m_value = self.M
        w0 = compute_w0(rcut / self.sigma, self.b)
        amp_factor = 4.0 * math.pi * logb

        with torch.no_grad():
            bw2 = torch.empty(
                m_value, dtype=self.bandwidth.dtype, device=self.bandwidth.device
            )
            bw2[0] = sigma2
            for mm in range(1, m_value):
                bw2[mm] = bw2[mm - 1] * b2

            amp = torch.empty_like(bw2)
            amp[0] = amp_factor * w0 * sigma2
            if m_value > 1:
                amp[1] = amp_factor * sigma2 * b2
            for mm in range(2, m_value):
                amp[mm] = amp[mm - 1] * b2

            if self.bandwidth.numel() != m_value or self.amp.numel() != m_value:
                raise ValueError(
                    f"cannot re-initialise to M={m_value}: the kernel currently "
                    f"stores {self.bandwidth.numel()} terms.  Construct a new "
                    "SogKernel instead of changing M in place."
                )
            self.bandwidth.copy_(bw2.sqrt())
            self.amp.copy_(amp)
        self._param_mode = "internal"
        self._stored_amp_is_internal = True
        self._bandwidth_is_squared = False
        self._version += 1
        return self

    # ── external binding ────────────────────────────────────────────────────

    def bind_external(
        self,
        amp: torch.Tensor,
        bandwidth: torch.Tensor,
        *,
        amp_is_internal: bool = True,
        bandwidth_is_squared: bool = False,
    ) -> "SogKernel":
        """Bind caller-owned tensors; gradients flow to the caller's parameters.

        The defaults describe a framework fitting net that stores the internal
        amplitude and the **pre-square** bandwidth (DeepMD's ``SOGEnergyFittingNet``
        layout), so ``bandwidth_sq()`` squares it and the never-negative guarantee
        survives.  Pass ``bandwidth_is_squared=True`` for a framework that already
        holds ``bw²``.
        """
        if not isinstance(amp, torch.Tensor) or not isinstance(bandwidth, torch.Tensor):
            raise TypeError("bind_external() requires torch.Tensor arguments.")
        if amp.numel() not in (1, bandwidth.numel()):
            raise ValueError(
                "`amp` should be scalar or have the same length as `bandwidth`."
            )
        # Drop any parameters this kernel created in owned mode; they would
        # otherwise linger in ``_parameters`` (invisible to ``.amp`` but still in
        # ``state_dict`` and in an optimizer's param groups) after the switch.
        self._parameters.pop("amp", None)
        self._parameters.pop("bandwidth", None)
        # Bypass nn.Module.__setattr__: these are the caller's tensors, not ours.
        object.__setattr__(self, "amp", amp)
        object.__setattr__(self, "bandwidth", bandwidth)
        self._stored_amp_is_internal = bool(amp_is_internal)
        self._param_mode = "internal" if amp_is_internal else "raw"
        self._bandwidth_is_squared = bool(bandwidth_is_squared)
        self._tensor_mode = "external"
        self._adopted = False
        self._version += 1
        return self

    @classmethod
    def adopt(cls, gaussian: nn.Module) -> "SogKernel":
        """Read-only view of the kernel parameters already held by a ``Gaussian``.

        Produces a :class:`SogKernel` whose ``amp``/``bandwidth`` are the *same*
        tensor objects as ``gaussian``'s, so no parameters are duplicated and an
        adopted kernel contributes nothing to ``state_dict``.  Changing ``M`` or
        calling :meth:`init_from_rcut` on an adopted kernel is refused.
        """
        obj = cls.__new__(cls)
        nn.Module.__init__(obj)

        obj.b = float(getattr(gaussian, "b", SOG_DEFAULT_B))
        obj.sigma = float(getattr(gaussian, "sigma", SOG_DEFAULT_SIGMA))
        obj.M = int(gaussian._bw2().reshape(-1).numel())
        obj.nlayers = int(getattr(gaussian, "nlayers", 1))
        obj.r_cut = getattr(gaussian, "rcut", None)
        obj.n_dl = getattr(gaussian, "n_dl", None)
        obj.cubes2_phi_max = getattr(gaussian, "cubes2_phi_max", None)
        obj.cubes2_order = int(getattr(gaussian, "cubes2_order", 4))
        obj.quads_order = int(getattr(gaussian, "quads_order", 6))
        obj.remove_self_interaction = bool(
            getattr(gaussian, "remove_self_interaction", True)
        )
        obj.norm_factor = float(getattr(gaussian, "norm_factor", E2_PER_ANGSTROM_TO_EV))
        obj.trainable = True
        obj.use_cubes2_fft = bool(getattr(gaussian, "use_cubes2_fft", False))
        obj.use_quads_fft = bool(getattr(gaussian, "use_quads_fft", False))

        obj._adopted = True
        obj._version = 0
        obj._tensor_mode = (
            "external"
            if getattr(gaussian, "_bandwidth_is_squared", False)
            else "owned"
        )
        # amp is already internal in the Gaussian; bandwidth keeps the source's
        # squared/unsquared convention.
        obj._bandwidth_is_squared = bool(
            getattr(gaussian, "_bandwidth_is_squared", False)
        )
        obj._param_mode = "internal"
        obj._stored_amp_is_internal = True
        object.__setattr__(obj, "amp", gaussian.amp)
        object.__setattr__(obj, "bandwidth", gaussian.bandwidth)
        return obj

    # ── solver interop ──────────────────────────────────────────────────────

    def sync_into(self, gaussian: nn.Module) -> None:
        """Re-bind ``gaussian``'s kernel tensors if this kernel's changed.

        ``Gaussian`` holds external tensors as plain attributes, which
        ``nn.Module._apply`` (``.to()``/``.cuda()``) does not move.  After the
        kernel's parameters move, the binding must be refreshed.  The steady-state
        cost is one integer comparison.
        """
        if getattr(gaussian, "_sog_kernel_version", -1) == self._version:
            return
        amp_i, bw2 = self.resolve()
        object.__setattr__(gaussian, "amp", amp_i)
        object.__setattr__(gaussian, "bandwidth", bw2)
        gaussian._bandwidth_is_squared = True
        gaussian._sog_kernel_version = self._version

    def _apply(self, fn, recurse: bool = True):
        out = super()._apply(fn, recurse)
        # The tensors may have been replaced (device/dtype change).
        self._version += 1
        return out

    # ── LAMMPS export ───────────────────────────────────────────────────────

    def spline_name(self) -> str:
        """The ``spline`` keyword value matching this kernel's solver choice."""
        if self.use_quads_fft:
            return "quads_6" if self.quads_order == 6 else "quads_4"
        return "cubes2_6" if self.cubes2_order == 6 else "cubes2_4"

    def to_lammps_dict(self, *, include_grid: bool = True) -> Dict[str, Any]:
        """Kernel parameters in the LAMMPS ``kspace_style sog`` convention.

        The C++ plugin consumes ``bandwidth`` as the **squared** bandwidth:
        ``K(k²) = Σ amp[m]·exp(-0.5·bandwidth[m]·k²)``.  This exporter always emits
        the squared values, whatever the internal storage format, so generated
        input files stay correct across format changes.  Prefer this over reading
        the raw ``.bandwidth`` attribute.

        Two naming traps this function handles for you:

        * The plugin's grid keyword is ``phi_max`` (``sog.cpp:540``).  The name
          ``cubes2_phi_max`` is emitted only as a same-value deprecated alias,
          because passing it to the plugin raises ``Unknown sog option``.
        * The term-count keyword is lowercase ``m`` (``sog.cpp:604``).  ``M`` is
          emitted alongside for consumers that expect the DeepMD spelling.
        """
        amp = self.amp_internal().detach().reshape(-1).double().tolist()
        bw2 = self.bandwidth_sq().detach().reshape(-1).double().tolist()
        out: Dict[str, Any] = {
            "b": float(self.b),
            "sigma": float(self.sigma),
            "m": int(self.M),  # plugin keyword (lowercase)
            "M": int(self.M),  # DeepMD-compatible alias
            "amp": amp,
            "bandwidth": bw2,  # SQUARED
            "remove_self_interaction": bool(self.remove_self_interaction),
            "spline": self.spline_name(),
        }
        if include_grid:
            phi = float(self.cubes2_phi_max) if self.cubes2_phi_max is not None else None
            out["phi_max"] = phi  # plugin keyword
            out["cubes2_phi_max"] = phi  # deprecated alias; the plugin rejects this name
            out["n_dl"] = float(self.n_dl) if self.n_dl is not None else None
        return out

    def to_lammps_tokens(
        self,
        *,
        extras: Optional[Mapping[str, Any]] = None,
        include_grid: bool = True,
    ) -> list:
        """``kspace_style sog`` keyword tokens for :meth:`to_lammps_dict`'s values.

        ``extras`` covers keywords this class does not model —
        ``mesh_oversample``, ``mesh_alias_extent``, ``use_gpu``, ``direct``,
        ``phi_accuracy``, ``bspline_order``.
        """
        d = self.to_lammps_dict(include_grid=include_grid)
        toks: list = ["amp"]
        toks += [repr(float(v)) for v in d["amp"]]
        toks += ["bandwidth"]
        toks += [repr(float(v)) for v in d["bandwidth"]]
        toks += ["m", str(int(d["m"]))]
        toks += ["b", repr(float(d["b"]))]
        toks += ["sigma", repr(float(d["sigma"]))]
        toks += ["spline", str(d["spline"])]
        toks += [
            "remove_self_interaction",
            "yes" if d["remove_self_interaction"] else "no",
        ]
        if include_grid and d.get("phi_max") is not None:
            toks += ["phi_max", repr(float(d["phi_max"]))]
        elif include_grid and d.get("n_dl") is not None:
            toks += ["n_dl", repr(float(d["n_dl"]))]
        for key, value in (extras or {}).items():
            if value is None or value is False:
                continue
            if value is True:
                toks += [str(key), "yes"]
            else:
                toks += [str(key), str(value)]
        return toks

    def to_lammps_input(
        self,
        *,
        accuracy: float = 1e-6,
        pair_style: Optional[str] = None,
        extras: Optional[Mapping[str, Any]] = None,
    ) -> str:
        """A ready-to-paste LAMMPS command block for this kernel.

        Emits the *pure keyword form* — never a numeric positional header.  The
        plugin treats a numeric ``arg[0]`` as the legacy 4-argument header
        ``accuracy b sigma M [n_dl]``, so a lone number before ``amp`` would be
        mis-parsed as ``b``.  With explicit ``amp``/``bandwidth`` the header
        scalars are vestigial anyway (``grid_method`` 0 + ``phi_max`` drive the
        mesh; ``accuracy`` only feeds auto-``n_dl``/legacy-PPPM refinement).
        The ``accuracy`` parameter is therefore accepted for API compatibility
        but not emitted.
        """
        del accuracy  # inexpressible in keyword form; see docstring
        lines = []
        if pair_style:
            lines.append(f"pair_style {pair_style}")
        lines.append(
            "kspace_style sog " + " ".join(self.to_lammps_tokens(extras=extras))
        )
        return "\n".join(lines)

    # ── (de)serialisation ───────────────────────────────────────────────────

    def to_state_dict(self) -> Dict[str, Any]:
        """Hyperparameters + kernel tensors + the format marker.

        Unlike a bare ``nn.Module.state_dict()`` this also carries the
        hyperparameters, which are plain attributes and would otherwise have to be
        re-supplied by hand.
        """
        return {
            "format": "sog-kernel",
            "version": BW_FORMAT_VERSION,
            "b": self.b,
            "sigma": self.sigma,
            "m": self.M,  # constructor keyword name
            "M": self.M,  # alias, matches to_lammps_dict()'s spelling
            "nlayers": self.nlayers,
            "r_cut": self.r_cut,
            "n_dl": self.n_dl,
            "cubes2_phi_max": self.cubes2_phi_max,
            "cubes2_order": self.cubes2_order,
            "quads_order": self.quads_order,
            "remove_self_interaction": self.remove_self_interaction,
            "norm_factor": self.norm_factor,
            "use_cubes2_fft": self.use_cubes2_fft,
            "use_quads_fft": self.use_quads_fft,
            "amp_is_internal": True,
            "bandwidth_is_squared": bool(self._bandwidth_is_squared),
            "amp": self.amp_internal().detach().clone(),
            "bandwidth": self.bandwidth.detach().clone(),
        }

    @classmethod
    def from_state_dict(cls, data: Mapping[str, Any]) -> "SogKernel":
        """Inverse of :meth:`to_state_dict`, with v1 up-conversion.

        A payload without the ``bandwidth_is_squared`` label is the v1 convention
        and stores a **squared** bandwidth; it is sqrt-recovered into the
        pre-square parameter, exactly as ``Gaussian._load_from_state_dict`` does
        for legacy checkpoints.

        The parameters are restored with ``copy_`` rather than by re-deriving them,
        so a round trip is bit-exact — owned mode's ``sqrt``-then-``square``
        storage would otherwise reintroduce rounding on every save/load.
        """
        data = dict(data)
        version = int(data.pop("version", 1))
        data.pop("format", None)

        amp = data.pop("amp")
        if not isinstance(amp, torch.Tensor):
            amp = torch.as_tensor(amp)
        bw_stored = data.pop("bandwidth")
        if not isinstance(bw_stored, torch.Tensor):
            bw_stored = torch.as_tensor(bw_stored)

        bw_is_squared = data.pop("bandwidth_is_squared", None)
        if bw_is_squared is None:
            # v1 payload: unlabelled, squared by convention.
            bw_is_squared = True
        if version < BW_FORMAT_VERSION and bw_is_squared:
            bw_pre = bw_stored.clamp(min=0).sqrt()
        elif bw_is_squared:
            bw_pre = bw_stored.clamp(min=0).sqrt()
        else:
            bw_pre = bw_stored

        data.pop("amp_is_internal", None)
        hyper = {k: v for k, v in data.items() if k in cls._HYPERPARAM_KEYS}
        # Build with a same-shaped placeholder, then write the stored tensors
        # verbatim: M comes from the payload so the shapes already agree.
        n_terms = int(bw_pre.numel())
        obj = cls(
            amp=torch.zeros(n_terms, dtype=amp.dtype),
            bandwidth=torch.ones(n_terms, dtype=bw_pre.dtype),
            kernel_param_mode="internal",
            kernel_tensor_mode="owned",
            **{**hyper, "m": int(hyper.get("m", hyper.get("M", n_terms)))},
        )
        with torch.no_grad():
            obj.amp.copy_(amp)
            obj.bandwidth.copy_(bw_pre)
        obj._stored_amp_is_internal = True
        obj._bandwidth_is_squared = False
        return obj

    _HYPERPARAM_KEYS = frozenset(
        {
            "b",
            "sigma",
            "m",
            "nlayers",
            "r_cut",
            "n_dl",
            "cubes2_phi_max",
            "cubes2_order",
            "quads_order",
            "remove_self_interaction",
            "norm_factor",
            "use_cubes2_fft",
            "use_quads_fft",
        }
    )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # Same cross-version contract as Gaussian: a ``bandwidth`` without the
        # marker is a legacy squared value and must be sqrt-recovered, otherwise
        # it would be squared again on use and silently become bw⁴.
        bw_key = prefix + "bandwidth"
        if bw_key in state_dict and prefix + "_bw_format_version" not in state_dict:
            state_dict = state_dict.copy()
            bw = state_dict[bw_key]
            if bw is not None:
                state_dict[bw_key] = bw.clamp(min=0).sqrt()
            state_dict[prefix + "_bw_format_version"] = torch.tensor(BW_FORMAT_VERSION)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
