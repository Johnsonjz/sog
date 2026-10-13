from __future__ import annotations

from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn

from .kernel import RCUT_TO_SIGMA, SogKernel
from .module import Atomwise, BEC, Gaussian


class Sog(nn.Module):
    """
    Important ``sog_arguments`` keys
    -------------------------------
    kernel_param_mode : {"raw", "internal"}
        Controls interpretation of ``amp``/``bandwidth``.
    kernel_tensor_mode : {"owned", "external", "auto"}
        Controls whether Gaussian owns kernel parameters or binds caller tensors.

    Legacy bool keys are still supported for compatibility:
    ``amp_is_internal``, ``bandwidth_is_squared``,
    ``use_external_kernel_tensors``.
    """

    def __init__(
        self,
        sog_arguments: Union[Dict[str, Any], str, None] = None,
        r_cut: Optional[float] = None,
        kernel: Optional[SogKernel] = None,
    ):
        super().__init__()
        self.kernel: Optional[SogKernel] = None

        if sog_arguments is None:
            sog_arguments = {}
        if isinstance(sog_arguments, str):
            import yaml

            with open(sog_arguments, "r", encoding="utf-8") as f:
                parsed = yaml.safe_load(f)
            sog_arguments = {} if parsed is None else parsed

        if r_cut is not None:
            sog_arguments = dict(sog_arguments)
            sog_arguments["r_cut"] = r_cut

        self._parse_arguments(sog_arguments)

        self.atomwise: nn.Module = (
            Atomwise(
                n_in=self.n_in,
                n_out=self.n_out,
                n_layers=self.n_layers,
                n_hidden=self.n_hidden,
                add_linear_nn=self.add_linear_nn,
                output_scaling_factor=self.output_scaling_factor,
            )
            if self.use_atomwise
            else _DummyAtomwise()
        )

        # When a SogKernel is supplied it becomes the source of the kernel
        # tensors and the solver binds them in external mode, so the parameters
        # live in exactly one place and gradients reach them directly.
        if kernel is not None:
            self.kernel = kernel
            amp_i, bw2 = kernel.resolve()
            gaussian_kwargs: Dict[str, Any] = dict(
                amp=amp_i,
                bandwidth=bw2,
                kernel_param_mode="internal",
                kernel_tensor_mode="external",
                trainable=False,
            )
        else:
            gaussian_kwargs = dict(
                amp=self.amp,
                bandwidth=self.bandwidth,
                kernel_param_mode=self.kernel_param_mode,
                kernel_tensor_mode=self.kernel_tensor_mode,
                trainable=self.trainable_kernel,
            )

        self.gaussian = Gaussian(
            n_dl=self.n_dl,
            cubes2_phi_max=self.cubes2_phi_max,
            cubes2_order=self.cubes2_order,
            b=self.b,
            sigma=self.sigma,
            m=self.m,
            rcut=self.r_cut,
            nlayers=self.nlayers,
            remove_self_interaction=self.remove_self_interaction,
            charge_neutral_lambda=self.charge_neutral_lambda,
            use_nufft=self.nufft,
            nufft_eps=self.nufft_eps,
            norm_factor=self.norm_factor,
            use_cubes2_fft=self.use_cubes2_fft,
            use_quads_fft=self.use_quads_fft,
            quads_order=self.quads_order,
            **gaussian_kwargs,
        )
        if self.kernel is not None:
            self.kernel.sync_into(self.gaussian)

        self.bec = BEC(
            remove_mean=self.remove_mean,
            epsilon_factor=self.epsilon_factor,
        )

    def _parse_arguments(self, args: Dict[str, Any]) -> None:
        self.use_atomwise = bool(args.get("use_atomwise", True))
        self.n_in = args.get("n_in", None)
        self.n_out = int(args.get("n_out", 1))
        self.n_layers = int(args.get("n_layers", 3))
        self.n_hidden = args.get("n_hidden", [32, 16])
        self.add_linear_nn = bool(args.get("add_linear_nn", True))
        self.output_scaling_factor = float(args.get("output_scaling_factor", 0.1))

        self.n_dl = float(args["n_dl"]) if args.get("n_dl") is not None else None
        self.cubes2_phi_max = args.get("cubes2_phi_max", None)
        if self.cubes2_phi_max is not None:
            self.cubes2_phi_max = float(self.cubes2_phi_max)
            # Conflict detection: user specified both explicitly
            if "n_dl" in args:
                raise ValueError(
                    "Cannot specify both `cubes2_phi_max` and `n_dl`. "
                    "Use `cubes2_phi_max` (φ = Δ/r_c, recommended) or remove "
                    "`n_dl` to auto-default from Predescu 2020 Table III."
                )
        self.cubes2_order = int(args.get("cubes2_order", 4))
        self.quads_order = int(args.get("quads_order", 6))
        self.amp = args.get("amp", None)
        self.bandwidth = args.get("bandwidth", None)

        # Preferred concise API.
        self.kernel_param_mode = str(args.get("kernel_param_mode", "raw")).strip().lower()
        self.kernel_tensor_mode = str(args.get("kernel_tensor_mode", "owned")).strip().lower()

        # Backward compatibility for older bool-style config keys.
        legacy_amp_is_internal = args.get("amp_is_internal", None)
        legacy_bw_is_squared = args.get("bandwidth_is_squared", None)
        if (legacy_amp_is_internal is not None) or (legacy_bw_is_squared is not None):
            amp_flag = bool(
                False if legacy_amp_is_internal is None else legacy_amp_is_internal
            )
            bw_flag = bool(
                False if legacy_bw_is_squared is None else legacy_bw_is_squared
            )
            if amp_flag != bw_flag:
                raise ValueError(
                    "Legacy keys `amp_is_internal` and `bandwidth_is_squared` "
                    "should be consistent (both true or both false)."
                )
            legacy_mode = "internal" if amp_flag else "raw"
            if self.kernel_param_mode not in {"raw", "internal"}:
                raise ValueError(
                    "`kernel_param_mode` should be 'raw' or 'internal'."
                )
            if args.get("kernel_param_mode", None) is not None and self.kernel_param_mode != legacy_mode:
                raise ValueError(
                    "Conflicting values between `kernel_param_mode` and legacy bool keys."
                )
            self.kernel_param_mode = legacy_mode

        legacy_external = args.get("use_external_kernel_tensors", None)
        if legacy_external is not None:
            legacy_tensor_mode = "external" if bool(legacy_external) else "owned"
            if self.kernel_tensor_mode not in {"owned", "external", "auto"}:
                raise ValueError(
                    "`kernel_tensor_mode` should be 'owned', 'external', or 'auto'."
                )
            if args.get("kernel_tensor_mode", None) is not None and self.kernel_tensor_mode != legacy_tensor_mode:
                raise ValueError(
                    "Conflicting values between `kernel_tensor_mode` and legacy bool key `use_external_kernel_tensors`."
                )
            self.kernel_tensor_mode = legacy_tensor_mode

        if self.kernel_param_mode not in {"raw", "internal"}:
            raise ValueError("`kernel_param_mode` should be 'raw' or 'internal'.")
        if self.kernel_tensor_mode not in {"owned", "external", "auto"}:
            raise ValueError(
                "`kernel_tensor_mode` should be 'owned', 'external', or 'auto'."
            )

        self.b = float(args.get("b", 2.0))
        r_cut_arg = args.get("r_cut", args.get("rcut", None))
        self.r_cut = float(r_cut_arg) if r_cut_arg is not None else None
        self.use_cubes2_fft = bool(args.get("use_cubes2_fft", False))
        self.use_quads_fft = bool(args.get("use_quads_fft", False))
        self.nlayers = int(args.get("nlayers", 1))
        if self.r_cut is not None:
            if self.r_cut <= 0.0:
                raise ValueError("`r_cut` should be positive.")
            self.sigma = self.r_cut * self.nlayers / RCUT_TO_SIGMA
        else:
            self.sigma = float(args.get("sigma", 2.180230445405648))
        self.m = int(args.get("m", 12))
        self.remove_self_interaction = bool(args.get("remove_self_interaction", True))
        self.charge_neutral_lambda = args.get("charge_neutral_lambda", None)
        self.charge_neutral = bool(args.get("charge_neutral", False))
        self.nufft = bool(args.get("nufft", args.get("use_nufft", False)))
        self.use_nufft = self.nufft
        self.nufft_eps = float(args.get("nufft_eps", 1e-4))
        self.norm_factor = float(args.get("norm_factor", 14.3996454784255))
        self.trainable_kernel = bool(args.get("trainable_kernel", True))

        self.remove_mean = bool(args.get("remove_mean", True))
        self.epsilon_factor = float(args.get("epsilon_factor", 1.0))

    def as_kernel(self) -> SogKernel:
        """A :class:`SogKernel` view of this model's kernel parameters.

        When the model was built from a ``SogKernel`` the original object is
        returned.  Otherwise a read-only view of ``self.gaussian``'s parameters is
        produced — same tensor objects, no duplicated parameters.  Useful for
        exporting to LAMMPS:
        :meth:`~sog.kernel.SogKernel.to_lammps_dict`.
        """
        if self.kernel is not None:
            return self.kernel
        return SogKernel.adopt(self.gaussian)

    def forward(
        self,
        positions: torch.Tensor,
        cell: Optional[torch.Tensor],
        desc: Optional[torch.Tensor] = None,
        latent_charges: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        compute_energy: bool = True,
        compute_force: bool = False,
        compute_virial: bool = False,
        compute_bec: bool = False,
        bec_output_index: Optional[int] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        if batch is None:
            batch = torch.zeros(positions.shape[0], dtype=torch.int64, device=positions.device)

        # ``Gaussian`` holds externally bound kernel tensors as plain attributes,
        # which ``.to()``/``.cuda()`` on the kernel does not move.  Refresh the
        # binding when the kernel's tensors changed (one int compare when not).
        if self.kernel is not None:
            self.kernel.sync_into(self.gaussian)

        if latent_charges is not None:
            if latent_charges.dim() == 1:
                latent_charges = latent_charges.unsqueeze(1)
            assert latent_charges.shape[0] == positions.shape[0]
        elif desc is not None:
            if not self.use_atomwise:
                raise ValueError("desc is provided but use_atomwise is False")
            assert desc.shape[0] == positions.shape[0]
            latent_charges = self.atomwise(desc, batch)
        else:
            raise ValueError("Either desc or latent_charges must be provided")

        if self.charge_neutral:
            # Hard per-frame per-channel charge neutrality: subtract the per-frame
            # mean so that sum_i q_i = 0 for each frame and each charge channel.
            # This mirrors DeepMD's _corr_head (lr_fitting.py:521-528).
            n_frames = int(batch.max().item()) + 1
            frame_sum = torch.zeros(
                n_frames, latent_charges.shape[1],
                dtype=latent_charges.dtype, device=latent_charges.device,
            )
            frame_count = torch.zeros(
                n_frames, dtype=latent_charges.dtype, device=latent_charges.device,
            )
            frame_sum.scatter_add_(
                0, batch.unsqueeze(-1).expand_as(latent_charges), latent_charges,
            )
            frame_count.scatter_add_(
                0, batch, torch.ones_like(latent_charges[:, 0]),
            )
            frame_mean = frame_sum / frame_count.clamp(min=1).unsqueeze(-1)
            latent_charges = latent_charges - frame_mean[batch]

        forces = None
        virial = None
        used_explicit = None

        if compute_energy or compute_force or compute_virial:
            bundle = self.gaussian.compute_bundle(
                q=latent_charges,
                r=positions,
                cell=cell,
                batch=batch,
                compute_force=compute_force,
                compute_virial=compute_virial,
            )
            energy = bundle["energy"]
            e_lr = energy if compute_energy else None
            used_explicit = bundle["used_explicit_derivatives"]
            autograd_safe = bool(bundle.get("autograd_safe", False))
            forces = bundle["forces"]
            virial = bundle["virial"]
            need_autograd = (compute_force and forces is None) or (
                compute_virial and virial is None
            )
            if need_autograd:
                if not autograd_safe:
                    raise RuntimeError(
                        "The solver took an explicit-derivative branch in some "
                        "frames and an autograd branch in others, so it returns "
                        "no forces for this mix.  Request them only on a uniform "
                        "configuration (use_cubes2_fft / use_quads_fft / use_nufft "
                        "for every frame, or for none)."
                    )
                ag_force, ag_virial = self._autograd_efv(
                    energy,
                    positions,
                    cell,
                    batch,
                    forces=forces,
                    need_virial=(compute_virial and virial is None),
                )
                if compute_force and forces is None:
                    forces = ag_force
                if compute_virial and virial is None:
                    virial = ag_virial
        else:
            e_lr = None

        bec = None
        if compute_bec:
            bec = self.bec(
                q=latent_charges,
                r=positions,
                cell=cell,
                batch=batch,
                output_index=bec_output_index,
            )

        return {
            "E_lr": e_lr,
            "latent_charges": latent_charges,
            "BEC": bec,
            # Additive: present (and None) unless compute_force/compute_virial asked
            # for them, so existing callers that only read "E_lr" are unaffected.
            "forces": forces,
            "virial": virial,
            "used_explicit_derivatives": used_explicit,
        }

    def _autograd_efv(
        self,
        energy: torch.Tensor,
        positions: torch.Tensor,
        cell: Optional[torch.Tensor],
        batch: torch.Tensor,
        forces: Optional[torch.Tensor],
        need_virial: bool,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Forces and virial by autograd, for solver paths with no analytic derivatives.

        Used only when ``compute_bundle`` produced no explicit derivatives while
        reporting ``autograd_safe`` — i.e. the energy came entirely from
        differentiable operations and differentiating it is the intended route.

        The virial is **not** simply ``-dE/dh``.  SOG's convention (and LAMMPS's)
        is the affine one: a homogeneous strain of the whole system, which holds
        the *fractional* coordinates fixed.  With ``h' = h(I + ε)`` that gives

            W = -dE/dε = Σ_{i∈f} F_i ⊗ r_i  -  (∂E/∂h)ᵀ h        (per frame f)

        The first term is the familiar per-atom position virial.  Dropping it
        still reproduces the diagonal of a cubic cell but is wrong off-diagonal,
        because ``∂E/∂h`` alone is the derivative at *fixed Cartesian* positions.
        Both terms are obtained in a single ``autograd.grad`` pass.
        """
        # Only ask autograd for what is genuinely missing.  When the solver
        # already produced forces (e.g. the NUFFT branch) they are used for the
        # position-virial term, so `positions` need not be a grad leaf at all --
        # only `cell` does, for the box-strain term.
        leaves = []
        want_force = forces is None
        if want_force:
            if not positions.requires_grad:
                raise ValueError(
                    "This solver path computes no analytic forces, so they must "
                    "come from autograd — but `positions` does not require grad.  "
                    "Set `positions.requires_grad_(True)` before the call and keep "
                    "the graph (do not wrap the call in torch.no_grad)."
                )
            leaves.append(positions)
        if need_virial:
            if cell is None:
                raise ValueError(
                    "A periodic `cell` is required to obtain the virial by autograd."
                )
            if not cell.requires_grad:
                raise ValueError(
                    "This solver path computes no analytic virial, so the "
                    "box-strain term (∂E/∂h)ᵀh must come from autograd — but `cell` "
                    "does not require grad.  Set `cell.requires_grad_(True)` before "
                    "the call and keep the graph (do not wrap in torch.no_grad)."
                )
            leaves.append(cell)

        if not leaves:
            return forces, None

        grads = torch.autograd.grad(
            energy.sum(),
            leaves,
            create_graph=torch.is_grad_enabled(),
            retain_graph=True,
        )
        force = forces
        if want_force:
            grad_r = grads[0]
            if grad_r is None:
                raise RuntimeError("the energy does not depend on `positions`.")
            force = -grad_r  # F = -dE/dr

        virial = None
        if need_virial:
            grad_h = grads[-1]
            if grad_h is None:
                raise RuntimeError("the energy does not depend on `cell`.")
            n_frames = int(energy.shape[0])
            outer = torch.einsum("ic,id->icd", force, positions)  # [N,3,3]
            pos_vir = torch.zeros(
                n_frames, 3, 3, dtype=outer.dtype, device=outer.device
            ).index_add_(0, batch, outer)
            box_vir = -torch.einsum("fga,fgb->fab", grad_h, cell)
            virial = pos_vir + box_vir

        return force, virial


class _DummyAtomwise(nn.Module):
    def forward(self, desc: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        del desc, batch
        raise ValueError("set use_atomwise=True to use the Atomwise module")
