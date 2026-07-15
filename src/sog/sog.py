from __future__ import annotations

from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn

from .module import Atomwise, BEC, Gaussian


RCUT_TO_SIGMA = 1.9892536839080267


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
    ):
        super().__init__()

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

        self.gaussian = Gaussian(
            n_dl=self.n_dl,
            cubes2_phi_max=self.cubes2_phi_max,
            cubes2_order=self.cubes2_order,
            amp=self.amp,
            bandwidth=self.bandwidth,
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
            trainable=self.trainable_kernel,
            kernel_param_mode=self.kernel_param_mode,
            kernel_tensor_mode=self.kernel_tensor_mode,
            use_cubes2_fft=self.use_cubes2_fft,
            use_quads_fft=self.use_quads_fft,
            quads_order=self.quads_order,
        )

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

    def forward(
        self,
        positions: torch.Tensor,
        cell: Optional[torch.Tensor],
        desc: Optional[torch.Tensor] = None,
        latent_charges: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        compute_energy: bool = True,
        compute_bec: bool = False,
        bec_output_index: Optional[int] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        if batch is None:
            batch = torch.zeros(positions.shape[0], dtype=torch.int64, device=positions.device)

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

        if compute_energy:
            e_lr = self.gaussian(
                q=latent_charges,
                r=positions,
                cell=cell,
                batch=batch,
            )
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
        }


class _DummyAtomwise(nn.Module):
    def forward(self, desc: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        del desc, batch
        raise ValueError("set use_atomwise=True to use the Atomwise module")
