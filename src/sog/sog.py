from __future__ import annotations

from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn

from .module import Atomwise, BEC, Gaussian


class Sog(nn.Module):
    """SOG plugin model with LES-like API."""

    def __init__(self, sog_arguments: Union[Dict[str, Any], str, None] = None):
        super().__init__()

        if sog_arguments is None:
            sog_arguments = {}
        if isinstance(sog_arguments, str):
            import yaml

            with open(sog_arguments, "r", encoding="utf-8") as f:
                parsed = yaml.safe_load(f)
            sog_arguments = {} if parsed is None else parsed

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
            amp=self.amp,
            bandwidth=self.bandwidth,
            b=self.b,
            sigma=self.sigma,
            m=self.m,
            remove_self_interaction=self.remove_self_interaction,
            charge_neutral_lambda=self.charge_neutral_lambda,
            use_nufft=self.use_nufft,
            nufft_eps=self.nufft_eps,
            norm_factor=self.norm_factor,
            trainable=self.trainable_kernel,
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

        self.n_dl = float(args.get("n_dl", 1.0))
        self.amp = args.get("amp", None)
        self.bandwidth = args.get("bandwidth", None)
        self.b = float(args.get("b", 1.62976708826776469))
        self.sigma = float(args.get("sigma", 2.180230445405648))
        self.m = int(args.get("m", 12))
        self.remove_self_interaction = bool(args.get("remove_self_interaction", True))
        self.charge_neutral_lambda = args.get("charge_neutral_lambda", None)
        self.use_nufft = bool(args.get("use_nufft", True))
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
        compute_force: bool = False,
        compute_virial: bool = False,
        compute_bec: bool = False,
        bec_output_index: Optional[int] = None,
        use_explicit_derivatives: bool = True,
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

        need_energy = bool(compute_energy or compute_force or compute_virial)

        e_lr = None
        used_explicit_derivatives = False
        need_force_internal = bool(compute_force or compute_virial)

        if compute_bec or need_force_internal:
            positions = positions.requires_grad_(True)

        if need_energy:
            if need_force_internal and use_explicit_derivatives:
                bundle = self.gaussian.compute_bundle(
                    q=latent_charges,
                    r=positions,
                    cell=cell,
                    batch=batch,
                    compute_force=True,
                    compute_virial=compute_virial,
                )
                e_lr = bundle["energy"]
                used_explicit_derivatives = bool(bundle["used_explicit_derivatives"])
            else:
                bundle = None
                e_lr = self.gaussian(
                    q=latent_charges,
                    r=positions,
                    cell=cell,
                    batch=batch,
                )
        else:
            bundle = None

        forces = None
        virial = None
        if need_force_internal:
            if used_explicit_derivatives and bundle is not None:
                forces = bundle["forces"]
                virial = bundle["virial"] if compute_virial else None
            else:
                assert e_lr is not None
                grad_pos = torch.autograd.grad(
                    outputs=[e_lr.sum()],
                    inputs=[positions],
                    retain_graph=(compute_virial or compute_bec),
                    create_graph=compute_virial,
                    allow_unused=False,
                )[0]
                assert grad_pos is not None
                forces = -grad_pos

                if compute_virial:
                    virial = self._batch_virial(forces, positions, batch)

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
            "forces": forces,
            "virial": virial,
            "BEC": bec,
            "used_explicit_derivatives": used_explicit_derivatives,
        }

    @staticmethod
    def _batch_virial(
        forces: torch.Tensor,
        positions: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        out = []
        for bid_t in torch.unique(batch):
            mask = batch == bid_t
            f_now = forces[mask]
            r_now = positions[mask]
            out.append(torch.einsum("ni,nj->ij", f_now, r_now))
        return torch.stack(out, dim=0)


class _DummyAtomwise(nn.Module):
    def forward(self, desc: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        del desc, batch
        raise ValueError("set use_atomwise=True to use the Atomwise module")
