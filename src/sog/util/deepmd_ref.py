from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from typing import Dict, Optional, Sequence

import torch


def ensure_deepmd_source_on_path(
    deepmd_source: Optional[str] = None,
) -> Optional[Path]:
    candidates = []

    if deepmd_source:
        candidates.append(Path(deepmd_source).expanduser().resolve())

    env_path = os.environ.get("DEEPMD_SOURCE_DIR")
    if env_path:
        candidates.append(Path(env_path).expanduser().resolve())

    try:
        workspace_root = Path(__file__).resolve().parents[4]
        candidates.append(workspace_root / "dp_pt" / "dp_devel" / "deepmd-kit-devel")
    except IndexError:
        pass

    picked = None
    for candidate in candidates:
        if candidate.is_dir() and (candidate / "deepmd").is_dir():
            picked = candidate
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            break

    try:
        import deepmd  # noqa: F401
    except Exception as exc:
        raise ImportError(
            "Cannot import deepmd. Set DEEPMD_SOURCE_DIR or pass --deepmd-source "
            "to point to deepmd-kit-devel."
        ) from exc

    return picked


class DeepMDSOGReference:
    def __init__(
        self,
        q_dim: int,
        amp: float,
        bandwidth: Sequence[float],
        n_dl: float,
        remove_self_interaction: bool,
        deepmd_source: Optional[str] = None,
    ):
        self._deepmd_source = ensure_deepmd_source_on_path(deepmd_source)

        from deepmd.pt.model.model.sog_model import SOGEnergyModel
        from deepmd.pt.model.task.sog_energy_fitting import SOGEnergyFittingNet

        self._sog_model_cls = SOGEnergyModel
        self._fitting = SOGEnergyFittingNet(
            var_name="energy",
            ntypes=1,
            dim_descrpt=1,
            dim_out_sr=1,
            dim_out_lr=int(q_dim),
            neuron_sr=[1],
            neuron_lr=[1],
            trainable=False,
            amp=float(amp),
            bandwidth=list(float(x) for x in bandwidth),
            n_dl=float(n_dl),
            remove_self_interaction=bool(remove_self_interaction),
        )

        self._model = object.__new__(SOGEnergyModel)
        self._model._kgrid_base_cache = {}
        fitting_ref = self._fitting

        def _get_fitting_net(model_self):
            del model_self
            return fitting_ref

        self._model.get_fitting_net = types.MethodType(_get_fitting_net, self._model)

    @property
    def deepmd_source(self) -> Optional[Path]:
        return self._deepmd_source

    @classmethod
    def from_gaussian(
        cls,
        gaussian: torch.nn.Module,
        q_dim: int,
        deepmd_source: Optional[str] = None,
    ) -> "DeepMDSOGReference":
        amp = float(gaussian.amp.detach().reshape(-1)[0].cpu().item())
        bw = gaussian.bandwidth.detach().reshape(-1).cpu().tolist()
        n_dl = float(gaussian.n_dl)
        remove_si = bool(gaussian.remove_self_interaction)
        return cls(
            q_dim=q_dim,
            amp=amp,
            bandwidth=bw,
            n_dl=n_dl,
            remove_self_interaction=remove_si,
            deepmd_source=deepmd_source,
        )

    def compute_bundle(
        self,
        coord: torch.Tensor,
        latent_charge: torch.Tensor,
        box: torch.Tensor,
        compute_force: bool = True,
        compute_virial: bool = True,
    ) -> Dict[str, Optional[torch.Tensor]]:
        if coord.dim() != 3:
            raise ValueError(f"`coord` should be [nf, nloc, 3], got {tuple(coord.shape)}")
        if latent_charge.dim() != 3:
            raise ValueError(
                "`latent_charge` should be [nf, nloc, nq], "
                f"got {tuple(latent_charge.shape)}"
            )
        if box.dim() != 3 or box.shape[-2:] != (3, 3):
            raise ValueError(f"`box` should be [nf, 3, 3], got {tuple(box.shape)}")

        raw = self._sog_model_cls._compute_sog_frame_correction_bundle(
            self._model,
            coord,
            latent_charge,
            box,
            need_force=bool(compute_force or compute_virial),
            need_virial=bool(compute_virial),
        )

        energy = raw["corr_redu"].reshape(coord.shape[0])

        force = None
        if compute_force or compute_virial:
            force = raw["force_local"]

        virial = None
        if compute_virial:
            virial = raw["virial_local"].sum(dim=1).reshape(coord.shape[0], 3, 3)

        return {
            "energy": energy,
            "forces": force,
            "virial": virial,
        }
