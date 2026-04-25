from __future__ import annotations

import argparse
from typing import Tuple

import torch

from sog import Sog
from sog.util.deepmd_ref import DeepMDSOGReference


def _error_stat(a: torch.Tensor, b: torch.Tensor) -> Tuple[float, float, float]:
    diff = (a - b).abs()
    rel = diff / b.abs().clamp_min(1e-12)
    rmse = torch.sqrt(torch.mean((a - b) ** 2))
    return float(diff.max().item()), float(rel.max().item()), float(rmse.item())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare sog core against deepmd SOG frame correction on the same input."
    )
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--natom", type=int, default=18)
    parser.add_argument("--ncharge", type=int, default=2)
    parser.add_argument("--deepmd-source", type=str, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    dtype = torch.float64

    model = Sog(
        {
            "use_atomwise": False,
            "use_nufft": True,
            "nufft_eps": 1e-4,
            "remove_self_interaction": True,
            "trainable_kernel": False,
        }
    )

    r = torch.rand(args.natom, 3, dtype=dtype)
    q = torch.rand(args.natom, args.ncharge, dtype=dtype) - 0.5
    q = q - q.mean(dim=0, keepdim=True)
    cell = torch.tensor(
        [[9.2, 0.0, 0.0], [0.7, 8.6, 0.0], [0.2, 0.4, 8.0]],
        dtype=dtype,
    ).unsqueeze(0)

    out_sog = model(
        positions=r.clone(),
        cell=cell,
        latent_charges=q,
        compute_energy=True,
        compute_force=True,
        compute_virial=True,
        use_explicit_derivatives=True,
    )

    if not out_sog["used_explicit_derivatives"]:
        raise RuntimeError("sog explicit derivative path is unavailable in current runtime")

    ref = DeepMDSOGReference.from_gaussian(
        model.gaussian,
        q_dim=q.shape[1],
        deepmd_source=args.deepmd_source,
    )

    out_dp = ref.compute_bundle(
        coord=r.unsqueeze(0),
        latent_charge=q.unsqueeze(0),
        box=cell,
        compute_force=True,
        compute_virial=True,
    )

    e_sog = out_sog["E_lr"]
    f_sog = out_sog["forces"]
    v_sog = out_sog["virial"]

    e_dp = out_dp["energy"]
    f_dp = out_dp["forces"]
    v_dp = out_dp["virial"]

    assert e_sog is not None and f_sog is not None and v_sog is not None
    assert f_dp is not None and v_dp is not None

    e_abs, e_rel, e_rmse = _error_stat(e_sog, e_dp)
    f_abs, f_rel, f_rmse = _error_stat(f_sog, f_dp[0])
    v_abs, v_rel, v_rmse = _error_stat(v_sog, v_dp)

    print("=== SOG vs deepmd reference (same input) ===")
    print(f"deepmd source : {ref.deepmd_source}")
    print(f"energy max_abs={e_abs:.6e} max_rel={e_rel:.6e} rmse={e_rmse:.6e}")
    print(f"force  max_abs={f_abs:.6e} max_rel={f_rel:.6e} rmse={f_rmse:.6e}")
    print(f"virial max_abs={v_abs:.6e} max_rel={v_rel:.6e} rmse={v_rmse:.6e}")


if __name__ == "__main__":
    main()
