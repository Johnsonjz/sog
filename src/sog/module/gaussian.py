from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

try:
    import pytorch_finufft

    HAS_PYTORCH_FINUFFT = True
except (ImportError, ModuleNotFoundError):
    HAS_PYTORCH_FINUFFT = False
    pytorch_finufft = None


E2_PER_ANGSTROM_TO_EV = 14.3996454784255
SOG_DEFAULT_B = 1.62976708826776469
SOG_DEFAULT_SIGMA = 2.180230445405648
SOG_DEFAULT_M = 12


class Gaussian(nn.Module):
    """Gaussian long-range SOG core.

    Periodic systems are computed in reciprocal space (NUFFT preferred), and
    non-periodic/singular-cell inputs fall back to a direct real-space kernel.
    """

    def __init__(
        self,
        n_dl: float = 1.0,
        amp: Optional[float] = None,
        bandwidth: Optional[torch.Tensor] = None,
        b: float = SOG_DEFAULT_B,
        sigma: float = SOG_DEFAULT_SIGMA,
        m: int = SOG_DEFAULT_M,
        remove_self_interaction: bool = True,
        charge_neutral_lambda: Optional[float] = None,
        use_nufft: bool = True,
        nufft_eps: float = 1e-4,
        norm_factor: float = E2_PER_ANGSTROM_TO_EV,
        trainable: bool = True,
        max_cache_size: int = 8,
    ):
        super().__init__()

        n_dl_value = float(n_dl)
        if (not math.isfinite(n_dl_value)) or n_dl_value <= 0.0:
            raise ValueError("`n_dl` should be a positive finite number.")
        self.n_dl = n_dl_value

        if bandwidth is None:
            if sigma <= 0.0:
                raise ValueError("`sigma` should be positive when `bandwidth` is not provided.")
            m_value = max(1, int(m))
            bw = sigma * torch.pow(
                torch.tensor(float(b), dtype=torch.get_default_dtype()),
                torch.arange(m_value, dtype=torch.get_default_dtype()),
            )
        else:
            bw = torch.as_tensor(bandwidth, dtype=torch.get_default_dtype()).reshape(-1)

        if bw.numel() == 0:
            raise ValueError("`bandwidth` should not be empty.")
        if not torch.isfinite(bw).all():
            raise ValueError("`bandwidth` should be finite.")
        if torch.any(bw <= 0.0):
            raise ValueError("`bandwidth` values should be positive.")
        bw2 = bw.square()

        if amp is None:
            if b <= 0.0:
                raise ValueError("`b` should be positive when `amp` is not provided.")
            coef1 = float(4.0 * torch.pi * math.log(b))
            amp_tensor = torch.full_like(bw2, fill_value=coef1)
        else:
            amp_tensor = torch.as_tensor(amp, dtype=torch.get_default_dtype()).reshape(-1)
        if amp_tensor.numel() == 0:
            raise ValueError("`amp` should not be empty.")
        if not torch.isfinite(amp_tensor).all():
            raise ValueError("`amp` should be finite.")

        # Allow scalar amp and broadcast it to all Gaussian terms.
        if amp_tensor.numel() == 1 and bw2.numel() > 1:
            amp_tensor = amp_tensor.expand_as(bw2).clone()
        elif amp_tensor.numel() != bw2.numel():
            raise ValueError(
                "`amp` should be scalar or have the same length as `bandwidth`."
            )

        self.amp = nn.Parameter(amp_tensor, requires_grad=trainable)
        self.bandwidth = nn.Parameter(bw2, requires_grad=trainable)

        self.remove_self_interaction = bool(remove_self_interaction)
        self.charge_neutral_lambda = charge_neutral_lambda
        self.use_nufft = bool(use_nufft)
        self.nufft_eps = float(nufft_eps)
        self.norm_factor = float(norm_factor)

        self._max_cache_size = max(1, int(max_cache_size))
        self._kgrid_base_cache: Dict[
            Tuple[str, str, int, int, int],
            Tuple[torch.Tensor, torch.Tensor, Tuple[int, int, int]],
        ] = {}

    @staticmethod
    def _device_key(device: torch.device) -> str:
        if device.index is None:
            return device.type
        return f"{device.type}:{device.index}"

    def _trim_cache(self) -> None:
        if len(self._kgrid_base_cache) > self._max_cache_size:
            oldest_key = next(iter(self._kgrid_base_cache.keys()))
            self._kgrid_base_cache.pop(oldest_key, None)

    def _get_cached_kgrid_base(
        self,
        nk: Tuple[int, int, int],
        runtime_device: torch.device,
        real_dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int, int]]:
        cache_key = (
            self._device_key(runtime_device),
            str(real_dtype),
            int(nk[0]),
            int(nk[1]),
            int(nk[2]),
        )
        cached = self._kgrid_base_cache.get(cache_key)
        if cached is not None:
            return cached

        n1 = torch.arange(-nk[0], nk[0] + 1, device=runtime_device, dtype=real_dtype)
        n2 = torch.arange(-nk[1], nk[1] + 1, device=runtime_device, dtype=real_dtype)
        n3 = torch.arange(-nk[2], nk[2] + 1, device=runtime_device, dtype=real_dtype)
        kx_grid, ky_grid, kz_grid = torch.meshgrid(n1, n2, n3, indexing="ij")

        k_grid_int = torch.stack((kx_grid, ky_grid, kz_grid), dim=0)
        zero_mask = (k_grid_int[0] == 0) & (k_grid_int[1] == 0) & (k_grid_int[2] == 0)
        output_shape = tuple(int(x) for x in kx_grid.shape)

        out = (k_grid_int, zero_mask, output_shape)
        self._kgrid_base_cache[cache_key] = out
        self._trim_cache()
        return out

    def forward(
        self,
        q: torch.Tensor,
        r: torch.Tensor,
        cell: Optional[torch.Tensor],
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.compute_bundle(
            q=q,
            r=r,
            cell=cell,
            batch=batch,
            compute_force=False,
            compute_virial=False,
        )["energy"]

    def compute_bundle(
        self,
        q: torch.Tensor,
        r: torch.Tensor,
        cell: Optional[torch.Tensor],
        batch: Optional[torch.Tensor] = None,
        compute_force: bool = False,
        compute_virial: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:
        if q.dim() == 1:
            q = q.unsqueeze(1)

        n, d = r.shape
        assert d == 3, "r dimension error"
        assert n == q.size(0), "q dimension error"

        if batch is None:
            batch = torch.zeros(n, dtype=torch.int64, device=r.device)

        if cell is not None and (cell.dim() != 3 or cell.shape[-2:] != (3, 3)):
            raise ValueError(f"`cell` should be [nbatch, 3, 3], got {tuple(cell.shape)}")

        need_force = bool(compute_force or compute_virial)
        energies = []
        force_full = (
            torch.zeros((n, 3), dtype=r.dtype, device=r.device) if need_force else None
        )
        virial_list = [] if compute_virial else None

        explicit_all = True

        for bid_t in torch.unique(batch):
            bid = int(bid_t.item())
            mask = batch == bid_t
            r_now = r[mask]
            q_now = q[mask]

            if r_now.shape[0] == 0:
                continue

            periodic = False
            box_now = None
            if cell is not None:
                box_now = cell[bid]
                det_now = torch.det(box_now)
                periodic = torch.abs(det_now) > torch.finfo(box_now.dtype).eps

            if periodic and need_force and self.use_nufft and HAS_PYTORCH_FINUFFT:
                assert box_now is not None
                state = self._prepare_triclinic_state(r_now, q_now, box_now)
                pot_now, force_now, virial_now = self._compute_periodic_nufft_bundle(
                    state,
                    need_force=True,
                    need_virial=compute_virial,
                )

                if self.remove_self_interaction:
                    pot_now = pot_now - torch.sum(q_now * q_now) * state["diag_sum"]

                pot_now = pot_now * self.norm_factor
                force_now = force_now * self.norm_factor
                if virial_now is not None:
                    virial_now = virial_now * self.norm_factor

                if force_full is not None:
                    force_full[mask] = force_now
                if virial_list is not None:
                    assert virial_now is not None
                    virial_list.append(virial_now)
            else:
                if need_force:
                    explicit_all = False

                if periodic:
                    assert box_now is not None
                    pot_now = self.compute_potential_triclinic(r_now, q_now, box_now)
                else:
                    pot_now = self.compute_potential_realspace(r_now, q_now)

                if virial_list is not None:
                    virial_list.append(torch.zeros((3, 3), dtype=r.dtype, device=r.device))

            if self.charge_neutral_lambda is not None:
                pot_now = pot_now + float(self.charge_neutral_lambda) * torch.mean(q_now).square()

            energies.append(pot_now)

        if len(energies) == 0:
            energy_out = torch.zeros(0, dtype=r.dtype, device=r.device)
        else:
            energy_out = torch.stack(energies, dim=0)

        used_explicit = bool(need_force and explicit_all)
        if need_force and not explicit_all:
            force_out = None
            virial_out = None
        else:
            force_out = force_full
            virial_out = torch.stack(virial_list, dim=0) if virial_list is not None else None

        return {
            "energy": energy_out,
            "forces": force_out,
            "virial": virial_out,
            "used_explicit_derivatives": used_explicit,
        }

    def compute_potential_realspace(self, r_raw: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        if q.dim() == 1:
            q = q.unsqueeze(1)

        n = r_raw.shape[0]
        r_ij = r_raw.unsqueeze(0) - r_raw.unsqueeze(1)
        r_sq = torch.sum(r_ij * r_ij, dim=-1, keepdim=True)

        amp = self.amp.to(dtype=r_raw.dtype, device=r_raw.device).view(1, 1, -1)
        bw2 = self.bandwidth.to(dtype=r_raw.dtype, device=r_raw.device).view(1, 1, -1)
        kernel = amp * torch.exp(-0.5 * r_sq / bw2)
        kernel = kernel.sum(dim=-1)

        diag = torch.arange(n, device=r_raw.device)
        kernel[diag, diag] = 0.0

        pair_q = q.unsqueeze(0) * q.unsqueeze(1)
        pot = 0.5 * torch.sum(pair_q * kernel.unsqueeze(-1))

        if not self.remove_self_interaction:
            k0 = amp.sum()
            pot = pot + 0.5 * torch.sum(q * q) * k0

        return pot * self.norm_factor

    def compute_potential_triclinic(
        self,
        r_raw: torch.Tensor,
        q: torch.Tensor,
        cell_now: torch.Tensor,
    ) -> torch.Tensor:
        state = self._prepare_triclinic_state(r_raw, q, cell_now)

        if self.use_nufft and HAS_PYTORCH_FINUFFT:
            pot = self._compute_periodic_nufft(
                state["r_in"],
                state["q"],
                state["kfac"],
                state["output_shape"],
                state["volume"],
            )
        else:
            pot = self._compute_periodic_direct(
                state["r_raw"],
                state["q"],
                state["g_cart"],
                state["kfac"],
                state["volume"],
                state["k_mode_mask"],
            )

        if self.remove_self_interaction:
            pot = pot - torch.sum(state["q"] * state["q"]) * state["diag_sum"]

        return pot * self.norm_factor

    def _prepare_triclinic_state(
        self,
        r_raw: torch.Tensor,
        q: torch.Tensor,
        cell_now: torch.Tensor,
    ) -> Dict[str, torch.Tensor | Tuple[int, int, int]]:
        if q.dim() == 1:
            q = q.unsqueeze(1)

        runtime_device = r_raw.device
        real_dtype = r_raw.dtype

        box = cell_now.to(dtype=real_dtype, device=runtime_device)
        volume = torch.det(box)
        if torch.abs(volume) <= torch.finfo(real_dtype).eps:
            raise ValueError("`cell` is singular (near-zero volume).")
        volume = torch.abs(volume)

        cell_inv = torch.linalg.inv(box)
        r_frac = torch.matmul(r_raw, cell_inv)
        r_frac = torch.remainder(r_frac + 0.5, 1.0) - 0.5

        pi_tensor = torch.tensor(torch.pi, dtype=real_dtype, device=runtime_device)
        point_limit = pi_tensor - 32.0 * torch.finfo(real_dtype).eps
        r_in = torch.clamp(
            2.0 * pi_tensor * r_frac,
            min=-point_limit,
            max=point_limit,
        ).contiguous()

        norms = torch.norm(box, dim=1)
        nk = tuple(max(1, int(v.item() / self.n_dl)) for v in norms)

        k_grid_int, zero_mask, output_shape = self._get_cached_kgrid_base(
            nk,
            runtime_device,
            real_dtype,
        )

        two_pi = 2.0 * pi_tensor
        n_dl_tensor = torch.as_tensor(self.n_dl, dtype=real_dtype, device=runtime_device)
        k_sq_max = (two_pi / n_dl_tensor) ** 2

        g_cart = two_pi * torch.einsum("ik,k...->i...", cell_inv, k_grid_int)
        k_sq = torch.sum(g_cart * g_cart, dim=0)
        k_mode_mask = (~zero_mask) & (k_sq <= k_sq_max)

        amp = self.amp.to(dtype=real_dtype, device=runtime_device).view(
            1,
            1,
            1,
            -1,
        )
        bw2 = self.bandwidth.to(dtype=real_dtype, device=runtime_device).view(
            1,
            1,
            1,
            -1,
        )
        kfac = amp * bw2 * torch.exp(-0.5 * bw2 * k_sq.unsqueeze(-1))
        kfac = kfac.sum(dim=-1).masked_fill(~k_mode_mask, 0.0)

        diag_sum = kfac.sum() / (2.0 * volume)

        return {
            "r_raw": r_raw,
            "q": q,
            "r_in": r_in,
            "g_cart": g_cart,
            "kfac": kfac,
            "k_mode_mask": k_mode_mask,
            "output_shape": output_shape,
            "volume": volume,
            "diag_sum": diag_sum,
        }

    def _compute_periodic_nufft(
        self,
        r_in: torch.Tensor,
        q: torch.Tensor,
        kfac: torch.Tensor,
        output_shape: Tuple[int, int, int],
        volume: torch.Tensor,
    ) -> torch.Tensor:
        q_t = q.transpose(0, 1).contiguous()
        complex_dtype = torch.complex128 if q.dtype == torch.float64 else torch.complex64
        charge = torch.complex(q_t, torch.zeros_like(q_t)).to(dtype=complex_dtype).contiguous()

        nufft_points = r_in.transpose(0, 1).contiguous()
        recon = pytorch_finufft.functional.finufft_type1(
            nufft_points,
            charge,
            output_shape=output_shape,
            eps=self.nufft_eps,
            isign=-1,
        )

        if recon.dim() == 3:
            recon = recon.unsqueeze(0)

        recon = torch.fft.fftshift(recon, dim=(1, 2, 3))
        rho_sq = recon.real.square() + recon.imag.square()

        return (kfac.unsqueeze(0) * rho_sq).sum() / (2.0 * volume)

    def _compute_periodic_nufft_bundle(
        self,
        state: Dict[str, torch.Tensor | Tuple[int, int, int]],
        need_force: bool,
        need_virial: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        r_in = state["r_in"]
        q = state["q"]
        kfac = state["kfac"]
        output_shape = state["output_shape"]
        volume = state["volume"]
        g_cart = state["g_cart"]
        r_raw = state["r_raw"]

        assert isinstance(r_in, torch.Tensor)
        assert isinstance(q, torch.Tensor)
        assert isinstance(kfac, torch.Tensor)
        assert isinstance(output_shape, tuple)
        assert isinstance(volume, torch.Tensor)
        assert isinstance(g_cart, torch.Tensor)
        assert isinstance(r_raw, torch.Tensor)

        q_t = q.transpose(0, 1).contiguous()
        real_dtype = q.dtype
        complex_dtype = torch.complex128 if real_dtype == torch.float64 else torch.complex64
        charge = torch.complex(q_t, torch.zeros_like(q_t)).to(dtype=complex_dtype).contiguous()

        nufft_points = r_in.transpose(0, 1).contiguous()
        recon = pytorch_finufft.functional.finufft_type1(
            nufft_points,
            charge,
            output_shape=output_shape,
            eps=self.nufft_eps,
            isign=-1,
        )

        if recon.dim() == 3:
            recon = recon.unsqueeze(0)

        recon = torch.fft.fftshift(recon, dim=(1, 2, 3))
        rho_sq = recon.real.square() + recon.imag.square()
        energy = (kfac.unsqueeze(0) * rho_sq).sum() / (2.0 * volume)

        if not need_force:
            return energy, torch.zeros((q.shape[0], 3), dtype=real_dtype, device=q.device), None

        conv = kfac.unsqueeze(0).to(dtype=complex_dtype) * recon
        grad_conv = (1j * g_cart.unsqueeze(1).to(dtype=complex_dtype)) * conv.unsqueeze(0)
        grad_conv = torch.fft.ifftshift(grad_conv, dim=(2, 3, 4))

        grad_field = pytorch_finufft.functional.finufft_type2(
            nufft_points,
            grad_conv,
            eps=self.nufft_eps,
            isign=1,
        )

        force = (
            -(q_t.unsqueeze(0) * grad_field.real.to(dtype=real_dtype)).sum(dim=1).transpose(0, 1)
            / volume
        )

        virial = None
        if need_virial:
            virial = torch.einsum("ni,nj->ij", force, r_raw)

        return energy, force, virial

    def _compute_periodic_direct(
        self,
        r_raw: torch.Tensor,
        q: torch.Tensor,
        g_cart: torch.Tensor,
        kfac: torch.Tensor,
        volume: torch.Tensor,
        k_mode_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        kvec = g_cart.reshape(3, -1).transpose(0, 1)
        kfac_flat = kfac.reshape(-1)

        if k_mode_mask is None:
            # Fallback for old call sites.
            mask = kfac_flat != 0
        else:
            # Keep semantics consistent with condition=(k==0) and cutoff filtering.
            mask = k_mode_mask.reshape(-1)

        if not torch.any(mask):
            return torch.zeros((), dtype=r_raw.dtype, device=r_raw.device)

        kvec = kvec[mask]
        kfac_flat = kfac_flat[mask]

        k_dot_r = torch.matmul(r_raw, kvec.transpose(0, 1))
        cos_k_dot_r = torch.cos(k_dot_r)
        sin_k_dot_r = torch.sin(k_dot_r)

        s_real = (q.unsqueeze(2) * cos_k_dot_r.unsqueeze(1)).sum(dim=0)
        s_imag = (q.unsqueeze(2) * sin_k_dot_r.unsqueeze(1)).sum(dim=0)
        s_sq = s_real.square() + s_imag.square()

        return (kfac_flat.unsqueeze(0) * s_sq).sum() / (2.0 * volume)

    def __repr__(self) -> str:
        return (
            "Gaussian("
            f"n_dl={self.n_dl}, "
            f"remove_self_interaction={self.remove_self_interaction}, "
            f"use_nufft={self.use_nufft and HAS_PYTORCH_FINUFFT}"
            ")"
        )
