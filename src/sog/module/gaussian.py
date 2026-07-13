from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

try:
    import pytorch_finufft

    HAS_PYTORCH_FINUFFT = True
except (ImportError, ModuleNotFoundError):
    HAS_PYTORCH_FINUFFT = False
    pytorch_finufft = None


E2_PER_ANGSTROM_TO_EV = 14.3996454784255
SOG_DEFAULT_B = 2
SOG_DEFAULT_SIGMA = 2.180230445405648
SOG_DEFAULT_M = 12
RCUT_TO_SIGMA = 1.9892536839080267  # r_c/σ for b=2, C¹ continuity (midtown-sog.md)


def _as_1d_tensor_keep_input(value: object) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.reshape(-1)
    return torch.as_tensor(value, dtype=torch.get_default_dtype()).reshape(-1)


def _normalize_mode(value: str, valid: set[str], name: str) -> str:
    mode = str(value).strip().lower()
    if mode not in valid:
        raise ValueError(
            f"`{name}` should be one of {sorted(valid)}, got {value!r}."
        )
    return mode


def _is_differentiable_tensor(value: object) -> bool:
    return isinstance(value, torch.Tensor) and bool(value.requires_grad)


def _compute_w0(r0: float, b: float, nterms: int = 500, tol: float = 1e-14) -> float:
    """Compute w0 — real-space correction factor (fastsog.cpp:318-328).

    w0 = (1/G(1,r0)) * (1/(2*ln(b)*r0) - Σ b^{-i} * G(1, b^{-i}*r0))
    where G(sigma, r) = exp(-r²/(2*sigma²))/sqrt(2*pi*sigma²)

    Uses adaptive convergence: stops when |term| < tol * |sum|.
    """
    import math as _math
    G = lambda s, r: _math.exp(-r * r / (2.0 * s * s)) / _math.sqrt(2.0 * _math.pi * s * s)
    s = 0.0
    abs_sum = 0.0
    for i in range(1, nterms + 1):
        bi = b ** (-i)
        term = bi * G(1.0, bi * r0)
        s += term
        abs_sum += abs(term)
        if i >= 10 and abs(term) < tol * max(abs_sum, 1e-30):
            break
    w0 = (1.0 / G(1.0, r0)) * (1.0 / (2.0 * _math.log(b) * r0) - s)
    return w0


class Gaussian(nn.Module):
    """Gaussian long-range SOG core.

    Periodic systems are computed in reciprocal space when NUFFT is enabled,
    non-periodic/singular-cell inputs fall back to a direct real-space kernel.

    Parameters
    ----------
    kernel_param_mode : str
        Parameter interpretation mode.
        - ``"raw"``: ``amp`` is raw coefficient and ``bandwidth`` is bw.
        - ``"internal"``: ``amp`` is internal amplitude and ``bandwidth`` is bw^2.
    kernel_tensor_mode : str
        Tensor ownership/binding mode.
        - ``"owned"``: Gaussian owns ``nn.Parameter`` tensors.
        - ``"external"``: Gaussian directly binds caller tensors.
        - ``"auto"``: external binding when differentiable tensor inputs are
          detected and ``trainable=False``.
    """

    _diag_count = 0  # per-process step counter for gate tracing

    def __init__(
        self,
        n_dl: Optional[float] = None,
        amp: Optional[float] = None,
        bandwidth: Optional[torch.Tensor] = None,
        b: float = SOG_DEFAULT_B,
        sigma: float = SOG_DEFAULT_SIGMA,
        m: int = SOG_DEFAULT_M,
        rcut: Optional[float] = None,
        nlayers: int = 1,
        remove_self_interaction: bool = True,
        charge_neutral_lambda: Optional[float] = None,
        use_nufft: bool = False,
        nufft_eps: float = 1e-6,
        norm_factor: float = E2_PER_ANGSTROM_TO_EV,
        trainable: bool = True,
        max_cache_size: int = 8,
        nufft: Optional[bool] = None,
        kernel_param_mode: str = "raw",
        kernel_tensor_mode: str = "owned",
        use_cubes2_fft: bool = False,
        cubes2_phi_max: Optional[float] = None,
        cubes2_order: int = 4,
        use_quads_fft: bool = False,
        quads_order: int = 6,
        **_kwargs: Any,
    ):
        super().__init__()

        if _kwargs:
            unknown = ", ".join(sorted(_kwargs.keys()))
            raise TypeError(f"Unexpected keyword arguments: {unknown}")

        param_mode = _normalize_mode(
            kernel_param_mode,
            {"raw", "internal"},
            "kernel_param_mode",
        )
        tensor_mode = _normalize_mode(
            kernel_tensor_mode,
            {"owned", "external", "auto"},
            "kernel_tensor_mode",
        )

        self.self_coeff = 0.0  # real-space self-energy (fastsog convention)

        if n_dl is not None:
            n_dl_value = float(n_dl)
            if (not math.isfinite(n_dl_value)) or n_dl_value <= 0.0:
                raise ValueError("`n_dl` should be a positive finite number.")
            self.n_dl = n_dl_value
        else:
            self.n_dl = None

        # Validate cubes2_phi_max
        if cubes2_phi_max is not None:
            phi_val = float(cubes2_phi_max)
            if not math.isfinite(phi_val) or phi_val <= 0.0:
                raise ValueError("`cubes2_phi_max` should be a positive finite number.")

        # Paper-consistent sigma: σ = rcut · nlayers / RCUT_TO_SIGMA (midtown-sog.md, b=2 C¹)
        use_fastsog_conv = (rcut is not None and rcut > 0)
        if use_fastsog_conv:
            sigma = float(rcut) * int(nlayers) / RCUT_TO_SIGMA

        if bandwidth is None:
            if sigma <= 0.0:
                raise ValueError("`sigma` should be positive when `bandwidth` is not provided.")
            m_value = max(1, int(m))
            if use_fastsog_conv:
                # fastsog.cpp convention: bandwidth[m] = sigma^2 * b^{2m}
                sigma2 = sigma * sigma
                b2 = float(b) * float(b)
                bw2 = torch.zeros(m_value, dtype=torch.get_default_dtype())
                bw2[0] = sigma2
                for mm in range(1, m_value):
                    bw2[mm] = bw2[mm - 1] * b2
            else:
                bw = sigma * torch.pow(
                    torch.tensor(float(b), dtype=torch.get_default_dtype()),
                    torch.arange(m_value, dtype=torch.get_default_dtype()),
                )
                bw2 = bw.square()
        else:
            bw_in = _as_1d_tensor_keep_input(bandwidth)

        if bandwidth is None:
            bw_check = bw2
        else:
            bw_check = bw_in

        if bw_check.numel() == 0:
            raise ValueError("`bandwidth` should not be empty.")
        if not torch.isfinite(bw_check).all():
            raise ValueError("`bandwidth` should be finite.")
        if torch.any(bw_check <= 0.0):
            raise ValueError("`bandwidth` values should be positive.")
        if bandwidth is not None:
            bw2 = bw_in if param_mode == "internal" else bw_in.square()

        if amp is None:
            if b <= 0.0:
                raise ValueError("`b` should be positive when `amp` is not provided.")
            if use_fastsog_conv:
                # fastsog.cpp convention (line 530-541):
                #   amp[0] = 4*pi*log(b) * w0 * sigma^2
                #   amp[1] = 4*pi*log(b) * sigma^2 * b^2
                #   amp[m] = amp[m-1] * b^2   (m >= 2)
                sigma2 = sigma * sigma
                b2 = float(b) * float(b)
                logb = math.log(float(b))
                r0 = float(rcut) / sigma
                w0 = _compute_w0(r0, float(b))
                amp_factor = 4.0 * math.pi * logb
                amp_tensor = torch.zeros(m_value, dtype=torch.get_default_dtype())
                amp_tensor[0] = amp_factor * w0 * sigma2
                if m_value > 1:
                    amp_tensor[1] = amp_factor * sigma2 * b2
                for mm in range(2, m_value):
                    amp_tensor[mm] = amp_tensor[mm - 1] * b2

                # Real-space self-energy (fastsog.cpp line 544-551)
                sum_b_inv = 0.0
                for mm in range(1, m_value):
                    sum_b_inv += float(b) ** (-mm)
                self.self_coeff = (logb / (math.sqrt(2.0 * math.pi) * sigma)) * (w0 + sum_b_inv)
            else:
                coef1 = float(4.0 * math.pi * math.log(float(b)))
                amp_tensor = torch.full_like(bw2, fill_value=coef1)
        else:
            amp_tensor = _as_1d_tensor_keep_input(amp)
            self.self_coeff = 0.0
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

        if (param_mode == "raw") or (amp is None):
            if not use_fastsog_conv:
                amp_tensor *= bw2

        use_external = tensor_mode == "external"
        if tensor_mode == "auto":
            use_external = (not trainable) and (
                _is_differentiable_tensor(amp) or _is_differentiable_tensor(bandwidth)
            )

        if use_external:
            # Keep references to external tensors so autograd can flow back to
            # the caller-owned parameters (e.g. upstream fitting nets).
            self.amp = amp_tensor
            self.bandwidth = bw2
        else:
            self.amp = nn.Parameter(amp_tensor, requires_grad=trainable)
            self.bandwidth = nn.Parameter(bw2, requires_grad=trainable)

        self.remove_self_interaction = bool(remove_self_interaction)
        self.charge_neutral_lambda = charge_neutral_lambda
        self.use_nufft = bool(use_nufft if nufft is None else nufft)
        self.nufft_eps = float(nufft_eps)
        self.norm_factor = float(norm_factor)

        # CubeS₂ + PyTorch FFT path (replaces direct sum during training)
        self.use_cubes2_fft = bool(use_cubes2_fft)
        self.cubes2_phi_max = float(cubes2_phi_max) if cubes2_phi_max is not None else None
        self.cubes2_order = int(cubes2_order)
        self.use_quads_fft = bool(use_quads_fft)
        self.quads_order = int(quads_order)
        self.rcut = float(rcut) if rcut is not None else None
        self.b = float(b)

        self._max_cache_size = max(1, int(max_cache_size))
        self._kgrid_base_cache: Dict[
            Tuple[str, str, int, int, int],
            Tuple[torch.Tensor, torch.Tensor, Tuple[int, int, int]],
        ] = {}
        # Cache for prefiltered direct-path k-space state, keyed by box + n_dl.
        # Avoids recomputing g_cart, kfac, diag_sum when the box is unchanged
        # (NVT training, multi-system with fixed boxes).
        self._direct_state_cache: Dict[
            Tuple[str, str, int, tuple],
            Dict[str, torch.Tensor],
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
            k_cached, zero_mask_cached, output_shape_cached = cached
            # Be defensive against stale caches carried across save/load or device moves.
            if (
                k_cached.device == runtime_device
                and k_cached.dtype == real_dtype
                and zero_mask_cached.device == runtime_device
            ):
                return cached
            self._kgrid_base_cache.pop(cache_key, None)

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

    @staticmethod
    def _greedy_group_by_nloc(
        nloc_per_frame: List[int], max_waste: float
    ) -> List[List[Tuple[int, int]]]:
        """Greedy grouping of frame indices by similar atom count.

        Sorts frames by *nloc*, then walks the sorted list, adding each frame
        to the current group as long as the group's padding waste stays ≤
        *max_waste*.  Returns a list of groups; each group is a list of
        ``(original_frame_index, nloc)`` tuples.
        """
        indexed = sorted(enumerate(nloc_per_frame), key=lambda x: x[1])
        groups: List[List[Tuple[int, int]]] = []
        cur = [indexed[0]]
        cur_max = indexed[0][1]
        cur_sum = indexed[0][1]
        for idx, nl in indexed[1:]:
            new_max = max(cur_max, nl)
            new_sum = cur_sum + nl
            new_nf = len(cur) + 1
            new_waste = (new_nf * new_max - new_sum) / new_sum if new_sum > 0 else 0.0
            if new_waste <= max_waste:
                cur.append((idx, nl))
                cur_max = new_max
                cur_sum = new_sum
            else:
                groups.append(cur)
                cur = [(idx, nl)]
                cur_max = nl
                cur_sum = nl
        groups.append(cur)
        return groups

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

        # ── Precompute per-frame scalars ONCE per batch (not per frame) to avoid the
        #    ~6 GPU→CPU syncs/frame that dominate the direct k-sum for small systems
        #    (bid.item, box_hash.tolist, 3× nk.item, bandwidth.min.item, volume>eps).
        #    The k-sum FLOPs are trivial; the per-frame sync/launch overhead is not. ──
        bids_py = torch.unique(batch).tolist()
        _direct_mode = (
            cell is not None
            and not self.use_cubes2_fft
            and not (self.use_nufft and HAS_PYTORCH_FINUFFT)
        )
        vols_all = None
        periodic_py = None
        n_dl_direct = None
        nk_by_frame = None
        if cell is not None:
            vols_all = torch.abs(torch.det(cell))  # [nf], differentiable (for autograd virial)
            periodic_py = (vols_all > torch.finfo(cell.dtype).eps).tolist()
            if _direct_mode:
                n_dl_direct = self._resolve_direct_n_dl(cell.dtype, cell.device)
                nk_all = (
                    torch.norm(cell, dim=2) / n_dl_direct
                ).to(torch.int64).clamp(min=1)
                nk_by_frame = nk_all.tolist()  # [nf][3]

        # ── Fast batched direct k-sum: uniform nloc + uniform nk + canonical batch ──
        # Collapses the per-frame Python loop (dozens of tiny kernel launches × nf)
        # into single batched ops. Energy only; forces/virial via autograd exactly as
        # in the loop path. Falls through to the loop for ragged/mixed-nk batches.
        #
        # ── diagnostic gate tracer (SOG_DIAG=1 to enable) ──
        _diag = getattr(Gaussian, "_diag_count", 0)
        _print_diag = _diag < 3 and __import__("os").environ.get("SOG_DIAG", "") == "1"
        Gaussian._diag_count = _diag + 1
        if (
            _direct_mode
            and getattr(self, "_enable_batched_direct", True)
            and nk_by_frame is not None
            and len(bids_py) > 0
            and all(periodic_py)
        ):
            nf_b = len(bids_py)
            nk0 = nk_by_frame[0]
            n_total = r.shape[0]
            _nk_ok = all(nkf == nk0 for nkf in nk_by_frame)
            # ── nk-relaxed batched path: use max nk across frames for the shared k-grid.
            #     Frames with smaller nk get extra zero-masked modes (the mode_mask
            #     is computed per-frame and zeros out |k|>k_max). This handles ±2% NPT
            #     box fluctuations without falling through to the per-frame loop.
            _nk_max = tuple(int(max(nkf[i] for nkf in nk_by_frame)) for i in range(3))

            # ── per-frame atom counts (needed for mixed-nloc batching) ──
            _nloc_by_frame = torch.bincount(batch, minlength=nf_b).tolist()
            _nloc_max = max(_nloc_by_frame)
            _nloc_min = min(_nloc_by_frame)
            _nloc_ok = (_nloc_min == _nloc_max)

            if _print_diag:
                print(f"[SOG-DIAG] batched gate: _direct_mode={_direct_mode} nf={nf_b} "
                      f"nk_by_frame={nk_by_frame} nk_ok={_nk_ok} nloc_ok={_nloc_ok} "
                      f"nk_max={_nk_max} ntot={n_total} nloc_bat={_nloc_max if _nloc_ok else f'mix[{_nloc_min},{_nloc_max}]'}")

            _batched_energy: Optional[torch.Tensor] = None
            _n_dl_float = float(n_dl_direct)

            # ── Three-level batched-direct decision ──
            # Level 0: uniform nloc + canonical batch → direct (zero-overhead fast path)
            # Level 1: non-uniform nloc, waste ≤ _waste_threshold → full padding
            # Level 2: waste > _waste_threshold → greedy grouping per group
            _waste_threshold = 1.0  # max acceptable padding overhead ratio

            if _nloc_ok:
                # Level 0: uniform nloc — try canonical fast path
                _canonical = torch.equal(
                    batch,
                    torch.arange(
                        nf_b, device=batch.device, dtype=batch.dtype
                    ).repeat_interleave(_nloc_max),
                )
                if _canonical:
                    _batched_energy = self._compute_bundle_direct_batched(
                        q, r, cell, nf_b, [_nloc_max] * nf_b,
                        _nk_max, _n_dl_float,
                    )
            elif n_total > 0:
                _waste = (nf_b * _nloc_max - n_total) / n_total
                if _waste <= _waste_threshold:
                    # Level 1: full-padding batched (atoms already in canonical order from PyG)
                    if _print_diag:
                        print(f"[SOG-DIAG] batched gate → Level 1 (full pad): waste={_waste:.2%}")
                    _batched_energy = self._compute_bundle_direct_batched(
                        q, r, cell, nf_b, _nloc_by_frame,
                        _nk_max, _n_dl_float,
                    )
                else:
                    # Level 2: greedy grouping
                    _groups = self._greedy_group_by_nloc(_nloc_by_frame, _waste_threshold)
                    if _print_diag:
                        _g_sizes = [len(g) for g in _groups]
                        print(f"[SOG-DIAG] batched gate → Level 2 (grouped): waste={_waste:.2%} "
                              f"groups={_g_sizes}")
                    _batched_energy = q.new_zeros(nf_b)  # type: ignore[union-attr]
                    for _group in _groups:
                        _g_indices = [gi for gi, _ in _group]
                        _g_nlocs = [gn for _, gn in _group]
                        _g_nf = len(_group)
                        if _g_nf == 0:
                            continue
                        # Gather atoms for this group (frames may be non-contiguous
                        # after sorting by nloc)
                        _g_r_parts = []
                        _g_q_parts = []
                        _cumsum = 0
                        _cumsum_map = {}
                        for _orig_idx in range(nf_b):
                            _cumsum_map[_orig_idx] = _cumsum
                            _cumsum += _nloc_by_frame[_orig_idx]
                        for _orig_idx, _ in _group:
                            _start = _cumsum_map[_orig_idx]
                            _nl = _nloc_by_frame[_orig_idx]
                            _g_r_parts.append(r[_start : _start + _nl])
                            _g_q_parts.append(q[_start : _start + _nl])
                        _g_r = torch.cat(_g_r_parts, dim=0)
                        _g_q = torch.cat(_g_q_parts, dim=0)
                        _g_cell = cell[_g_indices]
                        _g_energy = self._compute_bundle_direct_batched(
                            _g_q, _g_r, _g_cell, _g_nf, _g_nlocs,
                            _nk_max, _n_dl_float,
                        )
                        for _j, _orig_idx in enumerate(_g_indices):
                            _batched_energy[_orig_idx] = _g_energy[_j]

            if _batched_energy is not None:
                return {
                    "energy": _batched_energy,
                    "forces": None,
                    "virial": None,
                    "used_explicit_derivatives": False,
                }

        for bid in bids_py:
            mask = batch == bid
            r_now = r[mask]
            q_now = q[mask]

            if r_now.shape[0] == 0:
                continue

            periodic = False
            box_now = None
            volume = None
            if cell is not None:
                box_now = cell[bid]
                volume = vols_all[bid]
                periodic = bool(periodic_py[bid])

            if periodic and self.use_cubes2_fft:
                # ── CubeS₂ + PyTorch FFT path (autograd-compatible) ──
                from .cubes2_fft import (
                    XI_4,
                    Cubes2FFTFunction,
                    compute_cubes2_fft as _cubes2_fft,
                )
                from .cubes2_spline import _get_xi

                assert box_now is not None

                # Multi-channel: compute each channel separately and sum.
                # This avoids spurious cross-channel interactions that would
                # arise from replicating positions on a single FFT grid.
                nq_local = q_now.shape[1] if q_now.dim() > 1 else 1

                pot_now = 0.0
                f_ch_list = []
                v_ch_list = []

                for ch in range(nq_local):
                    q_ch = q_now[:, ch] if nq_local > 1 else q_now.reshape(-1)
                    r_fft = r_now
                    q_fft = q_ch.reshape(-1)

                    state = self._prepare_triclinic_state(
                        r_fft, q_fft, box_now, compute_spectral=False,
                        compute_r_in=False,
                        _volume=volume,
                    )

                    if need_force or compute_virial:
                        result = _cubes2_fft(
                            q=state["q"].reshape(-1),
                            r=state["r_raw"],
                            cell=box_now,
                            amp=self.amp,
                            bw2=self.bandwidth,
                            volume=state["volume"],
                            diag_sum=state["diag_sum"],
                            cubes2_phi_max=self.cubes2_phi_max,
                            n_dl=self.n_dl if self.cubes2_phi_max is None else None,
                            r_c=self.rcut,
                            b=self.b,
                            xi=_get_xi(self.cubes2_order),
                            order=self.cubes2_order,
                            remove_self_interaction=self.remove_self_interaction,
                            self_coeff=self.self_coeff,
                            norm_factor=self.norm_factor,
                            compute_force=True,
                            compute_virial=compute_virial,
                        )
                        pot_now += result["energy"]
                        if result["forces"] is not None:
                            f_ch = result["forces"]
                            if f_ch.dim() == 3:
                                f_ch = f_ch.squeeze(0)
                            f_ch_list.append(f_ch)
                        if result["virial"] is not None:
                            v_ch_list.append(result["virial"])
                    else:
                        q_flat = state["q"].reshape(-1)
                        volume_val = float(state["volume"].detach().item())
                        diag_sum_val = float(state["diag_sum"].detach().item())
                        e_ch = Cubes2FFTFunction.apply(
                            q_flat,
                            state["r_raw"],
                            box_now,
                            self.amp,
                            self.bandwidth,
                            volume_val,
                            diag_sum_val,
                            self.cubes2_phi_max,
                            self.n_dl if self.cubes2_phi_max is None else None,
                            self.rcut,
                            self.b,
                            self.cubes2_order,
                            _get_xi(self.cubes2_order),
                            self.remove_self_interaction,
                            self.self_coeff,
                            self.norm_factor,
                        )
                        pot_now += e_ch

                # Sum forces and virial over channels
                if f_ch_list:
                    force_full[mask] = torch.stack(f_ch_list, dim=0).sum(dim=0)
                if v_ch_list:
                    virial_list.append(torch.stack(v_ch_list, dim=0).sum(dim=0))

            elif periodic and self.use_quads_fft:
                # ── QuadS (separable quadrature spline) + PyTorch FFT path ──
                # Exact separable influence (Form A), better energy/force than CubeS₂;
                # explicit-force path (autograd through the spread floor is truncated, so
                # forces are taken explicitly, like the CubeS₂ force branch).
                from .quads_fft import compute_quads_fft as _quads_fft

                assert box_now is not None
                nq_local = q_now.shape[1] if q_now.dim() > 1 else 1
                pot_now = 0.0
                f_ch_list = []
                v_ch_list = []
                for ch in range(nq_local):
                    q_ch = q_now[:, ch] if nq_local > 1 else q_now.reshape(-1)
                    state = self._prepare_triclinic_state(
                        r_now, q_ch.reshape(-1), box_now, compute_spectral=False,
                        compute_r_in=False, _volume=volume,
                    )
                    result = _quads_fft(
                        q=state["q"].reshape(-1),
                        r=state["r_raw"],
                        cell=box_now,
                        amp=self.amp,
                        bw2=self.bandwidth,
                        volume=state["volume"],
                        diag_sum=state["diag_sum"],
                        cubes2_phi_max=self.cubes2_phi_max,
                        n_dl=self.n_dl if self.cubes2_phi_max is None else None,
                        r_c=self.rcut,
                        b=self.b,
                        order=self.quads_order,
                        remove_self_interaction=self.remove_self_interaction,
                        self_coeff=self.self_coeff,
                        norm_factor=self.norm_factor,
                        compute_force=need_force,
                        compute_virial=compute_virial,
                    )
                    pot_now += result["energy"]
                    if result["forces"] is not None:
                        f_ch = result["forces"]
                        if f_ch.dim() == 3:
                            f_ch = f_ch.squeeze(0)
                        f_ch_list.append(f_ch)
                    if result["virial"] is not None:
                        v_ch_list.append(result["virial"])
                if f_ch_list:
                    force_full[mask] = torch.stack(f_ch_list, dim=0).sum(dim=0)
                if v_ch_list:
                    virial_list.append(torch.stack(v_ch_list, dim=0).sum(dim=0))

            elif periodic and need_force and self.use_nufft and HAS_PYTORCH_FINUFFT:
                assert box_now is not None
                state = self._prepare_triclinic_state(r_now, q_now, box_now, _volume=volume)
                pot_now, force_now, virial_now = self._compute_periodic_nufft_bundle(
                    state,
                    need_force=True,
                    need_virial=compute_virial,
                )

                if self.remove_self_interaction:
                    pot_now = pot_now - torch.sum(q_now * q_now) * state["diag_sum"]
                # Real-space self-energy (fastsog convention)
                if self.self_coeff != 0.0:
                    pot_now = pot_now - torch.sum(q_now * q_now) * self.self_coeff

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
                    _nk_ov = (
                        tuple(nk_by_frame[bid]) if nk_by_frame is not None else None
                    )
                    pot_now = self.compute_potential_triclinic(
                        r_now, q_now, box_now, _volume=volume,
                        n_dl_override=n_dl_direct, nk_override=_nk_ov,
                        use_direct_cache=False,
                    )
                else:
                    pot_now = self.compute_potential_realspace(r_now, q_now)

                if virial_list is not None:
                    virial_list.append(torch.zeros((3, 3), dtype=r.dtype, device=r.device))

            # ── Physical k=0 correction (all paths: direct, FFT, NUFFT) ──
            # k=0 is excluded in ALL paths → add it back.
            # Charge term (always on): A·Q²/(2V) — physical cross-term.
            # Self term (tied to remove_self_interaction): −A·Σq_i²/(2V).
            #
            # IMPORTANT: Do NOT use raw amp_sum = Σ_m amp_m for k=0 kernel value.
            # amp[m] grows as b^(2m) (geometric growth, e.g. 2.17e6 for M=12).
            # At k=0, exp(−½ bw²·0) = 1 for ALL m → dominated by high-m "inactive" terms.
            # At any finite k (even the smallest on the grid), exp decay suppresses
            # high-m terms, giving kfac ≈ 4π/k² consistent with Coulomb physics.
            # The raw amp_sum at k=0 is a numerical artifact of the parameterization.
            #
            # Fix: evaluate the SOG kernel at the smallest physical |k| determined
            # by the reciprocal lattice vectors: k_min = min(|b1|, |b2|, |b3|).
            #   b_i = 2π · (a_j × a_k) / volume
            # This handles both orthorhombic and triclinic boxes correctly.
            # Gives kfac_eff ≈ Σ_m amp_m·exp(−½ bw²_m · k_min²), consistent
            # with the k≠0 energy (same kernel, same convention).
            if periodic and box_now is not None:
                amp = self.amp.to(dtype=q_now.dtype, device=q_now.device)
                bw2 = self.bandwidth.to(dtype=q_now.dtype, device=q_now.device)
                # Minimum non-zero k-vector magnitude from reciprocal lattice.
                # For orthorhombic: k_min = 2π / max(Lx, Ly, Lz).
                a1, a2, a3 = box_now[0], box_now[1], box_now[2]
                b1 = 2.0 * math.pi * torch.linalg.cross(a2, a3) / volume
                b2 = 2.0 * math.pi * torch.linalg.cross(a3, a1) / volume
                b3 = 2.0 * math.pi * torch.linalg.cross(a1, a2) / volume
                k_min_sq = torch.min(torch.stack([
                    torch.dot(b1, b1),
                    torch.dot(b2, b2),
                    torch.dot(b3, b3),
                ]))
                # Effective k=0 kernel value (regularized)
                kfac_eff = (amp * torch.exp(-0.5 * bw2 * k_min_sq)).sum()

                q_sum = q_now.sum()          # total charge (sum over atoms + channels)
                e_k0_charge = kfac_eff * (q_sum ** 2) / (2.0 * volume) * self.norm_factor
                pot_now = pot_now + e_k0_charge

                if self.remove_self_interaction:
                    q_sq_sum = (q_now ** 2).sum()
                    e_k0_self = -kfac_eff * q_sq_sum / (2.0 * volume) * self.norm_factor
                    pot_now = pot_now + e_k0_self

            # Optional user override penalty (additional regularization):
            if self.charge_neutral_lambda is not None and self.charge_neutral_lambda > 0:
                e_penalty_per_ch = float(self.charge_neutral_lambda) * (q_now.mean(dim=0) ** 2).sum()
                pot_now = pot_now + e_penalty_per_ch

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

    def _resolve_direct_n_dl(
        self, dtype: torch.dtype, device: torch.device
    ) -> float:
        """Direct-k-sum ``n_dl``, computed ONCE per batch (it is constant across
        frames — depends only on the kernel bandwidths, not the box). Returns
        ``self.n_dl`` if set, else the accuracy-derived value from the narrowest
        Gaussian. Replaces a per-frame ``bandwidth.min().item()`` sync with one."""
        if self.n_dl is not None:
            return float(self.n_dl)
        bw_min = self.bandwidth.to(dtype=dtype, device=device).min().item()
        _eps = 1e-5
        _kmax = math.sqrt(2.0 * math.log(1.0 / _eps) / max(bw_min, 1e-30))
        return 2.0 * math.pi / max(_kmax, 1e-30)

    def compute_potential_triclinic(
        self,
        r_raw: torch.Tensor,
        q: torch.Tensor,
        cell_now: torch.Tensor,
        _volume: Optional[torch.Tensor] = None,
        n_dl_override: Optional[float] = None,
        nk_override: Optional[Tuple[int, int, int]] = None,
        use_direct_cache: bool = True,
    ) -> torch.Tensor:
        state = self._prepare_triclinic_state(
            r_raw, q, cell_now, _volume=_volume,
            compute_r_in=self.use_nufft and HAS_PYTORCH_FINUFFT,
            prefilter=not (self.use_nufft and HAS_PYTORCH_FINUFFT),
            n_dl_override=n_dl_override,
            nk_override=nk_override,
            use_direct_cache=use_direct_cache,
        )

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
        # Real-space self-energy (fastsog convention)
        if self.self_coeff != 0.0:
            pot = pot - torch.sum(state["q"] * state["q"]) * self.self_coeff

        return pot * self.norm_factor

    def _prepare_triclinic_state(
        self,
        r_raw: torch.Tensor,
        q: torch.Tensor,
        cell_now: torch.Tensor,
        compute_spectral: bool = True,
        compute_r_in: bool = True,
        _volume: Optional[torch.Tensor] = None,
        prefilter: bool = False,
        n_dl_override: Optional[float] = None,
        nk_override: Optional[Tuple[int, int, int]] = None,
        use_direct_cache: bool = True,
    ) -> Dict[str, torch.Tensor | Tuple[int, int, int]]:
        """Prepare triclinic state for periodic computations.

        When compute_spectral=False (FFT path), skips the expensive k-space
        kernel computation (kfac, mask, diag_sum) — the FFT solver computes
        its own spectral kernel on the FFT grid.

        When compute_r_in=False (direct path), skips fractional coordinate
        computation — r_in is only needed for the NUFFT path.

        When prefilter=True (direct path), stores flat filtered k-vectors
        and kfac instead of full 3D grids — avoids computing kfac on masked
        (~50%) grid points. The NUFFT path needs the full 3D grid.
        """
        if q.dim() == 1:
            q = q.unsqueeze(1)

        runtime_device = r_raw.device
        real_dtype = r_raw.dtype

        box = cell_now.to(dtype=real_dtype, device=runtime_device)
        if _volume is not None:
            volume = _volume
        else:
            volume = torch.det(box)
            if torch.abs(volume) <= torch.finfo(real_dtype).eps:
                raise ValueError("`cell` is singular (near-zero volume).")
            volume = torch.abs(volume)

        if compute_r_in or compute_spectral:
            cell_inv = torch.linalg.inv(box)
        if compute_r_in:
            r_frac = torch.matmul(r_raw, cell_inv)
            r_frac = torch.remainder(r_frac + 0.5, 1.0) - 0.5

            pi_tensor = torch.tensor(torch.pi, dtype=real_dtype, device=runtime_device)
            point_limit = pi_tensor - 32.0 * torch.finfo(real_dtype).eps
            r_in = torch.clamp(
                2.0 * pi_tensor * r_frac,
                min=-point_limit,
                max=point_limit,
            ).contiguous()
        else:
            pi_tensor = torch.tensor(torch.pi, dtype=real_dtype, device=runtime_device)
            r_in = torch.empty(0, dtype=real_dtype, device=runtime_device)

        if not compute_spectral:
            # Fast path for FFT: skip expensive spectral kernel computation.
            # The FFT solver computes its own kfac on the FFT grid.
            return {
                "r_raw": r_raw,
                "q": q,
                "r_in": r_in,
                "g_cart": torch.empty(0),    # not used by FFT path
                "kfac": torch.empty(0),      # not used by FFT path
                "k_mode_mask": torch.empty(0, dtype=torch.bool),  # not used
                "output_shape": (0, 0, 0),   # not used by FFT path
                "volume": volume,
                "diag_sum": torch.tensor(0.0, dtype=real_dtype, device=runtime_device),
            }

        # Auto-compute n_dl from accuracy when not set (direct k-space path).
        # n_dl_override (precomputed once per batch by the caller) skips the per-frame
        # bandwidth.min().item() GPU→CPU sync.
        if n_dl_override is not None:
            _n_dl = n_dl_override
        else:
            _n_dl = self.n_dl
            if _n_dl is None:
                bw_min = self.bandwidth.to(dtype=real_dtype, device=runtime_device).min().item()
                # ε = 1e-5: n_dl = 2π / sqrt(2·ln(1/ε) / bw_min)
                _eps = 1e-5
                _kmax = math.sqrt(2.0 * math.log(1.0 / _eps) / max(bw_min, 1e-30))
                _n_dl = 2.0 * math.pi / max(_kmax, 1e-30)

        # nk_override (precomputed once per batch) skips 3 per-frame v.item() syncs.
        if nk_override is not None:
            nk = nk_override
        else:
            norms = torch.norm(box, dim=1)
            nk = tuple(max(1, int(v.item() / _n_dl)) for v in norms)

        # ── Direct-path cache: skip g_cart/kfac/diag_sum when box is unchanged ──
        # Disabled (use_direct_cache=False) for multi-frame training batches, where
        # every frame has a distinct box so the cache never hits and the box_hash
        # .tolist() is a pure per-frame sync.
        if prefilter and use_direct_cache:
            box_hash = tuple(
                float(x)
                for x in box.detach().reshape(-1).mul(1e6).round().div(1e6).tolist()
            )
            cache_key = (self._device_key(runtime_device), str(real_dtype),
                         int(round(_n_dl, 8)), box_hash)
            cached = self._direct_state_cache.get(cache_key)
            if cached is not None:
                g_cart_c, kfac_c, diag_c = cached
                if (g_cart_c.device == runtime_device and g_cart_c.dtype == real_dtype
                        and kfac_c.device == runtime_device
                        and diag_c.device == runtime_device):
                    return {
                        "r_raw": r_raw,
                        "q": q,
                        "r_in": r_in,
                        "g_cart": g_cart_c,
                        "kfac": kfac_c,
                        "k_mode_mask": torch.empty(0, dtype=torch.bool,
                                                   device=runtime_device),
                        "output_shape": (0, 0, 0),
                        "volume": volume,
                        "diag_sum": diag_c,
                    }

        k_grid_int, zero_mask, output_shape = self._get_cached_kgrid_base(
            nk,
            runtime_device,
            real_dtype,
        )

        two_pi = 2.0 * pi_tensor
        n_dl_tensor = torch.as_tensor(_n_dl, dtype=real_dtype, device=runtime_device)
        k_sq_max = (two_pi / n_dl_tensor) ** 2

        g_cart = two_pi * torch.einsum("ik,k...->i...", cell_inv, k_grid_int)
        k_sq = torch.sum(g_cart * g_cart, dim=0)
        k_mode_mask = (~zero_mask) & (k_sq <= k_sq_max)

        amp = self.amp.to(dtype=real_dtype, device=runtime_device).view(1, -1)
        bw2 = self.bandwidth.to(dtype=real_dtype, device=runtime_device).view(1, -1)

        if prefilter:
            # Direct path: compute kfac only on valid k-points (k≠0, k≤k_max).
            # Avoids wasted exp() on ~50% of the grid that would be masked.
            mask_flat = k_mode_mask.reshape(-1)
            g_cart_out = g_cart.reshape(3, -1)[:, mask_flat]         # [3, K]
            k_sq_flat = k_sq.reshape(-1)[mask_flat]                   # [K]
            kfac_out = (amp * torch.exp(-0.5 * bw2 * k_sq_flat.unsqueeze(-1))).sum(dim=-1)  # [K]
            diag_sum = kfac_out.sum() / (2.0 * volume)
            # Cache for future frames with the same box (when enabled).
            if use_direct_cache:
                self._direct_state_cache[cache_key] = (
                    g_cart_out.detach(), kfac_out.detach(), diag_sum.detach())
                if len(self._direct_state_cache) > self._max_cache_size:
                    oldest = next(iter(self._direct_state_cache.keys()))
                    self._direct_state_cache.pop(oldest, None)
        else:
            # NUFFT path / legacy: full 3D grid with masked_fill.
            amp_3d = amp.view(1, 1, 1, -1)
            bw2_3d = bw2.view(1, 1, 1, -1)
            kfac_out = amp_3d * torch.exp(-0.5 * bw2_3d * k_sq.unsqueeze(-1))
            kfac_out = kfac_out.sum(dim=-1).masked_fill(~k_mode_mask, 0.0)
            g_cart_out = g_cart
            diag_sum = kfac_out.sum() / (2.0 * volume)

        return {
            "r_raw": r_raw,
            "q": q,
            "r_in": r_in,
            "g_cart": g_cart_out,
            "kfac": kfac_out,
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

    def _compute_bundle_direct_batched(
        self,
        q: torch.Tensor,
        r: torch.Tensor,
        cell: torch.Tensor,
        nf: int,
        nloc_per_frame: List[int],
        nk: Tuple[int, int, int],
        n_dl: float,
    ) -> torch.Tensor:
        """Batched direct k-sum over (possibly mixed-nloc) frames → per-frame
        energy [nf].  Collapses the per-frame Python loop (dozens of tiny kernel
        launches × nf) into single batched ops.  For uniform atom counts the
        standard ``.view`` fast path is used; for mixed sizes, shorter frames are
        zero-padded to ``max(nloc_per_frame)`` — the zero-charge padding atoms
        contribute identically zero to the structure factor, so the energy is
        mathematically exact.  Differentiable w.r.t. *r* and *cell*."""
        _nloc_max = max(nloc_per_frame)
        _nloc_uniform = min(nloc_per_frame) == _nloc_max
        device = r.device
        dtype = r.dtype
        two_pi = 2.0 * math.pi

        if q.dim() == 1:
            q = q.unsqueeze(1)
        nch = q.shape[1]
        r_f = q.new_zeros(0)  # placeholder for type checkers
        _nloc_tensor = None   # per-frame atom counts as a tensor (used for mean)
        if _nloc_uniform:
            r_f = r.view(nf, _nloc_max, 3)
            q_f = q.view(nf, _nloc_max, nch)
        else:
            r_f = q.new_zeros(nf, _nloc_max, 3)
            q_f = q.new_zeros(nf, _nloc_max, nch)
            offset = 0
            for _i, _ni in enumerate(nloc_per_frame):
                if _ni > 0:
                    r_f[_i, :_ni] = r[offset : offset + _ni]
                    q_f[_i, :_ni] = q[offset : offset + _ni]
                    offset += _ni
            _nloc_tensor = torch.tensor(nloc_per_frame, device=device, dtype=dtype)

        volume = torch.abs(torch.det(cell))          # [nf], differentiable
        inv2v = 1.0 / (2.0 * volume)                 # [nf]
        cell_inv = torch.linalg.inv(cell)            # [nf,3,3]

        k_grid_int, zero_mask, _ = self._get_cached_kgrid_base(nk, device, dtype)
        k_int = k_grid_int.reshape(3, -1)            # [3, G]
        zero_flat = zero_mask.reshape(-1)            # [G]

        g_cart = two_pi * torch.einsum("fik,kG->fiG", cell_inv, k_int)  # [nf,3,G]
        k_sq = (g_cart * g_cart).sum(dim=1)          # [nf,G]
        k_sq_max = (two_pi / n_dl) ** 2
        mode_mask = (~zero_flat).unsqueeze(0) & (k_sq <= k_sq_max)      # [nf,G]

        amp = self.amp.to(dtype=dtype, device=device).view(1, 1, -1)    # [1,1,M]
        bw2 = self.bandwidth.to(dtype=dtype, device=device).view(1, 1, -1)
        kfac = (amp * torch.exp(-0.5 * bw2 * k_sq.unsqueeze(-1))).sum(dim=-1)  # [nf,G]
        kfac = kfac.masked_fill(~mode_mask, 0.0)

        # structure factor S(k) per frame/channel
        k_dot_r = torch.einsum("fnd,fdG->fnG", r_f, g_cart)            # [nf,nloc,G]
        cos_kr = torch.cos(k_dot_r)
        sin_kr = torch.sin(k_dot_r)
        s_real = torch.einsum("fnc,fnG->fcG", q_f, cos_kr)            # [nf,nch,G]
        s_imag = torch.einsum("fnc,fnG->fcG", q_f, sin_kr)
        s_sq = s_real.square() + s_imag.square()                      # [nf,nch,G]

        # full-grid reciprocal sum (factor 1) == loop's half-sphere × 2
        e_recip = (kfac.unsqueeze(1) * s_sq).sum(dim=(1, 2)) * inv2v   # [nf]
        diag_sum = kfac.sum(dim=1) * inv2v                            # [nf]
        q_sq_sum = (q_f * q_f).sum(dim=(1, 2))                        # [nf]

        pot = e_recip
        if self.remove_self_interaction:
            pot = pot - q_sq_sum * diag_sum
        if self.self_coeff != 0.0:
            pot = pot - q_sq_sum * self.self_coeff
        pot = pot * self.norm_factor

        # ── physical k=0 correction (regularized k_min from reciprocal lattice) ──
        a1, a2, a3 = cell[:, 0], cell[:, 1], cell[:, 2]
        vol_c = volume.unsqueeze(1)
        b1 = two_pi * torch.linalg.cross(a2, a3, dim=1) / vol_c
        b2 = two_pi * torch.linalg.cross(a3, a1, dim=1) / vol_c
        b3 = two_pi * torch.linalg.cross(a1, a2, dim=1) / vol_c
        k_min_sq = torch.stack(
            [(b1 * b1).sum(1), (b2 * b2).sum(1), (b3 * b3).sum(1)], dim=1
        ).min(dim=1).values                                          # [nf]
        amp0 = self.amp.to(dtype=dtype, device=device).view(1, -1)
        bw0 = self.bandwidth.to(dtype=dtype, device=device).view(1, -1)
        kfac_eff = (amp0 * torch.exp(-0.5 * bw0 * k_min_sq.unsqueeze(1))).sum(dim=1)  # [nf]
        q_sum = q_f.sum(dim=(1, 2))                                   # [nf]
        pot = pot + kfac_eff * q_sum.square() * inv2v * self.norm_factor
        if self.remove_self_interaction:
            pot = pot - kfac_eff * q_sq_sum * inv2v * self.norm_factor
        if self.charge_neutral_lambda is not None and self.charge_neutral_lambda > 0:
            _q_mean = (q_f.sum(dim=(1, 2)) / (_nloc_tensor if _nloc_tensor is not None else _nloc_max)).unsqueeze(1)  # [nf,1]
            pot = pot + float(self.charge_neutral_lambda) * (_q_mean ** 2).sum(dim=1)
        return pot  # [nf]

    def _compute_periodic_direct(
        self,
        r_raw: torch.Tensor,
        q: torch.Tensor,
        g_cart: torch.Tensor,
        kfac: torch.Tensor,
        volume: torch.Tensor,
        k_mode_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Detect flat vs 3D input: flat tensors are [3,K] / [K] (prefiltered),
        # 3D tensors are [3,nx,ny,nz] / [nx,ny,nz] (legacy / NUFFT path).
        if g_cart.dim() == 2:
            # Prefiltered flat input — k=0 and k>k_max already excluded.
            kvec = g_cart.transpose(0, 1)          # [K, 3]
            kfac_flat = kfac                        # [K]
        else:
            # Legacy 3D input — apply k_mode_mask.
            kvec = g_cart.reshape(3, -1).transpose(0, 1)
            kfac_flat = kfac.reshape(-1)

            if k_mode_mask is None:
                mask = kfac_flat != 0
            else:
                mask = k_mode_mask.reshape(-1)

            if not torch.any(mask):
                return torch.zeros((), dtype=r_raw.dtype, device=r_raw.device)

            kvec = kvec[mask]
            kfac_flat = kfac_flat[mask]

        if kvec.shape[0] == 0:
            return torch.zeros((), dtype=r_raw.dtype, device=r_raw.device)

        # Exploit Fourier space symmetry: K(k²)=K(|-k|²) and |S(-k)|²=|S(k)|².
        # Keep only half-sphere k-vectors, multiply result by 2.
        # k=0 is already excluded, so every remaining k has a distinct -k partner.
        half_mask = (
            (kvec[:, 0] > 0)
            | ((kvec[:, 0] == 0) & (kvec[:, 1] > 0))
            | ((kvec[:, 0] == 0) & (kvec[:, 1] == 0) & (kvec[:, 2] > 0))
        )
        kvec = kvec[half_mask]
        kfac_flat = kfac_flat[half_mask]
        # Each selected k represents a (k, -k) pair.
        energy_factor = 2.0

        k_dot_r = torch.matmul(r_raw, kvec.transpose(0, 1))
        cos_k_dot_r = torch.cos(k_dot_r)
        sin_k_dot_r = torch.sin(k_dot_r)

        # matmul structure factor: q.T @ cos  → [C,K] instead of broadcast [N,C,K]
        s_real = torch.matmul(q.transpose(0, 1).to(cos_k_dot_r.dtype), cos_k_dot_r)
        s_imag = torch.matmul(q.transpose(0, 1).to(sin_k_dot_r.dtype), sin_k_dot_r)
        s_sq = s_real.square() + s_imag.square()

        return energy_factor * (kfac_flat.unsqueeze(0) * s_sq).sum() / (2.0 * volume)

    def __repr__(self) -> str:
        return (
            "Gaussian("
            f"n_dl={self.n_dl}, "
            f"remove_self_interaction={self.remove_self_interaction}, "
            f"use_nufft={self.use_nufft and HAS_PYTORCH_FINUFFT}"
            ")"
        )
