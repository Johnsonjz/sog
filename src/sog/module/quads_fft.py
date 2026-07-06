"""QuadS³ — Separable Quadrature Spline FFT solver for anisotropic cells.

QuadS³ = product of 1D QuadS per dimension with independent ξ_i = σ_s/Δ_i.
Correctly handles orthorhombic cells where Δ_x ≠ Δ_y ≠ Δ_z.

Reference: Predescu, Bergdorf, Shaw, J. Chem. Phys. 153, 224117 (2020)
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch

from .cubes2_spline import _get_xi, _get_num_nodes
from .cubes2_fft import _resolve_grid, _next_fft_friendly, _compute_phi_max
from .gaussian import RCUT_TO_SIGMA


# ── 1D QuadS₄ weights (4th order, v=2, 4 nodes: offsets -1, 0, 1, 2) ──

def _quads_1d_L(theta: torch.Tensor, xi: float) -> torch.Tensor:
    """L(θ, ξ) for 1D QuadS — left-center weight."""
    xi2 = xi * xi
    return (-0.5 * theta**3 + 0.5 * theta**2
            - (9.0 * xi2 - 2.0) / 6.0 * theta
            + xi2 / 2.0)


def _quads_1d_R(theta: torch.Tensor, xi: float) -> torch.Tensor:
    """R(θ, ξ) for 1D QuadS — right-center weight."""
    xi2 = xi * xi
    return (1.0 / 6.0) * theta**3 + (3.0 * xi2 - 1.0) / 6.0 * theta


def _quads_1d_weights(theta: torch.Tensor, xi: float) -> Tuple[torch.Tensor, ...]:
    """4-node 1D QuadS weights at fractional coordinate θ ∈ [0,1).

    Returns (w_minus1, w_0, w_1, w_2) for grid offsets [-1, 0, 1, 2].
    """
    one_minus_theta = 1.0 - theta
    return (
        _quads_1d_R(one_minus_theta, xi),   # offset -1: far left
        _quads_1d_L(theta, xi),             # offset  0: left-center
        _quads_1d_R(theta, xi),             # offset  1: right-center
        _quads_1d_L(one_minus_theta, xi),   # offset  2: far right
    )


# ── Per-dimension ξ computation ──

def _quads_xi_per_dim(lx: float, ly: float, lz: float,
                      nx: int, ny: int, nz: int,
                      xi_opt: float = 2.0 / math.sqrt(15)) -> Tuple[float, float, float]:
    """Compute per-dimension ξ_i = ξ_opt * Δ_avg / Δ_i.

    For cubic cells, ξ_x = ξ_y = ξ_z = ξ_opt.
    """
    dx = lx / nx
    dy = ly / ny
    dz = lz / nz
    delta_avg = (lx * ly * lz / (nx * ny * nz)) ** (1.0 / 3.0)
    xi_x = xi_opt * delta_avg / dx
    xi_y = xi_opt * delta_avg / dy
    xi_z = xi_opt * delta_avg / dz
    return xi_x, xi_y, xi_z


# ── QuadS³ charge spreading ──

def quads_spread(q: torch.Tensor, r_frac: torch.Tensor,
                 nx: int, ny: int, nz: int,
                 xi_x: float, xi_y: float, xi_z: float) -> torch.Tensor:
    """Spread charges onto 3D grid using separable QuadS³ (64 nodes/atom).

    Args:
        q: charges [N]
        r_frac: fractional coordinates [N, 3] in [0,1)³
        nx, ny, nz: grid dimensions
        xi_x, xi_y, xi_z: per-dimension ξ parameters

    Returns:
        rho_grid: [nz, ny, nx] charge density
    """
    N = q.shape[0]
    dtype = q.dtype
    device = q.device

    rho = torch.zeros(nz, ny, nx, dtype=dtype, device=device)

    # Fractional grid coordinates
    fx = r_frac[:, 0] * nx
    fy = r_frac[:, 1] * ny
    fz = r_frac[:, 2] * nz

    # Integer cell indices
    ix0 = torch.floor(fx).long()
    iy0 = torch.floor(fy).long()
    iz0 = torch.floor(fz).long()

    # Fractional offsets θ ∈ [0,1)
    theta_x = fx - ix0.float()
    theta_y = fy - iy0.float()
    theta_z = fz - iz0.float()

    # 1D weights for each atom [N]
    wx_m1, wx_0, wx_1, wx_2 = _quads_1d_weights(theta_x, xi_x)
    wy_m1, wy_0, wy_1, wy_2 = _quads_1d_weights(theta_y, xi_y)
    wz_m1, wz_0, wz_1, wz_2 = _quads_1d_weights(theta_z, xi_z)

    # Offsets: -1, 0, 1, 2
    offsets = torch.tensor([-1, 0, 1, 2], device=device).long()
    wx_all = torch.stack([wx_m1, wx_0, wx_1, wx_2], dim=1)  # [N, 4]
    wy_all = torch.stack([wy_m1, wy_0, wy_1, wy_2], dim=1)  # [N, 4]
    wz_all = torch.stack([wz_m1, wz_0, wz_1, wz_2], dim=1)  # [N, 4]

    # Scatter-add to grid: 4×4×4 = 64 contributions per atom
    for dx_idx in range(4):
        ix = (ix0 + offsets[dx_idx]) % nx
        wx = wx_all[:, dx_idx]  # [N]
        for dy_idx in range(4):
            iy = (iy0 + offsets[dy_idx]) % ny
            wxy = wx * wy_all[:, dy_idx]  # [N]
            for dz_idx in range(4):
                iz = (iz0 + offsets[dz_idx]) % nz
                w = wxy * wz_all[:, dz_idx]  # [N]
                rho.index_put_((iz, iy, ix), w * q, accumulate=True)

    return rho


# ── QuadS³ influence function (MC sampling) ──

_QUADS_INFLUENCE_CACHE: Dict = {}


def precompute_quads_influence(
    nx: int, ny: int, nz: int,
    lx: float, ly: float, lz: float,
    xi_x: float, xi_y: float, xi_z: float,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float64,
    seed: int = 42,
    n_samples: int = 256,
) -> torch.Tensor:
    """MC-computed |Φ(k)|² for QuadS³ influence function deconvolution."""
    cache_key = (nx, ny, nz, round(lx, 6), round(ly, 6), round(lz, 6),
                 round(xi_x, 6), round(xi_y, 6), round(xi_z, 6),
                 str(device), str(dtype), seed, n_samples)
    cached = _QUADS_INFLUENCE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    rng = torch.Generator(device=device)
    rng.manual_seed(seed)

    influence_acc = torch.zeros(nz, ny, nx // 2 + 1, dtype=dtype, device=device)

    for _ in range(n_samples):
        theta_x = torch.rand(1, generator=rng, dtype=dtype, device=device).item()
        theta_y = torch.rand(1, generator=rng, dtype=dtype, device=device).item()
        theta_z = torch.rand(1, generator=rng, dtype=dtype, device=device).item()

        # Place single point charge at random fractional position
        r_frac = torch.tensor([[theta_x, theta_y, theta_z]], dtype=dtype, device=device)
        q = torch.ones(1, dtype=dtype, device=device)

        window = quads_spread(q, r_frac, nx, ny, nz, xi_x, xi_y, xi_z)
        window_k = torch.fft.rfftn(window)
        influence_acc += (window_k.real ** 2 + window_k.imag ** 2).to(dtype=dtype)

    influence_acc = influence_acc / n_samples
    influence_acc = influence_acc.clamp(min=1e-20)
    _QUADS_INFLUENCE_CACHE[cache_key] = influence_acc
    return influence_acc


# ── QuadS³ FFT solver (analogous to compute_cubes2_fft) ──

def compute_quads_fft(
    q: torch.Tensor,
    r: torch.Tensor,
    cell: torch.Tensor,
    amp: torch.Tensor,
    bw2: torch.Tensor,
    volume: torch.Tensor,
    diag_sum: torch.Tensor,
    cubes2_phi_max: Optional[float],
    n_dl: Optional[float],
    r_c: float,
    b: float,
    order: int = 4,
    remove_self_interaction: bool = True,
    self_coeff: float = 0.0,
    norm_factor: float = 1.0,
    compute_force: bool = False,
    compute_virial: bool = False,
) -> Dict[str, Optional[torch.Tensor]]:
    """Compute SOG long-range energy/forces using QuadS³ + FFT.

    Same interface as compute_cubes2_fft but uses separable QuadS³
    with per-dimension ξ for anisotropic cell handling.
    """
    runtime_device = r.device
    real_dtype = r.dtype

    lx = float(torch.norm(cell[:, 0]).item())
    ly = float(torch.norm(cell[:, 1]).item())
    lz = float(torch.norm(cell[:, 2]).item())
    vol_val = float(volume.detach().item())

    nx, ny, nz, k_sq_max = _resolve_grid(lx, ly, lz,
                                          n_dl=n_dl,
                                          cubes2_phi_max=cubes2_phi_max,
                                          r_c=r_c, b=b, spline_order=order)

    xi_opt = 2.0 / math.sqrt(15) if order == 4 else _get_xi(order)
    xi_x, xi_y, xi_z = _quads_xi_per_dim(lx, ly, lz, nx, ny, nz, xi_opt)

    # ── Fractional coordinates ──
    cell_inv = torch.linalg.inv(cell)
    r_frac = torch.matmul(r, cell_inv).contiguous()

    # ── k-space kernel ──
    two_pi = 2.0 * math.pi
    MX = torch.arange(nx // 2 + 1, device=runtime_device, dtype=real_dtype).view(1, -1, 1, 1)
    MY = torch.arange(ny, device=runtime_device, dtype=real_dtype).view(1, 1, -1, 1)
    MZ = torch.arange(nz, device=runtime_device, dtype=real_dtype).view(1, 1, 1, -1)

    MY_r = torch.where(MY > ny // 2, MY - ny, MY)
    MZ_r = torch.where(MZ > nz // 2, MZ - nz, MZ)

    KX = two_pi * (MX * cell_inv[0, 0] + MY_r * cell_inv[1, 0] + MZ_r * cell_inv[2, 0])
    KY = two_pi * (MX * cell_inv[0, 1] + MY_r * cell_inv[1, 1] + MZ_r * cell_inv[2, 1])
    KZ = two_pi * (MX * cell_inv[0, 2] + MY_r * cell_inv[1, 2] + MZ_r * cell_inv[2, 2])
    k_sq = (KX ** 2 + KY ** 2 + KZ ** 2).squeeze(0).permute(2, 1, 0)

    amp_dev = amp.to(device=runtime_device, dtype=real_dtype)
    bw2_dev = bw2.to(device=runtime_device, dtype=real_dtype)
    kfac_fft = (
        amp_dev.view(1, 1, 1, -1)
        * torch.exp(-0.5 * bw2_dev.view(1, 1, 1, -1) * k_sq.unsqueeze(-1))
    ).sum(dim=-1)
    kfac_fft[0, 0, 0] = 0.0
    kfac_fft = kfac_fft.masked_fill(k_sq > k_sq_max, 0.0)

    # ── QuadS³ influence function ──
    with torch.no_grad():
        influence_sq = precompute_quads_influence(
            nx, ny, nz, lx, ly, lz, xi_x, xi_y, xi_z,
            device=runtime_device, dtype=torch.float64,
        ).to(dtype=real_dtype)

    # rfftn weight (same as CubeS₂ path)
    _qw = torch.ones(1, 1, nx // 2 + 1, dtype=real_dtype, device=runtime_device)
    if nx % 2 == 0:
        _qw[:, :, 1:nx // 2] = 2.0
    else:
        _qw[:, :, 1:] = 2.0

    green_k = kfac_fft / influence_sq.clamp(min=1e-20)

    # Self-interaction (with rfftn weight)
    diag_sum_fft = (_qw * kfac_fft).sum() / (2.0 * vol_val)

    # ── Spread + FFT + Energy ──
    rho_grid = quads_spread(q, r_frac, nx, ny, nz, xi_x, xi_y, xi_z)
    rho_k = torch.fft.rfftn(rho_grid)
    rho_sq = rho_k.real ** 2 + rho_k.imag ** 2

    energy = (_qw * green_k * rho_sq).sum() / (2.0 * vol_val)
    if remove_self_interaction:
        energy = energy - (q * q).sum() * diag_sum_fft
    if self_coeff != 0.0:
        energy = energy - (q * q).sum() * self_coeff
    energy = energy * norm_factor

    result: Dict[str, Optional[torch.Tensor]] = {
        "energy": energy, "forces": None, "virial": None,
    }

    # ── Explicit forces ──
    if compute_force or compute_virial:
        conv_k = (_qw * green_k).to(dtype=torch.complex128) * rho_k.to(dtype=torch.complex128)

        KX3 = KX.squeeze(0).permute(2, 1, 0).to(dtype=torch.complex128)
        KY3 = KY.squeeze(0).permute(2, 1, 0).to(dtype=torch.complex128)
        KZ3 = KZ.squeeze(0).permute(2, 1, 0).to(dtype=torch.complex128)

        grad_kx = 1j * KX3 * conv_k
        grad_ky = 1j * KY3 * conv_k
        grad_kz = 1j * KZ3 * conv_k

        force_grid_x = torch.fft.irfftn(grad_kx, s=(nz, ny, nx))
        force_grid_y = torch.fft.irfftn(grad_ky, s=(nz, ny, nx))
        force_grid_z = torch.fft.irfftn(grad_kz, s=(nz, ny, nx))

        force_grid = torch.stack(
            [force_grid_x, force_grid_y, force_grid_z], dim=0,
        ).to(dtype=real_dtype)

        force = _quads_interpolate(q, r_frac, force_grid, nx, ny, nz,
                                    xi_x, xi_y, xi_z)
        force = force * float(nx * ny * nz) / vol_val * norm_factor
        result["forces"] = force

        if compute_virial:
            result["virial"] = torch.einsum("ni,nj->ij", force, r)

    return result


def _quads_interpolate(q: torch.Tensor, r_frac: torch.Tensor,
                       grid: torch.Tensor,
                       nx: int, ny: int, nz: int,
                       xi_x: float, xi_y: float, xi_z: float) -> torch.Tensor:
    """Interpolate forces from grid back to atoms using QuadS³ weights."""
    N = q.shape[0]
    dtype = grid.dtype
    device = grid.device

    fx = r_frac[:, 0] * nx
    fy = r_frac[:, 1] * ny
    fz = r_frac[:, 2] * nz

    ix0 = torch.floor(fx).long()
    iy0 = torch.floor(fy).long()
    iz0 = torch.floor(fz).long()

    theta_x = fx - ix0.float()
    theta_y = fy - iy0.float()
    theta_z = fz - iz0.float()

    wx_m1, wx_0, wx_1, wx_2 = _quads_1d_weights(theta_x, xi_x)
    wy_m1, wy_0, wy_1, wy_2 = _quads_1d_weights(theta_y, xi_y)
    wz_m1, wz_0, wz_1, wz_2 = _quads_1d_weights(theta_z, xi_z)

    offsets = torch.tensor([-1, 0, 1, 2], device=device).long()
    wx_all = torch.stack([wx_m1, wx_0, wx_1, wx_2], dim=1)
    wy_all = torch.stack([wy_m1, wy_0, wy_1, wy_2], dim=1)
    wz_all = torch.stack([wz_m1, wz_0, wz_1, wz_2], dim=1)

    force = torch.zeros(N, 3, dtype=dtype, device=device)

    for dx_idx in range(4):
        ix = (ix0 + offsets[dx_idx]) % nx
        for dy_idx in range(4):
            iy = (iy0 + offsets[dy_idx]) % ny
            for dz_idx in range(4):
                iz = (iz0 + offsets[dz_idx]) % nz
                w = wx_all[:, dx_idx] * wy_all[:, dy_idx] * wz_all[:, dz_idx]
                for d in range(3):
                    force[:, d] += w * grid[d, iz, iy, ix]

    return force
