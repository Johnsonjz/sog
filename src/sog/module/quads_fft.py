"""QuadS³ — Separable Quadrature-Spline FFT solver for SOG long-range electrostatics.

QuadS is the separable (tensor-product) midtown spline. Because it is separable, its
Fourier influence factorizes, |Ŵ(k)|² = |Ŵx(kx)|²·|Ŵy(ky)|²·|Ŵz(kz)|², and the exact
SPME-style deconvolution (Form A) is *stable* — divide the kernel by the exact window
influence, `green = K(k²)/|Ŵ(k)|²`. The window then cancels exactly in the energy
(`green·|ρ̂|² = K·|S|²`), so the mesh energy AND stress reduce to the direct k-sum
truncated at Nyquist. This is the property CubeS₂ (non-separable window) lacks: its
influence does not factorize and dividing by it is unstable, so CubeS₂ must use the
approximate Form-B variance-subtraction deconv (accurate for energy, ~2.5% for the stress).

Reference: Predescu, Bergdorf, Shaw, J. Chem. Phys. 153, 224117 (2020); §A/§8.3 of
papers/midtown/midtown-sog.md. Weights come from `quads_spline.py` (order 4 & 6).

Public API preserved: `compute_quads_fft(...)` (no `xi` kwarg — ξ○ is per-order),
`precompute_quads_influence(...)` (now the exact analytic influence), `_QUADS_INFLUENCE_CACHE`.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch

from .cubes2_fft import _resolve_grid, mesh_reciprocal_virial
from .quads_spline import (
    quads_xi, quads_offsets, quads_num_nodes_1d, quads_offset_polys, quads_1d_weights,
)


# ── Separable charge spreading (batched scatter_add over the (2ν)³ nodes) ──

def quads_spread(q: torch.Tensor, r_frac: torch.Tensor,
                 nx: int, ny: int, nz: int,
                 xi: float, order: int,
                 rho_scale: float = 1.0) -> torch.Tensor:
    """Spread charges onto the [nz,ny,nx] grid using separable QuadS (2ν nodes/dim)."""
    offs = quads_offsets(order)
    K = len(offs)
    device, dtype = q.device, q.dtype
    q = q.reshape(-1)
    N = q.shape[0]

    fx = r_frac[:, 0] * nx
    fy = r_frac[:, 1] * ny
    fz = r_frac[:, 2] * nz
    ix0 = torch.floor(fx).long()
    iy0 = torch.floor(fy).long()
    iz0 = torch.floor(fz).long()
    tx = fx - ix0.to(dtype)
    ty = fy - iy0.to(dtype)
    tz = fz - iz0.to(dtype)

    wx = quads_1d_weights(tx, xi, order)  # [N,K]
    wy = quads_1d_weights(ty, xi, order)
    wz = quads_1d_weights(tz, xi, order)

    offs_t = torch.tensor(offs, device=device, dtype=torch.long)
    gx = torch.remainder(ix0[:, None] + offs_t[None, :], nx)  # [N,K]
    gy = torch.remainder(iy0[:, None] + offs_t[None, :], ny)
    gz = torch.remainder(iz0[:, None] + offs_t[None, :], nz)

    # 3-D node weights and flat indices via broadcasting: [N, Kx, Ky, Kz]
    w3 = wx[:, :, None, None] * wy[:, None, :, None] * wz[:, None, None, :]
    idx = (gz[:, None, None, :] * (ny * nx)
           + gy[:, None, :, None] * nx
           + gx[:, :, None, None])  # [N,Kx,Ky,Kz]

    rho = torch.zeros(nz * ny * nx, dtype=dtype, device=device)
    vals = (rho_scale * q)[:, None, None, None] * w3
    rho.scatter_add_(0, idx.reshape(-1), vals.reshape(-1))
    return rho.reshape(nz, ny, nx)


def quads_interpolate(q: torch.Tensor, r_frac: torch.Tensor,
                      grad_grid: torch.Tensor,
                      nx: int, ny: int, nz: int,
                      xi: float, order: int) -> torch.Tensor:
    """Gather forces from the [3,nz,ny,nx] grid (force = −q·Σ_nodes w·∂grid)."""
    offs = quads_offsets(order)
    device, dtype = q.device, grad_grid.dtype
    q = q.reshape(-1)

    fx = r_frac[:, 0] * nx
    fy = r_frac[:, 1] * ny
    fz = r_frac[:, 2] * nz
    ix0 = torch.floor(fx).long()
    iy0 = torch.floor(fy).long()
    iz0 = torch.floor(fz).long()
    tx = fx - ix0.to(dtype)
    ty = fy - iy0.to(dtype)
    tz = fz - iz0.to(dtype)

    wx = quads_1d_weights(tx, xi, order)
    wy = quads_1d_weights(ty, xi, order)
    wz = quads_1d_weights(tz, xi, order)

    offs_t = torch.tensor(offs, device=device, dtype=torch.long)
    gx = torch.remainder(ix0[:, None] + offs_t[None, :], nx)
    gy = torch.remainder(iy0[:, None] + offs_t[None, :], ny)
    gz = torch.remainder(iz0[:, None] + offs_t[None, :], nz)

    w3 = wx[:, :, None, None] * wy[:, None, :, None] * wz[:, None, None, :]
    idx = (gz[:, None, None, :] * (ny * nx)
           + gy[:, None, :, None] * nx
           + gx[:, :, None, None]).reshape(len(q), -1)   # [N, K³]
    w3f = w3.reshape(len(q), -1)

    out = torch.empty(len(q), 3, dtype=dtype, device=device)
    for d in range(3):
        gflat = grad_grid[d].reshape(-1)
        out[:, d] = -(w3f * gflat[idx]).sum(dim=1) * q
    return out


# ── 1-D Fourier integrals I_p(α)=∫₀¹ tᵖ e^{iαt} dt (vectorized, Taylor+recurrence) ──

def _I_p(alpha: torch.Tensor, pmax: int) -> List[torch.Tensor]:
    """Return [I_0, …, I_pmax] as complex tensors over the real mode array `alpha`."""
    small = alpha.abs() < 1e-3
    a_safe = torch.where(small, torch.ones_like(alpha), alpha)
    ia = (1j * alpha.to(torch.complex128))
    eia = torch.exp(ia)
    inv_ia = 1.0 / (1j * a_safe.to(torch.complex128))

    # Stable recurrence for |α| not small: I_0=(e^{iα}-1)/(iα), I_p=(e^{iα}-p·I_{p-1})/(iα)
    I = [(eia - 1.0) * inv_ia]
    for p in range(1, pmax + 1):
        I.append((eia - p * I[-1]) * inv_ia)

    # Taylor for small |α|: I_p = Σ_n (iα)^n / (n!·(p+n+1))
    for p in range(pmax + 1):
        s = torch.ones_like(ia) / (p + 1)
        pw = torch.ones_like(ia)
        fact = 1.0
        for n in range(1, 12):
            pw = pw * ia
            fact *= n
            s = s + pw / (fact * (p + n + 1))
        I[p] = torch.where(small, s, I[p])
    return I


# ── Exact separable QuadS influence |Ŵ(k)|² (Form A, SPME-style, deterministic) ──

_QUADS_INFLUENCE_CACHE: Dict = {}


def _phi1d_abs2(modes: torch.Tensor, N: int, order: int, xi: float,
                offs: List[int], polys: Dict) -> torch.Tensor:
    """|Ŵ₁(k)|² for a 1-D axis at the given signed integer modes."""
    deg = order  # 2ν monomial terms
    alpha = (2.0 * math.pi / N) * modes.to(torch.float64)
    Ip = _I_p(alpha, deg - 1)                       # I_p(α) = ∫₀¹ tᵖ e^{+iαt} dt
    phi = torch.zeros_like(alpha, dtype=torch.complex128)
    for off in offs:
        c = polys[off]
        g = torch.zeros_like(alpha, dtype=torch.complex128)
        for p in range(deg):
            if c[p] != 0.0:
                g = g + float(c[p]) * Ip[p]
        # Ŵ(k)=Σ_off e^{+iα·off}·∫c_off(θ)e^{−iαθ}dθ ; the θ-integral is conj(g) (c real, FFT e^{−i}).
        phi = phi + torch.exp(1j * alpha * off) * g.conj()
    return (phi.real ** 2 + phi.imag ** 2)


def precompute_quads_influence_analytic(
    nx: int, ny: int, nz: int,
    lx: float, ly: float, lz: float,
    order: int,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Exact separable |Ŵ(k)|² on the rfft grid [nz, ny, nx//2+1]."""
    xi = quads_xi(order)
    key = (nx, ny, nz, order, round(xi, 12), str(device), str(dtype))
    cached = _QUADS_INFLUENCE_CACHE.get(key)
    if cached is not None:
        return cached

    offs = quads_offsets(order)
    polys = quads_offset_polys(xi, order)
    dev64 = torch.device("cpu")  # small 1-D work; assemble on device at the end

    mx = torch.arange(0, nx // 2 + 1, device=dev64)                      # rfft x: ≥0
    iy = torch.arange(ny, device=dev64); my = torch.where(iy > ny // 2, iy - ny, iy)
    iz = torch.arange(nz, device=dev64); mz = torch.where(iz > nz // 2, iz - nz, iz)

    Sx = _phi1d_abs2(mx, nx, order, xi, offs, polys)
    Sy = _phi1d_abs2(my, ny, order, xi, offs, polys)
    Sz = _phi1d_abs2(mz, nz, order, xi, offs, polys)

    infl = (Sz[:, None, None] * Sy[None, :, None] * Sx[None, None, :]).clamp(min=1e-20)
    infl = infl.to(device=device, dtype=dtype)
    _QUADS_INFLUENCE_CACHE[key] = infl
    return infl


def precompute_quads_influence(nx, ny, nz, lx, ly, lz, xi_x=None, xi_y=None, xi_z=None,
                               device=torch.device("cpu"), dtype=torch.float64,
                               order: int = 4, **_ignored):
    """Backward-compatible name → exact analytic separable influence (was MC)."""
    return precompute_quads_influence_analytic(nx, ny, nz, lx, ly, lz, order, device, dtype)


# ── Main QuadS FFT solver (mirrors compute_cubes2_fft; Form-A green) ──

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
    """SOG long-range energy/forces/virial via QuadS spreading + FFT, exact Form-A green."""
    device, dtype = r.device, r.dtype
    if q.dim() > 1:
        q = q.reshape(-1)
    xi = quads_xi(order)

    # ── Grid sizing (φ-based or n_dl-based) — identical to the CubeS₂ path ──
    with torch.no_grad():
        box_norms = torch.norm(cell, dim=1)
        lx = float(box_norms[0].item()); ly = float(box_norms[1].item()); lz = float(box_norms[2].item())
        nx, ny, nz, k_sq_max = _resolve_grid(lx, ly, lz, n_dl=n_dl,
                                             cubes2_phi_max=cubes2_phi_max, r_c=r_c, b=b,
                                             spline_order=order, min_grid=8)

    cell_inv = torch.linalg.inv(cell)
    r_frac = torch.remainder(r @ cell_inv, 1.0)

    # ── k-space grid + spectral kernel (triclinic-correct, same convention as CubeS₂) ──
    mx = torch.arange(0, nx // 2 + 1, device=device, dtype=torch.float64)
    my = torch.cat([torch.arange(0, ny // 2 + 1, device=device, dtype=torch.float64),
                    torch.arange(-ny // 2 + 1, 0, device=device, dtype=torch.float64)]) if ny > 1 \
        else torch.zeros(1, device=device, dtype=torch.float64)
    mz = torch.cat([torch.arange(0, nz // 2 + 1, device=device, dtype=torch.float64),
                    torch.arange(-nz // 2 + 1, 0, device=device, dtype=torch.float64)]) if nz > 1 \
        else torch.zeros(1, device=device, dtype=torch.float64)
    MX, MY, MZ = torch.meshgrid(mx, my, mz, indexing="ij")
    two_pi = 2.0 * math.pi
    KX = two_pi * (MX * cell_inv[0, 0] + MY * cell_inv[1, 0] + MZ * cell_inv[2, 0])
    KY = two_pi * (MX * cell_inv[0, 1] + MY * cell_inv[1, 1] + MZ * cell_inv[2, 1])
    KZ = two_pi * (MX * cell_inv[0, 2] + MY * cell_inv[1, 2] + MZ * cell_inv[2, 2])
    k_sq = (KX ** 2 + KY ** 2 + KZ ** 2).permute(2, 1, 0)          # [nz,ny,nx//2+1]

    amp_dev = amp.to(device=device, dtype=dtype)
    bw2_dev = bw2.to(device=device, dtype=dtype)
    kfac_fft = (amp_dev.view(1, 1, 1, -1)
                * torch.exp(-0.5 * bw2_dev.view(1, 1, 1, -1) * k_sq.unsqueeze(-1))).sum(dim=-1)
    kfac_fft[0, 0, 0] = 0.0
    kfac_fft = kfac_fft.masked_fill(k_sq > k_sq_max, 0.0)

    # ── Form-A green: divide by the EXACT separable window influence ──
    influence = precompute_quads_influence_analytic(nx, ny, nz, lx, ly, lz, order,
                                                    device=device, dtype=dtype)
    green_k = kfac_fft / influence
    green_k[0, 0, 0] = 0.0
    green_k = green_k.masked_fill(k_sq > k_sq_max, 0.0)

    _rfft_w = torch.ones(1, 1, nx // 2 + 1, dtype=dtype, device=device)
    if nx % 2 == 0:
        _rfft_w[:, :, 1:nx // 2] = 2.0
    else:
        _rfft_w[:, :, 1:] = 2.0

    N3 = float(nx * ny * nz)
    s2 = 1.0 / (N3 * N3)
    scaleinv = 1.0 / N3
    vol_val = volume.item() if isinstance(volume, torch.Tensor) else float(volume)
    diag_sum_fft = (_rfft_w * kfac_fft).sum() / (2.0 * vol_val)

    # ── Spread + FFT + energy (rho_scale = N/V, as in CubeS₂) ──
    rho_grid = quads_spread(q, r_frac, nx, ny, nz, xi, order, rho_scale=N3 / vol_val)
    rho_k = torch.fft.rfftn(rho_grid)
    rho_sq = rho_k.real ** 2 + rho_k.imag ** 2

    energy = 0.5 * vol_val * s2 * (_rfft_w * green_k * rho_sq).sum()
    if remove_self_interaction:
        energy = energy - (q * q).sum() * diag_sum_fft
    if self_coeff != 0.0:
        energy = energy - (q * q).sum() * self_coeff
    energy = energy * norm_factor

    result: Dict[str, Optional[torch.Tensor]] = {"energy": energy, "forces": None, "virial": None}

    # ── Explicit forces: ik·(1/N³)·green·ρ → irfftn → gather (fixed: no rfft weight here) ──
    if compute_force or compute_virial:
        conv_k = (scaleinv * green_k).to(torch.complex128) * rho_k.to(torch.complex128)
        KX3 = KX.permute(2, 1, 0).to(torch.complex128)
        KY3 = KY.permute(2, 1, 0).to(torch.complex128)
        KZ3 = KZ.permute(2, 1, 0).to(torch.complex128)
        force_grid = torch.stack([
            torch.fft.irfftn(1j * KX3 * conv_k, s=(nz, ny, nx)),
            torch.fft.irfftn(1j * KY3 * conv_k, s=(nz, ny, nx)),
            torch.fft.irfftn(1j * KZ3 * conv_k, s=(nz, ny, nx)),
        ], dim=0).to(dtype)
        force = quads_interpolate(q, r_frac, force_grid, nx, ny, nz, xi, order)
        force = force * N3 * norm_factor
        result["forces"] = force
        if compute_virial:
            kfacv_fft = (amp_dev.view(1, 1, 1, -1) * bw2_dev.view(1, 1, 1, -1)
                         * torch.exp(-0.5 * bw2_dev.view(1, 1, 1, -1) * k_sq.unsqueeze(-1))).sum(dim=-1)
            kfacv_fft[0, 0, 0] = 0.0
            kfacv_fft = kfacv_fft.masked_fill(k_sq > k_sq_max, 0.0)
            green_virial = (kfacv_fft / influence).masked_fill(k_sq > k_sq_max, 0.0)
            green_virial[0, 0, 0] = 0.0
            result["virial"] = mesh_reciprocal_virial(
                rho_sq, green_k, green_virial, kfac_fft, kfacv_fft,
                KX.permute(2, 1, 0), KY.permute(2, 1, 0), KZ.permute(2, 1, 0),
                _rfft_w, vol_val, s2, norm_factor, float((q * q).sum()), remove_self_interaction)

    return result
