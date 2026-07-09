#!/usr/bin/env python3
"""Compare two CubeS2 FFT energy forms against the direct k-sum reference.

Form A (SPME-division):     green_k = K(k²) / |Φ_analytic(k)|²   (fixed monomials)
Form B (variance-subtract): green_k = Σ amp·exp(-½(bw-2σ_s²)k²)  (paper Eq. 70)

Both use the same CubeS2 spread (rho_scale = N_g/V) and the same k-grid.
Reference: direct k-sum E = (1/2V) Σ_{k≠0} K(k²)|S(k)|² (converged, no self term).

Run over a grid-refinement sequence to show convergence to `direct`.
"""
import sys
sys.path.insert(0, "/root/code/sog/src")
import math
import numpy as np
import torch

torch.set_default_dtype(torch.float64)
E2 = 14.3996454784255

from sog.module.cubes2_fft import cubes2_spread, _next_fft_friendly
from sog.module.cubes2_spline import _get_xi
from sog.module.influence_analytic import precompute_influence_analytic

XI4 = _get_xi(4)
B = 1.6297670882677646
SIGMA = 2.180230445405648
RCUT = 5.0


def geometric():
    amp = np.array([4*math.pi*math.log(B)*(SIGMA**2*B**(2*m)) for m in range(12)])
    bw = np.array([SIGMA**2*B**(2*m) for m in range(12)])
    return amp, bw


def trained():
    amp = np.array([1.05852560e+01, 5.87989331e+01, 1.81409068e+02, 5.32747690e+02,
                    1.45220412e+03, 3.85725858e+03, 1.02454217e+04, 2.72132823e+04,
                    7.22823085e+04, 1.91991986e+05, 5.09957740e+05, 1.35451954e+06])
    bw = np.array([7.77989631e+00, 2.89900118e+01, 5.64525176e+01, 1.20836924e+02,
                   2.36598317e+02, 6.28434810e+02, 1.66921132e+03, 4.43366022e+03,
                   1.17764256e+04, 3.12798441e+04, 8.30836690e+04, 2.20681920e+05])
    return amp, bw


def make_system(nmol=64, seed=1):
    rng = np.random.default_rng(seed)
    box_l = (nmol ** (1/3)) * 3.1
    coords = rng.uniform(0, box_l, size=(nmol*3, 3))
    q = np.tile([-0.8, 0.4, 0.4], nmol)
    pert = rng.normal(0, 0.1, size=q.shape); pert -= pert.mean()
    q = q + pert; q -= q.mean()
    return coords, q, box_l


def kgrid(nx, ny, nz, L):
    mx = torch.arange(0, nx//2+1).double()
    my = torch.cat([torch.arange(0, ny//2+1), torch.arange(-ny//2+1, 0)]).double()
    mz = torch.cat([torch.arange(0, nz//2+1), torch.arange(-nz//2+1, 0)]).double()
    MX, MY, MZ = torch.meshgrid(mx, my, mz, indexing="ij")
    tp = 2*math.pi/L
    ksq = ((MX*tp)**2 + (MY*tp)**2 + (MZ*tp)**2).permute(2, 1, 0)
    return ksq


def E_direct(amp, bw, coords, q, L, kmax_mode=20):
    """Exact reciprocal energy (1/2V) Σ_{k≠0} K|S|² over integer grid."""
    amp = torch.tensor(amp); bw = torch.tensor(bw)
    r = torch.tensor(coords); qt = torch.tensor(q)
    rng = torch.arange(-kmax_mode, kmax_mode+1).double()
    MX, MY, MZ = torch.meshgrid(rng, rng, rng, indexing="ij")
    tp = 2*math.pi/L
    kvec = torch.stack([MX.reshape(-1)*tp, MY.reshape(-1)*tp, MZ.reshape(-1)*tp], 1)
    ksq = (kvec**2).sum(1)
    mask = ksq > 1e-12
    kvec = kvec[mask]; ksq = ksq[mask]
    K = (amp.view(1, -1)*torch.exp(-0.5*bw.view(1, -1)*ksq.view(-1, 1))).sum(1)
    kr = r @ kvec.t()
    S = (qt.view(-1, 1)*torch.exp(-1j*kr)).sum(0)
    Ssq = (S.real**2 + S.imag**2)
    V = L**3
    return (K*Ssq).sum().item()/(2*V)*E2


def _spread_rho(coords, q, nx, ny, nz, L):
    r = torch.tensor(coords); qt = torch.tensor(q); cell = torch.eye(3)*L
    rf = (r @ torch.linalg.inv(cell)) % 1.0
    V = L**3; N3 = nx*ny*nz
    rho = cubes2_spread(qt, rf, nx, ny, nz, xi=XI4, order=4, rho_scale=N3/V)
    rk = torch.fft.rfftn(rho)
    return rk.real**2 + rk.imag**2, V, N3


def _rfft_w(nx):
    w = torch.ones(1, 1, nx//2+1)
    if nx % 2 == 0:
        w[:, :, 1:nx//2] = 2.0
    else:
        w[:, :, 1:] = 2.0
    return w


def E_formA(amp, bw, coords, q, nx, ny, nz, L):
    """SPME-division: green = K/|Φ|², masked where K negligible."""
    amp = torch.tensor(amp); bw = torch.tensor(bw)
    ksq = kgrid(nx, ny, nz, L)
    K = (amp.view(1,1,1,-1)*torch.exp(-0.5*bw.view(1,1,1,-1)*ksq.unsqueeze(-1))).sum(-1)
    K[0,0,0] = 0.0
    inf = precompute_influence_analytic(nx, ny, nz, L, L, L, xi=XI4)
    # mask modes where K is below rel tol (avoid /|Φ|² blow-up where kernel is ~0)
    Kmax = K.max()
    green = torch.where(K > 1e-12*Kmax, K/inf.clamp(min=1e-20), torch.zeros_like(K))
    rho_sq, V, N3 = _spread_rho(coords, q, nx, ny, nz, L)
    s2 = 1.0/N3**2; w = _rfft_w(nx)
    return 0.5*V*s2*(w*green*rho_sq).sum().item()*E2


def E_formB(amp, bw, coords, q, nx, ny, nz, L):
    """Variance-subtraction: green = Σ amp exp(-½(bw-2σ_s²)k²), no division."""
    amp = torch.tensor(amp); bw = torch.tensor(bw)
    ksq = kgrid(nx, ny, nz, L)
    sigma_s2 = (XI4*L/nx)**2  # cubic grid: σ_s = ξ₀·Δ, Δ=L/nx
    bw_eff = (bw - 2*sigma_s2).clamp(min=1e-8)
    green = (amp.view(1,1,1,-1)*torch.exp(-0.5*bw_eff.view(1,1,1,-1)*ksq.unsqueeze(-1))).sum(-1)
    green[0,0,0] = 0.0
    rho_sq, V, N3 = _spread_rho(coords, q, nx, ny, nz, L)
    s2 = 1.0/N3**2; w = _rfft_w(nx)
    return 0.5*V*s2*(w*green*rho_sq).sum().item()*E2


def run(name, amp, bw, coords, q, L):
    edir = E_direct(amp, bw, coords, q, L)
    print(f"\n{'='*74}\n{name}   (direct reference = {edir:.6f} eV)\n{'='*74}")
    print(f"  σ_min = {math.sqrt(min(bw)):.3f} Å,  bw_min = {min(bw):.3f}")
    print(f"  {'nx':>4} {'Δ(Å)':>7} {'FormA (÷|Φ|²)':>16} {'relA':>10} "
          f"{'FormB (var-sub)':>16} {'relB':>10}")
    for nx in [16, 20, 24, 32, 40, 48, 64]:
        nx = _next_fft_friendly(nx)
        delta = L/nx
        ea = E_formA(amp, bw, coords, q, nx, nx, nx, L)
        eb = E_formB(amp, bw, coords, q, nx, nx, nx, L)
        ra = abs(ea-edir)/abs(edir); rb = abs(eb-edir)/abs(edir)
        print(f"  {nx:>4} {delta:>7.3f} {ea:>16.5f} {ra:>10.2e} {eb:>16.5f} {rb:>10.2e}")


def main():
    coords, q, L = make_system(nmol=64, seed=1)
    print(f"System: {len(q)} charges, box {L:.2f}³ Å³, Σq={q.sum():.2e}")
    run("GEOMETRIC u-series", *geometric(), coords, q, L)
    run("TRAINED (non-geometric)", *trained(), coords, q, L)


if __name__ == "__main__":
    main()
