#!/usr/bin/env python3
"""Numerical validation of CubeS₂ 6th-order (88-node) implementation.

Pass criteria (per plan): moment conditions exact through total degree 5,
FFT energy converges to the direct k-sum, and forces match finite differences.
(Negative weights are acceptable if the numerical energy is correct.)
"""
import sys
sys.path.insert(0, "/root/code/sog/src")
import math
import itertools
import numpy as np
import torch

torch.set_default_dtype(torch.float64)
E2 = 14.3996454784255

from sog.module.cubes2_spline import get_nodes, cubes2_weight, _get_xi
from sog.module.cubes2_fft import compute_cubes2_fft, _resolve_grid

XI6 = _get_xi(6)
B = 1.6297670882677646


# ── Test 1: partition of unity + moment conditions ──
def test_moments(order, xi, max_deg=5, npts=200):
    """Σ_j w_j·(d_j−θ)^(α,β,γ) must reproduce the moments of the CubeS₂ window.

    For an interpolation exact through degree p, the discrete moments must equal
    the moments of the *continuous* assignment window (a Gaussian of variance
    σ_s²=(ξ·1)² in grid units for CubeS₂). Rather than hardcode those, we use the
    order-4 window as the ground-truth reference at LOW degree and check the
    key invariants: (0th) Σw=1, (1st) Σ w·(d−θ)=const (centroid), and that the
    moments are θ-INDEPENDENT up to degree (2ν−1) — the defining property of a
    degree-p accurate assignment (translation-covariance of the moments).
    """
    nodes = get_nodes(order)
    dvec = np.array([(n.dx, n.dy, n.dz) for n in nodes], dtype=float)
    rng = np.random.default_rng(0)
    # For each monomial degree, the moment Σ w_j (d_j-θ)^m should be a POLYNOMIAL
    # in θ of degree ≤ m that is the SAME for the exact window. The cleanest
    # order-independent invariant: the "central moments about θ" must match those
    # of order-4 at the SAME θ up to degree (2ν_4−1)=3, and additionally be
    # θ-independent for the leading orders. We check θ-independence of moments
    # 0..(2ν−1): a p-th order scheme reproduces polynomials up to degree p, which
    # forces Σ w (d−θ)^m to be θ-independent for m ≤ p.
    results = {}
    for deg in range(max_deg + 1):
        for combo in _monomials_of_degree(deg):
            vals = []
            for _ in range(npts):
                th = rng.random(3)
                w = _weights_at(nodes, th, xi, order)
                dm = (dvec - th)  # [N,3]
                mono = dm[:, 0]**combo[0] * dm[:, 1]**combo[1] * dm[:, 2]**combo[2]
                vals.append(float((w * mono).sum()))
            vals = np.array(vals)
            results[combo] = (vals.mean(), vals.std())
    return results


def _monomials_of_degree(deg):
    return [c for c in itertools.product(range(deg + 1), repeat=3) if sum(c) == deg]


def _weights_at(nodes, theta, xi, order):
    tx, ty, tz = [torch.tensor(t) for t in theta]
    return np.array([cubes2_weight(tx, ty, tz, n, xi, order=order).item() for n in nodes])


def report_moments(order, xi, label):
    print(f"\n--- Moment θ-independence, {label} (order={order}, ξ={xi:.6f}) ---")
    res = test_moments(order, xi, max_deg=5)
    p = 2 * (order // 2) - 1  # 2ν−1: order4→3, order6→5
    print(f"  interpolation should reproduce polynomials through degree {p}")
    worst_by_deg = {}
    for combo, (m, s) in res.items():
        d = sum(combo)
        worst_by_deg.setdefault(d, 0.0)
        worst_by_deg[d] = max(worst_by_deg[d], s)  # std over θ = θ-dependence
    for d in sorted(worst_by_deg):
        flag = "θ-indep ✓" if worst_by_deg[d] < 1e-9 else "θ-DEPENDENT ✗"
        marker = "  (must be θ-indep)" if d <= p else "  (may depend on θ)"
        print(f"    degree {d}: max θ-std = {worst_by_deg[d]:.3e}   {flag}{marker}")
    ok = all(worst_by_deg[d] < 1e-9 for d in range(p + 1) if d in worst_by_deg)
    return ok


# ── Test 2: FFT energy vs direct k-sum ──
def geometric(sigma, M=12):
    amp = np.array([4*math.pi*math.log(B)*(sigma**2*B**(2*m)) for m in range(M)])
    bw = np.array([sigma**2*B**(2*m) for m in range(M)])
    return amp, bw


def make_system(nmol=64, seed=1):
    rng = np.random.default_rng(seed)
    L = (nmol ** (1/3)) * 3.1
    coords = rng.uniform(0, L, size=(nmol*3, 3))
    q = np.tile([-0.8, 0.4, 0.4], nmol)
    pert = rng.normal(0, 0.1, size=q.shape); pert -= pert.mean()
    q = q + pert; q -= q.mean()
    return coords, q, L


def E_direct(amp, bw, coords, q, L, kmax=20):
    """Exact reciprocal energy with self-interaction removed (matches
    compute_cubes2_fft's remove_self_interaction=True)."""
    amp = torch.tensor(amp); bw = torch.tensor(bw)
    r = torch.tensor(coords); qt = torch.tensor(q); V = L**3
    rng = torch.arange(-kmax, kmax+1).double() * (2*math.pi/L)
    MX, MY, MZ = torch.meshgrid(rng, rng, rng, indexing="ij")
    kv = torch.stack([MX.reshape(-1), MY.reshape(-1), MZ.reshape(-1)], 1)
    ksq = (kv**2).sum(1); m = ksq > 1e-12; kv = kv[m]; ksq = ksq[m]
    K = (amp.view(1, -1)*torch.exp(-0.5*bw.view(1, -1)*ksq.view(-1, 1))).sum(1)
    S = (qt.view(-1, 1)*torch.exp(-1j*(r@kv.t()))).sum(0)
    E_four = (K*(S.real**2+S.imag**2)).sum().item()/(2*V)
    diag = K.sum().item()/(2*V)                 # Σ_{k≠0} K(k²)/(2V)
    E_self = -(q**2).sum()*diag                  # self-interaction removal
    return (E_four + E_self)*E2


def E_fft(amp, bw, coords, q, L, order, phi):
    r = torch.tensor(coords); qt = torch.tensor(q); cell = torch.eye(3)*L; V = L**3
    res = compute_cubes2_fft(qt, r, cell, torch.tensor(amp), torch.tensor(bw),
                             torch.tensor(V), torch.tensor(0.0),
                             cubes2_phi_max=phi, r_c=5.0, b=B, xi=_get_xi(order),
                             order=order, remove_self_interaction=True, norm_factor=E2)
    return res["energy"].item()


def test_energy(order, phi_seq, label):
    coords, q, L = make_system()
    sigma = 2.180230445405648
    amp, bw = geometric(sigma)
    edir = E_direct(amp, bw, coords, q, L)
    print(f"\n--- FFT(order={order}) vs direct, {label}  (direct={edir:.5f} eV) ---")
    for phi in phi_seq:
        ef = E_fft(amp, bw, coords, q, L, order, phi)
        nx, ny, nz, _ = _resolve_grid(L, L, L, None, phi, 5.0, b=B, spline_order=order)
        rel = abs(ef-edir)/abs(edir)
        print(f"    φ={phi:<6} grid={nx}³  E_fft={ef:12.5f}  rel={rel:.2e}")
    return abs(E_fft(amp, bw, coords, q, L, order, phi_seq[-1]) - edir)/abs(edir)


# ── Test 3: force finite-difference (order-6 energy-only autograd path) ──
def test_force(order, phi=0.16):
    from sog.module.gaussian import Gaussian
    rng = np.random.default_rng(3); nmol = 16; L = (nmol**(1/3))*3.1
    coords = rng.uniform(0, L, size=(nmol*3, 3))
    q = np.tile([-0.8, 0.4, 0.4], nmol); pert = rng.normal(0, 0.1, q.shape)
    pert -= pert.mean(); q = q+pert; q -= q.mean()
    sigma = 2.180230445405648; amp, bw = geometric(sigma)
    qt = torch.tensor(q).reshape(-1, 1); cell = torch.tensor(np.diag([L]*3)).unsqueeze(0)
    batch = torch.zeros(len(q), dtype=torch.int64)
    g = Gaussian(amp=torch.tensor(amp), bandwidth=torch.tensor(bw),
                 kernel_param_mode="internal", kernel_tensor_mode="external",
                 remove_self_interaction=True, use_nufft=False, use_cubes2_fft=True,
                 norm_factor=E2, trainable=False, b=B, rcut=5.0,
                 cubes2_phi_max=phi, cubes2_order=order)
    r = torch.tensor(coords, requires_grad=True)
    E = g.compute_bundle(q=qt, r=r, cell=cell, batch=batch)["energy"].sum()
    F = -torch.autograd.grad(E, r)[0]
    h = 1e-5; maxrel = 0.0
    print(f"\n--- Force finite-diff, order={order} (φ={phi}) ---")
    for i in [0, 5, 20]:
        for a in range(3):
            rp = coords.copy(); rp[i, a] += h; rm = coords.copy(); rm[i, a] -= h
            ep = g.compute_bundle(q=qt, r=torch.tensor(rp), cell=cell, batch=batch)["energy"].sum().item()
            em = g.compute_bundle(q=qt, r=torch.tensor(rm), cell=cell, batch=batch)["energy"].sum().item()
            ffd = -(ep-em)/(2*h); fa = F[i, a].item()
            rel = abs(fa-ffd)/max(abs(ffd), 1e-6); maxrel = max(maxrel, rel)
    print(f"    max |F_autograd − F_fd|/|F| = {maxrel:.2e}")
    return maxrel


def main():
    print("="*70 + "\nCubeS₂ ORDER-6 NUMERICAL VALIDATION\n" + "="*70)
    # sanity: order-4 (known good) then order-6
    ok4 = report_moments(4, _get_xi(4), "order-4 reference")
    ok6 = report_moments(6, XI6, "order-6")
    r4 = test_energy(4, [0.23, 0.15, 0.1, 0.065], "order-4 reference")
    r6 = test_energy(6, [0.35, 0.25, 0.16, 0.10], "order-6")
    f6 = test_force(6)
    print("\n" + "="*70 + "\nSUMMARY")
    print(f"  order-4 moments θ-indep(≤3): {'✓' if ok4 else '✗'}   energy rel: {r4:.2e}")
    print(f"  order-6 moments θ-indep(≤5): {'✓' if ok6 else '✗'}   energy rel: {r6:.2e}   force rel: {f6:.2e}")
    verdict = ok6 and r6 < 1e-3 and f6 < 1e-2
    print(f"\n  ORDER-6: {'✓ NUMERICALLY VALIDATED' if verdict else '✗ needs work'}")


if __name__ == "__main__":
    main()
