"""Verify mixed-nloc (zero-padded) batched direct k-sum:

  (1) energy identity:  padded batched ≡ per-frame loop  (within fp tolerance)
  (2) force identity:    padded batched ≡ per-frame loop  (via autograd)
  (3) charge_neutral_lambda: correct with mixed nloc
  (4) greedy grouping:   groups correctly formed, energy matches loop

CPU-only, tiny system, no GPU needed.
"""

import torch

from sog.module.gaussian import (
    Gaussian,
    SOG_DEFAULT_B,
    SOG_DEFAULT_M,
    SOG_DEFAULT_SIGMA,
)

dtype = torch.float64

# nloc patterns: uniform (Level 0), mild mix (Level 1 full pad), large outlier
# (Level 2 greedy grouping).
MIXED_PATTERNS = [
    [4, 4, 4, 4],
    [3, 5, 4, 4, 6, 3, 5],
    [2, 2, 2, 20, 2, 2],
]


def make_gaussian(amp=None, bw=None, **kwargs):
    """Create a minimal Gaussian kernel."""
    g = Gaussian(
        amp=amp if amp is not None else torch.tensor([1.0], dtype=dtype),
        bandwidth=bw if bw is not None else torch.tensor([0.5], dtype=dtype),
        b=SOG_DEFAULT_B, sigma=SOG_DEFAULT_SIGMA, m=SOG_DEFAULT_M,
        remove_self_interaction=True,
        **kwargs,
    )
    g.eval()
    return g


def build_mixed_batch(nloc_by_frame, nch=1, box_scale=10.0):
    """Build a mixed-nloc batch with PyG-style canonical ordering."""
    nf = len(nloc_by_frame)
    n_total = sum(nloc_by_frame)
    r = torch.randn((n_total, 3), dtype=dtype) * 2.0
    q = torch.randn((n_total, nch), dtype=dtype)
    batch = torch.cat([
        torch.full((nl,), i, dtype=torch.long) for i, nl in enumerate(nloc_by_frame)
    ])
    # Simple cubic cells scaled per frame
    cell = torch.eye(3, dtype=dtype).unsqueeze(0).repeat(nf, 1, 1) * box_scale
    return q, r, cell, batch, nf, nloc_by_frame


def run_per_frame_loop(g, q, r, cell, batch, nf):
    """Simulate the per-frame loop from compute_bundle for reference."""
    energies = []
    for bid in range(nf):
        mask = batch == bid
        r_now = r[mask]
        q_now = q[mask]
        if r_now.shape[0] == 0:
            energies.append(torch.tensor(0.0, dtype=dtype))
            continue
        # Use compute_bundle with batched disabled to force per-frame loop
        g._enable_batched_direct = False
        e = g(q=q_now, r=r_now, cell=cell[bid:bid + 1])
        energies.append(e.squeeze())
    g._enable_batched_direct = True
    return torch.stack(energies)


def test_energy_identity_padded_vs_loop():
    """(1) padded batched ≡ per-frame loop, all three grouping levels."""
    torch.manual_seed(42)
    for nloc_by_frame in MIXED_PATTERNS:
        nf = len(nloc_by_frame)
        q, r, cell, batch, _, _ = build_mixed_batch(
            nloc_by_frame, nch=2, box_scale=10.0 + sum(nloc_by_frame) * 0.05
        )
        g = make_gaussian()
        g._enable_batched_direct = True

        e_batched = g(q=q, r=r, cell=cell, batch=batch).detach()
        e_loop = run_per_frame_loop(g, q, r, cell, batch, nf)

        diff = (e_batched - e_loop).abs().max().item()
        assert diff < 1e-12, (
            f"energy mismatch for nloc={nloc_by_frame}: max|dE|={diff:.2e}\n"
            f"  batched: {e_batched}\n  loop: {e_loop}"
        )


def test_force_identity_via_autograd():
    """(2) grad(padded batched) ≡ grad(per-frame loop)."""
    torch.manual_seed(42)
    for nloc_by_frame, desc in [
        ([3, 5, 4, 4, 6, 3, 5], "Level 1 (full pad)"),
        ([2, 2, 2, 20, 2, 2], "Level 2 (grouped)"),
    ]:
        nf = len(nloc_by_frame)
        q, r, cell, batch, _, _ = build_mixed_batch(
            nloc_by_frame, nch=1, box_scale=10.0
        )
        g = make_gaussian()
        g._enable_batched_direct = True

        r_b = r.clone().requires_grad_(True)
        e_b = g(q=q, r=r_b, cell=cell, batch=batch).sum()
        f_b = -torch.autograd.grad(e_b, r_b, create_graph=False)[0]

        r_l = r.clone().requires_grad_(True)
        e_l = run_per_frame_loop(g, q, r_l, cell, batch, nf).sum()
        f_l = -torch.autograd.grad(e_l, r_l, create_graph=False)[0]

        max_df = (f_b - f_l).abs().max().item()
        assert max_df < 1e-10, (
            f"force mismatch for {desc} (nloc={nloc_by_frame}): max|dF|={max_df:.2e}"
        )


def test_charge_neutral_lambda_mixed_nloc():
    """(3) charge_neutral_lambda path with mixed nloc."""
    torch.manual_seed(42)
    g_cnl = make_gaussian(charge_neutral_lambda=1.0)
    for nloc_by_frame in [[3, 5, 4, 4], [2, 2, 20, 2]]:
        nf = len(nloc_by_frame)
        q, r, cell, batch, _, _ = build_mixed_batch(
            nloc_by_frame, nch=1, box_scale=10.0
        )
        g_cnl._enable_batched_direct = True
        e_b = g_cnl(q=q, r=r, cell=cell, batch=batch).detach()
        e_l = run_per_frame_loop(g_cnl, q, r, cell, batch, nf)
        diff = (e_b - e_l).abs().max().item()
        assert diff < 1e-12, (
            f"charge_neutral_lambda mismatch for nloc={nloc_by_frame}: "
            f"max|dE|={diff:.2e}\n  batched: {e_b}\n  loop: {e_l}"
        )


def test_greedy_grouping_structure():
    """(4) greedy grouping: every frame exactly once; uniform -> single group."""
    test_cases = [
        ([4, 4, 4, 4], [[(0, 4), (1, 4), (2, 4), (3, 4)]]),  # all same -> 1 group
        ([2, 2, 20, 2], None),                               # outlier -> 2 groups
        ([10, 12, 100, 11], None),                           # outlier -> 2 groups
    ]
    for nloc_list, expected in test_cases:
        groups = Gaussian._greedy_group_by_nloc(nloc_list, max_waste=1.0)
        all_indices = sorted(idx for g in groups for idx, _ in g)
        assert all_indices == list(range(len(nloc_list))), (
            f"nloc={nloc_list}: frames not covered exactly once: {groups}"
        )
        if expected is not None:
            assert groups == expected, (
                f"nloc={nloc_list}: expected {expected}, got {groups}"
            )
