"""Verify mixed-nloc (zero-padded) batched direct k-sum:

  (1) energy identity:  padded batched ≡ per-frame loop  (within fp tolerance)
  (2) force identity:    padded batched ≡ per-frame loop  (via autograd)
  (3) charge_neutral_lambda: correct with mixed nloc
  (4) greedy grouping:   groups correctly formed, energy matches loop

CPU-only, tiny system, no GPU needed.
"""
import math, os, sys
import torch

# Add sog/src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sog.module.gaussian import Gaussian
from sog.module.gaussian import SOG_DEFAULT_B, SOG_DEFAULT_SIGMA, SOG_DEFAULT_M

dtype = torch.float64
torch.manual_seed(42)

# ── helpers ──
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
        e = g(q=q_now, r=r_now, cell=cell[bid:bid+1])
        energies.append(e.squeeze())
    g._enable_batched_direct = True
    return torch.stack(energies)


# ── (1) energy identity: padded batched ≡ per-frame loop ──
print("=== (1) energy identity: padded batched vs per-frame loop ===")
for nloc_by_frame in [
    [4, 4, 4, 4],            # uniform (Level 0)
    [3, 5, 4, 4, 6, 3, 5],  # mild mix (Level 1: waste ~57% for max=6)
    [2, 2, 2, 20, 2, 2],    # large outlier (Level 2: waste >> 1.0)
]:
    nf = len(nloc_by_frame)
    q, r, cell, batch, _, _ = build_mixed_batch(
        nloc_by_frame, nch=2, box_scale=10.0 + sum(nloc_by_frame) * 0.05
    )
    g = make_gaussian()
    g._enable_batched_direct = True

    # Batched path (with padding if needed)
    e_batched = g(q=q, r=r, cell=cell, batch=batch).detach()

    # Per-frame loop reference
    e_loop = run_per_frame_loop(g, q, r, cell, batch, nf)

    diff = (e_batched - e_loop).abs().max().item()
    status = "PASS" if diff < 1e-12 else "FAIL"
    nloc_str = f"nloc={nloc_by_frame}"
    print(f"  {nloc_str:45s}  max|dE|={diff:.2e}  [{status}]")
    if diff >= 1e-12:
        print(f"    batched: {e_batched}")
        print(f"    loop:    {e_loop}")
        raise AssertionError(f"Energy mismatch for nloc={nloc_by_frame}")

print("  -> ALL PASS\n")


# ── (2) force identity via autograd ──
print("=== (2) force identity: grad(padded batched) vs grad(per-frame) ===")
for nloc_by_frame, desc in [
    ([3, 5, 4, 4, 6, 3, 5], "Level 1 (full pad)"),
    ([2, 2, 2, 20, 2, 2], "Level 2 (grouped)"),
]:
    nf = len(nloc_by_frame)
    q, r, cell, batch, _, nloc_list = build_mixed_batch(
        nloc_by_frame, nch=1, box_scale=10.0
    )
    g = make_gaussian()
    g._enable_batched_direct = True

    # Batched forces
    r_b = r.clone().requires_grad_(True)
    e_b = g(q=q, r=r_b, cell=cell, batch=batch).sum()
    f_b = -torch.autograd.grad(e_b, r_b, create_graph=False)[0]

    # Per-frame forces
    r_l = r.clone().requires_grad_(True)
    e_l = run_per_frame_loop(g, q, r_l, cell, batch, nf).sum()
    f_l = -torch.autograd.grad(e_l, r_l, create_graph=False)[0]

    max_df = (f_b - f_l).abs().max().item()
    rms_df = (f_b - f_l).pow(2).mean().sqrt().item()
    status = "PASS" if max_df < 1e-10 else "FAIL"
    print(f"  {desc:35s}  max|dF|={max_df:.2e}  rms|dF|={rms_df:.2e}  [{status}]")
    if max_df >= 1e-10:
        raise AssertionError(f"Force mismatch for {desc}")

print("  -> ALL PASS\n")


# ── (3) charge_neutral_lambda correctness with mixed nloc ──
print("=== (3) charge_neutral_lambda with mixed nloc ===")
# Use non-zero lambda and verify padded batched matches per-frame
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
    status = "PASS" if diff < 1e-12 else "FAIL"
    print(f"  nloc={nloc_by_frame!s:30s}  max|dE|={diff:.2e}  [{status}]")
    if diff >= 1e-12:
        # If this fails, the q_f.mean fix might be wrong
        print(f"    batched: {e_b}")
        print(f"    loop:    {e_l}")
        raise AssertionError("charge_neutral_lambda mismatch with mixed nloc")

print("  -> ALL PASS\n")


# ── (4) greedy grouping structure ──
print("=== (4) greedy grouping correctness ===")
from sog.module.gaussian import Gaussian as G

# Test that grouping produces expected structure
test_cases = [
    ([4, 4, 4, 4], [[(0,4),(1,4),(2,4),(3,4)]]),   # all same → 1 group
    ([2, 2, 20, 2], None),                            # outlier → 2 groups
    ([10, 12, 100, 11], None),                        # outlier → 2 groups
]
for nloc_list, expected in test_cases:
    groups = G._greedy_group_by_nloc(nloc_list, max_waste=1.0)
    n_groups = len(groups)
    # For cases with expected=None, just check every frame appears exactly once
    all_indices = sorted([idx for g in groups for idx, _ in g])
    ok = all_indices == list(range(len(nloc_list)))

    if expected is not None:
        ok = ok and groups == expected

    if ok:
        print(f"  nloc={nloc_list!s:30s}  groups={n_groups}  [PASS]")
    else:
        print(f"  nloc={nloc_list!s:30s}  groups={n_groups} → {groups}  [FAIL]")
        if expected:
            print(f"    expected: {expected}")
        raise AssertionError("Grouping mismatch")

print("  -> ALL PASS\n")

print("=" * 60)
print("OVERALL: ALL PASS")
