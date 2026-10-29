# SOG integration guide

The porting procedure, the four coupling points, the deployment protocols and
the traps an agent must know all live on the **repo homepage
([`../README.md`](../README.md))**.  This directory holds the numbered
specifications that page points to.

## Pages

| Page | Covers |
|---|---|
| [01-overview](01-overview.md) | physics + data flow — read this first |
| [02-python-api](02-python-api.md) | `Sog` / `SogKernel` API, kernel modes, batch/charge validation |
| [03-training-adapter](03-training-adapter.md) | training port: `E_lr` in the loss, the external-k-space seed |
| [04-parameter-export](04-parameter-export.md) | kernel parameters → LAMMPS input / `.pt2` |
| [05-lammps-plugin](05-lammps-plugin.md) | `kspace_style sog`, `fix sog/response`, keyword reference |
| [06-cpp-porting](06-cpp-porting.md) | the two C++ contracts (`SogChargeSource` / `SogChargeResponse`) |
| [07-pipeline](07-pipeline.md) | end-to-end: train → freeze → MD |
| [08-validation](08-validation.md) | **numerical acceptance** — the parity numbers a correct port reproduces |
| [09-in-graph-kspace](09-in-graph-kspace.md) | the `sogkspace` op, grid sizing, `box_tol`, share semantics |
| [10-export-and-artifacts](10-export-and-artifacts.md) | Mode A vs Mode B, protocols, 12/20-input ABIs, AOTI, metadata |
| [11-lr-sync](11-lr-sync.md) | mirror of `deepmd-kit-LR`: vendored / patched / deferred |

## Reading paths

- **Training only**: 01 → 02 → 03 → 04
- **MD only**: 04 → 05 → 06 (the last only if your charges are latent)
- **Deployment (PyTorch 2.x)**: 09 → 10 → 11, then `test/v2/`
- **Full port**: 01 → 02 → 03 → 04 → 05 → 06 → 07, proved with 08

A runnable MACE migration lives in [`../example/mace/`](../example/mace/).
