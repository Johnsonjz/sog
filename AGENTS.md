# AGENTS.md

The agent-facing migration guide for this repository is the **README (the
GitHub homepage)** — see its section
[**"Port SOG into your framework (for agents)"**](README.md#port-sog-into-your-framework-for-agents).
It has the port scope table, the four coupling points, the `sog.v2` deployment
protocols, how to drive an agent, and the non-obvious traps.

| Need | Go to |
|---|---|
| Port scope, coupling points, protocols, traps | [`README.md`](README.md) |
| The numbered specifications (01–11) | [`docs/README.md`](docs/README.md) |
| In-graph `sogkspace` op: build (MPI!), self-energy, grid, `box_tol` | [`docs/09`](docs/09-in-graph-kspace.md) |
| A runnable MACE migration | [`example/mace/`](example/mace/) |
| Numerical acceptance criteria | [`docs/08-validation.md`](docs/08-validation.md) |

## Must know (easily missed)

- **Self-energy is closed-form.**  `E_self = ½·Σq²·g(0)`, with
  `g(0) = Σ_m amp_m (2π bw²_m)^(-3/2)` (`amp` internal, `bw²` squared).  It is
  exactly `∝ q²` (volume-extensive) and has **zero** self virial — never use a
  mesh sum `Σ_k K/V` (not extensive; injects spurious shear).  Applies to the
  training solvers (`Gaussian`/`Sog`, CubeS₂/QuadS via `sog.module._self`), the
  in-graph op, and `lmp/sog.cpp` alike.
- **The in-graph `op/` requires MPI.**  Configure with `-DSOG_WITH_MPI=ON` and
  an MPI whose `MPI_Comm` is a pointer (OpenMPI / Intel MPI); MPICH's `int`
  handle fails the vendored `set_comm` cast.  Both builds link `${CMAKE_DL_LIBS}`
  for the NCCL `dlopen` in `sog_gpu.cu`.
- `amp` is the **internal** amplitude and `bandwidth` is **bw²**; the response
  seed is `Σ v·q` **not** `½ Σ v·q`, divided by the owned node count; the virial
  order is `(xx,yy,zz,xy,xz,yz)` (DeepMD maps `{0,4,8,1,2,5}`).

Full list: the README's "Non-obvious facts".  Symptom → cause:
[`docs/10`](docs/10-export-and-artifacts.md).  This repository mirrors
`deepmd-kit-LR` at the pin in [`lmp/SYNC_NOTE.md`](lmp/SYNC_NOTE.md).
