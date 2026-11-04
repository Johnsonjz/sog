// sog_kspace_op.cc — differentiable-op wrapper around the SOG device kspace.
//
// Registers ``sogkspace::solve_auto`` so an exported (AOTI/torch.export)
// artifact can put the SOG long-range solve INSIDE its own graph, which makes
// the short-range and charge-response forces come out of a single backward and
// removes the duplicated model forward of the 2F+1B pair/kspace/fix
// decomposition (measured: -33% at 12 288 atoms; see
// sog_work/bench/SPEED_TEST.md sections 6.7.1-6.7.3).
//
// Registered under its own `sogkspace` namespace (not `deepmd_export`): the op
// is self-contained, and exported artifacts reference the schema by name, so a
// private namespace keeps the SOG module independent of the exportable-op ABI.
// `common.cc::load_op_library` dlopen's the library; the fragment name is not
// part of the loading contract.
//
// The kernels are the deployed ones: this target compiles
// ``source/lmp/sog_gpu.cu`` unchanged (the same code ``kspace_style sog/kk``
// runs), so the op reproduces the deployed discretization exactly rather than
// re-deriving it.
//
#include <chrono>
#include <torch/library.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "sog_gpu.cuh"
#include "sog_green.h"

#ifdef USE_MPI
#include <mpi.h>
#endif

namespace {

// One cached, page-locked arena per handle.  The distributed solve stages the
// mesh through host memory several times per step: pageable H<->D runs at a few
// GB/s while pinned runs at ~20, and allocating ~0.5 GB of std::vector every
// call also pays the page-fault cost.  The arena is sized for the current
// decomposition (every buffer is ~n_g/np) and reused across steps.
struct PinnedArena {
  double *base = nullptr;
  size_t cap = 0, used = 0;
  bool pinned = false;
  PinnedArena() = default;
  PinnedArena(const PinnedArena &) = delete;
  PinnedArena &operator=(const PinnedArena &) = delete;
  ~PinnedArena() { release(); }
  void release() {
    if (base) {
      if (pinned) cudaFreeHost(base); else std::free(base);
    }
    base = nullptr; cap = used = 0; pinned = false;
  }
  void ensure(size_t n) {
    if (base != nullptr && n <= cap) { used = 0; return; }
    release();
    void *p = nullptr;
    if (cudaHostAlloc(&p, n * sizeof(double), cudaHostAllocDefault) ==
        cudaSuccess) {
      base = static_cast<double *>(p);
      pinned = true;
    } else {
      base = static_cast<double *>(std::malloc(n * sizeof(double)));
    }
    TORCH_CHECK(base != nullptr, "sogkspace: host arena allocation failed");
    cap = n;
    used = 0;
  }
  double *alloc(size_t n) {
    TORCH_CHECK(used + n <= cap, "sogkspace: host arena overflow");
    double *p = base + used;
    used += n;
    return p;
  }
};

struct Entry {
  SogGpuState *s = nullptr;
  int nx = 0, ny = 0, nz = 0;
  // Host copies kept so a re-setup with the same handle can be skipped.
  std::vector<double> ge, gf, gv;
  // ── self-contained (solve_auto) state: grid sizing + box_tol caching,
  //    mirroring SOGKSpace::ensure_fft_plan's Case-1 skip ──
  bool have_green = false;
  double glx = 0.0, gly = 0.0, glz = 0.0;   // box length at the last rebuild
  int spline = 4;
  std::vector<double> amp, bw;              // parameter signature
  // host staging buffers for the non-CUDA-aware replicated-grid reduction
  std::vector<double> rho_host, rho_recv;
  PinnedArena arena;                        // distributed-mesh staging
  // Cached green layout: full grid (replicated) or this rank's y-slice
  // (distributed).  The slab depends on the rank's y range, so it is part of
  // the reuse key even when the grid and parameters are unchanged.
  bool dist_mode = false;
  int slab_ylo = 0, slab_YL = 0;
};

// number of z planes in an owned+ghost brick
inline int ZB_of(int ZL, int gh_lo, int gh_hi) { return ZL + gh_lo + gh_hi; }

std::mutex g_mu;
std::unordered_map<int64_t, Entry> g_states;

// A build without MPI still works (single rank): the replicated-grid path is
// compiled out and the op simply solves the whole system locally.
#ifdef USE_MPI
// Communicator used by the replicated-grid Allreduce.  MPI_COMM_WORLD until the
// framework registers its own communicator (LAMMPS `world`) via set_comm.
MPI_Comm g_world = MPI_COMM_WORLD;
inline int mpi_rank_count() {
  int inited = 0;
  MPI_Initialized(&inited);
  if (!inited || g_world == MPI_COMM_NULL) return 1;
  int n = 1;
  MPI_Comm_size(g_world, &n);
  return n;
}
inline void mpi_sum(double *buf, int n) {
  MPI_Allreduce(MPI_IN_PLACE, buf, n, MPI_DOUBLE, MPI_SUM, g_world);
}
inline int mpi_rank() {
  int inited = 0;
  MPI_Initialized(&inited);
  if (!inited || g_world == MPI_COMM_NULL) return 0;
  int r = 0;
  MPI_Comm_rank(g_world, &r);
  return r;
}
// Two-direction halo exchange with the periodic neighbours.
//
// `send_prev` goes to prev as ITS `recv_next` (same tag), and `send_next` goes
// to next as ITS `recv_prev`.  The two directions carry DIFFERENT kinds of
// plane, so they must never be paired with each other:
//
//   * at np == 2 prev == next == the same rank, and a single-tag Sendrecv pair
//     pairs my LOWER-ghost send with the peer's recv that expects its UPPER
//     ghost.  Both are the same byte count, so MPI does not complain; the
//     result is a silent lower/upper swap (measured: ~1% spectral-norm error,
//     0.26% in E_lr, zero at np == 1);
//   * distinct tags inside two Sendrecv calls DEADLOCK, because the peer posts
//     the matching receive in its *other* call.
//
// Non-blocking calls with one tag per direction fix both: every recv is posted
// before any wait, and the tags make the pairing unambiguous at any np.
inline void mpi_exchange2(const double *send_prev, const double *send_next,
                          double *recv_prev, double *recv_next, int count,
                          int prev, int next, int tag_prev, int tag_next) {
  MPI_Request req[4];
  int n = 0;
  MPI_Irecv(recv_prev, count, MPI_DOUBLE, prev, tag_next, g_world, &req[n++]);
  MPI_Irecv(recv_next, count, MPI_DOUBLE, next, tag_prev, g_world, &req[n++]);
  MPI_Isend(send_prev, count, MPI_DOUBLE, prev, tag_prev, g_world, &req[n++]);
  MPI_Isend(send_next, count, MPI_DOUBLE, next, tag_next, g_world, &req[n++]);
  MPI_Waitall(n, req, MPI_STATUSES_IGNORE);
}
#else
inline int mpi_rank_count() { return 1; }
inline void mpi_sum(double *, int) {}
inline int mpi_rank() { return 0; }
inline void mpi_exchange2(const double *, const double *, double *, double *,
                          int, int, int, int, int) {}
#endif

// ── process-wide NCCL bootstrap (shared by neutralize/qmean/solve_auto) ─────
// The first collective of the step is neutralize's owned mean, so the comm is
// initialized there (or in solve_auto when the Python harness calls it
// directly).  Any failure leaves the MPI fallback in place.
static int g_nccl_tried = 0, g_nccl_ok = 0;
static bool ensure_nccl_global() {
  if (g_nccl_tried) return g_nccl_ok != 0;
  g_nccl_tried = 1;
  const int np = mpi_rank_count();
  if (np <= 1) { g_nccl_ok = 0; return false; }
#ifdef USE_MPI
  // EVERY rank must agree before any NCCL collective, otherwise the ranks that
  // cannot init block the others inside ncclCommInitRank.  A rank is ready only
  // if the user did not disable NCCL and the library actually loaded here.
  int local_ok = 0;
  if (!(std::getenv("SOG_NCCL") != nullptr &&
        std::atoi(std::getenv("SOG_NCCL")) == 0)) {
    local_ok = sog_gpu_nccl_available() ? 1 : 0;
  }
  int all_ok = 0;
  MPI_Allreduce(&local_ok, &all_ok, 1, MPI_INT, MPI_MIN, g_world);
  if (!all_ok) { g_nccl_ok = 0; return false; }
  // This deployment is single-node; NCCL's bootstrap otherwise picks a real
  // NIC and crashes.  Only default the loopback variables for a single-node
  // job and only when the caller has not set them.
  if (std::getenv("SOG_NCCL_NOENV") == nullptr) {
    int world = 0, shared = 0;
    MPI_Comm_size(g_world, &world);
    MPI_Comm sh = MPI_COMM_NULL;
    MPI_Comm_split_type(g_world, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &sh);
    if (sh != MPI_COMM_NULL) {
      MPI_Comm_size(sh, &shared);
      MPI_Comm_free(&sh);
    }
    if (shared == world) {
      if (std::getenv("NCCL_SOCKET_IFNAME") == nullptr)
        setenv("NCCL_SOCKET_IFNAME", "lo", 1);
      if (std::getenv("NCCL_IB_DISABLE") == nullptr)
        setenv("NCCL_IB_DISABLE", "1", 1);
    }
  }
  int ok = 0;
  unsigned char id[128] = {0};
  if (mpi_rank() == 0) ok = (sog_gpu_nccl_get_id(id) == 0) ? 1 : 0;
  MPI_Bcast(&ok, 1, MPI_INT, 0, g_world);
  if (ok) {
    MPI_Bcast(id, 128, MPI_BYTE, 0, g_world);
    ok = (sog_gpu_nccl_init(np, mpi_rank(), id) == 0) ? 1 : 0;
  }
  g_nccl_ok = ok;
#else
  g_nccl_ok = 0;
#endif
  return g_nccl_ok != 0;
}

// In-place all-reduce of a contiguous fp64 CUDA tensor (device NCCL, or one
// host round trip when NCCL is unavailable).
static void global_sum_dev(at::Tensor &t) {
  if (mpi_rank_count() <= 1 || t.numel() == 0) return;
  if (ensure_nccl_global()) {
    TORCH_CHECK(sog_gpu_nccl_allreduce_dev(t.data_ptr<double>(),
                                           static_cast<size_t>(t.numel())) == 0,
                "sogkspace: NCCL reduction failed");
    return;
  }
  at::Tensor c = t.to(at::kCPU).contiguous();
  mpi_sum(c.data_ptr<double>(), static_cast<int>(t.numel()));
  t.copy_(c.to(t.device()));
}

Entry &entry(int64_t h) {
  auto it = g_states.find(h);
  TORCH_CHECK(it != g_states.end(), "sogkspace: unknown handle ", h,
              " (call setup first)");
  return it->second;
}

void setup_impl(int64_t h, int64_t nx, int64_t ny, int64_t nz,
                const at::Tensor &ge, const at::Tensor &gf,
                const at::Tensor &gs, const at::Tensor &gv,
                const at::Tensor &gsv) {
  std::lock_guard<std::mutex> lock(g_mu);
  Entry &e = g_states[h];
  if (e.s == nullptr) {
    e.s = sog_gpu_create();
    TORCH_CHECK(e.s != nullptr, "sogkspace: sog_gpu_create failed");
  }
  auto to_host = [](const at::Tensor &t) {
    at::Tensor c = t.detach().to(at::kCPU).contiguous().to(at::kDouble);
    const double *p = c.data_ptr<double>();
    return std::vector<double>(p, p + c.numel());
  };
  e.ge = to_host(ge);
  e.gf = to_host(gf);
  e.gv = to_host(gv);
  // `gs` / `gsv` (the bare K and K_v "self" tables) are still accepted so the
  // op schema -- and therefore every already-frozen .pt2 graph -- stays valid,
  // but they are no longer used: the self-energy term is closed-form now and
  // needs no per-mode table.  See sog_gpu.cu::sog_gpu_solve_dev.
  (void)gs;
  (void)gsv;
  const size_t ngrid = static_cast<size_t>(nx) * ny * nz;
  TORCH_CHECK(static_cast<size_t>(e.ge.size()) == ngrid,
              "sogkspace: green_energy size ", e.ge.size(), " != ngrid ", ngrid);
  e.nx = static_cast<int>(nx);
  e.ny = static_cast<int>(ny);
  e.nz = static_cast<int>(nz);
  sog_gpu_setup(e.s, e.nx, e.ny, e.nz, e.ge.data(), e.gf.data(), e.gv.data());
}

// q      : (N,)   contiguous double, CUDA
// pos    : (N,3)  contiguous double, CUDA   (absolute coords)
// box    : (9,)   contiguous double, CUDA   row-major h [x y z | xy xz yz]
// returns (v (N,), F (N,3), vir6 (6,))
std::vector<at::Tensor> solve_impl(int64_t h, const at::Tensor &q,
                                   const at::Tensor &pos, const at::Tensor &box,
                                   double qscale, int64_t spline,
                                   double self_coeff,
                                   bool remove_self) {
  Entry &e = entry(h);
  const int nlocal = static_cast<int>(q.numel());
  TORCH_CHECK(pos.numel() == static_cast<int64_t>(nlocal) * 3,
              "sogkspace: pos must be (N,3)");
  const at::cuda::OptionalCUDAGuard guard(at::device_of(q));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // Orthogonal cells only: the device kernels take (lx, ly, lz, boxlo).
  at::Tensor b = box.detach().to(at::kCPU).contiguous().to(at::kDouble);
  const double *bh = b.data_ptr<double>();
  TORCH_CHECK(bh[1] == 0.0 && bh[2] == 0.0 && bh[3] == 0.0 && bh[5] == 0.0 &&
                  bh[6] == 0.0 && bh[7] == 0.0,
              "sogkspace: triclinic cells are not supported by the device "
              "kernels yet");
  const double lx = bh[0], ly = bh[4], lz = bh[8];

  // The kernels expect x in [0, L) with boxlo = 0.  The caller wraps on device
  // (sog_op.py) so this op never round-trips coordinates through the host.
  at::Tensor x = pos.detach().contiguous().to(at::kDouble);

  at::Tensor v = at::empty({nlocal}, q.options().dtype(at::kDouble));
  at::Tensor F = at::empty({nlocal, 3}, q.options().dtype(at::kDouble));
  at::Tensor vir = at::empty({6}, q.options().dtype(at::kDouble));
  at::Tensor vir_cpu = at::empty({6}, q.options().dtype(at::kDouble).device(at::kCPU));
  double vir6[6] = {0, 0, 0, 0, 0, 0};
  const double boxlo[3] = {0.0, 0.0, 0.0};
  at::Tensor qs = q.detach().contiguous().to(at::kDouble);
  at::Tensor xs = x.contiguous();
  // Sigma q^2 for the self term, reduced on device inside the opaque op so the
  // traced graph never sees a host-side scalar.
  const double qsqsum = at::sum(qs * qs).item<double>();
  sog_gpu_compute_dev(e.s, nlocal, xs.data_ptr<double>(), boxlo, lx, ly, lz,
                      qs.data_ptr<double>(), qscale, static_cast<int>(spline),
                      /*want_potential=*/1, F.data_ptr<double>(), vir6,
                      v.data_ptr<double>(), self_coeff, qsqsum,
                      remove_self ? 1 : 0);
  (void)stream;
  // sog_gpu_compute_dev returns the 6-virial on the HOST; `vir` must be copied
  // into device memory (never memcpy into a CUDA tensor).
  std::memcpy(vir_cpu.data_ptr<double>(), vir6, 6 * sizeof(double));
  vir.copy_(vir_cpu);
  return {v, F, vir};
}

// ── self-contained variant: grid sizing + green build + box_tol caching all
//    inside the op, so a consumer only supplies (q, pos, box) and the SOG
//    parameters.  This is what production needs: the deployed chain gets this
//    behaviour from SOGKSpace::ensure_fft_plan, including the box_tol rule that

// ── Distributed (slab) k-space solve ────────────────────────────────────────
// Each rank owns a z-slab of the mesh, so per-rank k-space memory is n_g/np
// instead of n_g.  The solve stays INSIDE the frozen graph, so the caller
// still gets one forward and one backward for SR+LR.
//
// Layouts (match sog_gpu.cu):
//   brick (nx, ny, ZB)  z slowest, owned + ghost planes in z
//   "xy"  (nx, ny, ZL)  z slowest, one contiguous (x,y) plane per local z
//   "z"   (nz, nx, YL)  z FASTEST, so the z transform is one contiguous batch
//
// Invariants that MUST hold (SOG_DISTRIBUTED_MESH_THEORY.md): global n_g in
// scaleinv/rho_scale, global grid index for the green tables, global reduction
// of the accumulators.  Breaking any of them still yields a plausible number.
static double solve_auto_dist(Entry &e, int nlocal, const double *x,
                              const double *q, double lx, double ly, double lz,
                              double qscale, int spline, double self_coeff,
                              double qsqsum, bool remove_self, double *v_out,
                              double *F_out, double vir6[6], bool full_brick) {
  const int nx = e.nx, ny = e.ny, nz = e.nz;
  const int np = mpi_rank_count(), me = mpi_rank();
  const double volume = lx * ly * lz;
  const size_t ngrid = (size_t)nx * ny * nz;
  const double rho_scale = (double)ngrid / volume;
  const double scaleinv = 1.0 / (double)ngrid;

  const int zlo = me * nz / np, zhi = (me + 1) * nz / np - 1, ZL = (me + 1) * nz / np - me * nz / np;
  // The (x,y) transform is real-to-complex: only Nh = nx/2+1 columns along x
  // are stored and transported (the -kx half is implicit).
  const int Nh = nx / 2 + 1;
  const int ylo = me * ny / np, YL = (me + 1) * ny / np - me * ny / np;
  // Ghost width = the interpolation ORDER on both sides, exactly as LAMMPS's
  // Grid3d does.  Using the one-sided stencil reach (1 below / 2 above for
  // spline 4) is WRONG here: LAMMPS assigns atoms to the CONTINUOUS
  // subdomain [me*L/np, (me+1)*L/np), while this mesh slab is
  // [me*n_g/np, (me+1)*n_g/np).  When n_g/np is not an integer the two
  // boundaries differ by up to one grid cell, so an atom owned by this rank
  // can sit at iz0 = zhi+1 and its stencil then reaches zhi+nord-1.  Measured
  // symptom of the too-narrow ghost: a 1.3% spectral norm error (and 0.26% in
  // E_lr) at np=2 that is zero at np=1.  PPPM avoids it with ghost = nord.
  const int nord_gh = (spline == 6) ? 6 : 4;
  const int gh_lo = nord_gh, gh_hi = nord_gh;
  // full_brick: a processor grid that is not 1x1xnp.  Every rank spreads into
  // the FULL z range and the density/potential are made global with one
  // all-reduce each; the transforms still use this rank's 1-D slab.
  const int zb_lo = full_brick ? -gh_lo : zlo - gh_lo;
  const int zb_hi = full_brick ? nz - 1 + gh_hi : zhi + gh_hi;
  const int ZB = full_brick ? nz + gh_lo + gh_hi
                            : ZB_of(zhi - zlo + 1, gh_lo, gh_hi);
  const int plane = nx * ny;
  TORCH_CHECK(ZL >= gh_hi, "sogkspace: slab too thin for the stencil");

  // The mesh is split by MPI rank, which is only correct when the framework's
  // atom decomposition is 1 x 1 x np (rank == z-slab).  Verify it cheaply:
  // every local atom must lie in this rank's continuous z-subdomain, whose grid
  // coordinate range is [zlo, zhi+1].  With a grid split along x (LAMMPS'
  // default when it has a choice) the atoms span the whole box, and without
  // this check the solver silently returns a density with the wrong shape.
  if (np > 1 && !full_brick) {
    // The caller may hand us ghost atoms as well; they carry q == 0 and do not
    // enter the density, so they must not be range-checked (a rank's ghosts
    // span the whole box).
    double zmin = 1e300, zmax = -1e300;
    for (int i = 0; i < nlocal; ++i) {
      if (q[i] == 0.0) continue;
      double fz = x[3 * i + 2] / lz * nz;   // boxlo is (0,0,0) in this solver
      fz -= nz * std::floor(fz / nz);
      zmin = std::min(zmin, fz);
      zmax = std::max(zmax, fz);
    }
    // Compare against the CONTINUOUS z-subdomain [me*nz/np, (me+1)*nz/np],
    // not the (even-aligned) mesh slab, so this only fires for a grid that
    // does not split z at all.
    const double zgb = (double)me * nz / np, zgt = (double)(me + 1) * nz / np;
    TORCH_CHECK(zmin > zmax || (zmin >= zgb - 1.0 && zmax <= zgt + 1.0),
                "sogkspace distributed mesh: rank ", me, " owns atoms spanning "
                "z-grid [", zmin, ", ", zmax, "] but its slab is [", zlo, ", ",
                zhi, "]. The distributed solver needs the LAMMPS processor grid "
                "to be 1 x 1 x nprocs (rank == z slab); add `processors 1 1 ",
                np, "` to the input.");
  }
  const int prev = (me - 1 + np) % np, next = (me + 1) % np;
  const double boxlo[3] = {0.0, 0.0, 0.0};
  const int gp = std::max(gh_lo, gh_hi);
  const bool dbg = std::getenv("SOG_DIST_DEBUG") != nullptr;
  auto t_prev = std::chrono::steady_clock::now();
  auto mark = [&](const char *what) {
    if (dbg) {
      const auto now = std::chrono::steady_clock::now();
      const double ms =
          std::chrono::duration<double, std::milli>(now - t_prev).count();
      t_prev = now;
      std::fprintf(stderr,
                   "[sog-dist] r%d/%d %s +%.3f ms | z %d..%d ZL %d | y %d YL %d"
                   " | gh %d/%d\n",
                   me, np, what, ms, zlo, zhi, ZL, ylo, YL, gh_lo, gh_hi);
      std::fflush(stderr);
    }
  };
  mark("enter");

  // ---- 1. spread into the DEVICE brick (no download) ----
  // The brick stays on the GPU across the halo exchanges: only the ghost
  // planes cross the bus, and the slab conversions are kernels.  The host
  // arena holds only the two all-to-all message matrices and the small halo
  // buffers.
  // With NCCL the packed messages are device buffers (d_pot / d_rho); the host
  // matrices are only needed by the MPI fallback.
  const bool use_nccl = (np > 1) && (sog_gpu_nccl_ok() != 0);
  const size_t n_snd = use_nccl ? 0 : 2 * (size_t)Nh * ny * ZL;
  const size_t n_rcv = use_nccl ? 0 : 2 * (size_t)nz * Nh * YL;
  const size_t n_halo4 = 4 * (size_t)plane * gp;
  e.arena.ensure(2 * n_snd + 2 * n_rcv + n_halo4 + 64);
  double *snd = n_snd ? e.arena.alloc(n_snd) : nullptr;
  double *back = n_snd ? e.arena.alloc(n_snd) : nullptr;
  double *rcv = n_rcv ? e.arena.alloc(n_rcv) : nullptr;
  double *snd_back = n_rcv ? e.arena.alloc(n_rcv) : nullptr;
  double *h0 = e.arena.alloc((size_t)plane * gp);
  double *h1 = e.arena.alloc((size_t)plane * gp);
  double *h2 = e.arena.alloc((size_t)plane * gp);
  double *h3 = e.arena.alloc((size_t)plane * gp);
  const int zoff = zlo - zb_lo;                  // brick plane of owned start
  const size_t pl = (size_t)plane;
  sog_gpu_spread_brick_dev(e.s, nlocal, x, boxlo, lx, ly, lz, q, rho_scale,
                           spline, nx, ny, nz, 0, nx - 1, 0, ny - 1, zb_lo,
                           zb_hi);
  mark("spread done");
  // ---- 2. reverse halo on the device brick (overlap-add) ----
  if (full_brick) {
    // No z neighbour: fold the periodic images inside this rank's own full
    // brick, then sum the bricks so every rank holds the global density.
    for (int j = 0; j < gh_lo; ++j) {          // lower ghost -> last owned
      sog_gpu_brick_get(e.s, h0, pl * (size_t)j, pl);
      sog_gpu_brick_add(e.s, h0, pl * (size_t)(nz + j), pl);
    }
    for (int j = 0; j < gh_hi; ++j) {          // upper ghost -> first owned
      sog_gpu_brick_get(e.s, h0, pl * (size_t)(nz + gh_lo + j), pl);
      sog_gpu_brick_add(e.s, h0, pl * (size_t)(gh_lo + j), pl);
    }
#ifdef USE_MPI
    // Each rank needs only ITS slab of the global density: a reduce-scatter
    // (half the traffic of an all-reduce).  The slab chunks tile [0,nz) in
    // rank order, so the send buffer is just the owned planes.
    std::vector<int> zcnt(np), zdisp(np);
    for (int r = 0; r < np; ++r) {
      const int lo = r * nz / np, hi = (r + 1) * nz / np;
      zcnt[r] = (hi - lo) * plane;
      zdisp[r] = lo * plane;
    }
    std::vector<double> owned((size_t)nz * plane), mine((size_t)ZL * plane);
    sog_gpu_brick_get(e.s, owned.data(), (size_t)gh_lo * plane,
                      owned.size());
    MPI_Reduce_scatter(owned.data(), mine.data(), zcnt.data(), MPI_DOUBLE,
                       MPI_SUM, g_world);
    sog_gpu_brick_set(e.s, mine.data(), (size_t)zoff * plane, mine.size());
#endif
    mark("full-brick density sum done");
  } else if (np == 1) {
    // Lower ghosts are periodic images of the last owned planes and vice
    // versa; fold them across explicitly.
    sog_gpu_brick_get(e.s, h0, 0, pl * gh_lo);
    sog_gpu_brick_add(e.s, h0, pl * (size_t)(zoff + ZL - gh_lo), pl * gh_lo);
    sog_gpu_brick_get(e.s, h0, pl * (size_t)(zoff + ZL), pl * gh_hi);
    sog_gpu_brick_add(e.s, h0, pl * (size_t)zoff, pl * gh_hi);
  } else if (use_nccl) {
    TORCH_CHECK(sog_gpu_nccl_halo_rev(e.s, prev, next, pl, zoff, ZL, gh_lo,
                                      gh_hi) == 0,
                "sogkspace: NCCL reverse halo failed");
  } else {
    std::fill(h0, h0 + (size_t)plane * gp, 0.0);
    std::fill(h1, h1 + (size_t)plane * gp, 0.0);
    std::fill(h2, h2 + (size_t)plane * gp, 0.0);
    std::fill(h3, h3 + (size_t)plane * gp, 0.0);
    sog_gpu_brick_get(e.s, h0, 0, pl * gh_lo);                       // lower ghost
    sog_gpu_brick_get(e.s, h1, pl * (size_t)(zoff + ZL), pl * gh_hi);  // upper ghost
    // recv_prev = prev's upper ghost -> my first owned planes;
    // recv_next = next's lower ghost -> my last owned planes.
    mpi_exchange2(h0, h1, h3, h2, plane * gp, prev, next, 11, 12);
    sog_gpu_brick_add(e.s, h3, pl * (size_t)zoff, pl * gh_hi);
    sog_gpu_brick_add(e.s, h2, pl * (size_t)(zoff + ZL - gh_lo), pl * gh_lo);
  }
  mark("reverse halo done");
  // ---- 3-5. (x,y) FFT + device pack, then the all-to-all ----
  std::vector<int> scnt(np), sdisp(np), rcnt(np), rdisp(np);
  for (int d = 0; d < np; ++d)
    scnt[d] = Nh * ((d + 1) * ny / np - d * ny / np) * ZL * 2;
  for (int r = 0; r < np; ++r)
    rcnt[r] = Nh * YL * ((r + 1) * nz / np - r * nz / np) * 2;
  sdisp[0] = 0; for (int d = 1; d < np; ++d) sdisp[d] = sdisp[d - 1] + scnt[d - 1];
  rdisp[0] = 0; for (int r = 1; r < np; ++r) rdisp[r] = rdisp[r - 1] + rcnt[r - 1];
  TORCH_CHECK(sog_gpu_slab_setup(e.s, nx, ny, nz, ZL, YL) == 0,
              "sogkspace: slab cuFFT plan failed");
  double acc[7] = {0, 0, 0, 0, 0, 0, 0};
  if (use_nccl) {
    // Device-resident: the packed blocks never leave the GPU.
    TORCH_CHECK(sog_gpu_slab_xyfft_pack_brick_dev(e.s, nx, ny, nz, ZL, np,
                                                  zoff) == 0,
                "sogkspace: slab (x,y) FFT + pack from the brick failed");
    mark("fft xy + pack done");
    TORCH_CHECK(sog_gpu_nccl_alltoall_fwd(e.s, scnt.data(), sdisp.data(),
                                          rcnt.data(), rdisp.data(), np) == 0,
                "sogkspace: NCCL forward all-to-all failed");
    mark("alltoallv done");
    TORCH_CHECK(sog_gpu_slab_unpack_green_izpack_dev(
                    e.s, nx, ny, nz, ylo, YL, lx, ly, lz, e.ge.data(),
                    e.gf.data(), e.gv.data(), scaleinv, np) == 0,
                "sogkspace: slab unpack / z FFT / green / inverse failed");
    mark("unpack+zfft+green+izpack done");
    TORCH_CHECK(sog_gpu_nccl_allreduce_red(e.s) == 0,
                "sogkspace: NCCL accumulator reduction failed");
    // Do NOT download yet: the copy is a full sync.  It is done just before
    // the energy/virial formulas so the reverse all-to-all, unpack and gather
    // can be enqueued first.
    mark("sum done");
    TORCH_CHECK(sog_gpu_nccl_alltoall_bwd(e.s, rcnt.data(), rdisp.data(),
                                          scnt.data(), sdisp.data(), np) == 0,
                "sogkspace: NCCL inverse all-to-all failed");
    TORCH_CHECK(sog_gpu_slab_unpack_ixyfft_to_brick_dev(
                    e.s, nx, ny, nz, ZL, np, zoff, ZB) == 0,
                "sogkspace: slab unpack + inverse (x,y) FFT to the brick failed");
    mark("inverse done");
  } else {
    TORCH_CHECK(sog_gpu_slab_xyfft_pack_brick(e.s, nx, ny, nz, ZL, np, zoff,
                                              snd) == 0,
                "sogkspace: slab (x,y) FFT + pack from the brick failed");
    mark("fft xy + pack done");
    MPI_Alltoallv(snd, scnt.data(), sdisp.data(), MPI_DOUBLE, rcv,
                  rcnt.data(), rdisp.data(), MPI_DOUBLE, g_world);
    mark("alltoallv done");
    // ---- 6. unpack -> FFT(z) -> green -> FFT^-1(z) -> pack ----
    TORCH_CHECK(sog_gpu_slab_unpack_green_izpack(
                    e.s, nx, ny, nz, ylo, YL, lx, ly, lz, e.ge.data(),
                    e.gf.data(), e.gv.data(), scaleinv, np, rcv, snd_back,
                    acc) == 0,
                "sogkspace: slab unpack / z FFT / green / inverse failed");
    mark("unpack+zfft+green+izpack done");
    mpi_sum(acc, 7);
    mark("sum done");
    MPI_Alltoallv(snd_back, rcnt.data(), rdisp.data(), MPI_DOUBLE,
                  back, scnt.data(), sdisp.data(), MPI_DOUBLE, g_world);
    // ---- 7. unpack -> FFT^-1(x,y) -> device brick ----
    TORCH_CHECK(sog_gpu_slab_unpack_ixyfft_to_brick(e.s, nx, ny, nz, ZL, np,
                                                    zoff, ZB, back) == 0,
                "sogkspace: slab unpack + inverse (x,y) FFT to the brick failed");
    mark("inverse done");
  }
  // ---- 8. forward halo on the device brick ----
  if (full_brick) {
    // Sum the disjoint 1-D slabs into the full potential, then fill the
    // periodic ghosts from the owned edge planes.
#ifdef USE_MPI
    // Every rank needs the full potential: all-gather the disjoint slabs
    // (half the traffic of an all-reduce).
    std::vector<int> zcnt(np), zdisp(np);
    for (int r = 0; r < np; ++r) {
      const int lo = r * nz / np, hi = (r + 1) * nz / np;
      zcnt[r] = (hi - lo) * plane;
      zdisp[r] = lo * plane;
    }
    std::vector<double> mine((size_t)ZL * plane), full((size_t)nz * plane);
    sog_gpu_brick_get(e.s, mine.data(), (size_t)zoff * plane, mine.size());
    MPI_Allgatherv(mine.data(), (int)mine.size(), MPI_DOUBLE, full.data(),
                   zcnt.data(), zdisp.data(), MPI_DOUBLE, g_world);
    sog_gpu_brick_set(e.s, full.data(), (size_t)gh_lo * plane, full.size());
#endif
    for (int j = 0; j < gh_lo; ++j) {
      sog_gpu_brick_get(e.s, h0, pl * (size_t)(nz + j), pl);
      sog_gpu_brick_set(e.s, h0, pl * (size_t)j, pl);
    }
    for (int j = 0; j < gh_hi; ++j) {
      sog_gpu_brick_get(e.s, h0, pl * (size_t)(gh_lo + j), pl);
      sog_gpu_brick_set(e.s, h0, pl * (size_t)(nz + gh_lo + j), pl);
    }
  } else if (np == 1) {
    sog_gpu_brick_get(e.s, h0, pl * (size_t)zoff, pl * gh_hi);   // first owned
    sog_gpu_brick_set(e.s, h0, pl * (size_t)(zoff + ZL), pl * gh_hi);
    sog_gpu_brick_get(e.s, h0, pl * (size_t)(zoff + ZL - gh_lo), pl * gh_lo);
    sog_gpu_brick_set(e.s, h0, 0, pl * gh_lo);
  } else if (use_nccl) {
    TORCH_CHECK(sog_gpu_nccl_halo_fwd(e.s, prev, next, pl, zoff, ZL, gh_lo,
                                      gh_hi) == 0,
                "sogkspace: NCCL forward halo failed");
  } else {
    std::fill(h0, h0 + (size_t)plane * gp, 0.0);
    std::fill(h1, h1 + (size_t)plane * gp, 0.0);
    std::fill(h2, h2 + (size_t)plane * gp, 0.0);
    std::fill(h3, h3 + (size_t)plane * gp, 0.0);
    sog_gpu_brick_get(e.s, h0, pl * (size_t)zoff, pl * gh_hi);   // first owned
    sog_gpu_brick_get(e.s, h1, pl * (size_t)(zoff + ZL - gh_lo), pl * gh_lo);
    mpi_exchange2(h0, h1, h2, h3, plane * gp, prev, next, 13, 14);
    sog_gpu_brick_set(e.s, h2, 0, pl * gh_lo);                   // lower ghost
    sog_gpu_brick_set(e.s, h3, pl * (size_t)(zoff + ZL), pl * gh_hi);  // upper
  }
  mark("forward halo done");
  // ---- 9. gather, then the analytic self term on the potential ----
  std::vector<double> force(3 * (size_t)nlocal), vpot((size_t)nlocal);
  sog_gpu_gather_brick_dev(e.s, nlocal, x, boxlo, lx, ly, lz, q, qscale,
                           spline, nx, ny, nz, 0, nx - 1, 0, ny - 1, zb_lo,
                           zb_hi, force.data(), vpot.data());
  const double rsi = remove_self ? 2.0 * self_coeff : 0.0;
  for (int i = 0; i < nlocal; ++i) {
    v_out[i] = vpot[i] - qscale * q[i] * rsi;
    F_out[3 * i + 0] = force[3 * i + 0];
    F_out[3 * i + 1] = force[3 * i + 1];
    F_out[3 * i + 2] = force[3 * i + 2];
  }
  // ---- 10. energy / virial: same formulas as the replicated path ----
  if (use_nccl) sog_gpu_red_download(e.s, acc);   // one sync, as late as possible
  const double e_kneq0 = 0.5 * volume * acc[0];
  const double e_self = remove_self ? self_coeff * qsqsum : 0.0;
  for (int j = 0; j < 6; ++j) vir6[j] = 0.5 * volume * qscale * acc[1 + j];
  if (dbg) {
    std::fprintf(stderr, "[sog-dist] r%d acc:", me);
    for (int t = 0; t < 7; ++t) std::fprintf(stderr, " %.4e", acc[t]);
    std::fprintf(stderr, "\n");
    double vmax = 0, fmax = 0;
    for (int i = 0; i < nlocal; ++i) {
      vmax = std::max(vmax, std::fabs(v_out[i]));
      for (int c = 0; c < 3; ++c) fmax = std::max(fmax, std::fabs(F_out[3 * i + c]));
    }
    auto stat = [](const std::vector<double> &v, const char *nm) {
      double mx = 0.0;
      long bad = 0;
      for (double t : v) {
        if (!std::isfinite(t)) ++bad;
        else mx = std::max(mx, std::fabs(t));
      }
      std::fprintf(stderr, "   %s: n=%zu max=%.4e nonfinite=%ld\n", nm, v.size(), mx, bad);
      std::fflush(stderr);
    };
    std::fprintf(stderr, "[sog-dist] r%d vmax=%.4e fmax=%.4e\n", me, vmax, fmax);
    stat(e.ge, "ge");
    stat(e.gf, "gf");
    stat(e.gv, "gv");
    std::fflush(stderr);
  }
  mark("done");
  if (dbg) {
    std::fprintf(stderr, "[sog-dist] r%d E_lr=%.9f e_k=%.9f e_self=%.9f vir=", me,
                 qscale * (e_kneq0 - e_self), qscale * e_kneq0, qscale * e_self);
    for (int j = 0; j < 6; ++j) std::fprintf(stderr, " %.9e", vir6[j]);
    std::fprintf(stderr, "\n");
    std::fflush(stderr);
  }
  return qscale * (e_kneq0 - e_self);
}

//    keeps NPT from rebuilding the 3D green table every step (~60 ms).
std::vector<at::Tensor> solve_auto_impl(int64_t h, const at::Tensor &q,
                                        const at::Tensor &pos, const at::Tensor &box,
                                        const at::Tensor &amp_t, const at::Tensor &bw_t,
                                        double rcut, int64_t spline,
                                        double phi_accuracy, double box_tol,
                                        double qscale, double self_coeff,
                                        bool remove_self) {
  std::lock_guard<std::mutex> lock(g_mu);
  Entry &e = g_states[h];
  if (e.s == nullptr) {
    e.s = sog_gpu_create();
    TORCH_CHECK(e.s != nullptr, "sogkspace: sog_gpu_create failed");
  }
  const int nlocal = static_cast<int>(q.numel());
  TORCH_CHECK(pos.numel() == static_cast<int64_t>(nlocal) * 3,
              "sogkspace: pos must be (N,3)");

  at::Tensor b = box.detach().to(at::kCPU).contiguous().to(at::kDouble);
  const double *bh = b.data_ptr<double>();
  TORCH_CHECK(bh[1] == 0.0 && bh[2] == 0.0 && bh[3] == 0.0 && bh[5] == 0.0 &&
                  bh[6] == 0.0 && bh[7] == 0.0,
              "sogkspace: tilted cells are not supported (the deployed SOG "
              "solver rejects them too: 'sog supports only orthorhombic boxes')");
  const double lx = bh[0], ly = bh[4], lz = bh[8];

  auto to_host = [](const at::Tensor &t) {
    at::Tensor c = t.detach().to(at::kCPU).contiguous().to(at::kDouble);
    const double *p = c.data_ptr<double>();
    return std::vector<double>(p, p + c.numel());
  };
  std::vector<double> amp = to_host(amp_t);
  std::vector<double> bw = to_host(bw_t);
  TORCH_CHECK(amp.size() == bw.size() && !amp.empty(),
              "sogkspace: amp/bandwidth must be non-empty and the same length");

  // ── Exact self-energy coefficient ───────────────────────────────────────
  // The k-space energy is the Fourier form of a pair potential; summing over
  // the complete reciprocal lattice reproduces the pair sum, whose
  // (i == j, R == 0) term is 0.5 * qsum_sq * g(0) with g(0) the real-space
  // multi-Gaussian kernel at zero separation:
  //     g(0) = sum_m amp_m (2 pi bw_m)^(-3/2),
  // so E_self = self_coeff * qsum_sq with self_coeff = 0.5 * that sum.  This is
  // algebraically identical to the RBSOG closed form in sog.cpp (verified to
  // 15 digits).  `bw` here is the SQUARED bandwidth -- make_model passes
  // fit.get_bandwidth_sq().  Do NOT use the FFT mesh sum instead: several bands
  // are extremely narrow in k, so a finite mesh under-counts g(0) by a
  // box-dependent amount and the energy stops being volume-extensive.
  //
  // The `self_coeff` ARGUMENT is kept, and added, so the op schema and every
  // frozen .pt2 graph stay valid; the deployed make_model passes 0.0.
  double self_coeff_exact = self_coeff;
  for (size_t m = 0; m < amp.size(); ++m) {
    self_coeff_exact += 0.5 * amp[m] * std::pow(2.0 * M_PI * bw[m], -1.5);
  }

  // ── grid sizing (deployed phi_max rule) ──
  const int nx = sog_green::grid_axis(lx, bw, rcut, static_cast<int>(spline), phi_accuracy);
  const int ny = sog_green::grid_axis(ly, bw, rcut, static_cast<int>(spline), phi_accuracy);
  const int nz = sog_green::grid_axis(lz, bw, rcut, static_cast<int>(spline), phi_accuracy);

  const bool grid_same = (nx == e.nx && ny == e.ny && nz == e.nz);
  const bool params_same = (e.amp == amp && e.bw == bw && e.spline == spline);
  double drift = 1.0;
  if (e.have_green) {
    drift = std::max({std::fabs(lx - e.glx) / std::max(std::fabs(e.glx), 1e-30),
                      std::fabs(ly - e.gly) / std::max(std::fabs(e.gly), 1e-30),
                      std::fabs(lz - e.glz) / std::max(std::fabs(e.glz), 1e-30)});
  }
  // ── mesh strategy ───────────────────────────────────────────────────────
  // Decided BEFORE the green tables are built: the distributed solver keeps a
  // y-slice of them, the replicated one keeps the full grid, and the two
  // layouts cannot be mixed inside one state.
  const int nranks_pre = mpi_rank_count();
  // The distributed solver assumes LAMMPS owns atoms in z-slabs, i.e. a
  // `processors 1 1 nprocs` grid.  The pair exports the actual grid in
  // SOG_PROCGRID=px,py,pz,ix,iy,iz; when it is not a z-only grid we keep the
  // replicated mesh (correct, just slower) instead of aborting.  SOG_DIST_MESH
  // still overrides: =1 forces (and errors on an incompatible grid), =0 forces
  // the replicated path.
  bool dist_env_set = false, dist_env_val = false;
  if (const char *e = std::getenv("SOG_DIST_MESH")) {
    dist_env_set = true;
    dist_env_val = std::atoi(e) != 0;
  }
  bool grid_ok = true, full_brick = false;
  if (std::getenv("SOG_FORCE_FULLBRICK") != nullptr) full_brick = true;
  if (const char *pg = std::getenv("SOG_PROCGRID")) {
    int px = 0, py = 0, pz = 0, ix = 0, iy = 0, iz = 0;
    if (std::sscanf(pg, "%d,%d,%d,%d,%d,%d", &px, &py, &pz, &ix, &iy, &iz) == 6) {
      grid_ok = (px > 0 && py > 0 && pz > 0 && px * py * pz == nranks_pre &&
                 ix >= 0 && ix < px && iy >= 0 && iy < py && iz >= 0 && iz < pz);
      // A z-only grid lets each rank own exactly its mesh z-slab (fast halo
      // path).  Any other grid still works, but the density must be global
      // before the (1-D) transforms: spread into the FULL brick, one world
      // all-reduce, then the same slab machinery; the potential is summed back
      // the same way.
      full_brick = grid_ok && !(px == 1 && py == 1 && pz == nranks_pre);
    } else {
      grid_ok = false;
    }
  }
  const bool dist_mode = dist_env_set ? (dist_env_val && nranks_pre > 1)
                                     : (nranks_pre > 1 && grid_ok);
  if (dist_env_set && dist_env_val && nranks_pre > 1 && !grid_ok) {
    TORCH_CHECK(false,
                "sogkspace: SOG_DIST_MESH=1 needs the LAMMPS processor grid "
                "1 x 1 x nprocs (SOG_PROCGRID=", std::getenv("SOG_PROCGRID"),
                "); add `processors 1 1 ", nranks_pre, "` to the input.");
  }
  if (!dist_mode && nranks_pre > 1 && !grid_ok && !dist_env_set) {
    // Auto mode with a grid that is not z-only: the replicated mesh is
    // correct, just slower.  Say so once instead of degrading silently.
    static int logged_grid = 0;
    if (mpi_rank() == 0 && !logged_grid) {
      logged_grid = 1;
      std::fprintf(stderr,
                   "[sog-dist] LAMMPS processor grid is not 1x1xnprocs; using "
                   "the replicated mesh (add `processors 1 1 %d` to use the "
                   "slab mesh)\n", nranks_pre);
    }
  }
  // Distributed mode keeps only this rank's y-slice of the green tables and
  // slab-sized device scratch (sog_gpu_setup_slab); the replicated mode keeps
  // the full grid.  The two layouts must never be mixed inside one state.
  const int rn = nranks_pre, rme = mpi_rank();
  int gylo = 0, gYL = ny;
  if (dist_mode) {
    gylo = rme * ny / rn;
    gYL = (rme + 1) * ny / rn - gylo;
  }
  const bool reuse = e.have_green && grid_same && params_same && drift <= box_tol &&
                     e.dist_mode == dist_mode &&
                     (!dist_mode || (e.slab_ylo == gylo && e.slab_YL == gYL));
  if (!reuse) {
    if (dist_mode) {
      // Phase (b): the slab tables keep only the compact x half-spectrum
      // (Nh = nx/2+1); the kernel indexes kh + Nh*(iyl + YL*iz).
      sog_green::build_green_slab_half(nx, ny, nz, lx, ly, lz, amp, bw,
                                       static_cast<int>(spline), gylo, gYL,
                                       e.ge, e.gf, e.gv);
      const int gZL = (rme + 1) * nz / rn - rme * nz / rn;
      const size_t ggn = static_cast<size_t>(nx / 2 + 1) * gYL * nz;
      sog_gpu_setup_slab(e.s, nx, ny, nz, gYL, gZL, e.ge.data(), e.gf.data(),
                         e.gv.data(), ggn);
    } else {
      sog_green::build_green(nx, ny, nz, lx, ly, lz, amp, bw,
                             static_cast<int>(spline), e.ge, e.gf, e.gv);
      sog_gpu_setup(e.s, nx, ny, nz, e.ge.data(), e.gf.data(), e.gv.data());
    }
    e.dist_mode = dist_mode;
    e.slab_ylo = gylo;
    e.slab_YL = gYL;
    e.nx = nx;
    e.ny = ny;
    e.nz = nz;
    e.glx = lx;
    e.gly = ly;
    e.glz = lz;
    e.amp = amp;
    e.bw = bw;
    e.spline = static_cast<int>(spline);
    e.have_green = true;
  }

  // ── solve ──
  const at::cuda::OptionalCUDAGuard guard(at::device_of(q));
  at::Tensor x = pos.detach().contiguous().to(at::kDouble);
  at::Tensor qs = q.detach().contiguous().to(at::kDouble);
  at::Tensor v = at::empty({nlocal}, q.options().dtype(at::kDouble));
  at::Tensor F = at::empty({nlocal, 3}, q.options().dtype(at::kDouble));
  at::Tensor vir = at::empty({6}, q.options().dtype(at::kDouble));
  at::Tensor vir_cpu = at::empty({6}, q.options().dtype(at::kDouble).device(at::kCPU));
  double vir6[6] = {0, 0, 0, 0, 0, 0};
  const double boxlo[3] = {0.0, 0.0, 0.0};
  // ── multi-rank: replicated full grid, one Allreduce, then the usual
  //    per-rank solve (the hook sog_kokkos.cpp already uses).  Every rank
  //    spreads only its OWN atoms; the summed density is the global one, so
  //    E_k and the virial are identical on all ranks by construction while
  //    v/F are gathered for local atoms only.
  const int nranks = nranks_pre;
  const bool multi = nranks > 1;

  // The full-LR artifact consumes only v/F/vir: the op's own energy (and thus
  // Sum q^2, which enters only the self term of that discarded scalar) is
  // needed for the debug print alone.  Computing it unconditionally costs a
  // full device->host sync every step.
  double qsqsum = 0.0;
  if (std::getenv("SOG_DIST_DEBUG") != nullptr) {
    qsqsum = at::sum(qs * qs).item<double>();
    double qsq_glob = qsqsum;
    mpi_sum(&qsq_glob, 1);      // no-op at np == 1
    qsqsum = qsq_glob;
  }
  if (dist_mode) {
    // NCCL bootstrap (device collectives for the transposes/halos).  Rank 0
    // creates the unique id; every rank gets it with one Bcast.  Any failure
    // (library missing, duplicate devices, no loopback) leaves nccl_state at -1
    // and solve_auto_dist uses the host-staged MPI path.
    if (nranks > 1) ensure_nccl_global();
    at::Tensor xh = x.detach().to(at::kCPU).contiguous().to(at::kDouble);
    at::Tensor qh = qs.detach().to(at::kCPU).contiguous().to(at::kDouble);
    std::vector<double> vh(static_cast<size_t>(nlocal));
    std::vector<double> Fh(3 * static_cast<size_t>(nlocal));
    solve_auto_dist(e, nlocal, xh.data_ptr<double>(), qh.data_ptr<double>(), lx,
                    ly, lz, qscale, static_cast<int>(spline), self_coeff_exact,
                    qsqsum, remove_self, vh.data(), Fh.data(), vir6,
                    full_brick);
    v.copy_(at::from_blob(vh.data(), {nlocal},
                          at::TensorOptions().dtype(at::kDouble)).to(v.device()));
    F.copy_(at::from_blob(Fh.data(), {nlocal, 3},
                          at::TensorOptions().dtype(at::kDouble)).to(F.device()));
    const double share = 1.0 / static_cast<double>(nranks);
    for (int k = 0; k < 6; ++k) vir6[k] *= share;
    std::memcpy(vir_cpu.data_ptr<double>(), vir6, 6 * sizeof(double));
    vir.copy_(vir_cpu);
    return {v, F, vir};
  }
  if (multi) {
    const double volume = lx * ly * lz;
    const size_t ngrid = static_cast<size_t>(e.nx) * e.ny * e.nz;
    const double rho_scale = static_cast<double>(ngrid) / volume;
    sog_gpu_spread_full_dev(e.s, nlocal, x.data_ptr<double>(), boxlo, lx, ly, lz,
                            qs.data_ptr<double>(), rho_scale,
                            static_cast<int>(spline));
    const size_t nrho = 2 * ngrid;
    // CUDA-aware MPI reduces the device mesh in place; otherwise stage through
    // host memory (~31 MB at 125^3), matching sog_kokkos.cpp.
    double *d_rho = getenv("SOG_KK_GPUAWARE_MPI") ? sog_gpu_rho_device_ptr(e.s)
                                                  : nullptr;
    if (d_rho != nullptr) {
#ifdef USE_MPI
      MPI_Allreduce(MPI_IN_PLACE, d_rho, static_cast<int>(nrho), MPI_DOUBLE,
                    MPI_SUM, g_world);
#endif
    } else {
      e.rho_host.resize(nrho);
      e.rho_recv.resize(nrho);
      sog_gpu_rho_download(e.s, e.rho_host.data());
      e.rho_recv = e.rho_host;
      mpi_sum(e.rho_recv.data(), static_cast<int>(nrho));
      sog_gpu_rho_upload(e.s, e.rho_recv.data());
    }
    const double e_repl = sog_gpu_solve_dev(
        e.s, nlocal, x.data_ptr<double>(), boxlo, lx, ly, lz,
        qs.data_ptr<double>(), qscale, static_cast<int>(spline),
        /*want_pot=*/1, F.data_ptr<double>(), vir6, v.data_ptr<double>(),
        self_coeff_exact, qsqsum, remove_self ? 1 : 0);
    if (std::getenv("SOG_DIST_DEBUG"))
      std::fprintf(stderr, "[sog-repl] E=%.6f vir0=%.6e vir1=%.6e\n", e_repl,
                   vir6[0], vir6[1]);
  } else {
    const double e_repl = sog_gpu_compute_dev(
        e.s, nlocal, x.data_ptr<double>(), boxlo, lx, ly, lz,
        qs.data_ptr<double>(), qscale, static_cast<int>(spline),
        /*want_potential=*/1, F.data_ptr<double>(), vir6, v.data_ptr<double>(),
        self_coeff_exact, qsqsum, remove_self ? 1 : 0);
    if (std::getenv("SOG_DIST_DEBUG"))
      std::fprintf(stderr, "[sog-repl] E=%.6f vir0=%.6e vir1=%.6e\n", e_repl,
                   vir6[0], vir6[1]);
  }
  // The box-strain virial is a GLOBAL quantity (it is built from the summed
  // density), unlike the energy and the force above, which are this rank's
  // share. Split it evenly so a caller that sums over ranks recovers the
  // exact total -- that is what makes the artifact usable as a LAMMPS pair
  // contribution. np == 1 leaves it untouched.
  const double vir_share = 1.0 / static_cast<double>(nranks);
  for (int k = 0; k < 6; ++k) vir6[k] *= vir_share;
  std::memcpy(vir_cpu.data_ptr<double>(), vir6, 6 * sizeof(double));
  vir.copy_(vir_cpu);
  return {v, F, vir};
}

// ── global charge neutralisation ────────────────────────────────────────────
// SOG drops k = 0, so E_k depends only on the ZERO-MEAN charge.  The model
// neutralises when it traces WITHOUT a comm_dict (per-rank projection), but the
// with-comm artifacts deliberately keep the raw charge and let the consumer do
// a GLOBAL projection -- the deployed chain gave that job to the LAMMPS kspace
// solver.  With the kspace moved inside the graph the op must own it, because:
//   * a per-rank mean is a different number on every rank, so the concatenation
//     of per-rank-neutralised charges is NOT globally neutral; and
//   * sum_i v_i != 0 for a non-neutral charge set, which makes the reported
//     E_k = 1/2 sum_i v_i q_i pick up a spurious (mean q) * (sum v) / 2.
//
// Ghost rows are excluded from BOTH the sum and the count: each atom must be
// counted once, and the owned rows are the [0, n_local) prefix of the node axis
// (the same convention node_ownership_mask uses).
at::Tensor neutralize_impl(const at::Tensor &q, const at::Tensor &n_local) {
  at::Tensor qs = q.detach().contiguous().to(at::kDouble);
  const int64_t ntot = qs.dim() > 0 ? qs.size(0) : 0;
  if (ntot == 0) return qs.to(q.options());
  const int64_t nch = qs.numel() / ntot;
  at::Tensor q2 = qs.reshape({ntot, nch});
  // Owned mask on the device: rows [0, n_local) contribute, ghosts are zero.
  at::Tensor nl = n_local.reshape({1}).to(qs.options());
  at::Tensor idx = at::arange(ntot, qs.options()).reshape({ntot, 1});
  at::Tensor owned = (idx < nl).to(at::kDouble);
  at::Tensor sums = (q2 * owned).sum(0);          // per-channel owned sums
  at::Tensor cnt = owned.sum().reshape({1});
  // One device all-reduce for the sums and one for the count, so every rank
  // uses the same global mean.
  global_sum_dev(sums);
  global_sum_dev(cnt);
  at::Tensor mean = sums / cnt;                   // (nch,)
  // Owned rows get the global mean removed; GHOST rows must contribute
  // NOTHING to the k-space solve (each atom is counted once, by its owner).
  // Returning ``q - mean`` for every row leaves each ghost carrying ``-mean``,
  // and when the raw latent charge is far from neutral (mean ~ -1.13 for the
  // 1536-atom water model) that injects |ghosts| * |mean| ~ 4500 spurious
  // charge units into the Allreduced density -- the observed ~1e8 eV garbage at
  // np>1.  Mirrors the model-side ``neutralize_latent_charge``
  // (deepmd/dpmodel/utils/neighbor_graph/graph.py), which also zeroes ghosts.
  at::Tensor out = (q2 - mean.reshape({1, nch})) * owned;
  return out.reshape(qs.sizes()).to(q.options());
}

at::Tensor neutralize_fake(const at::Tensor &q, const at::Tensor &n_local) {
  return at::empty_symint(q.sym_sizes(), q.options());
}

// ── global owned mean, for the autograd rule of ``neutralize`` ──────────────
// The exact VJP of ``qn_i = owned_i (q_i - mean)`` is
//     dL/dq_i = owned_i (g_i - mean_owned(g)),
// so the backward needs the SAME global mean the forward subtracted.  This op
// returns it per charge channel (trailing dims preserved; the node axis is
// summed over the owned prefix [0, n_local)).  It is deliberately VALUE-ONLY
// and carries a Meta fake impl: it runs INSIDE a traced backward, where a real
// host reduction is illegal, so the trace must see it as one opaque op that
// only reaches its CUDA body at run time.
at::Tensor qmean_impl(const at::Tensor &q, const at::Tensor &n_local) {
  at::Tensor qs = q.detach().contiguous().to(at::kDouble);
  const int64_t ntot = qs.dim() > 0 ? qs.size(0) : 0;
  const std::vector<int64_t> trailing(qs.sizes().begin() +
                                          (qs.dim() > 0 ? 1 : 0),
                                      qs.sizes().end());
  if (ntot == 0) return at::zeros(trailing, q.options());
  const int64_t nch = qs.numel() / ntot;
  at::Tensor q2 = qs.reshape({ntot, nch});
  at::Tensor nl = n_local.reshape({1}).to(qs.options());
  at::Tensor idx = at::arange(ntot, qs.options()).reshape({ntot, 1});
  at::Tensor owned = (idx < nl).to(at::kDouble);
  at::Tensor sums = (q2 * owned).sum(0);
  at::Tensor cnt = owned.sum().reshape({1});
  global_sum_dev(sums);
  global_sum_dev(cnt);
  at::Tensor mean = (sums / cnt).reshape(trailing);
  return mean.to(q.options());
}

at::Tensor qmean_fake(const at::Tensor &q, const at::Tensor &n_local) {
  return at::empty_symint(q.sym_sizes().slice(q.dim() > 0 ? 1 : 0), q.options());
}

std::vector<at::Tensor> solve_auto_fake(int64_t h, const at::Tensor &q,
                                        const at::Tensor &pos, const at::Tensor &box,
                                        const at::Tensor &amp_t, const at::Tensor &bw_t,
                                        double rcut, int64_t spline,
                                        double phi_accuracy, double box_tol,
                                        double qscale, double self_coeff,
                                        bool remove_self) {
  const c10::SymInt n = q.sym_size(0);
  auto o = q.options().dtype(at::kDouble);
  return {at::empty_symint({n}, o), at::empty_symint({n, 3}, o),
          at::empty({6}, o)};
}

std::vector<at::Tensor> solve_fake(int64_t h, const at::Tensor &q,
                                   const at::Tensor &pos, const at::Tensor &box,
                                   double qscale, int64_t spline,
                                   double self_coeff,
                                   bool remove_self) {
  // sym_size, not numel(): the trace runs this under FakeTensor with a symbolic
  // node axis, where numel() is illegal.
  const c10::SymInt n = q.sym_size(0);
  auto o = q.options().dtype(at::kDouble);
  return {at::empty_symint({n}, o), at::empty_symint({n, 3}, o), at::empty({6}, o)};
}

void setup_fake(int64_t, int64_t, int64_t, int64_t, const at::Tensor &,
                const at::Tensor &, const at::Tensor &, const at::Tensor &,
                const at::Tensor &) {}

// Diagnostic only (never traced): the mesh the last solve_auto call used.
at::Tensor last_grid_impl(int64_t h) {
  std::lock_guard<std::mutex> lock(g_mu);
  auto it = g_states.find(h);
  TORCH_CHECK(it != g_states.end(), "sogkspace: unknown handle ", h);
  at::Tensor g = at::empty({3}, at::TensorOptions().dtype(at::kLong));
  g[0] = it->second.nx;
  g[1] = it->second.ny;
  g[2] = it->second.nz;
  return g;
}

void destroy_impl(int64_t h) {
  std::lock_guard<std::mutex> lock(g_mu);
  auto it = g_states.find(h);
  if (it != g_states.end()) {
    if (it->second.s) sog_gpu_destroy(it->second.s);
    g_states.erase(it);
  }
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(sogkspace, m) {
  m.def(
      "setup(int h, int nx, int ny, int nz, Tensor green_energy, "
      "Tensor green_force, Tensor green_self, Tensor green_virial, "
      "Tensor green_self_virial) -> ()");
  m.def(
      "solve(int h, Tensor q, Tensor pos, Tensor box, float qscale, "
      "int spline, float self_coeff, bool remove_self) "
      "-> Tensor[]");
  m.def(
      "solve_auto(int h, Tensor q, Tensor pos, Tensor box, Tensor amp, "
      "Tensor bandwidth, float rcut, int spline, float phi_accuracy, "
      "float box_tol, float qscale, float self_coeff, bool remove_self) "
      "-> Tensor[]");
  m.def("neutralize(Tensor q, Tensor n_local) -> Tensor");
  m.def("qmean(Tensor q, Tensor n_local) -> Tensor");
  m.def("last_grid(int h) -> Tensor");
  m.def("set_comm(int comm) -> ()");
  m.def("destroy(int h) -> ()");
}

TORCH_LIBRARY_IMPL(sogkspace, CUDA, m) {
  m.impl("setup", &setup_impl);
  m.impl("solve", &solve_impl);
  m.impl("solve_auto", &solve_auto_impl);
  m.impl("neutralize", &neutralize_impl);
  m.impl("qmean", &qmean_impl);
  m.impl("last_grid", &last_grid_impl);
  m.impl("destroy", &destroy_impl);
}

TORCH_LIBRARY_IMPL(sogkspace, CPU, m) {
  m.impl("setup", &setup_impl);
  m.impl("destroy", &destroy_impl);
  m.impl("qmean", &qmean_impl);
}

TORCH_LIBRARY_IMPL(sogkspace, Meta, m) {
  m.impl("setup", &setup_fake);
  m.impl("solve", &solve_fake);
  m.impl("solve_auto", &solve_auto_fake);
  m.impl("neutralize", &neutralize_fake);
  m.impl("qmean", &qmean_fake);
}

// last_grid takes no tensor arguments, so it never reaches a device key: give
// it a catch-all registration (diagnostic query, never traced).
void set_comm_impl(int64_t comm_ptr) {
#ifdef USE_MPI
  std::lock_guard<std::mutex> lock(g_mu);
  g_world = reinterpret_cast<MPI_Comm>(static_cast<intptr_t>(comm_ptr));
#else
  (void)comm_ptr;
#endif
}

TORCH_LIBRARY_IMPL(sogkspace, CatchAll, m) {
  m.impl("last_grid", &last_grid_impl);
  m.impl("set_comm", &set_comm_impl);
}
