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
#include <torch/library.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cmath>
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

struct Entry {
  SogGpuState *s = nullptr;
  int nx = 0, ny = 0, nz = 0;
  // Host copies kept so a re-setup with the same handle can be skipped.
  std::vector<double> ge, gf, gs, gv, gsv;
  // ── self-contained (solve_auto) state: grid sizing + box_tol caching,
  //    mirroring SOGKSpace::ensure_fft_plan's Case-1 skip ──
  bool have_green = false;
  double glx = 0.0, gly = 0.0, glz = 0.0;   // box length at the last rebuild
  int spline = 4;
  std::vector<double> amp, bw;              // parameter signature
  // host staging buffers for the non-CUDA-aware replicated-grid reduction
  std::vector<double> rho_host, rho_recv;
};

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
#else
inline int mpi_rank_count() { return 1; }
inline void mpi_sum(double *, int) {}
#endif

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
  e.gs = to_host(gs);
  e.gv = to_host(gv);
  e.gsv = to_host(gsv);
  const size_t ngrid = static_cast<size_t>(nx) * ny * nz;
  TORCH_CHECK(static_cast<size_t>(e.ge.size()) == ngrid,
              "sogkspace: green_energy size ", e.ge.size(), " != ngrid ", ngrid);
  e.nx = static_cast<int>(nx);
  e.ny = static_cast<int>(ny);
  e.nz = static_cast<int>(nz);
  sog_gpu_setup(e.s, e.nx, e.ny, e.nz, e.ge.data(), e.gf.data(), e.gs.data(),
                e.gv.data(), e.gsv.data());
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
  const bool reuse = e.have_green && grid_same && params_same && drift <= box_tol;
  if (!reuse) {
    sog_green::build_green(nx, ny, nz, lx, ly, lz, amp, bw, static_cast<int>(spline),
                           e.ge, e.gf, e.gs, e.gv, e.gsv);
    sog_gpu_setup(e.s, nx, ny, nz, e.ge.data(), e.gf.data(), e.gs.data(),
                  e.gv.data(), e.gsv.data());
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
  const int nranks = mpi_rank_count();
  const bool multi = nranks > 1;

  double qsqsum = at::sum(qs * qs).item<double>();
  if (multi) {
    double qsq_glob = qsqsum;
    mpi_sum(&qsq_glob, 1);
    qsqsum = qsq_glob;
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
    sog_gpu_solve_dev(e.s, nlocal, x.data_ptr<double>(), boxlo, lx, ly, lz,
                      qs.data_ptr<double>(), qscale, static_cast<int>(spline),
                      /*want_pot=*/1, F.data_ptr<double>(), vir6,
                      v.data_ptr<double>(), self_coeff, qsqsum,
                      remove_self ? 1 : 0);
  } else {
    sog_gpu_compute_dev(e.s, nlocal, x.data_ptr<double>(), boxlo, lx, ly, lz,
                        qs.data_ptr<double>(), qscale, static_cast<int>(spline),
                        /*want_potential=*/1, F.data_ptr<double>(), vir6,
                        v.data_ptr<double>(), self_coeff, qsqsum,
                        remove_self ? 1 : 0);
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
  const int64_t nlocal = n_local.item<int64_t>();
  at::Tensor qs = q.detach().contiguous().to(at::kDouble);
  const int64_t ntot = qs.numel();
  TORCH_CHECK(nlocal >= 0 && nlocal <= ntot,
              "sogkspace::neutralize: n_local out of range");
  double sum = nlocal > 0 ? at::sum(qs.narrow(0, 0, nlocal)).item<double>() : 0.0;
  double cnt = static_cast<double>(nlocal);
  // One reduction for the pair so every rank uses the same global mean.
  double red[2] = {sum, cnt};
  mpi_sum(red, 2);
  const double mean = red[1] > 0.0 ? red[0] / red[1] : 0.0;
  // Owned rows get the global mean removed; GHOST rows must contribute
  // NOTHING to the k-space solve (each atom is counted once, by its owner).
  // Returning ``q - mean`` for every row leaves each ghost carrying ``-mean``,
  // and when the raw latent charge is far from neutral (mean ~ -1.13 for the
  // 1536-atom water model) that injects |ghosts| * |mean| ~ 4500 spurious
  // charge units into the Allreduced density -- the observed ~1e8 eV garbage at
  // np>1.  Mirrors the model-side ``neutralize_latent_charge``
  // (deepmd/dpmodel/utils/neighbor_graph/graph.py), which also zeroes ghosts.
  at::Tensor out = qs.sub(mean);
  if (nlocal < ntot) {
    out.narrow(0, nlocal, ntot - nlocal).zero_();
  }
  return out.to(q.options());
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
  const int64_t nlocal = n_local.reshape(-1)[0].item<int64_t>();
  at::Tensor qs = q.detach().contiguous().to(at::kDouble);
  const int64_t ntot = qs.dim() > 0 ? qs.size(0) : 0;
  TORCH_CHECK(nlocal >= 0 && nlocal <= ntot,
              "sogkspace::qmean: n_local out of range");
  std::vector<int64_t> sumshape(qs.sizes().begin() + (qs.dim() > 0 ? 1 : 0),
                                qs.sizes().end());
  at::Tensor part =
      nlocal > 0 ? qs.narrow(0, 0, nlocal).sum(0)
                 : at::zeros(sumshape, qs.options());
  at::Tensor part_cpu = part.to(at::kCPU).contiguous();
  const int64_t nch = part_cpu.numel();
  std::vector<double> buf(part_cpu.data_ptr<double>(),
                          part_cpu.data_ptr<double>() + nch);
  // One reduction for the per-channel sums and one for the owned count, so
  // every rank divides by the same global owned count.
  mpi_sum(buf.data(), static_cast<int>(nch));
  double cnt = static_cast<double>(nlocal);
  mpi_sum(&cnt, 1);
  const double denom = cnt > 0.0 ? cnt : 1.0;
  for (int64_t k = 0; k < nch; ++k) buf[k] /= denom;
  // Build the result on the HOST (the reduction above is host-side) and move
  // it to the caller's device; never memcpy into a CUDA tensor.
  at::Tensor out = at::empty(sumshape, qs.options().device(at::kCPU));
  std::memcpy(out.data_ptr<double>(), buf.data(), nch * sizeof(double));
  return out.to(q.options());
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
