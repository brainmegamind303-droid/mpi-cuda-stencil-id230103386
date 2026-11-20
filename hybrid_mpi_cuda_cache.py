import os
import sys
import json
import math
import time
import re
import hashlib
import platform
import subprocess
import numpy as np
from mpi4py import MPI
from numba import cuda, njit, float64, uint64

TILE_DIM = 16
SHARED_DIM = TILE_DIM + 2          # 18x18 shared tile
GLOBAL_NX = 512
GLOBAL_NY = 512
LOCAL_N = 256                      # interior per rank (256x256)
PAD_N = LOCAL_N + 2                # 258x258 with ghost layer
STENCIL_STEPS = 300
SHM_ITERS = 40_000_000
SHM_SLOTS = 64
LAMBDA_DAMP = 0.15
CFL_FACTOR = 0.8                   # dt = CFL_FACTOR * dt_max
SRC_AMP = 50.0
CHECK_STEPS = (0, 100, 200, 300)

TAG_UP, TAG_DOWN, TAG_LEFT, TAG_RIGHT = 10, 11, 12, 13   # tag = direction of travel


def get_gpu_uuid() -> str:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
        encoding="utf-8",
    )
    return out.strip().splitlines()[0].strip()


def detect_cpu_line_size() -> int:
    path = "/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return int(f.read().strip())
    return 64


def detect_cpu_model() -> str:
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def log(ctx, *args):
    if ctx["rank"] == 0:
        print(*args, flush=True)


# ==============================================================================
# TASK 1: Hybrid MPI Cartesian Grid + Node-Local MPI-3 Shared-Memory Window
# ==============================================================================
def init_hybrid_context(github_user: str) -> dict:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if size != 4:
        raise RuntimeError(f"Requires exactly 4 MPI ranks (2x2 grid), got {size}.")

    # Non-periodic 2x2 Cartesian communicator
    cart_comm = comm.Create_cart(dims=[2, 2], periods=[False, False], reorder=False)
    coords = cart_comm.Get_coords(cart_comm.Get_rank())          # (row, col)
    up, down = cart_comm.Shift(0, 1)                             # PROC_NULL at edges
    left, right = cart_comm.Shift(1, 1)

    # GPU binding
    cuda.select_device(0)
    gpu_uuid = comm.bcast(get_gpu_uuid() if rank == 0 else None, root=0)

    # Deterministic 64-bit instance seed
    line_bytes = detect_cpu_line_size()
    cpu_sig = f"{platform.machine()}::{os.cpu_count()}::{detect_cpu_model()}::L{line_bytes}"
    digest = hashlib.sha256(
        f"{github_user.strip()}::{gpu_uuid}::{cpu_sig}".encode("utf-8")
    ).digest()
    seed = int.from_bytes(digest[:8], "big")

    # Node-local communicator + 64-byte aligned shared window (64 x uint64)
    node_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
    node_rank = node_comm.Get_rank()
    node_size = node_comm.Get_size()
    if node_size != 4:
        raise RuntimeError(f"Expected all 4 ranks on one node, node_comm size={node_size}.")

    nbytes = SHM_SLOTS * 8
    alloc = (nbytes + 64) if node_rank == 0 else 0     # +64 slack for manual alignment
    win = MPI.Win.Allocate_shared(alloc, 1, MPI.INFO_NULL, node_comm)
    buf, _disp = win.Shared_query(0)
    shm_raw = np.frombuffer(buf, dtype=np.uint8)
    off = (-shm_raw.ctypes.data) % 64
    shm_view = shm_raw[off:off + nbytes].view(np.uint64)
    if not shm_view.flags.writeable:
        raise RuntimeError("Shared window view is not writable.")
    assert shm_view.ctypes.data % 64 == 0, "shared window is not 64-byte aligned"
    assert shm_view.shape == (SHM_SLOTS,) and shm_view.dtype == np.uint64
    if node_rank == 0:
        shm_view[:] = 0
    node_comm.Barrier()

    return {
        "comm": comm, "rank": rank, "size": size,
        "cart_comm": cart_comm, "coords": tuple(coords),
        "up": up, "down": down, "left": left, "right": right,
        "gpu_uuid": gpu_uuid, "cpu_sig": cpu_sig, "line_bytes": line_bytes,
        "seed": seed, "github_user": github_user.strip(),
        "node_comm": node_comm, "node_rank": node_rank, "node_size": node_size,
        "win": win, "shm_raw": shm_raw, "shm_view": shm_view,
        "params": {},
    }


# ==============================================================================
# TASK 2: Rank-local fields + GPU halo pack/unpack kernels
# ==============================================================================
def initialize_fields(ctx: dict):
    """Rank-local (258,258) float64 arrays built from GLOBAL coordinates (no global grid)."""
    cy, cx = ctx["coords"]
    seed = ctx["seed"]
    phi = 2.0 * math.pi * (((seed >> 16) & 0xFFFF) / 65535.0)
    frac = (seed & 0xFFFF) / 65535.0
    x0 = 0.30 + 0.40 * frac                      # seed-bound bump centre

    idx = np.arange(PAD_N, dtype=np.float64)
    gi = cy * LOCAL_N + idx - 1.0                # global row index (ghost: -1 .. 512)
    gj = cx * LOCAL_N + idx - 1.0
    dx = 1.0 / (GLOBAL_NX + 1)                   # 1/513
    y = (gi + 1.0) * dx
    x = (gj + 1.0) * dx
    Y, X = np.meshgrid(y, x, indexing="ij")

    kappa = 1.0 + 0.25 * (1.0 + np.sin(6.0 * math.pi * X + phi) * np.cos(4.0 * math.pi * Y + phi))
    S = SRC_AMP * np.exp(-((X - 0.30) ** 2 + (Y - 0.70) ** 2) / 0.01)
    u = np.exp(-((X - x0) ** 2 + (Y - 0.50) ** 2) / 0.02)
    u[0, :] = 0.0; u[-1, :] = 0.0; u[:, 0] = 0.0; u[:, -1] = 0.0   # ghosts (Dirichlet 0 / filled by halos)
    return (np.ascontiguousarray(u), np.ascontiguousarray(kappa), np.ascontiguousarray(S))


@cuda.jit
def pack_halos_kernel(u, send_top, send_bottom, send_left, send_right):
    k = cuda.grid(1)
    if k < LOCAL_N:
        i = k + 1
        send_top[k] = u[1, i]              # contiguous row 1
        send_bottom[k] = u[LOCAL_N, i]     # contiguous row 256
        send_left[k] = u[i, 1]             # strided column 1
        send_right[k] = u[i, LOCAL_N]      # strided column 256


@cuda.jit
def unpack_halos_kernel(u, recv_top, recv_bottom, recv_left, recv_right,
                        has_up, has_down, has_left, has_right):
    k = cuda.grid(1)
    if k < LOCAL_N:
        i = k + 1
        if has_up:
            u[0, i] = recv_top[k]
        if has_down:
            u[LOCAL_N + 1, i] = recv_bottom[k]
        if has_left:
            u[i, 0] = recv_left[k]
        if has_right:
            u[i, LOCAL_N + 1] = recv_right[k]


# ==============================================================================
# TASK 3: Tiled shared-memory stencil + dual-stream overlap
# ==============================================================================
@cuda.jit
def tiled_stencil_kernel(u, u_new, kappa, src, i0, i1, j0, j1, inv_dx2, dt, lam):
    """Updates cells [i0..i1] x [j0..j1] using an 18x18 shared tile."""
    tile = cuda.shared.array((SHARED_DIM, SHARED_DIM), dtype=float64)
    ty = cuda.threadIdx.y
    tx = cuda.threadIdx.x
    i = i0 + cuda.blockIdx.y * TILE_DIM + ty
    j = j0 + cuda.blockIdx.x * TILE_DIM + tx
    ny = u.shape[0]
    nx = u.shape[1]

    if i < ny and j < nx:
        tile[ty + 1, tx + 1] = u[i, j]
        if ty == 0 and i > 0:
            tile[0, tx + 1] = u[i - 1, j]
        if ty == TILE_DIM - 1 and i + 1 < ny:
            tile[SHARED_DIM - 1, tx + 1] = u[i + 1, j]
        if tx == 0 and j > 0:
            tile[ty + 1, 0] = u[i, j - 1]
        if tx == TILE_DIM - 1 and j + 1 < nx:
            tile[ty + 1, SHARED_DIM - 1] = u[i, j + 1]
    cuda.syncthreads()     # ALL threads reach the barrier (no early return above)

    if i >= i0 and i <= i1 and j >= j0 and j <= j1:
        uc = tile[ty + 1, tx + 1]
        un_ = tile[ty, tx + 1]
        us_ = tile[ty + 2, tx + 1]
        uw_ = tile[ty + 1, tx]
        ue_ = tile[ty + 1, tx + 2]
        kc = kappa[i, j]
        ke = 0.5 * (kc + kappa[i, j + 1])
        kw = 0.5 * (kc + kappa[i, j - 1])
        kn = 0.5 * (kc + kappa[i - 1, j])
        ks = 0.5 * (kc + kappa[i + 1, j])
        lap = (ke * (ue_ - uc) - kw * (uc - uw_) + ks * (us_ - uc) - kn * (uc - un_)) * inv_dx2
        u_new[i, j] = uc + dt * (lap + src[i, j] - lam * uc)


def make_state(ctx, u0, k0, S0):
    f64 = np.float64
    st = {
        "d_u": cuda.to_device(u0),
        "d_un": cuda.to_device(np.zeros_like(u0)),
        "d_kappa": cuda.to_device(k0),
        "d_S": cuda.to_device(S0),
        "stream_interior": cuda.stream(),
        "stream_boundary": cuda.stream(),
    }
    for name in ("top", "bottom", "left", "right"):
        st[f"d_send_{name}"] = cuda.device_array(LOCAL_N, dtype=f64)
        st[f"d_recv_{name}"] = cuda.device_array(LOCAL_N, dtype=f64)
        st[f"h_send_{name}"] = cuda.pinned_array(LOCAL_N, dtype=f64)
        st[f"h_recv_{name}"] = cuda.pinned_array(LOCAL_N, dtype=f64)
    cuda.synchronize()
    return st


def launch_region(ctx, st, i0, i1, j0, j1, stream):
    P = ctx["params"]
    gx = (j1 - j0 + TILE_DIM) // TILE_DIM
    gy = (i1 - i0 + TILE_DIM) // TILE_DIM
    tiled_stencil_kernel[(gx, gy), (TILE_DIM, TILE_DIM), stream](
        st["d_u"], st["d_un"], st["d_kappa"], st["d_S"],
        i0, i1, j0, j1, P["inv_dx2"], P["dt"], P["lam"])


def halo_exchange(ctx, st, stream):
    """Pack -> D2H (pinned) -> Irecv/Isend -> Waitall -> H2D -> unpack, all GPU work on `stream`."""
    cart = ctx["cart_comm"]
    reqs = []
    # Post receives first. Data that travelled DOWN arrives from the UP neighbour, etc.
    reqs.append(cart.Irecv(st["h_recv_top"], source=ctx["up"], tag=TAG_DOWN))
    reqs.append(cart.Irecv(st["h_recv_bottom"], source=ctx["down"], tag=TAG_UP))
    reqs.append(cart.Irecv(st["h_recv_left"], source=ctx["left"], tag=TAG_RIGHT))
    reqs.append(cart.Irecv(st["h_recv_right"], source=ctx["right"], tag=TAG_LEFT))

    pack_halos_kernel[1, LOCAL_N, stream](
        st["d_u"], st["d_send_top"], st["d_send_bottom"], st["d_send_left"], st["d_send_right"])
    for name in ("top", "bottom", "left", "right"):
        st[f"d_send_{name}"].copy_to_host(st[f"h_send_{name}"], stream=stream)
    stream.synchronize()

    reqs.append(cart.Isend(st["h_send_top"], dest=ctx["up"], tag=TAG_UP))
    reqs.append(cart.Isend(st["h_send_bottom"], dest=ctx["down"], tag=TAG_DOWN))
    reqs.append(cart.Isend(st["h_send_left"], dest=ctx["left"], tag=TAG_LEFT))
    reqs.append(cart.Isend(st["h_send_right"], dest=ctx["right"], tag=TAG_RIGHT))
    MPI.Request.Waitall(reqs)

    for name in ("top", "bottom", "left", "right"):
        st[f"d_recv_{name}"].copy_to_device(st[f"h_recv_{name}"], stream=stream)
    unpack_halos_kernel[1, LOCAL_N, stream](
        st["d_u"], st["d_recv_top"], st["d_recv_bottom"], st["d_recv_left"], st["d_recv_right"],
        ctx["up"] != MPI.PROC_NULL, ctx["down"] != MPI.PROC_NULL,
        ctx["left"] != MPI.PROC_NULL, ctx["right"] != MPI.PROC_NULL)


def step_serialized(ctx, st):
    s = cuda.default_stream()
    halo_exchange(ctx, st, s)
    launch_region(ctx, st, 1, LOCAL_N, 1, LOCAL_N, s)
    s.synchronize()
    st["d_u"], st["d_un"] = st["d_un"], st["d_u"]


def step_overlapped(ctx, st):
    si, sb = st["stream_interior"], st["stream_boundary"]
    # deep interior [2..255, 2..255] needs no ghost cells -> runs concurrently with MPI traffic
    launch_region(ctx, st, 2, LOCAL_N - 1, 2, LOCAL_N - 1, si)
    halo_exchange(ctx, st, sb)
    # 4 outer border strips (need ghost cells)
    launch_region(ctx, st, 1, 1, 1, LOCAL_N, sb)                      # top row
    launch_region(ctx, st, LOCAL_N, LOCAL_N, 1, LOCAL_N, sb)          # bottom row
    launch_region(ctx, st, 2, LOCAL_N - 1, 1, 1, sb)                  # left column
    launch_region(ctx, st, 2, LOCAL_N - 1, LOCAL_N, LOCAL_N, sb)      # right column
    sb.synchronize()
    si.synchronize()
    st["d_u"], st["d_un"] = st["d_un"], st["d_u"]


def global_energy(ctx, st) -> float:
    u = st["d_u"].copy_to_host()
    dx = ctx["params"]["dx"]
    e = float(np.sum(u[1:LOCAL_N + 1, 1:LOCAL_N + 1] ** 2)) * dx * dx
    return ctx["cart_comm"].allreduce(e, op=MPI.SUM)


def run_mode(ctx, mode, u0, k0, S0, steps=STENCIL_STEPS):
    stepper = step_serialized if mode == "serialized" else step_overlapped
    st = make_state(ctx, u0, k0, S0)               # warm-up (JIT compile)
    for _ in range(2):
        stepper(ctx, st)
    cuda.synchronize()
    ctx["cart_comm"].Barrier()
    del st

    st = make_state(ctx, u0, k0, S0)
    energies = {0: global_energy(ctx, st)}
    elapsed = 0.0
    for step in range(1, steps + 1):
        t0 = time.perf_counter()
        stepper(ctx, st)
        elapsed += time.perf_counter() - t0
        if step in CHECK_STEPS:
            energies[step] = global_energy(ctx, st)
    u_final = st["d_u"].copy_to_host()
    ms = ctx["cart_comm"].allreduce(elapsed * 1e3, op=MPI.MAX)
    return u_final, energies, ms


# ==============================================================================
# TASK 4: Host cache-line false-sharing sweep on the MPI-3 shared window
# ==============================================================================
@njit(nogil=True)
def host_shm_accumulate(shm_view, idx, iters, sink):
    # In-place read-modify-write on shared memory. The store to `sink` may alias
    # `shm_view` (numba arrays carry no noalias), so LLVM cannot promote
    # shm_view[idx] to a register: every iteration performs a real load + store.
    one = np.uint64(1)
    for _ in range(iters):
        shm_view[idx] = shm_view[idx] + one
        sink[0] = shm_view[idx]


def benchmark_shm_strides(ctx, iters=SHM_ITERS, reps=3):
    node_comm = ctx["node_comm"]
    r = ctx["node_rank"]
    shm = ctx["shm_view"]
    sink = np.zeros(1, dtype=np.uint64)

    host_shm_accumulate(shm, r, 1000, sink)        # compile + warm-up
    node_comm.Barrier()
    if r == 0:
        shm[:] = 0
    node_comm.Barrier()

    base = shm.ctypes.data
    out = {}
    for stride in (1, 8, 16):
        idx = r * stride
        times = []
        for _ in range(reps):
            shm[idx] = 0
            node_comm.Barrier()
            t0 = time.perf_counter()
            host_shm_accumulate(shm, idx, iters, sink)
            t1 = time.perf_counter()
            node_comm.Barrier()
            times.append(t1 - t0)
            assert int(shm[idx]) == iters, "lost update / wrong count"
        t_max = node_comm.allreduce(float(np.median(times)), op=MPI.MAX)
        lines = [(base + 8 * rk * stride) // 64 for rk in range(4)]
        out[stride] = {
            "bytes_spacing": 8 * stride,
            "seconds": t_max,
            "ns_per_update": t_max / iters * 1e9,
            "cpu_line_index_per_rank": [int(x) for x in lines],
            "distinct_cpu_lines": len(set(lines)),
        }
    return out


# ==============================================================================
# Assembly audits
# ==============================================================================
def _flatten_text(obj) -> str:
    if isinstance(obj, dict):
        return "\n".join(_flatten_text(v) for v in obj.values())
    return str(obj)


def audit_ptx() -> dict:
    ptx = _flatten_text(tiled_stencil_kernel.inspect_asm())
    return {
        "st.shared.f64": len(re.findall(r"\bst\.shared(?:\.v\d)?\.f64\b", ptx)),
        "ld.shared.f64": len(re.findall(r"\bld\.shared(?:\.v\d)?\.f64\b", ptx)),
        "bar.sync": len(re.findall(r"\bbar\.sync\b", ptx)),
    }


def audit_llvm() -> dict:
    ir = _flatten_text(host_shm_accumulate.inspect_llvm())
    return {
        "load_i64": len(re.findall(r"\bload i64\b", ir)),
        "store_i64": len(re.findall(r"\bstore i64\b", ir)),
        "volatile": len(re.findall(r"\bvolatile\b", ir)),
        "ir_lines": ir.count("\n"),
    }


# ==============================================================================
# TASK 5: Orchestration, verification, manifest
# ==============================================================================
def main():
    if len(sys.argv) < 2 or not sys.argv[1].strip():
        raise SystemExit("usage: hybrid_mpi_cuda_cache.py <GITHUB_USERNAME>")
    ctx = init_hybrid_context(sys.argv[1])
    cart = ctx["cart_comm"]

    log(ctx, "=" * 78)
    log(ctx, f"GitHub user      : {ctx['github_user']}")
    log(ctx, f"GPU UUID         : {ctx['gpu_uuid']}")
    log(ctx, f"CPU signature    : {ctx['cpu_sig']}")
    log(ctx, f"INSTANCE_SEED    : {ctx['seed']} (0x{ctx['seed']:016x})")
    log(ctx, f"Cart coords      : rank0={ctx['coords']}  node_size={ctx['node_size']}")
    log(ctx, f"SHM base % 64    : {ctx['shm_view'].ctypes.data % 64}  (0 = aligned)")
    log(ctx, "=" * 78)

    u0, k0, S0 = initialize_fields(ctx)
    dx = 1.0 / (GLOBAL_NX + 1)
    kappa_max = cart.allreduce(float(k0[1:LOCAL_N + 1, 1:LOCAL_N + 1].max()), op=MPI.MAX)
    dt_max = dx * dx / (4.0 * kappa_max)
    dt = CFL_FACTOR * dt_max
    ctx["params"] = {"dx": dx, "inv_dx2": 1.0 / (dx * dx), "dt": dt, "lam": LAMBDA_DAMP}
    assert dt <= dt_max, "CFL violated"
    log(ctx, f"kappa_max={kappa_max:.12f} dx={dx:.12e} dt_max={dt_max:.12e} dt={dt:.12e} "
             f"(dt/dt_max={dt / dt_max:.3f})")

    # --- stencil runs ---
    u_ser, e_ser, ms_ser = run_mode(ctx, "serialized", u0, k0, S0)
    u_ovl, e_ovl, ms_ovl = run_mode(ctx, "overlapped", u0, k0, S0)

    local_diff = float(np.max(np.abs(u_ser[1:LOCAL_N + 1, 1:LOCAL_N + 1]
                                     - u_ovl[1:LOCAL_N + 1, 1:LOCAL_N + 1])))
    max_diff = cart.allreduce(local_diff, op=MPI.MAX)
    assert max_diff < 1e-12, f"parity failure: max_diff={max_diff}"

    log(ctx, f"gpu_serialized_ms = {ms_ser:.3f}  ({ms_ser / STENCIL_STEPS:.4f} ms/step)")
    log(ctx, f"gpu_overlapped_ms = {ms_ovl:.3f}  ({ms_ovl / STENCIL_STEPS:.4f} ms/step)")
    log(ctx, f"serialized/overlapped speedup = {ms_ser / ms_ovl:.4f}")
    log(ctx, f"max_diff (overlapped vs serialized, all ranks) = {max_diff:.3e}  [< 1e-12 OK]")
    for s in CHECK_STEPS:
        log(ctx, f"  global energy step {s:3d}: serialized={e_ser[s]:.15e}  overlapped={e_ovl[s]:.15e}")

    # --- PTX audit ---
    ptx_counts = audit_ptx()
    log(ctx, f"PTX audit: {ptx_counts}")

    # --- shared-memory false sharing sweep ---
    cart.Barrier()
    shm = benchmark_shm_strides(ctx)
    for stride, d in shm.items():
        log(ctx, f"stride={stride:2d} ({d['bytes_spacing']:3d} B): {d['ns_per_update']:.3f} ns/update, "
                 f"{d['seconds'] * 1e3:.2f} ms, distinct CPU lines={d['distinct_cpu_lines']}, "
                 f"lines={d['cpu_line_index_per_rank']}")
    ratio_1_8 = shm[1]["seconds"] / shm[8]["seconds"]
    ratio_16_8 = shm[16]["seconds"] / shm[8]["seconds"]
    log(ctx, f"false-sharing slowdown stride1/stride8 = {ratio_1_8:.3f}x ; stride16/stride8 = {ratio_16_8:.3f}x")

    # --- LLVM IR audit ---
    llvm_counts = audit_llvm()
    log(ctx, f"LLVM IR audit (host_shm_accumulate): {llvm_counts}")

    # --- manifest ---
    if ctx["rank"] == 0:
        row_stride_bytes = PAD_N * 8
        manifest = {
            "github_username": ctx["github_user"],
            "instance_seed": ctx["seed"],
            "gpu_uuid": ctx["gpu_uuid"],
            "cpu_signature": ctx["cpu_sig"],
            "cpu_line_bytes": ctx["line_bytes"],
            "mpi_ranks": 4, "grid": [2, 2],
            "local_shape": [PAD_N, PAD_N], "global_interior": [GLOBAL_NY, GLOBAL_NX],
            "steps": STENCIL_STEPS, "shm_iters": SHM_ITERS,
            "kappa_max": kappa_max, "dx": dx, "dt_max": dt_max, "dt": dt,
            "cfl_ratio": dt / dt_max, "lambda_damp": LAMBDA_DAMP,
            "gpu_serialized_ms": ms_ser, "gpu_overlapped_ms": ms_ovl,
            "overlap_speedup": ms_ser / ms_ovl,
            "max_diff": max_diff,
            "energy_serialized": {str(k): v for k, v in e_ser.items()},
            "energy_overlapped": {str(k): v for k, v in e_ovl.items()},
            "ptx_counts": ptx_counts,
            "llvm_counts": llvm_counts,
            "shm_strides": {str(k): v for k, v in shm.items()},
            "false_sharing_ratio_stride1_vs_8": ratio_1_8,
            "padding_ratio_stride16_vs_8": ratio_16_8,
            "halo_traffic": {
                "row_stride_bytes": row_stride_bytes,
                "packed_bytes_per_step_per_rank": 4 * LOCAL_N * 8,
                "full_array_bytes": PAD_N * PAD_N * 8,
                "column_sectors_32B": LOCAL_N,
                "column_bytes_fetched_L2": LOCAL_N * 32,
                "column_bytes_useful": LOCAL_N * 8,
            },
        }
        canon = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
        manifest["manifest_sha256"] = hashlib.sha256(canon.encode("utf-8")).hexdigest()
        with open("hybrid_lab_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
        print("Manifest written: hybrid_lab_manifest.json "
              f"(sha256={manifest['manifest_sha256']})", flush=True)

    # --- clean shutdown of the shared window ---
    cart.Barrier()
    node_comm = ctx["node_comm"]
    win = ctx["win"]
    ctx["shm_view"] = None
    ctx["shm_raw"] = None
    del shm
    node_comm.Barrier()
    win.Free()
    log(ctx, "Done. Window freed.")


if __name__ == "__main__":
    main()
