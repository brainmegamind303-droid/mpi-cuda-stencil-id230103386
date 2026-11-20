import os
import sys
import json
import math
import time
import re
import hashlib
import subprocess
import numpy as np
from mpi4py import MPI
from numba import cuda, float64

# Compile-time constants for 2D shared-memory tiling (16x16 interior + 1-cell halo apron)
TILE_DIM = 16
SHARED_DIM = TILE_DIM + 2  # 18
GLOBAL_NX = 512
GLOBAL_NY = 512
STEPS = 400
LAMBDA_DAMP = 0.15

# Derived constants
PROC_DIMS = (2, 2)
NY_LOC = GLOBAL_NY // PROC_DIMS[0]   # 256 owned rows
NX_LOC = GLOBAL_NX // PROC_DIMS[1]   # 256 owned cols
PAD_NY = NY_LOC + 2                  # 258
PAD_NX = NX_LOC + 2                  # 258
DX = 1.0 / (GLOBAL_NX + 1)           # 1/513
DY = 1.0 / (GLOBAL_NY + 1)           # 1/513
KAPPA_MIN, KAPPA_MAX_BOUND = 0.08, 0.24
CFL_SAFETY = 0.8
DT_MAX_BOUND = DX * DX / (4.0 * KAPPA_MAX_BOUND)
DT = CFL_SAFETY * DT_MAX_BOUND
DIAG_EVERY = 100
WARMUP_STEPS = 4

# Message tags (named by direction of travel of the data)
TAG_UP, TAG_DOWN, TAG_LEFT, TAG_RIGHT = 10, 11, 12, 13


def get_gpu_uuid() -> str:
    """Query active NVIDIA GPU UUID via nvidia-smi."""
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
        encoding="utf-8",
    )
    return out.strip().splitlines()[0].strip()


def derive_student_seed(github_user: str, gpu_uuid: str) -> int:
    """Derive deterministic 64-bit integer seed bound to student GitHub handle and GPU UUID."""
    digest = hashlib.sha256(f"{github_user.strip()}::{gpu_uuid}".encode("utf8")).hexdigest()
    return int(digest[:16], 16)


# ==============================================================================
# TASK 1: Distributed Cartesian Topology & Hardware Context Initialization
# ==============================================================================
def init_distributed_context(github_user: str):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if size != 4:
        raise RuntimeError(f"Expected exactly 4 MPI ranks (2x2 grid), got {size}.")

    # Bind process to GPU 0 on Google Colab
    cuda.select_device(0)
    dev = cuda.get_current_device()
    dev_name = dev.name.decode() if isinstance(dev.name, bytes) else str(dev.name)
    cc = tuple(dev.compute_capability)

    gpu_uuid = get_gpu_uuid() if rank == 0 else None
    gpu_uuid = comm.bcast(gpu_uuid, root=0)
    instance_seed = derive_student_seed(github_user, gpu_uuid)

    # Non-periodic 2x2 Cartesian communicator
    cart_comm = comm.Create_cart(dims=list(PROC_DIMS), periods=[False, False], reorder=False)
    cart_rank = cart_comm.Get_rank()
    row_rank, col_rank = cart_comm.Get_coords(cart_rank)

    # Shift(direction, disp=1) -> (source, dest): source has lower coordinate along that axis.
    nbr_up, nbr_down = cart_comm.Shift(0, 1)
    nbr_left, nbr_right = cart_comm.Shift(1, 1)

    return {
        "comm": comm,
        "cart_comm": cart_comm,
        "rank": cart_rank,
        "size": size,
        "row_rank": row_rank,
        "col_rank": col_rank,
        "nbr_up": nbr_up,
        "nbr_down": nbr_down,
        "nbr_left": nbr_left,
        "nbr_right": nbr_right,
        "github_user": github_user.strip(),
        "gpu_uuid": gpu_uuid,
        "instance_seed": instance_seed,
        "device_name": dev_name,
        "compute_capability": f"{cc[0]}.{cc[1]}",
    }


# ==============================================================================
# TASK 2: Seeded Heterogeneous Subdomain Initialization (rank-local arrays only)
# ==============================================================================
def initialize_subdomains(ctx):
    seed = ctx["instance_seed"]
    rng = np.random.default_rng(seed)  # identical on all ranks -> identical global parameters

    # Seed-derived physical parameters (same on every rank, no communication needed)
    pulses = []
    for _ in range(3):
        cx, cy = rng.uniform(0.2, 0.8, size=2)
        amp = rng.uniform(0.8, 1.2)
        sig = rng.uniform(0.05, 0.09)
        pulses.append((cx, cy, amp, sig))
    phase_x, phase_y = rng.uniform(0.0, 2.0 * np.pi, size=2)
    src_cx, src_cy = rng.uniform(0.2, 0.8, size=2)
    src_amp = 0.01 * rng.uniform(0.5, 1.5)
    src_sig = 0.04

    # Global physical coordinates of local indices 0..257 (node g = rank_offset + local index,
    # g = 0 and g = 513 are the physical Dirichlet boundary nodes). No global array allocated.
    gi = ctx["row_rank"] * NY_LOC + np.arange(PAD_NY, dtype=np.float64)
    gj = ctx["col_rank"] * NX_LOC + np.arange(PAD_NX, dtype=np.float64)
    y = gi * DY
    x = gj * DX
    Y, X = np.meshgrid(y, x, indexing="ij")  # shape (258, 258): axis 0 = y (rows), axis 1 = x (cols)

    # Heterogeneous diffusivity in [0.08, 0.24]
    kappa = 0.16 + 0.08 * np.sin(2.0 * np.pi * 2.0 * X + phase_x) * np.cos(2.0 * np.pi * 3.0 * Y + phase_y)
    kappa = np.ascontiguousarray(np.clip(kappa, KAPPA_MIN, KAPPA_MAX_BOUND), dtype=np.float64)

    # Initial Gaussian thermal pulses
    u0 = np.zeros((PAD_NY, PAD_NX), dtype=np.float64)
    for cx, cy, amp, sig in pulses:
        u0 += amp * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2.0 * sig * sig))

    # Localized source field
    S = src_amp * np.exp(-((X - src_cx) ** 2 + (Y - src_cy) ** 2) / (2.0 * src_sig * src_sig))
    S = np.ascontiguousarray(S, dtype=np.float64)

    # Homogeneous Dirichlet u = 0 on outer global edges (neighbor == PROC_NULL)
    if ctx["nbr_up"] == MPI.PROC_NULL:
        u0[0, :] = 0.0
    if ctx["nbr_down"] == MPI.PROC_NULL:
        u0[PAD_NY - 1, :] = 0.0
    if ctx["nbr_left"] == MPI.PROC_NULL:
        u0[:, 0] = 0.0
    if ctx["nbr_right"] == MPI.PROC_NULL:
        u0[:, PAD_NX - 1] = 0.0
    u0 = np.ascontiguousarray(u0)

    # Global maximum of kappa over owned cells (for the CFL analysis, Q6)
    local_kmax = np.array([kappa[1:NY_LOC + 1, 1:NX_LOC + 1].max()], dtype=np.float64)
    global_kmax = np.zeros(1, dtype=np.float64)
    ctx["cart_comm"].Allreduce(local_kmax, global_kmax, op=MPI.MAX)

    return {
        "u0_host": u0,
        "kappa_host": kappa,
        "S_host": S,
        "d_u_a": cuda.to_device(u0),
        "d_u_b": cuda.to_device(u0),
        "d_kappa": cuda.to_device(kappa),
        "d_S": cuda.to_device(S),
        "kappa_max": float(global_kmax[0]),
        "pulses": [tuple(float(v) for v in p) for p in pulses],
        "source": (float(src_cx), float(src_cy), float(src_amp), float(src_sig)),
    }


# ==============================================================================
# TASK 3: GPU Halo Pack/Unpack Kernels & Pinned MPI Exchange
# Buffer layout (float64, 1D, contiguous):
#   [ up : nx | down : nx | left : ny | right : ny ]   total = 2*nx + 2*ny
# send: up = row 1, down = row ny, left = column 1, right = column nx
# recv: up -> ghost row 0, down -> ghost row ny+1, left -> ghost col 0, right -> ghost col nx+1
# ==============================================================================
@cuda.jit
def pack_halos_kernel(u, sendbuf, ny, nx):
    t = cuda.grid(1)
    total = 2 * nx + 2 * ny
    if t >= total:
        return
    if t < nx:                       # top owned row (row 1)
        sendbuf[t] = u[1, 1 + t]
    elif t < 2 * nx:                 # bottom owned row (row ny)
        k = t - nx
        sendbuf[t] = u[ny, 1 + k]
    elif t < 2 * nx + ny:            # left owned column (column 1)  - strided gather
        k = t - 2 * nx
        sendbuf[t] = u[1 + k, 1]
    else:                            # right owned column (column nx) - strided gather
        k = t - 2 * nx - ny
        sendbuf[t] = u[1 + k, nx]


@cuda.jit
def unpack_halos_kernel(u, recvbuf, ny, nx, has_up, has_down, has_left, has_right):
    t = cuda.grid(1)
    total = 2 * nx + 2 * ny
    if t >= total:
        return
    if t < nx:
        if has_up:
            u[0, 1 + t] = recvbuf[t]
    elif t < 2 * nx:
        if has_down:
            k = t - nx
            u[ny + 1, 1 + k] = recvbuf[t]
    elif t < 2 * nx + ny:
        if has_left:
            k = t - 2 * nx
            u[1 + k, 0] = recvbuf[t]
    else:
        if has_right:
            k = t - 2 * nx - ny
            u[1 + k, nx + 1] = recvbuf[t]


def create_halo_buffers(ny=NY_LOC, nx=NX_LOC):
    total = 2 * nx + 2 * ny
    return {
        "ny": ny,
        "nx": nx,
        "total": total,
        "d_send": cuda.device_array(total, dtype=np.float64),
        "d_recv": cuda.device_array(total, dtype=np.float64),
        "h_send": cuda.pinned_array(total, dtype=np.float64),   # page-locked
        "h_recv": cuda.pinned_array(total, dtype=np.float64),   # page-locked
    }


def exchange_halos_async(ctx, bufs, d_u, stream_boundary):
    """Stage A + start of Stage B: pack on GPU, async D2H into pinned memory,
    synchronize the boundary stream, then post non-blocking Irecv/Isend.
    Returns the list of MPI requests (completed later by finish_halo_exchange)."""
    ny, nx, total = bufs["ny"], bufs["nx"], bufs["total"]
    threads = 128
    blocks = (total + threads - 1) // threads

    # Stage A: pack (strided gather on device) + async D2H DMA on stream_boundary
    pack_halos_kernel[blocks, threads, stream_boundary](d_u, bufs["d_send"], ny, nx)
    bufs["d_send"].copy_to_host(bufs["h_send"], stream=stream_boundary)
    stream_boundary.synchronize()  # host send buffers are now valid

    h_send, h_recv = bufs["h_send"], bufs["h_recv"]
    off_up, off_down, off_left, off_right = 0, nx, 2 * nx, 2 * nx + ny

    cart = ctx["cart_comm"]
    reqs = []
    # Receives (data travelling toward me)
    reqs.append(cart.Irecv(h_recv[off_up:off_up + nx], source=ctx["nbr_up"], tag=TAG_DOWN))
    reqs.append(cart.Irecv(h_recv[off_down:off_down + nx], source=ctx["nbr_down"], tag=TAG_UP))
    reqs.append(cart.Irecv(h_recv[off_left:off_left + ny], source=ctx["nbr_left"], tag=TAG_RIGHT))
    reqs.append(cart.Irecv(h_recv[off_right:off_right + ny], source=ctx["nbr_right"], tag=TAG_LEFT))
    # Sends
    reqs.append(cart.Isend(h_send[off_up:off_up + nx], dest=ctx["nbr_up"], tag=TAG_UP))
    reqs.append(cart.Isend(h_send[off_down:off_down + nx], dest=ctx["nbr_down"], tag=TAG_DOWN))
    reqs.append(cart.Isend(h_send[off_left:off_left + ny], dest=ctx["nbr_left"], tag=TAG_LEFT))
    reqs.append(cart.Isend(h_send[off_right:off_right + ny], dest=ctx["nbr_right"], tag=TAG_RIGHT))
    return reqs


def finish_halo_exchange(ctx, bufs, d_u, reqs, stream_boundary):
    """Stage C (first half): Waitall, async H2D on stream_boundary, unpack ghost cells."""
    MPI.Request.Waitall(reqs)
    bufs["d_recv"].copy_to_device(bufs["h_recv"], stream=stream_boundary)
    threads = 128
    blocks = (bufs["total"] + threads - 1) // threads
    unpack_halos_kernel[blocks, threads, stream_boundary](
        d_u, bufs["d_recv"], bufs["ny"], bufs["nx"],
        ctx["nbr_up"] != MPI.PROC_NULL,
        ctx["nbr_down"] != MPI.PROC_NULL,
        ctx["nbr_left"] != MPI.PROC_NULL,
        ctx["nbr_right"] != MPI.PROC_NULL,
    )


# ==============================================================================
# TASK 4: 2D Shared-Memory Tiled Stencil & Dual-Stream Orchestrator
# Bounding box is INCLUSIVE: updates cells with r_start<=i<=r_end, c_start<=j<=c_end
# ==============================================================================
@cuda.jit
def tiled_stencil_kernel(u, un, kappa, S, r_start, r_end, c_start, c_end,
                         dt, inv_dx2, inv_dy2, lam):
    smem = cuda.shared.array(shape=(SHARED_DIM, SHARED_DIM), dtype=float64)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    i = r_start + cuda.blockIdx.y * TILE_DIM + ty
    j = c_start + cuda.blockIdx.x * TILE_DIM + tx
    nrows = u.shape[0]
    ncols = u.shape[1]

    # Center cell
    if i < nrows and j < ncols:
        smem[ty + 1, tx + 1] = u[i, j]
    # 1-cell halo apron (edge threads of the block)
    if ty == 0 and i >= 1 and j < ncols:
        smem[0, tx + 1] = u[i - 1, j]
    if ty == TILE_DIM - 1 and i + 1 < nrows and j < ncols:
        smem[SHARED_DIM - 1, tx + 1] = u[i + 1, j]
    if tx == 0 and j >= 1 and i < nrows:
        smem[ty + 1, 0] = u[i, j - 1]
    if tx == TILE_DIM - 1 and j + 1 < ncols and i < nrows:
        smem[ty + 1, SHARED_DIM - 1] = u[i, j + 1]

    cuda.syncthreads()  # every thread reaches the barrier (no early return above)

    if i >= r_start and i <= r_end and j >= c_start and j <= c_end:
        c = smem[ty + 1, tx + 1]
        un_c = smem[ty + 2, tx + 1]   # i+1
        us_c = smem[ty, tx + 1]       # i-1
        ue_c = smem[ty + 1, tx + 2]   # j+1
        uw_c = smem[ty + 1, tx]       # j-1

        k0 = kappa[i, j]
        k_n = 0.5 * (kappa[i + 1, j] + k0)
        k_s = 0.5 * (kappa[i - 1, j] + k0)
        k_e = 0.5 * (kappa[i, j + 1] + k0)
        k_w = 0.5 * (kappa[i, j - 1] + k0)

        div_y = (k_n * (un_c - c) - k_s * (c - us_c)) * inv_dy2
        div_x = (k_e * (ue_c - c) - k_w * (c - uw_c)) * inv_dx2
        un[i, j] = c + dt * (div_x + div_y - lam * c * c * c + S[i, j])


def launch_stencil(stream, fld, d_in, d_out, r0, r1, c0, c1):
    nr = r1 - r0 + 1
    nc = c1 - c0 + 1
    if nr <= 0 or nc <= 0:
        return
    grid = ((nc + TILE_DIM - 1) // TILE_DIM, (nr + TILE_DIM - 1) // TILE_DIM)
    tiled_stencil_kernel[grid, (TILE_DIM, TILE_DIM), stream](
        d_in, d_out, fld["d_kappa"], fld["d_S"],
        r0, r1, c0, c1,
        float(DT), float(1.0 / (DX * DX)), float(1.0 / (DY * DY)), float(LAMBDA_DAMP),
    )


def step_overlapped(ctx, fld, bufs, streams, d_in, d_out):
    """One dual-stream step: interior compute overlaps with MPI halo exchange."""
    ny, nx = NY_LOC, NX_LOC
    stream_interior, stream_boundary = streams

    # Stage A + B(post): pack, async D2H, sync boundary stream, post Irecv/Isend
    reqs = exchange_halos_async(ctx, bufs, d_in, stream_boundary)

    # Stage B (concurrent): deep interior [2..ny-1, 2..nx-1] on stream_interior
    launch_stencil(stream_interior, fld, d_in, d_out, 2, ny - 1, 2, nx - 1)

    # Stage C: wait, async H2D, unpack ghosts, boundary ring stencil on stream_boundary
    finish_halo_exchange(ctx, bufs, d_in, reqs, stream_boundary)
    launch_stencil(stream_boundary, fld, d_in, d_out, 1, 1, 1, nx)             # row 1
    launch_stencil(stream_boundary, fld, d_in, d_out, ny, ny, 1, nx)           # row ny
    launch_stencil(stream_boundary, fld, d_in, d_out, 2, ny - 1, 1, 1)         # column 1
    launch_stencil(stream_boundary, fld, d_in, d_out, 2, ny - 1, nx, nx)       # column nx

    stream_boundary.synchronize()
    stream_interior.synchronize()


def step_serialized(ctx, fld, bufs, streams, d_in, d_out):
    """Reference step: fully serialized exchange -> full-interior stencil on a single stream."""
    ny, nx = NY_LOC, NX_LOC
    _, stream_boundary = streams

    reqs = exchange_halos_async(ctx, bufs, d_in, stream_boundary)
    MPI.Request.Waitall(reqs)  # explicit serialization: nothing computes while messages are in flight
    finish_halo_exchange(ctx, bufs, d_in, [], stream_boundary)
    launch_stencil(stream_boundary, fld, d_in, d_out, 1, ny, 1, nx)  # entire owned interior
    stream_boundary.synchronize()


# ==============================================================================
# TASK 5: Distributed Reductions, PTX Audit & Comparative Benchmark
# ==============================================================================
def global_diagnostics(ctx, d_new, d_old, step):
    """Global thermal energy E = sum u*dx*dy, L2 norm ||u||_2 and L2 residual ||(u^n - u^{n-1})/dt||_2.
    Computed with Allreduce(SUM) over a 3-vector (no field gather)."""
    new_h = d_new.copy_to_host()[1:NY_LOC + 1, 1:NX_LOC + 1]
    local = np.zeros(3, dtype=np.float64)
    local[0] = new_h.sum() * DX * DY
    local[1] = (new_h * new_h).sum() * DX * DY
    if step > 0:
        old_h = d_old.copy_to_host()[1:NY_LOC + 1, 1:NX_LOC + 1]
        diff = (new_h - old_h) / DT
        local[2] = (diff * diff).sum() * DX * DY
    glob = np.zeros(3, dtype=np.float64)
    ctx["cart_comm"].Allreduce(local, glob, op=MPI.SUM)
    return {
        "step": step,
        "energy": float(glob[0]),
        "l2_norm": float(math.sqrt(glob[1])),
        "l2_residual": float(math.sqrt(glob[2])),
    }


def reset_state(fld):
    fld["d_u_a"].copy_to_device(fld["u0_host"])
    fld["d_u_b"].copy_to_device(fld["u0_host"])


def run_mode(ctx, fld, bufs, streams, mode, steps=STEPS):
    stepper = step_overlapped if mode == "overlapped" else step_serialized
    cart = ctx["cart_comm"]

    # Warm-up (JIT compilation + MPI/CUDA warm paths), then restore identical initial state
    reset_state(fld)
    u_cur, u_nxt = fld["d_u_a"], fld["d_u_b"]
    for _ in range(WARMUP_STEPS):
        stepper(ctx, fld, bufs, streams, u_cur, u_nxt)
        u_cur, u_nxt = u_nxt, u_cur
    cuda.synchronize()
    reset_state(fld)
    u_cur, u_nxt = fld["d_u_a"], fld["d_u_b"]

    trace = [global_diagnostics(ctx, u_cur, None, 0)]
    wall_ms = 0.0
    event_ms = 0.0
    cart.Barrier()

    done = 0
    while done < steps:
        chunk = min(DIAG_EVERY, steps - done)
        ev_start, ev_end = cuda.event(), cuda.event()
        cart.Barrier()
        t0 = time.perf_counter()
        ev_start.record(stream=streams[1])
        for _ in range(chunk):
            stepper(ctx, fld, bufs, streams, u_cur, u_nxt)
            u_cur, u_nxt = u_nxt, u_cur
        ev_end.record(stream=streams[1])
        ev_end.synchronize()
        t1 = time.perf_counter()
        wall_ms += (t1 - t0) * 1e3
        event_ms += ev_start.elapsed_time(ev_end)
        done += chunk
        # after swap: u_cur = u^{done}, u_nxt = u^{done-1}
        trace.append(global_diagnostics(ctx, u_cur, u_nxt, done))

    final_interior = u_cur.copy_to_host()[1:NY_LOC + 1, 1:NX_LOC + 1].copy()
    return {"trace": trace, "wall_ms": wall_ms, "event_ms": event_ms, "final": final_interior}


def ptx_audit():
    """Inspect PTX emitted for tiled_stencil_kernel (must be called after it has been launched)."""
    asm = tiled_stencil_kernel.inspect_asm()
    if isinstance(asm, dict):
        ptx = "\n".join(asm.values())
    else:
        ptx = str(asm)
    counts = {
        "st.shared.f64": len(re.findall(r"st\.shared(?:\.\w+)*\.f64", ptx)),
        "ld.shared.f64": len(re.findall(r"ld\.shared(?:\.\w+)*\.f64", ptx)),
        "bar.sync": len(re.findall(r"bar\.sync", ptx)),
        "st.shared (any)": len(re.findall(r"st\.shared", ptx)),
        "ld.shared (any)": len(re.findall(r"ld\.shared", ptx)),
    }
    passed = counts["st.shared (any)"] > 0 and counts["ld.shared (any)"] > 0 and counts["bar.sync"] > 0
    return counts, passed


def run_solver(ctx):
    cart = ctx["cart_comm"]
    rank = ctx["rank"]

    fld = initialize_subdomains(ctx)
    bufs = create_halo_buffers()
    streams = (cuda.stream(), cuda.stream())  # (stream_interior, stream_boundary)

    kmax = fld["kappa_max"]
    dt_max = DX * DX / (4.0 * kmax)
    if rank == 0:
        print("=" * 78)
        print(f"GITHUB_USERNAME     : {ctx['github_user']}")
        print(f"GPU                 : {ctx['device_name']} (CC {ctx['compute_capability']})")
        print(f"GPU UUID            : {ctx['gpu_uuid']}")
        print(f"INSTANCE_SEED       : {ctx['instance_seed']}")
        print(f"MPI ranks           : {ctx['size']} in a {PROC_DIMS[0]}x{PROC_DIMS[1]} non-periodic Cartesian grid")
        print(f"Local padded array  : ({PAD_NY}, {PAD_NX}) float64 | owned {NY_LOC}x{NX_LOC}")
        print(f"dx = dy = 1/513     : {DX:.9e}")
        print(f"kappa_max (run)     : {kmax:.9f}")
        print(f"dt_max = dx^2/(4k)  : {dt_max:.6e}")
        print(f"dt used             : {DT:.6e}  (dt/dt_max = {DT / dt_max:.4f})")
        print("=" * 78)
    for r in range(ctx["size"]):
        cart.Barrier()
        if r == rank:
            print(f"[rank {rank}] coords=({ctx['row_rank']},{ctx['col_rank']}) up={ctx['nbr_up']} "
                  f"down={ctx['nbr_down']} left={ctx['nbr_left']} right={ctx['nbr_right']}", flush=True)
        cart.Barrier()

    res_ser = run_mode(ctx, fld, bufs, streams, "serialized")
    res_ovl = run_mode(ctx, fld, bufs, streams, "overlapped")

    # Parity between serialized and overlapped final states (global max-abs difference)
    local_diff = np.array([np.max(np.abs(res_ser["final"] - res_ovl["final"]))], dtype=np.float64)
    parity = np.zeros(1, dtype=np.float64)
    cart.Allreduce(local_diff, parity, op=MPI.MAX)
    parity_max_diff = float(parity[0])

    # Worst-case (slowest rank) timings
    t_loc = np.array([res_ser["wall_ms"], res_ovl["wall_ms"], res_ser["event_ms"], res_ovl["event_ms"]])
    t_max = np.zeros(4)
    cart.Allreduce(t_loc, t_max, op=MPI.MAX)

    counts, ptx_ok = ptx_audit()

    if rank == 0:
        for name, res in (("serialized", res_ser), ("overlapped", res_ovl)):
            print(f"\nALLREDUCE TRACE [{name}]")
            for t in res["trace"]:
                print(f"  step {t['step']:4d} | E = {t['energy']:.10e} | "
                      f"||u||_2 = {t['l2_norm']:.10e} | L2 residual = {t['l2_residual']:.10e}")
        energies = [t["energy"] for t in res_ovl["trace"]]
        monotonic = all(energies[k + 1] < energies[k] for k in range(len(energies) - 1))
        print(f"\nEnergy monotonically decreasing (overlapped): {monotonic}")
        print(f"\nserialized_ms  (wall, slowest rank) : {t_max[0]:.3f}")
        print(f"overlapped_ms  (wall, slowest rank) : {t_max[1]:.3f}")
        print(f"serialized CUDA-event ms            : {t_max[2]:.3f}")
        print(f"overlapped CUDA-event ms            : {t_max[3]:.3f}")
        print(f"overlap speedup (wall)              : {t_max[0] / t_max[1]:.4f}x")
        print(f"\nparity_max_diff (serialized vs overlapped) : {parity_max_diff:.3e}  "
              f"[{'PASS' if parity_max_diff < 1e-12 else 'FAIL'}]")
        print("\nPTX AUDIT (tiled_stencil_kernel.inspect_asm())")
        for k, v in counts.items():
            print(f"  {k:18s}: {v}")
        print(f"  ptx_audit_passed  : {ptx_ok}")

        manifest = {
            "github_username": ctx["github_user"],
            "gpu_uuid": ctx["gpu_uuid"],
            "gpu_name": ctx["device_name"],
            "compute_capability": ctx["compute_capability"],
            "instance_seed": ctx["instance_seed"],
            "mpi_ranks": ctx["size"],
            "cartesian_dims": list(PROC_DIMS),
            "global_grid": [GLOBAL_NY, GLOBAL_NX],
            "local_padded_shape": [PAD_NY, PAD_NX],
            "steps": STEPS,
            "dx": DX,
            "dt": DT,
            "kappa_max": kmax,
            "dt_max": dt_max,
            "lambda_damp": LAMBDA_DAMP,
            "serialized_ms": t_max[0],
            "overlapped_ms": t_max[1],
            "serialized_cuda_event_ms": t_max[2],
            "overlapped_cuda_event_ms": t_max[3],
            "parity_max_diff": parity_max_diff,
            "parity_passed": parity_max_diff < 1e-12,
            "ptx_counts": counts,
            "ptx_audit_passed": bool(ptx_ok),
            "energy_trace_serialized": res_ser["trace"],
            "energy_trace_overlapped": res_ovl["trace"],
            "energy_monotonic_decreasing": bool(monotonic),
        }
        with open("submission_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        print("\nsubmission_manifest.json written.")
        print("ALL TASKS COMPLETED.")

    return parity_max_diff, counts, ptx_ok


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: mpirun -np 4 python3 distributed_cuda_solver.py <github_username>")
    ctx = init_distributed_context(sys.argv[1])
    run_solver(ctx)


if __name__ == "__main__":
    main()
