import os
import sys
import re
import json
import time
import hashlib
import platform
import threading
import subprocess
import numpy as np
from numba import njit, uint64

try:
    import resource
except ImportError:  # Windows
    resource = None

# Replace with your exact Git handle before executing
GIT_USERNAME = "brainmegamind303-droid"
NUM_THREADS = 4
ITERATIONS_PER_THREAD = 60_000_000
STRIDES = (1, 4, 8, 16)          # element strides: 8 B, 32 B, 64 B, 128 B for uint64
REPEATS = 3                      # each layout is measured REPEATS times; median is reported
CACHE_LINE_BYTES_DEFAULT = 64
MANIFEST_PATH = "cache_lab_manifest.json"


def detect_cache_line_size() -> int:
    """Query L1 data cache coherency line size in bytes from OS sysfs/sysctl."""
    linux_path = "/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size"
    try:
        if os.path.exists(linux_path):
            with open(linux_path, "r", encoding="utf-8") as f:
                return int(f.read().strip())
        if sys.platform == "darwin":
            out = subprocess.check_output(["sysctl", "-n", "hw.cachelinesize"], encoding="utf-8")
            return int(out.strip())
    except Exception:
        pass
    return CACHE_LINE_BYTES_DEFAULT


def detect_cpu_model() -> str:
    """Best-effort CPU model string (Linux /proc/cpuinfo, macOS sysctl, else platform.processor())."""
    try:
        if sys.platform.startswith("linux") and os.path.exists("/proc/cpuinfo"):
            with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
        if sys.platform == "darwin":
            out = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], encoding="utf-8")
            return out.strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def _linux_physical_core_representatives(allowed):
    """Return one logical CPU per physical core (avoids SMT siblings that would share an L1)."""
    seen, reps = set(), []
    for cpu in allowed:
        path = f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list"
        try:
            with open(path, "r", encoding="utf-8") as f:
                key = f.read().strip()
        except OSError:
            key = f"cpu{cpu}"
        if key not in seen:
            seen.add(key)
            reps.append(cpu)
    return reps


def select_affinity_cores(n_threads: int, logical_cores: int):
    """Pick n_threads logical CPUs, preferring distinct physical cores. Returns (cores, note)."""
    if hasattr(os, "sched_getaffinity"):  # Linux
        allowed = sorted(os.sched_getaffinity(0))
        reps = _linux_physical_core_representatives(allowed)
        if len(reps) >= n_threads:
            return reps[:n_threads], "distinct physical cores (SMT siblings avoided)"
        return allowed[:n_threads], ("WARNING: fewer than 4 distinct physical cores in the allowed set; "
                                     "threads may share a core/L1 via SMT, weakening false sharing")
    if sys.platform == "win32":
        if logical_cores >= 2 * n_threads:
            return [2 * k for k in range(n_threads)], "heuristic: every 2nd logical CPU (assumes SMT pairs)"
        return list(range(n_threads)), "sequential logical CPUs (SMT layout unknown)"
    return list(range(n_threads)), "affinity unsupported on this OS (macOS): OS scheduler decides"


def set_current_thread_affinity(cpu: int) -> bool:
    """Pin the *calling thread* to one logical CPU. Never raises; returns True on success."""
    try:
        if hasattr(os, "sched_setaffinity"):          # Linux: pid 0 == calling thread
            os.sched_setaffinity(0, {cpu})
            return True
        if sys.platform == "win32" and cpu < 64:
            import ctypes
            k32 = ctypes.windll.kernel32
            k32.GetCurrentThread.restype = ctypes.c_void_p
            k32.SetThreadAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            k32.SetThreadAffinityMask.restype = ctypes.c_size_t
            return bool(k32.SetThreadAffinityMask(k32.GetCurrentThread(), 1 << cpu))
    except Exception:
        pass
    return False


# ==============================================================================
# TASK 1: Hardware Topology & Student Identity Context
# ==============================================================================
def init_system_context(git_user: str) -> dict:
    assert git_user != "REPLACE_WITH_YOUR_GIT_USERNAME" and len(git_user.strip()) > 0, (
        "Set GIT_USERNAME to your real Git handle before running."
    )
    logical_cores = os.cpu_count() or 1
    if logical_cores < NUM_THREADS:
        raise RuntimeError(f"Requires at least {NUM_THREADS} logical cores; detected {logical_cores}.")

    line_size = detect_cache_line_size()
    cpu_model = detect_cpu_model()
    cpu_sig = f"{platform.machine()}::{platform.processor()}::{logical_cores}::{cpu_model}"

    digest = hashlib.sha256(f"{git_user.strip()}::{cpu_sig}".encode("utf8")).hexdigest()
    instance_seed = int(digest[:16], 16)  # deterministic 64-bit integer

    cores, affinity_note = select_affinity_cores(NUM_THREADS, logical_cores)
    return {
        "git_username": git_user.strip(),
        "logical_cores": logical_cores,
        "cache_line_size": line_size,
        "cpu_model": cpu_model,
        "cpu_signature": cpu_sig,
        "instance_seed": instance_seed,
        "affinity_cores": cores,
        "affinity_note": affinity_note,
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
    }


def make_aligned_buffer(n_elems: int, align_bytes: int = 64) -> np.ndarray:
    """Allocate a zeroed uint64 array whose first element is aligned to `align_bytes`."""
    extra = align_bytes // 8
    raw = np.zeros(n_elems + extra, dtype=np.uint64)
    offset = ((-raw.ctypes.data) % align_bytes) // 8
    buf = raw[offset:offset + n_elems]
    assert buf.ctypes.data % align_bytes == 0, "buffer is not cache-line aligned"
    return buf


def thread_delta(seed: int) -> int:
    """Per-thread increment (odd, 1..15); identical formula is used inside both kernels."""
    return (seed & 15) | 1


# ==============================================================================
# TASK 2: GIL-Free Native Counter Kernel
# ==============================================================================
@njit(nogil=True)
def worker_strided_accumulate(buffer, slot, iters, seed):
    """In-place read-modify-write on buffer[idx] in every iteration.

    The address of the next access (idx) depends on the value just loaded from memory
    (v >> 63 is always 0 at run time because the counter stays far below 2**63, but the
    compiler cannot prove it). Because the address is not a loop-invariant, LLVM cannot
    promote buffer[slot] to a register, so each iteration performs a real load + store.
    """
    delta = (seed & np.uint64(15)) | np.uint64(1)
    shift = np.uint64(63)
    idx = slot
    for _ in range(iters):
        v = buffer[idx]
        buffer[idx] = v + delta
        idx = slot + np.int64(v >> shift)


# ==============================================================================
# TASK 4: Thread-Local Register Privatization Baseline
# ==============================================================================
@njit(nogil=True)
def worker_private_accumulate(buffer, slot, iters, seed):
    """Accumulate in a local scalar (register) and store to shared memory exactly once."""
    delta = (seed & np.uint64(15)) | np.uint64(1)
    shift = np.uint64(63)
    acc = np.uint64(0)
    for _ in range(iters):
        # (acc >> 63) is always 0 at run time; it only prevents LLVM from collapsing the
        # loop into the closed form iters*delta, so the baseline keeps a real dependency chain.
        acc = acc + delta + (acc >> shift)
    buffer[slot] = acc


# ==============================================================================
# TASK 3: Multi-Threaded Stride Sweep
# ==============================================================================
def _run_threads(kernel, buffer, stride, thread_seeds, cores, iters):
    """Launch NUM_THREADS native threads, release them together, return (wall_ns, per_thread_ns, pinned)."""
    barrier = threading.Barrier(NUM_THREADS + 1)
    per_thread_ns = [0] * NUM_THREADS
    pinned = [False] * NUM_THREADS

    def runner(t):
        pinned[t] = set_current_thread_affinity(cores[t % len(cores)])
        barrier.wait()
        s = time.perf_counter_ns()
        kernel(buffer, t * stride, iters, np.uint64(thread_seeds[t]))
        per_thread_ns[t] = time.perf_counter_ns() - s

    threads = [threading.Thread(target=runner, args=(t,)) for t in range(NUM_THREADS)]
    for th in threads:
        th.start()
    barrier.wait()
    t0 = time.perf_counter_ns()
    for th in threads:
        th.join()
    wall_ns = time.perf_counter_ns() - t0
    return wall_ns, per_thread_ns, pinned


def _rusage_snapshot():
    if resource is None:
        return None
    ru = resource.getrusage(resource.RUSAGE_SELF)
    return ru.ru_nvcsw, ru.ru_nivcsw


def _measure_layout(kernel, stride, ctx, thread_seeds, iters):
    n_elems = NUM_THREADS * max(stride, 1) + 16
    runs = []
    final_values = None
    for _ in range(REPEATS):
        buf = make_aligned_buffer(n_elems, 64)
        before = _rusage_snapshot()
        wall_ns, per_thread_ns, pinned = _run_threads(
            kernel, buf, stride, thread_seeds, ctx["affinity_cores"], iters)
        after = _rusage_snapshot()
        vol = invol = None
        if before is not None:
            vol, invol = after[0] - before[0], after[1] - before[1]
        runs.append({
            "wall_ns": wall_ns,
            "per_thread_ns": per_thread_ns,
            "voluntary_ctx_switches": vol,
            "involuntary_ctx_switches": invol,
            "affinity_applied": pinned,
        })
        vals = [int(buf[t * stride]) for t in range(NUM_THREADS)]
        if final_values is None:
            final_values = vals
        elif vals != final_values:
            raise AssertionError("final counters changed between repeats of the same layout")
    walls = sorted(r["wall_ns"] for r in runs)
    median_ns = walls[len(walls) // 2]
    return {
        "stride_elements": stride,
        "stride_bytes": stride * 8,
        "runs": runs,
        "median_wall_ns": median_ns,
        "min_wall_ns": walls[0],
        "ns_per_update_per_thread": median_ns / iters,
        "ns_per_update_aggregate": median_ns / (iters * NUM_THREADS),
        "final_values": final_values,
    }


def run_stride_benchmark(ctx: dict, iters: int = ITERATIONS_PER_THREAD) -> dict:
    seed = ctx["instance_seed"]
    # Distinct deterministic per-thread seeds derived from INSTANCE_SEED
    thread_seeds = [(seed + 0x9E3779B97F4A7C15 * (t + 1)) & 0xFFFFFFFFFFFFFFFF for t in range(NUM_THREADS)]

    # JIT warm-up (compile both kernels outside the timed region)
    warm = make_aligned_buffer(NUM_THREADS * 16 + 16, 64)
    worker_strided_accumulate(warm, 0, 1000, np.uint64(thread_seeds[0]))
    worker_private_accumulate(warm, 0, 1000, np.uint64(thread_seeds[0]))

    results = {"thread_seeds": thread_seeds, "layouts": {}}
    names = {1: "A_contiguous", 4: "B_half_line", 8: "C_line_padded", 16: "D_prefetch_safe"}
    for stride in STRIDES:
        res = _measure_layout(worker_strided_accumulate, stride, ctx, thread_seeds, iters)
        results["layouts"][names[stride]] = res
        print(f"[stride={stride:2d} ({stride * 8:3d} B)] median wall = {res['median_wall_ns']:>15,d} ns | "
              f"{res['ns_per_update_per_thread']:.3f} ns/update/thread | runs(ns)={[r['wall_ns'] for r in res['runs']]}",
              flush=True)

    # Task 4 baseline: private register accumulation, one final store per thread (slots spaced by 8 -> 64 B)
    priv = _measure_layout(worker_private_accumulate, 8, ctx, thread_seeds, iters)
    results["layouts"]["P_private_register"] = priv
    print(f"[private baseline      ] median wall = {priv['median_wall_ns']:>15,d} ns | "
          f"{priv['ns_per_update_per_thread']:.3f} ns/update/thread", flush=True)

    L = results["layouts"]
    results["slowdown_stride1_vs_stride8"] = L["A_contiguous"]["median_wall_ns"] / L["C_line_padded"]["median_wall_ns"]
    results["slowdown_stride4_vs_stride8"] = L["B_half_line"]["median_wall_ns"] / L["C_line_padded"]["median_wall_ns"]
    results["ratio_stride16_vs_stride8"] = L["D_prefetch_safe"]["median_wall_ns"] / L["C_line_padded"]["median_wall_ns"]
    results["slowdown_stride1_vs_private"] = L["A_contiguous"]["median_wall_ns"] / max(priv["median_wall_ns"], 1)
    results["slowdown_stride8_vs_private"] = L["C_line_padded"]["median_wall_ns"] / max(priv["median_wall_ns"], 1)
    return results


# ==============================================================================
# TASK 5: Exact State Parity & LLVM IR Verification
# ==============================================================================
def audit_llvm_and_parity(ctx: dict, bench: dict, iters: int = ITERATIONS_PER_THREAD) -> dict:
    # --- Exact state parity -----------------------------------------------------
    thread_seeds = bench["thread_seeds"]
    expected = [iters * thread_delta(s) for s in thread_seeds]
    reference = None
    parity_ok = True
    per_layout = {}
    for name, res in bench["layouts"].items():
        vals = res["final_values"]
        ok = vals == expected
        per_layout[name] = {"final_values": vals, "matches_expected": ok}
        parity_ok &= ok
        if reference is None:
            reference = vals
        parity_ok &= (vals == reference)

    # --- LLVM IR inspection -----------------------------------------------------
    ir_dict = worker_strided_accumulate.inspect_llvm()
    ir = "\n".join(ir_dict.values()) if isinstance(ir_dict, dict) else str(ir_dict)

    # Isolate the native compute function (skip the `cpython::` boxing wrapper and the cfunc wrapper)
    native_body, in_native = [], False
    for ln in ir.splitlines():
        if ln.startswith("define"):
            in_native = ("worker_strided_accumulate" in ln and "7cpython" not in ln and "cfunc." not in ln)
        if in_native:
            native_body.append(ln)
        if in_native and ln.strip() == "}":
            in_native = False
    native_ir = "\n".join(native_body)

    store_i64 = len(re.findall(r"\bstore i64\b", native_ir))
    load_i64 = len(re.findall(r"\bload i64\b", native_ir))
    # Python-object fallback would show up as CPython C-API calls inside the compute function itself
    pyobj_calls = len(re.findall(r"call [^\n]*@\"?Py", native_ir))
    # The argument-unboxing wrapper releases the GIL around the native call (nogil=True evidence)
    gil_release_calls = len(re.findall(r"@PyEval_SaveThread", ir))
    nopython_sigs = [str(s) for s in worker_strided_accumulate.nopython_signatures]
    llvm_ok = (store_i64 > 0 and load_i64 > 0 and pyobj_calls == 0
               and gil_release_calls > 0 and len(nopython_sigs) > 0)

    # Short excerpt around the loop's store for ANALYSIS.md (first store i64 with a few lines of context)
    lines = native_ir.splitlines()
    excerpt = []
    candidates = [i for i, ln in enumerate(lines) if re.search(r"\bstore i64\b", ln)]
    main_loop = [i for i in candidates if ".prol" not in lines[i] and ".epil" not in lines[i]]
    pick = (main_loop or candidates or [None])[0]
    if pick is not None:
        excerpt = lines[max(0, pick - 6): pick + 4]

    audit = {
        "expected_final_values": expected,
        "parity_per_layout": per_layout,
        "parity_bit_exact": bool(parity_ok),
        "llvm_store_i64_count": store_i64,
        "llvm_load_i64_count": load_i64,
        "llvm_python_object_calls": pyobj_calls,
        "llvm_gil_release_calls_PyEval_SaveThread": gil_release_calls,
        "numba_nopython_signatures": nopython_sigs,
        "llvm_audit_passed": bool(llvm_ok),
        "llvm_excerpt": excerpt,
    }
    return audit


def write_manifest(ctx, bench, audit, path=MANIFEST_PATH):
    L = bench["layouts"]
    manifest = {
        "git_username": ctx["git_username"],
        "instance_seed": ctx["instance_seed"],
        "logical_cores": ctx["logical_cores"],
        "cache_line_size": ctx["cache_line_size"],
        "cpu_model": ctx["cpu_model"],
        "cpu_signature": ctx["cpu_signature"],
        "platform": ctx["platform"],
        "python": ctx["python"],
        "numpy": ctx["numpy"],
        "affinity_cores": ctx["affinity_cores"],
        "affinity_note": ctx["affinity_note"],
        "num_threads": NUM_THREADS,
        "iterations_per_thread": ITERATIONS_PER_THREAD,
        "repeats": REPEATS,
        "timings_ns_median": {k: v["median_wall_ns"] for k, v in L.items()},
        "ns_per_update_per_thread": {k: v["ns_per_update_per_thread"] for k, v in L.items()},
        "slowdown_stride1_vs_stride8": bench["slowdown_stride1_vs_stride8"],
        "slowdown_stride4_vs_stride8": bench["slowdown_stride4_vs_stride8"],
        "ratio_stride16_vs_stride8": bench["ratio_stride16_vs_stride8"],
        "slowdown_stride1_vs_private": bench["slowdown_stride1_vs_private"],
        "layouts": L,
        "audit": audit,
        "verification_passed": bool(audit["parity_bit_exact"] and audit["llvm_audit_passed"]),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def main():
    ctx = init_system_context(GIT_USERNAME)
    print("=" * 78)
    print(f"GIT_USERNAME      : {ctx['git_username']}")
    print(f"CPU model         : {ctx['cpu_model']}")
    print(f"Logical cores     : {ctx['logical_cores']}")
    print(f"Cache line size   : {ctx['cache_line_size']} B")
    print(f"INSTANCE_SEED     : {ctx['instance_seed']}")
    print(f"Affinity cores    : {ctx['affinity_cores']}  ({ctx['affinity_note']})")
    print(f"Threads x iters   : {NUM_THREADS} x {ITERATIONS_PER_THREAD:,}")
    print("=" * 78)

    bench = run_stride_benchmark(ctx)
    audit = audit_llvm_and_parity(ctx, bench)
    manifest = write_manifest(ctx, bench, audit)

    L = bench["layouts"]
    print("\nSUMMARY (median wall-clock, ns)")
    for k, v in L.items():
        print(f"  {k:20s} {v['median_wall_ns']:>16,d} ns   {v['ns_per_update_per_thread']:8.3f} ns/update/thread")
    print(f"\nslowdown stride=1 vs stride=8 : {bench['slowdown_stride1_vs_stride8']:.3f}x")
    print(f"slowdown stride=4 vs stride=8 : {bench['slowdown_stride4_vs_stride8']:.3f}x")
    print(f"stride=16 / stride=8          : {bench['ratio_stride16_vs_stride8']:.3f}x")
    print(f"stride=1 vs private baseline  : {bench['slowdown_stride1_vs_private']:.3f}x")
    print("\nCONTEXT SWITCHES (median run per layout: voluntary / involuntary)")
    for k, v in L.items():
        r = sorted(v["runs"], key=lambda x: x["wall_ns"])[len(v["runs"]) // 2]
        print(f"  {k:20s} {r['voluntary_ctx_switches']} / {r['involuntary_ctx_switches']}")
    print(f"\nPARITY bit-exact across all layouts: {audit['parity_bit_exact']}")
    print(f"LLVM: store i64 = {audit['llvm_store_i64_count']}, load i64 = {audit['llvm_load_i64_count']}, "
          f"python-object calls = {audit['llvm_python_object_calls']}, "
          f"PyEval_SaveThread (GIL release) = {audit['llvm_gil_release_calls_PyEval_SaveThread']}  -> passed = {audit['llvm_audit_passed']}")
    print("LLVM excerpt around first `store i64`:")
    for ln in audit["llvm_excerpt"]:
        print("   ", ln)
    print(f"\n{MANIFEST_PATH} written | verification_passed = {manifest['verification_passed']}")
    if not manifest["verification_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
