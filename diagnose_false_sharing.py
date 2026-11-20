"""Optional diagnostics (NOT part of the graded deliverable): where does false sharing show up on this CPU?
Runs stride=1 vs stride=8 for several thread->CPU placements. Edit CONFIGS to match your CPU layout
(check Task Manager -> Performance -> CPU -> Logical processors; on a hybrid Intel CPU the first
logical CPUs are usually P-core SMT pairs (0,1)(2,3)... and the last ones are E-cores)."""
import os
import numpy as np
import false_sharing_lab as m

ITERS = int(os.environ.get("DIAG_ITERS", 30_000_000))
REPS = 3
CONFIGS = {
    "4 P-cores, one thread each [0,2,4,6]": [0, 2, 4, 6],
    "2 P-cores, SMT siblings    [0,1,2,3]": [0, 1, 2, 3],
    "4 E-cores                  [8,9,10,11]": [8, 9, 10, 11],
    "2 P + 2 E (cross-cluster)  [0,2,8,10]": [0, 2, 8, 10],
    "unpinned (OS decides)": [None],
}

def median_ns(stride, cores, seeds):
    walls, per_thread = [], None
    for _ in range(REPS):
        buf = m.make_aligned_buffer(m.NUM_THREADS * stride + 16, 64)
        w, pt, _ = m._run_threads(m.worker_strided_accumulate, buf, stride, seeds, cores, ITERS)
        walls.append(w); per_thread = pt
    walls.sort()
    return walls[len(walls) // 2], per_thread

seeds = [(12345 + 0x9E3779B97F4A7C15 * (t + 1)) & 0xFFFFFFFFFFFFFFFF for t in range(m.NUM_THREADS)]
warm = m.make_aligned_buffer(64)
m.worker_strided_accumulate(warm, 0, 1000, np.uint64(seeds[0]))   # JIT warm-up
print(f"iters/thread = {ITERS:,}, repeats = {REPS}")
for name, cores in CONFIGS.items():
    a, pa = median_ns(1, cores, seeds)
    c, pc = median_ns(8, cores, seeds)
    print(f"{name:42s} stride1 = {a/1e6:9.1f} ms | stride8 = {c/1e6:9.1f} ms | ratio = {a/c:6.3f}x"
          f" | per-thread stride1 (ms) = {[round(x/1e6, 1) for x in pa]}")
