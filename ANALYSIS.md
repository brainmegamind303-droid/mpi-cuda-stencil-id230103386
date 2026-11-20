# Cache Line & False Sharing Mechanics in Multi-Core Python: Analysis

**Student Git handle:** `brainmegamind303-droid`
**Host:** CPU model = `Intel64 Family 6 Model 186 Stepping 2, GenuineIntel` | logical cores = 12 | cache line = 64 B | OS = Windows
**INSTANCE_SEED:** 12850887399679581476
**Pinned logical CPUs:** [0, 2, 4, 6] (heuristic "every 2nd logical CPU", which assumes SMT pairs (0,1)(2,3)(4,5)(6,7); this mapping was not verified independently)
**Setup:** 4 threads x 60,000,000 updates per thread, each layout measured 3 times, **median** wall-clock reported. All values come from `cache_lab_manifest.json` (`verification_passed = true`).

## Measured data (from my run)

| Layout | Stride | Individual runs (ns) | Median wall time (ns) | ns / update / thread | ns / update (aggregate, 240 M updates) |
|---|---|---|---|---|---|
| A contiguous | 1 (8 B) | 216,910,500 / 199,942,500 / 190,064,100 | 199,942,500 | 3.332 | 0.833 |
| B half-line | 4 (32 B) | 198,952,200 / 197,429,400 / 196,948,400 | 197,429,400 | 3.290 | 0.823 |
| C line-padded | 8 (64 B) | 187,281,000 / 185,905,500 / 218,410,700 | 187,281,000 | 3.121 | 0.780 |
| D prefetch-safe | 16 (128 B) | 187,571,400 / 179,205,700 / 191,235,100 | 187,571,400 | 3.126 | 0.782 |
| P private register | final store only | (see manifest) | 42,821,200 | 0.714 | 0.178 |

- Slowdown stride 1 vs stride 8 = 199,942,500 / 187,281,000 = **1.068x**; stride 4 vs stride 8 = **1.054x**; stride 16 vs stride 8 = **1.002x**.
- stride 1 vs private baseline = **4.669x**; stride 8 vs private baseline = 187,281,000 / 42,821,200 = **4.374x**.
- Context switches: not available on Windows (the `resource` module does not exist there), so the manifest stores `null`.
- Parity: final counters are bit-exact across all five layouts and equal to `iters * delta` per thread (`parity_bit_exact = true`).

**Important caveat about these measurements:** the run-to-run spread inside a single layout (e.g. 185.9 to 218.4 ms for stride 8, 190.1 to 216.9 ms for stride 1) is as large as the difference between layouts. Comparing the best runs gives only 190,064,100 / 185,905,500 = 1.022x. Therefore, on this machine and with this kernel, false sharing is at most a few percent and is not clearly separable from noise. I report this as measured and do not claim the multi-fold slowdown that is typical on other hardware.

---

## Question 1 (MESI Protocol State Transitions)

With `stride = 1` the buffer is 64-byte aligned (asserted in code), so `buffer[0]` (byte offset 0) and `buffer[1]` (byte offset 8) are in the **same 64-byte cache line** (`floor((Base + 8k)/64)` is identical for k = 0..7). Coherence works on whole lines, so the two stores conflict although the addresses differ.

Assume neither core holds the line in a writable state:

1. **Core 0 stores `buffer[0]`.** C0 issues a Read-For-Ownership (RFO); other copies are invalidated. C0: `I -> M` (or `E -> M` silently if it already held the line Exclusive, or `S -> M` through an upgrade/invalidate request). C1: `I`.
2. **Core 1 stores `buffer[1]` immediately afterwards.** C1's copy is `I`, so it issues an RFO for the same line. C0 holds it in `M` and supplies the dirty data (cache-to-cache transfer / write-back) and moves `M -> I`. C1: `I -> M`.
3. **Core 0's next store** finds its copy `I` and repeats the RFO: C0 `I -> M`, C1 `M -> I`. Repeating this gives the ping-pong `M(C0) -> M(C1) -> M(C0) -> ...`; in layout A four cores compete for the same line.
   (The `S` state appears only when a core *reads* a line another core holds in `M`/`E`; then both copies become `S`. In my kernel each iteration performs a load and a store to the same line, so a stolen line costs both.)

**Why the RFO stalls the store buffer.** A store retires into the core's store buffer and can be written into L1 only when the line is held in `M`/`E`. x86 uses TSO (stores become globally visible in program order), so the **oldest store must obtain ownership and commit before younger stores can drain**. While the RFO of the oldest store is in flight, younger stores queue behind it; when the store buffer is full, the pipeline cannot allocate new stores and stalls. A cross-core ownership transfer typically costs tens of nanoseconds.

**What my measurement shows.** The measured penalty is small: stride 1 costs 3.332 ns/update/thread versus 3.121 ns for stride 8, an extra 0.211 ns per update (and within run-to-run noise). If every iteration had to wait for a cross-core ownership transfer, I would expect tens of ns per update. So on this CPU the ownership transfer is **not** on the critical path of each iteration (see the hypothesis in Question 2).

## Question 2 (Quantitative Slowdown & Coherency Overhead)

| Regime | Time (ns) | ns/update/thread = T / 60,000,000 | ns/update aggregate = T / 240,000,000 |
|---|---|---|---|
| stride = 1 | 199,942,500 | 3.332 | 0.833 |
| stride = 4 | 197,429,400 | 3.290 | 0.823 |
| stride = 8 | 187,281,000 | 3.121 | 0.780 |
| stride = 16 | 187,571,400 | 3.126 | 0.782 |
| private register | 42,821,200 | 0.714 | 0.178 |

**Slowdown of stride 1 relative to stride 8:** `199,942,500 / 187,281,000` = **1.068x** (manifest: `slowdown_stride1_vs_stride8`). Stride 4 vs stride 8: 1.054x (2-way sharing: threads {0,1} on one line, {2,3} on another).

**Interpretation of the regimes**
- *Private register baseline* (0.714 ns/update): a pure ALU dependency chain (add + shift + add) with no memory access inside the loop.
- *stride = 8 / 16* (about 3.12 ns/update): each thread owns its own line, so no coherence traffic; the cost is the L1 load -> add -> store chain with **store-to-load forwarding**, made longer because the next address depends on the loaded value. `stride 8 / private = 4.374x` isolates this store-forwarding latency from coherence.
- *stride = 1* (3.332 ns/update): the same chain plus the false-sharing effect, which adds only about 0.211 ns per update (1.068x), comparable to the measurement noise.

**Hypothesis for the small effect (not verified with hardware counters).** Loads in my loop can be satisfied by store-to-load forwarding from the core's own store buffer, so the dependency chain does not wait for the line. Ownership is needed only when stores commit to L1, and once a core owns the line it can commit several queued stores to that same line back-to-back, amortizing one RFO over many stores. In addition, the loop is already latency-bound (about 3.1 ns per iteration even without sharing), so a modest coherence cost can be hidden behind that chain. Other possible contributors are the hybrid core layout of this CPU and the Windows scheduler; I did not isolate them. I did not use performance counters (e.g. HITM events), so the mechanism remains a hypothesis consistent with the data.

## Question 3 (L2 Spatial Prefetcher & 128-Byte Pairing)

Measured: stride = 8 -> 187,281,000 ns, stride = 16 -> 187,571,400 ns, ratio `T16/T8` = **1.002** (manifest: `ratio_stride16_vs_stride8`). The individual runs overlap (stride 8: 185.9 to 218.4 ms; stride 16: 179.2 to 191.2 ms), so there is **no measurable difference**.

**Mechanism.** Some Intel cores include an L2 "adjacent cache line" (spatial) prefetcher: when a 64-byte line is fetched into L2, hardware may also fetch its buddy line so the pair forms an aligned **128-byte sector** (even/odd line indices). With stride = 8 the counters sit at 0, 64, 128 and 192 B, so threads 0/1 share one 128-byte pair (lines 0 and 1) and threads 2/3 another (lines 2 and 3). A demand access or RFO to line 0 could cause the prefetcher to pull line 1 (thread 1's counter) into a **Shared** state in the other core, so thread 1's next store has to invalidate it again: **secondary invalidations** although no two threads write the same 64-byte line. With stride = 16 (128 B spacing) each counter lies in its own 128-byte sector and the buddy line is unused padding, so this interference cannot occur.

**When 64-byte padding still suffers:** (1) the CPU has an adjacent-line / 128-byte prefetcher enabled and neighbouring threads' counters fall in the same 128-byte aligned pair (layout C); (2) the access pattern triggers the prefetcher (continuous stores/RFOs do); (3) the cores involved share the L2 structure where the speculative fetch happens; (4) the buffer base is 64-byte but not 128-byte aligned, which changes which counters pair up.

**My result:** `T16/T8 = 1.002` means I observed no sign of 128-byte pairing interference on this CPU in this setup. This is consistent with the overall small false-sharing effect of Question 2: if the ownership transfers are largely hidden, additional prefetch-induced invalidations also stay invisible. It does not prove the prefetcher is absent or disabled.

## Question 4 (Compiler Register Promotion vs. Volatile Memory Stores)

**Why a compiler wants to hoist.** In a loop such as `for i in range(n): buffer[slot] = buffer[slot] + delta`, the address `&buffer[slot]` is loop-invariant. LLVM's loop-invariant code motion / **scalar promotion** can keep the value in a register for the whole loop, load once before it and store once after it, reducing the loop to a register add (often even to the closed form `n*delta`). This removes all memory traffic and thus all false sharing, because the compiler assumes no other thread observes the intermediate values (Numba exposes no `volatile` or atomic here).

**How my kernel prevents it.** The address of each access depends on the value just loaded:
```python
v = buffer[idx]
buffer[idx] = v + delta
idx = slot + np.int64(v >> 63)     # always 0 at run time, but not provably 0 for the compiler
```
The accessed location is **not loop-invariant** (`idx` is a loop-carried value that depends on a load), so promotion is impossible and every iteration performs `load -> add -> store`. At run time `v >> 63` is 0 because the counters stay far below 2^63 (at most 60,000,000 x 15 = 9e8), so every thread still updates exactly `buffer[slot]`, and final values equal `iters * delta` (verified bit-exactly).

**LLVM IR evidence from `worker_strided_accumulate.inspect_llvm()` in my run:**
- `store i64` count in the native function: **5**; `load i64` count: **5** (more than one because the loop is unrolled/has peeled copies; the suffix `.1` in `%.168.1` in the excerpt indicates an unrolled copy).
- Python C-API calls inside the compute function: **0** (native `i64` code, no Python object fallback).
- `PyEval_SaveThread` calls in the boxing wrapper: **2** (the GIL is released around the native call, i.e. `nogil=True` works).
- Excerpt of the loop body (`load i64` -> `add i64` -> `store i64` -> `lshr i64 ..., 63` -> address update):
```llvm
%.168 = icmp slt i64 %idx.2.085, 0
%.169 = select i1 %.168, i64 %arg.buffer.5.0, i64 0
%4 = getelementptr i64, ptr %arg.buffer.4, i64 %idx.2.085
%.183 = getelementptr i64, ptr %4, i64 %.169
%.184 = load i64, ptr %.183, align 8
%.186 = add i64 %.24, %.184
store i64 %.186, ptr %.183, align 8
%.219 = lshr i64 %.184, 63
%.221 = add nsw i64 %.219, %arg.slot
%.168.1 = icmp slt i64 %.221, 0
```
(`icmp slt ..., 0` / `select` implement Python's negative-index wraparound.) By contrast, `worker_private_accumulate` keeps the accumulator in a register and has a single store after the loop, which is why it runs at 0.714 ns/update.

## Question 5 (CPython GIL Masking vs. nogil=True Execution)

In standard CPython the **Global Interpreter Lock** lets only one thread execute Python bytecode at a time. Four pure-Python threads incrementing `buf[i] += 1` take turns (the interpreter requests a switch about every 5 ms, `sys.getswitchinterval()`, or when a thread blocks). At any moment only one core performs the stores, and the cache line stays in that core's L1 for many consecutive iterations before the lock is handed over. There is **no simultaneous store contention and therefore no sustained RFO ping-pong**: layout A and layout C would look alike, so false sharing is *masked*. Moreover, each Python-level increment costs tens of nanoseconds of interpreter overhead (bytecode dispatch, reference counting, object handling), which dwarfs a coherence miss that would occur only about once per GIL hand-off.

Increasing the thread count makes the same per-thread work **slower** because of *software* lock contention, not hardware cache contention: GIL hand-offs involve condition-variable/futex-style signalling, OS context switches and wake-up latency (the convoy effect), the cache and branch-predictor state of the interpreter is lost when another core resumes it, and waiting threads add no throughput. With `@njit(nogil=True)` the GIL is released (the `PyEval_SaveThread` calls found in the IR wrapper), the four native loops truly run in parallel on separate cores, and the remaining bottleneck is the hardware MESI ownership transfer per store, which the stride sweep is designed to expose. (On my CPU that effect turned out small, see Questions 1 to 3.)
(This applies to the standard GIL build; a free-threaded CPython 3.13+ build without the GIL would behave differently.) I did not run a pure-Python thread experiment, so this answer is explanatory and not based on a measurement of my own.

---

## AI assistance disclosure
The code (`false_sharing_lab.py`) and the first draft of this analysis were generated with the help of an AI assistant (Claude, Anthropic) from the lab specification. I ran the program on my own machine; all timings, the seed, the LLVM IR counts and the manifest in this repository come from my own execution, and I reviewed the text before submission.
