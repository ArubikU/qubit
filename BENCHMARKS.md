# Benchmarks

Hardware: AMD Ryzen 5 5600H (6C/12T, dual-channel DDR4), NVIDIA RTX 3060
Laptop 6 GB (sm_86, ~336 GB/s), Windows 11. Baseline: Qiskit Aer 0.17.2
(`statevector` method, complex128, OpenMP). qubit uses complex64 unless
noted. Times are median of 5 runs after one discarded warmup; laptop
thermal throttling still adds run-to-run variance of up to 2x, so treat
single-digit-percent differences as noise.

All engines were checked against each other for correctness first:
identical expectation values / probabilities to 5-6 decimals on QFT,
GHZ, QAOA and Bell-pair circuits, and qubit's three CPU backends agree
bit-exactly with each other under a shared seed (300-trial differential
test) and with the GPU backend (100 trials).

## Time (ms), 20 qubits

| circuit  | qubit GPU | qubit CPU | Aer CPU | GPU vs Aer |
|----------|-----------|-----------|---------|------------|
| QFT      | 13.2      | 67        | 242     | **18x**    |
| QAOA-4   | 15.1      | 77        | 93      | **6.2x**   |
| random   | 14.6      | 49        | 92      | **6.3x**   |
| GHZ      | 8.3       | —         | 26      | 3.1x       |
| pairs    | 19.9      | —         | 23      | 1.2x       |

## Time (ms), 24 qubits

| circuit  | qubit GPU | qubit CPU | Aer CPU | GPU vs Aer |
|----------|-----------|-----------|---------|------------|
| QFT      | 236       | 4322      | 4371    | **18.5x**  |
| QAOA-4   | 179       | 2552      | 2646    | **14.8x**  |
| random   | 175       | 2256      | 2632    | **15x**    |
| GHZ      | 112       | —         | 592     | 5.3x       |
| pairs    | 93        | —         | 395     | 4.2x       |

Aer's CPU and qubit's CPU converge at 24 qubits: both are pinned to the
same DDR4 bandwidth wall (~512 MB of traffic per gate pass). The GPU's
advantage is exactly its memory system — 336 GB/s VRAM versus ~50 GB/s
host RAM — which is why the gap grows with qubit count.

## Time (ms), 28 qubits — the standard comparison point

Every engine fits at 28 qubits (qubit f32: 2 GB; Aer complex128: 4 GB),
so this is the anchor table for cross-engine comparison.

End-to-end single-shot with state retrieval. qubit-GPU and Aer-GPU both
complex64 on the same RTX 3060; Aer-GPU via WSL2 (qiskit-aer-gpu 0.15.1,
cuStateVec). Aer-GPU-fp64 is Aer's default complex128 (4 GB at 28q).

qubit GPU column is the dense-forced backend (dense-vs-dense); the
planner would auto-route GHZ/pairs to blocks/groups (see Qrack table).

| circuit  | qubit GPU | Aer-GPU (fp32) | Aer-GPU (fp64) | Aer CPU |
|----------|-----------|----------------|----------------|---------|
| QFT      | 4,290     | 13,990         | 16,474         | 83,297  |
| QAOA-4   | 3,935     | 20,779         | 19,893         | 56,050  |
| random   | 4,159     | 23,443         | 22,953         | 51,647  |
| GHZ      | 1,450     | 10,614         | 15,463         | 11,248  |
| pairs    | 1,526     | 6,909          | 15,413         | 7,760   |

qubit-GPU is 3-7x faster than Aer-GPU at matched complex64 precision.
Caveat: these are end-to-end latencies (include state read-back and, for
Aer, Python/result-object overhead a C++ engine avoids), not isolated
kernel throughput. Read as user-visible single-run latency, not a
cuStateVec kernel comparison.

## vs Qrack (closest peer, same RTX 3060)

Qrack is structure-aware (Schmidt decomposition + stabilizer), so
low-entanglement states never densify. Times below are Qrack
gate-application only (excludes read-back, favoring Qrack).

The planner auto-selects the backend from static analysis (no user
action). Qrack times are gate-only (exclude read-back, favoring Qrack).

Single authoritative median-5 RTX-3060 dense set (QFT 4290, QAOA 3935,
random 4159, GHZ dense 1450, pairs dense 1526) is used across all
tables here and in the paper.

| 28q    | qubit (backend)   | Qrack   |
|--------|-------------------|---------|
| QFT    | 4,290 (dense-gpu) | 18,953  |
| QAOA-4 | 3,935 (dense-gpu) | 13,665  |
| random | 4,159 (dense-gpu) | 24,189  |
| GHZ    | 20 (blocks-gpu)   | 0.4     |
| pairs  | 0.08 (groups-cpu) | 0.5     |

Reading (honest): the planner routes each circuit to its best backend.
Dense circuits (QFT/QAOA/random): qubit 3.5-6x faster than Qrack. Pairs:
qubit factors it (groups) and beats Qrack. GHZ: qubit's ZERO tier gives
20ms/1MB (down from 1450ms dense) but Qrack's stabilizer tableau is
asymptotically optimal for Clifford states and wins; we don't implement
a stabilizer backend. Net: qubit wins 4 of 5; the loss is pure-Clifford
GHZ. Core contribution stays the DENSE regime (no structure to exploit).
Run: `QRACK_OCL_DEFAULT_DEVICE=0 py bench/qrack_bench.py 28`.

## Kernel-only (gate application, no read-back) — rebuts apples-to-oranges

Fair kernel-vs-kernel at 28q: qubit gate-application time (CUDA events,
excludes init + read-back) vs Aer internal result.time_taken (excludes
Python + read-back). qubit fusion OFF here (conservative).

| 28q    | qubit gate-only | Aer-GPU internal | ratio |
|--------|-----------------|------------------|-------|
| QFT    | 3,742           | 12,284           | 3.3x  |
| QAOA-4 | 4,997           | 21,108           | 4.2x  |
| random | 4,863           | 26,424           | 5.4x  |

qubit is 3.3-5.4x faster even with all framework/read-back overhead
removed from both sides -> the end-to-end win is not a WSL2/Python
artifact. Gap >3x exceeds the ~2x thermal variance and reproduces on T4.
Run: bench/kernel_bench.cu (qubit), aer_gpu_one.py col 7 (Aer internal).

## Cross-architecture (Tesla T4, Turing sm_75, free cloud)

Same suite on a Tesla T4 (16 GB) confirms the method is not tied to the
3060. Correctness identical (50 GPU-vs-CPU trials + 25 exact + 25 lossy
all pass via bench/colab_m6.sh); GHZ-31 exact in 1 MB; 28q echo
fidelity 0.9998 with memory halving at full compression. Dense times
~1.1-2x slower than the 3060 (older card), same qualitative behavior.

| 28q dense | RTX 3060 | Tesla T4 |
|-----------|----------|----------|
| QFT       | 4,290    | 6,048    |
| QAOA-4    | 3,935    | 4,956    |
| random    | 4,159    | 4,760    |
| GHZ       | 1,450    | 2,898    |
| pairs     | 1,526    | 2,787    |

## Capacity ceiling (6 GB VRAM)

| qubits | state size | result            |
|--------|------------|-------------------|
| 28     | 2 GB       | full suite runs (QFT 4.5 s) |
| 29     | 4 GB       | 29-qubit GHZ in 4.7 s — the complex64 ceiling |
| 30     | 8 GB       | does not fit dense; needs FP16 tier or structured backends |

Memory per amplitude: qubit complex64 = 8 B (complex128 available via
`Precision::F64` on CPU); Aer statevector = 16 B always. At equal VRAM
qubit holds one more qubit than a complex128 engine.

## Compressed tiers at scale (blocks-gpu, Loschmidt echo)

Echo circuit `H^n RZ RZ' H^n` — ideal P(0) = 1, so the measured value is
the true end-to-end fidelity at any size; the intermediate state is
uniform (every block nonzero: worst case for ZERO, real work for
COMPRESSED).

| n  | budget | time   | peak state (dense f32) | echo fidelity |
|----|--------|--------|------------------------|---------------|
| 28 | 2 (full compress) | 9.9 s | 1024 MB (2048 MB) | 0.999788    |
| 28 | 0.5 (partial)     | 9.9 s | 2048 MB (2048 MB) | 0.999794    |
| 29 | 10 (full compress)| 143 s | 2048 MB (4096 MB) | 0.999799    |
| 31 (GHZ) | 0 (exact ZERO)| <1 s | 1 MB (16384 MB)   | exact        |

- 29 qubits: the full uniform state lives int16-compressed in half the
  dense footprint; dense f32 would still fit this card, but at 30
  qubits (8 GB dense) compression is the only option — the run
  completes (measured fidelity 0.999791) but Windows WDDM starts paging
  VRAM near the 6 GB limit and time balloons; treated as the capacity
  edge, not a headline row.
- Batched chunk retier (2 kernels + 3 copies per 64-block chunk,
  replacing ~6 synchronizing calls per block) cut the 28-qubit echo
  from 51 s to 9.9 s (5.2x).

## Budget sweep, 28-qubit echo (the fidelity-memory-time tradeoff)

Source: bench/results/curve28.csv (single run per budget; laptop
thermal variance applies). This is the data behind the paper's central
figure and the echo-28 capacity rows.

| budget | time    | peak MB | echo fidelity | worst-case bound | random-walk estimate |
|--------|---------|---------|---------------|------------------|----------------------|
| 0      | 3.8 s   | 2048    | 1.000000      | 1.000            | 1.000                |
| 1e-3   | 7.6 s   | 2048    | 1.000001      | 0.996            | 0.99995              |
| 1e-2   | 12.1 s  | 2048    | 0.999998      | 0.961            | 0.99982              |
| 0.1    | 14.9 s  | 2048    | 0.999934      | 0.669            | 0.99840              |
| 0.5    | 10.1 s  | 2048    | 0.999794      | 0.111            | 0.98778              |
| >= 2   | 10.6 s  | **1024** | 0.999788     | 0.000            | 0.96182              |

Memory halves (2048 to 1024 MB) only once the budget (>=2) covers full
compression of every block; below that some blocks stay FULL. The
guaranteed bound reaches 0 at budget>=2 (deep-circuit regime), while
measured fidelity stays 0.9998 throughout.

## Budget sweep, 26-qubit echo (earlier run, same shape)

| budget | time   | peak MB | echo fidelity | worst-case bound | random-walk estimate |
|--------|--------|---------|---------------|------------------|----------------------|
| 0      | 1.8 s  | 512     | 1.000000      | 1.000            | 1.000                |
| 1e-3   | 1.8 s  | 512     | 0.999995      | 0.996            | 0.99992              |
| 1e-2   | 2.0 s  | 512     | 0.999998      | 0.961            | 0.99975              |
| 0.1    | 2.3 s  | 512     | 0.999857      | 0.669            | 0.9957               |
| 0.5    | 2.4 s  | 512     | 0.999789      | 0.111            | 0.9739               |
| >= 2   | 2.5 s  | **256** | 0.999789      | 0.051            | 0.9605               |

Reading: measured fidelity stays ~0.9998 across the whole sweep; the
adversarial bound collapses with depth while the random-walk estimate
(sqrt of summed squared injections; independence assumption, reported
alongside the guarantee, never instead of it) stays informative.
Memory halves once the budget covers full compression — int16 is
exactly half of complex64, as designed.

## Structured circuits (the point of the backend zoo)

- 80 qubits as twenty 4-qubit GHZ clusters: `groups-cpu` runs it in
  **2.5 KB** peak state memory, 0.1 ms. Dense would need 2^80 amplitudes
  (~10^13 TB) — not runnable by any statevector engine.
- Lossy tiered blocks (`blocks-cpu`, int16 block compression under a
  global L2 error budget): random 10-16 qubit circuits at a 0.998
  fidelity floor measure true fidelity 0.99996 with peak-memory savings
  up to 98%. GPU port of this tier is the active research direction.

## CPU optimization history (single day, same hardware, 20q)

| stage                          | QFT  | QAOA-4 | random |
|--------------------------------|------|--------|--------|
| naive loop                     | 291  | 484    | 457    |
| + OpenMP + pair enumeration    | 89   | 168    | 171    |
| + 2-qubit gate fusion          | 67   | 81     | 49     |
| + diagonal-gate kernels        | ~67  | ~77    | ~49    |
| AVX2 + /fp:fast                | no measurable effect (bandwidth-bound) |

## Positioning

- **Correctness**: validated against Qiskit Aer to float precision.
- **Dense speed**: GPU 4-18x faster than Aer CPU on the same laptop;
  gap widens with qubit count. (Fair fight pending: Aer-GPU/cuQuantum
  need a CUDA-enabled Aer build — planned.)
- **Memory**: 2x smaller state than complex128 engines; 29 dense qubits
  on a 6 GB consumer card.
- **Structure exploitation**: sleeping qubits, dynamic entanglement
  groups and tiered lossy blocks run circuits dense engines cannot,
  with exact or budget-bounded fidelity, chosen automatically by the
  planner.
