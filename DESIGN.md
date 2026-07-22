# qubit — Design

GPU-accelerated quantum circuit simulator with automatic backend selection.
The user declares qubits, gates, and a fidelity target; the framework picks
the cheapest state representation that honors that target, or fails with a
descriptive error explaining exactly what did not fit and what would.

## Problem statement

Simulating n qubits exactly requires storing 2^n complex amplitudes: each
additional qubit doubles memory. On a 6 GB GPU this caps out at 29 qubits
(complex64, in-place kernels). The exponential wall is a theorem, not an
implementation deficiency — but most circuits of practical interest do not
occupy the full 2^n space at all times. qubit exploits that structure
automatically instead of asking the user to choose representations, cutoff
parameters, or bond dimensions by hand.

## Architecture

```
 user code (C++)
      |
      v
 +-----------+   records gates, never executes      (include/qubit/qubit.h)
 |  Circuit  |
 +-----------+
      |
      v
 +-----------+   static analysis, zero state memory
 | Analyzer  |   - sleeping qubits (never touched)
 +-----------+   - interaction graph components
      |          - per-cut chi bounds
      v
 +-----------+   cost model over available backends
 |  Planner  |   picks representation, or throws
 +-----------+   qubit::Error with alternatives
      |
      v
 +-----------+   executes gate stream, monitors,
 |  Runtime  |   samples measurement outcomes
 +-----------+
      |
      v
 backends: dense-cpu | dense-gpu | (planned: groups, blocks, mps)
```

### Circuit — deferred execution

`Circuit` is a recorder. Every gate call appends to a gate list; nothing
touches state memory until `run()`. This is required, not cosmetic: the
analyzer must see the complete circuit to bound entanglement and choose a
backend before the first amplitude is allocated.

Every operation lowers to one of three primitives:

| Primitive | Payload                              | Covers                          |
|-----------|--------------------------------------|---------------------------------|
| `U1`      | 2x2 unitary + control-qubit list     | all gates (H, Pauli, rotations, CNOT, CZ, Toffoli, custom unitaries) |
| `Measure` | target, classical bit id             | mid-circuit measurement          |
| `Reset`   | target                               | measure + force to zero, qubit reuse |

Named gates are sugar that builds the 2x2 matrix. Backends therefore
implement exactly one gate kernel plus measurement, which keeps the backend
contract small and every new backend cheap to add. Classical control
(`if_result`) tags gates with a condition bit; the runtime skips them when
the bit disagrees.

### Analyzer — static cost bounds

Runs in O(gates + qubits), allocates no state. Produces:

- **Sleeping qubits**: qubits no gate touches. A sleeping qubit remains
  exactly `|0>` and factors out of the state; it is elided from simulation
  entirely (see Runtime).
- **Interaction groups**: connected components of the graph whose edges are
  multi-qubit gates. Disjoint groups never entangle and could be simulated
  as independent state vectors (planned `groups` backend); memory is the
  sum, not the product, of component sizes.
- **Chi bounds per cut**: for every bipartition between adjacent qubit
  indices, a two-qubit gate crossing the cut can at most double the Schmidt
  rank across it. Hence `chi <= 2^crossings`, additionally capped by
  `2^min(|left|, |right|)`. This bounds MPS memory ahead of time from the
  gate list alone.

The bound is worst-case; the true Schmidt rank is often far lower (e.g. a
GHZ circuit has crossings on every cut yet chi = 2 exactly). Consequently
bounds are used to *admit* cheap plans, never to reject: refusal based on a
pessimistic bound would reject circuits that run fine.

### Planner — representation choice and failure policy

The user chooses a `Device` (Auto/CPU/GPU) and optionally grants
fidelity headroom; the planner owns everything else. The compression
budget is derived by inverting the fidelity bound
(`D = (1-sqrt(F))/(1+sqrt(F))`), so the user-facing contract holds by
construction. The ladder:

1. Compute live-qubit count L (declared minus sleeping).
2. GPU path (Auto with CUDA built, or forced): dense-gpu if `2^L * 8`
   fits in 90% of free VRAM; otherwise tiered blocks-gpu — exact via
   the ZERO tier on block-sparse states, budget-compressed when the
   user granted headroom.
3. CPU path: dense-cpu if it fits the host budget, else groups-cpu if
   the analyzer's component sum fits, else blocks-cpu (guarded so a
   truly dense huge state errors instead of thrashing RAM).
4. Otherwise: descriptive error listing what was needed, what was
   available, and which knobs would change the outcome.

Failure is a first-class outcome. The error message states: how many live
qubits, bytes needed versus available, and concrete alternatives (raise the
budget, build the GPU backend, or wait for structured backends — including
the analyzer's group count and chi bound so the user can see whether a
structured backend would help). Silent degradation is prohibited; lossy
backends must report achieved fidelity in `RunStats`.

Planned extension: optimistic execution for lossy backends. Run with the
chosen chi/compression budget while monitoring accumulated truncation
error; if the fidelity target becomes unreachable mid-circuit, abort with a
report of the gate index, the offending cut, the achievable fidelity, and
suggested remediations. Static bounds admit; runtime monitoring enforces.

### Runtime

- **Sleeping-qubit elision**: live qubits are remapped to a compact index
  space `0..L-1` before execution; gate targets/controls are translated per
  gate. All user-facing indices (`prob`, `amplitude`, `counts` keys,
  `expectation_z`) are translated back, with sleeping qubits reading as 0.
  Declared-but-unused qubits are free.
- **Shot handling**: circuits without measurement are deterministic — the
  state evolves once and `shots` outcomes are sampled from the final
  distribution (sorted uniforms, single cumulative pass, O(2^L + s log s)).
  Circuits with mid-circuit measurement branch on outcomes, so each shot
  re-executes the gate stream with fresh randomness.
- **RNG**: single seeded mt19937_64 for reproducible runs.

### Backend contract

```
init(nq)                  prepare |0...0> over nq compact qubits
apply(gate)               controlled 2x2 update
measure(q, u) -> bit      collapse using uniform u from the runtime RNG
reset(q, u)               measure, then X if outcome was 1
state() -> vector<cf>     full compact amplitude vector (host copy)
memory_bytes()            current state footprint
```

The RNG stays in the runtime (backends receive uniforms as arguments) so
that identical seeds produce identical outcomes across backends — this is
what makes cross-backend differential testing possible.

### dense-cpu

Straightforward loop over pair indices: for a gate on target t, indices
with bit t clear own the pair `(i, i | 1<<t)`; control masks filter
subspaces. Serves as the correctness reference for every other backend.

### dense-gpu (CUDA)

Same algorithm, one thread per index, state resident in VRAM for the whole
circuit. Host round-trips are limited to measurement scalars (8 bytes) and
the final `state()` copy. Kernels:

- `k_apply`: pair update with control-mask filtering. Warp divergence from
  the ownership test is acceptable; global-memory bandwidth dominates.
- `k_prob1`: P(bit = 1) via shared-memory block reduction, one atomicAdd
  per block, accumulated in double (float accumulation over 2^29 terms
  loses digits that measurement decisions need).
- `k_collapse`: zero the dead branch, rescale survivors.

In-place updates need no second buffer, so the ceiling on a 6 GB card is
29 qubits at complex64. Requires `-DQUBIT_CUDA`, compiled with nvcc for
`sm_86`.

## Memory model and roadmap

Measured/estimated capacity on a 6 GB GPU:

| Representation           | Cost                       | Capacity (6 GB)          | Status  |
|--------------------------|----------------------------|--------------------------|---------|
| dense complex64          | 2^L * 8 B                  | 29 qubits                | done    |
| dynamic groups (CPU)     | sum over factors 2^size * 8 | width-unbounded; largest cluster <= 40 | done |
| dense mixed FP16/FP32    | 2^L * 4 B                  | 30 qubits                | planned |
| tiered blocks            | per-block ZERO/compressed/full | 31-40+, circuit-dependent | done, CPU (blocks.h) and GPU (blocks_gpu.cu), all 3 tiers, measured |
| MPS                      | L * 2 * chi^2 * 8          | unbounded in L; chi-bound | planned |

### groups-cpu (implemented)

The state is a product of factors, each a dense vector over the qubits
entangled so far. Every qubit starts as its own 2-amplitude factor.

- **Merge**: a multi-qubit gate spanning factors tensors them into one
  (memory multiplies). Factor qubit order is concatenation order; gate
  kernels translate global qubit ids to local bit positions.
- **Classical-control short-circuit**: a control qubit whose singleton
  factor is a definite basis state never forces a merge — |0> makes the
  gate a no-op, |1> drops the control. Toffoli chains over classical
  ancillas stay cheap.
- **Split**: measurement projects inside the owning factor, then peels
  the measured qubit off into a fresh singleton; the factor halves.
  Splits happen only at measurement, where they are exact and O(2^k).
  Detecting incidental separability after a unitary would need a rank
  check per gate and is deliberately not attempted.
- **Queries without materialization**: amplitudes are products of
  per-factor amplitudes; Z-string expectations are products of
  per-factor expectations; sampling draws each factor independently and
  ORs the bits. `state()` refuses above 26 qubits.
- The analyzer's interaction-graph component sum is an exact upper bound
  on this backend's peak memory, so the planner can admit it statically.

An 80-qubit circuit of twenty 4-qubit GHZ clusters runs in 2.5 KB peak;
its dense equivalent would be 2^80 amplitudes.
- **Tiered blocks** (CPU in `include/qubit/blocks.h`, GPU in
  `src/blocks_gpu.cu`; both implemented, all three tiers; verified exact
  against dense with a zero budget, and against a rigorous fidelity
  bound with a nonzero one. GPU capacity results measured: GHZ-31 exact
  in 1 MB, echo-28/29 compressed): partition the dense
  vector into fixed-size blocks, each ZERO (no storage), COMPRESSED
  (block-scaled int16 pairs, 4 B/amplitude), or FULL (complex64).
  Blocks demote to ZERO when their norm vanishes — including via
  destructive interference mid-circuit — and demote to COMPRESSED only
  when the exactly-measured quantization error fits the remaining
  global budget. Gates promote the blocks they touch; a hot block stops
  being recompressed once the budget runs dry, with no heuristics.
  Gates on qubit k with `2^k < blocksize` stay intra-block; higher
  qubits pair blocks, including ZERO/FULL pairs that both become
  nonzero.

  **Error accounting is in L2 norm, not squared norm.** A quantization
  error delta_i injected at step i is carried through all later gates
  with its norm intact (unitaries preserve norms), so injections
  compose by the triangle inequality: `||delta|| <= D = sum ||delta_i||`.
  Squared-norm accounting understates the worst case by up to the
  number of injections (measured ~600x on random circuits before the
  fix). The reported bound is `F >= (1-D)^2 / (1+D)^2`, from
  `<psi|psi~> >= (1-D)/(1+D)` after renormalization. Measured on random
  circuits: true fidelity 0.99996 against a 0.998 guarantee at a 5e-4
  budget, with peak-memory savings up to 98% on compressible states.
- **MPS**: tensor chain with per-bond chi. SVD truncation gives optimal
  lossy compression with an exactly known error per cut; cuSOLVER provides
  the batched SVDs. Long-range gates pay in swaps or chi growth, so the
  planner should reorder qubits to shorten interaction distance before
  choosing this backend.

All planned backends slot in behind the existing `Backend` interface and
planner; none change the user-facing API.

## Non-goals

- Noise/decoherence modeling (density matrices square the memory cost;
  out of scope for v1).
- Transpilation to hardware gate sets or connectivity graphs.
- Distributed multi-GPU simulation.

## Validation

- Analytic invariants: Bell correlation `<Z0 Z1> = +1`; Grover 3-qubit
  target probability after 2 iterations = sin^2(5 asin(1/sqrt(8))) =
  0.9453; GHZ counts confined to all-zeros/all-ones.
- Cross-backend: identical seed must produce identical counts on dense-cpu
  and dense-gpu.
- Planned: differential testing against OpenQASM circuits executed by a
  reference simulator.
