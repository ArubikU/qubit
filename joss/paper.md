---
title: 'qubit: tiered-compression quantum circuit simulation on consumer GPUs'
tags:
  - quantum computing
  - quantum circuit simulation
  - GPU
  - CUDA
  - high-performance computing
authors:
  - name: Piero Jose Alarcon Dueñas
    orcid: 0009-0008-7075-5501
    affiliation: 1
affiliations:
  - name: Independent Researcher, Lima, Peru
    index: 1
date: 21 July 2026
bibliography: paper.bib
---

# Summary

`qubit` is a state-vector quantum circuit simulator that targets the
single consumer GPU most researchers and students actually own, rather
than the data-center accelerators assumed by mainstream tools. It
represents the quantum state as a set of fixed-size blocks, each stored
in one of three tiers: `ZERO` (all-zero, no storage), `COMPRESSED`
(block-scaled `int16`, 4 bytes per amplitude), or `FULL` (`complex64`,
8 bytes per amplitude). The entire structure stays resident in GPU
memory; promotion and demotion between tiers are CUDA kernels, so no
compressed state is streamed across the PCIe bus. Compression is
governed by an *a priori* error budget: because unitary evolution
preserves norms, per-event quantization errors compose linearly in
$L_2$ norm, yielding the fidelity bound $F \ge (1-D)^2/(1+D)^2$ for
total injected error $D$. Inverting this bound turns a user-specified
fidelity target into a compression budget the runtime never exceeds.

The user writes an ordinary C++ program: declare qubits, add gates,
call `run()`. An automatic planner analyzes the circuit statically and
routes it to the cheapest applicable representation — dense,
entanglement-factorized, or tiered-block — with no manual configuration.
The library is header-only for CPU use and compiles with a single
additional CUDA translation unit for GPU use, natively on both Linux
and Windows.

# Statement of need

Exact state-vector simulation stores $2^n$ complex amplitudes for $n$
qubits, so the reachable qubit count is fixed by memory: a 6 GB GPU
caps out at 29 qubits in `complex64`. Established simulators such as
Qiskit Aer [@qiskit], cuQuantum [@cuquantum], QuEST [@quest], and
Qulacs [@qulacs] are highly optimized but store the state densely, and
the tools that push past the memory ceiling either target
supercomputers with disk/host-memory compression [@wu2019] or
data-center GPUs [@bmqsim]. Structure-aware simulators (matrix product
states, the stabilizer formalism, and Qrack [@qrack]) shrink only
low-entanglement states and gain nothing on genuinely dense ones.

The consumer segment — a 6–12 GB laptop or desktop GPU, frequently on
Windows — is largely unserved: the tools that run there do not exceed
the dense ceiling, and the tools that exceed it assume hardware few
students or independent researchers can access. `qubit` fills this gap.
On an RTX 3060 Laptop GPU (6 GB) it runs a 31-qubit GHZ circuit exactly
in 1 MB via the `ZERO` tier, and holds a 29-qubit fully-dense state in
half its footprint at measured fidelity 0.9998 via the `COMPRESSED`
tier — the regime where no factorization or tableau helps and dense
amplitude compression is the only route past the ceiling. It is
useful for teaching, algorithm prototyping, and research on limited
hardware, and its automatic planner makes those gains available without
the user choosing a representation.

# Key features

- Three-tier (`ZERO`/`COMPRESSED`/`FULL`) blocked state vector, fully
  VRAM-resident, with batched per-gate scheduling.
- A priori fidelity contract: the user sets a target fidelity; the
  enforced compression budget guarantees it by construction.
- Automatic planner over dense, entanglement-factorized, and
  tiered-block backends from an $O(\text{gates}+\text{qubits})$ static
  analysis.
- CPU (header-only) and CUDA backends; native Windows and Linux builds.
- Differential and lossy test suites, and a benchmark suite comparing
  against Qiskit Aer and Qrack.

# Acknowledgements

We thank the maintainers of the open-source quantum simulation
ecosystem, whose tools served as baselines and references.

# References
