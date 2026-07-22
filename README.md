# qubit

Quantum circuit simulator for consumer GPUs. Single C++ header; the
engine decides how to represent the state (dense, entanglement groups,
or compressed tiered blocks) so a 6 GB card runs circuits a plain
statevector cannot hold — up to and past 30 qubits, with an exactly
guaranteed fidelity bound when compression is involved.

## Install

CPU only: copy `include/qubit/` into your include path. Done.

GPU (NVIDIA, CUDA 12.x): additionally compile one file with your app:

```
nvcc -O2 -std=c++17 -arch=sm_86 -DQUBIT_CUDA -I include \
     your_app.cpp src/qubit_gpu.cu -o your_app
```

## Use

```cpp
#include <qubit/qubit.h>

int main() {
    qubit::Circuit c(30);          // declare qubits; unused ones cost nothing
    c.h(0);
    for (int q = 0; q + 1 < 30; q++)
        c.cnot(q, q + 1);          // gates: h x y z s t rx ry rz phase
                                   //        cnot cz swap toffoli unitary
    auto r = qubit::run(c);        // engine picks device and representation
    r.print_counts();
    printf("P(all zeros) = %f\n", r.prob(0));
}
```

That is the whole API. Options when you need them:

```cpp
qubit::RunOptions o;
o.device   = qubit::Device::GPU;   // Auto (default) | CPU | GPU
o.fidelity = 0.999;                // < 1 lets the engine compress, with a
                                   // guaranteed lower bound on state fidelity
o.shots    = 4096;
o.seed     = 42;
auto r = qubit::run(c, o);
```

Mid-circuit measurement, qubit reuse, and classical control:

```cpp
int b = c.measure(0);              // collapse qubit 0 into classical bit b
c.if_result(b, [&]{ c.x(1); });    // gates conditioned on the outcome
c.reset(0);                        // measure + force |0>: recycle the qubit
```

Inspect a circuit's cost before running it:

```cpp
qubit::analyze(c).print();         // sleeping qubits, entanglement groups,
                                   // per-cut chi bounds, memory per backend
```

Results: `r.counts()`, `r.prob(idx)`, `r.amplitude(idx)`,
`r.expectation_z({q0, q1, ...})`, `r.stats` (backend used, peak memory,
time).

Everything else — which qubits sleep, when groups merge or split, which
blocks compress, error budgeting — is the engine's business, not yours.

## Layout

```
include/qubit/qubit.h   the library (header-only CPU engine + API)
src/qubit_gpu.cu        the GPU engine (compile only for CUDA builds)
examples/               bell, grover, clusters, benchmark
tests/                  differential + lossy validation
bench/                  benchmark suite vs Qiskit Aer
DESIGN.md               architecture
BENCHMARKS.md           measured results
LICENSE                 MIT
```

## Numbers (RTX 3060 Laptop, 6 GB)

- 4-18x faster than Qiskit Aer (CPU) at 20-24 qubits.
- 29-qubit dense state (4 GB) in seconds; 30-qubit circuits beyond the
  dense VRAM ceiling complete via compressed tiers with measured
  fidelity 0.9998.
- 80-qubit clustered circuits in 2.5 KB via entanglement groups.

See BENCHMARKS.md for methodology and full tables.

## Reproducing the paper's results

All results come from deterministic, seeded runs on a single machine
(RTX 3060 Laptop 6 GB, CUDA 12.6, MSVC 2019, Windows 11).

```
# CPU-only tests and examples (no CUDA needed)
powershell -File build.ps1
bin\difftest_cpu.exe 300        # 3 CPU backends agree, seeded
bin\lossytest.exe 30            # lossy tier vs fidelity bound

# GPU build (needs nvcc)
powershell -File build.ps1 -Gpu
bin\difftest_gpu.exe 100        # GPU == CPU, seeded
bin\blocks_gpu_test.exe 25      # tiered GPU: exact / lossy / capacity

# benchmark tables (median of 5)
bin\benchsuite_gpu.exe 28 gpu 5      # qubit GPU  -> Table 2
py bench\bench_qiskit.py 28 statevector   # Aer CPU baseline
bin\curve.exe 28                     # budget sweep -> Figure 1 data
py bench\plot_curve.py bench\results\curve28.csv   # the figure
```

Seeds are fixed in each harness (see the `seed` field in `RunOptions`
and the `meta` generators in `tests/`). The CSVs behind the reported
tables are in `bench/results/`.

Scope and limitations are stated honestly: evaluation is on consumer
and data-center single GPUs (no multi-GPU), the worst-case fidelity
bound is conservative (tight at low circuit depth), and the tiered
backend is a capacity tool, slower than dense when the dense state
fits. See DESIGN.md and BENCHMARKS.md.
