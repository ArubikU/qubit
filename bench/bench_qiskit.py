"""Qiskit Aer baseline: same circuit families as benchsuite.cpp.

Output CSV: circuit,qubits,backend,time_ms,peak_mem_bytes,check
Memory is the theoretical statevector size (Aer does not report peak).

Usage: py bench_qiskit.py [max_qubits] [method]
  method: statevector | matrix_product_state | automatic (default statevector)
"""
import math
import sys
import time

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def qft(n):
    c = QuantumCircuit(n)
    for q in range(n):
        c.h(q)
    for i in range(n - 1, -1, -1):
        c.h(i)
        for j in range(i - 1, -1, -1):
            c.cp(math.pi / float(1 << (i - j)), j, i)
    return c


def ghz(n):
    c = QuantumCircuit(n)
    c.h(0)
    for q in range(n - 1):
        c.cx(q, q + 1)
    return c


def qaoa(n, p=4):
    c = QuantumCircuit(n)
    for q in range(n):
        c.h(q)
    for r in range(p):
        for q in range(n - 1):
            c.cx(q, q + 1)
            c.rz(0.4 + 0.1 * r, q + 1)
            c.cx(q, q + 1)
        for q in range(n):
            c.rx(0.7 - 0.05 * r, q)
    return c


def random_dense(n, depth=10, seed=42):
    import random
    g = random.Random(seed)
    c = QuantumCircuit(n)
    for _ in range(depth):
        for k in range(n):
            sel = g.randrange(3)
            if sel == 0:
                c.h(k)
            elif sel == 1:
                c.t(k)
            else:
                c.ry(g.uniform(0, 6.2832), k)
        for _ in range(n // 2):
            a, b = g.randrange(n), g.randrange(n)
            if a != b:
                c.cx(a, b)
    return c


def pairs(n):
    c = QuantumCircuit(n)
    for q in range(0, n - 1, 2):
        c.h(q)
        c.cx(q, q + 1)
    return c


def main():
    maxq = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    method = sys.argv[2] if len(sys.argv) > 2 else "statevector"
    device = sys.argv[3] if len(sys.argv) > 3 else "CPU"
    sim = AerSimulator(method=method, device=device)
    tag = f"aer-{method}-{device.lower()}"

    print("circuit,qubits,backend,time_ms,peak_mem_bytes,check")
    for n in range(8, maxq + 1, 4):
        for name, build in [("qft", qft), ("ghz", ghz), ("qaoa4", qaoa),
                            ("random", random_dense), ("pairs", pairs)]:
            c = build(n)
            c.save_statevector()
            try:
                tc = transpile(c, sim)
                t0 = time.perf_counter()
                res = sim.run(tc, shots=1).result()
                dt = (time.perf_counter() - t0) * 1000.0
                sv = res.get_statevector()
                if name == "ghz":
                    check = abs(sv[0]) ** 2
                else:
                    # <Z0 Z1>
                    import numpy as np
                    probs = np.abs(np.asarray(sv)) ** 2
                    idx = np.arange(len(probs))
                    par = ((idx & 1) ^ ((idx >> 1) & 1)).astype(float)
                    check = float(((1 - 2 * par) * probs).sum())
                mem = (2 ** n) * 16  # Aer statevector is complex128
                print(f"{name},{n},{tag},{dt:.2f},{mem},{check:.6f}",
                      flush=True)
            except Exception:
                print(f"{name},{n},SKIP,,,", flush=True)


if __name__ == "__main__":
    main()
