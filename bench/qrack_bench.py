"""Qrack head-to-head on the SAME GPU as qubit (fair same-hardware baseline).

Qrack is the closest peer (approximate single-GPU sim). We time exact
Qrack gate application on the same five circuit families. Force the
NVIDIA device with QRACK_OCL_DEFAULT_DEVICE=0 (Qrack's default here is
the AMD iGPU).

Times are GATE APPLICATION ONLY, excluding state read-back -- this
favors Qrack, since the qubit/Aer numbers include read-back. Reported
honestly as such.

Usage: set QRACK_OCL_DEFAULT_DEVICE=0 ; py qrack_bench.py <maxq>
"""
import math, sys, time

QUBIT_FAMILIES = ["qft", "ghz", "qaoa4", "random", "pairs"]

def build(sim, name, n, Pauli):
    import random
    if name == "ghz":
        sim.h(0)
        for q in range(n - 1):
            sim.mcx([q], q + 1)
    elif name == "pairs":
        for q in range(0, n - 1, 2):
            sim.h(q); sim.mcx([q], q + 1)
    elif name == "qft":
        for q in range(n):
            sim.h(q)
        for i in range(n - 1, -1, -1):
            sim.h(i)
            for j in range(i - 1, -1, -1):
                th = math.pi / float(1 << (i - j))
                # controlled-phase: diag(1, e^{i th}) on target i, control j
                sim.mcmtrx([j], [1, 0, 0, complex(math.cos(th), math.sin(th))], i)
    elif name == "qaoa4":
        for q in range(n):
            sim.h(q)
        for r in range(4):
            for q in range(n - 1):
                sim.mcx([q], q + 1)
                sim.r(Pauli.PauliZ, 0.4 + 0.1 * r, q + 1)
                sim.mcx([q], q + 1)
            for q in range(n):
                sim.r(Pauli.PauliX, 0.7 - 0.05 * r, q)
    elif name == "random":
        g = random.Random(42)
        for _ in range(10):
            for k in range(n):
                s = g.randrange(3)
                if s == 0: sim.h(k)
                elif s == 1: sim.t(k)
                else: sim.r(Pauli.PauliY, g.uniform(0, 6.2832), k)
            for _ in range(n // 2):
                a, b = g.randrange(n), g.randrange(n)
                if a != b: sim.mcx([a], b)

def main():
    from pyqrack import QrackSimulator, Pauli
    maxq = int(sys.argv[1]) if len(sys.argv) > 1 else 28
    print("circuit,qubits,backend,time_ms,check")
    for n in range(20, maxq + 1, 4):
        for name in QUBIT_FAMILIES:
            try:
                sim = QrackSimulator(n)
                t0 = time.perf_counter()
                build(sim, name, n, Pauli)
                _ = sim.prob(0)  # light query, forces evaluation
                dt = (time.perf_counter() - t0) * 1000.0
                chk = sim.prob(0)
                print(f"{name},{n},qrack-gpu,{dt:.2f},{chk:.6f}", flush=True)
                del sim
            except Exception as e:
                print(f"{name},{n},FAIL,{type(e).__name__}: {str(e)[:80]}", flush=True)

if __name__ == "__main__":
    main()
