"""Run one circuit family at a fixed qubit count on Aer-GPU.
Usage: aer_gpu_one.py <qubits> <circuit>   circuit in {qft,ghz,qaoa4,random,pairs}
Isolates a single run so a GPU OOM does not take down a whole sweep.
"""
import sys, time
import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator
import bench_qiskit as B

n = int(sys.argv[1])
which = sys.argv[2] if len(sys.argv) > 2 else "qft"
prec = sys.argv[3] if len(sys.argv) > 3 else "double"
builders = {"qft": B.qft, "ghz": B.ghz, "qaoa4": B.qaoa,
            "random": B.random_dense, "pairs": B.pairs}

sim = AerSimulator(method="statevector", device="GPU", precision=prec)
c = builders[which](n)
c.save_statevector()
try:
    tc = transpile(c, sim)
    t0 = time.perf_counter()
    res = sim.run(tc, shots=1).result()
    dt = (time.perf_counter() - t0) * 1000.0
    sv = res.get_statevector()
    p0 = abs(sv[0]) ** 2
    b = 8 if prec == "single" else 16
    # result.time_taken is Aer's internal simulation wall time, excluding
    # Python-side overhead and the get_statevector read-back -> kernel-fair
    try:
        internal_ms = res.results[0].time_taken * 1000.0
    except Exception:
        internal_ms = float("nan")
    print(f"{which},{n},aer-sv-gpu-{prec},{dt:.2f},{(2**n)*b},{p0:.6f},{internal_ms:.2f}",
          flush=True)
except Exception as e:
    print(f"{which},{n},FAIL,{type(e).__name__}: {str(e)[:120]}", flush=True)
