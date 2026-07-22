/*
 * Memory/throughput stress: GHZ state of n qubits.
 * H on qubit 0 plus a CNOT chain gives (|0...0> + |1...1>)/sqrt(2):
 * maximal n-partite entanglement, yet only two nonzero amplitudes.
 *
 * Usage: benchmark [qubits]
 */
#include <qubit/qubit.h>
#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv)
{
	int n = (argc > 1) ? std::atoi(argv[1]) : 24;

	qubit::Circuit c(n);
	c.h(0);
	for (int q = 0; q + 1 < n; q++)
		c.cnot(q, q + 1);

	qubit::analyze(c).print();

	qubit::RunOptions opt;
	opt.shots = 16;
	try {
		auto r = qubit::run(c, opt);
		printf("\n%d-qubit GHZ:\n", n);
		r.print_counts();
		printf("\nbackend: %s | memory: %.1f MB | %.1f ms\n",
		       r.stats.backend.c_str(),
		       r.stats.memory_peak_bytes / 1048576.0, r.stats.time_ms);
	} catch (const qubit::Error& e) {
		printf("\n%s\n", e.what());
		return 1;
	}
	return 0;
}
