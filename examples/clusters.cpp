/*
 * 80 qubits as 20 independent 4-qubit GHZ clusters.
 *
 * Dense simulation would need 2^80 amplitudes (~9.7e12 TB). The grouped
 * backend never merges beyond 4 qubits, so the whole state is
 * 20 * 2^4 amplitudes = 2.5 KB. Entanglement is real within clusters
 * (<ZZ> = +1) and absent across them (<ZZ> = 0).
 *
 * Also exercises measurement-induced splitting: measuring one qubit of
 * a GHZ cluster collapses and disentangles it; the factor shrinks.
 */
#include <qubit/qubit.h>
#include <cstdio>

int main()
{
	const int clusters = 20, width = 4, n = clusters * width;

	qubit::Circuit c(n);
	for (int k = 0; k < clusters; k++) {
		int base = k * width;
		c.h(base);
		for (int q = 0; q < width - 1; q++)
			c.cnot(base + q, base + q + 1);
	}

	qubit::analyze(c).print();

	qubit::RunOptions opt;
	opt.shots = 8;
	auto r = qubit::run(c, opt);

	printf("\n%d qubits, %d shots:\n", n, opt.shots);
	r.print_counts();

	printf("\nintra-cluster <Z0 Z3>:   %+.4f  (same GHZ, ideal +1)\n",
	       r.expectation_z({0, 3}));
	printf("cross-cluster <Z0 Z4>:   %+.4f  (independent, ideal 0)\n",
	       r.expectation_z({0, 4}));
	printf("\nbackend: %s | peak memory: %.3f KB | %.1f ms\n",
	       r.stats.backend.c_str(),
	       r.stats.memory_peak_bytes / 1024.0, r.stats.time_ms);
	printf("dense equivalent: 2^%d amplitudes = %.1e TB\n",
	       n, std::ldexp(8.0, n) / 1e12);
	return 0;
}
