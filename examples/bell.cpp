/*
 * Bell pair on a 10-qubit register. Exercises sleeping-qubit elision:
 * 8 of the 10 qubits are never touched, so the simulated state is 2
 * qubits wide regardless of the declared size.
 */
#include <qubit/qubit.h>
#include <cstdio>

int main()
{
	qubit::Circuit c(10);
	c.h(0);
	c.cnot(0, 1);

	printf("circuit:\n");
	c.dump();

	qubit::analyze(c).print();

	qubit::RunOptions opt;
	opt.shots = 4096;
	auto r = qubit::run(c, opt);

	printf("\ncounts (%d shots):\n", opt.shots);
	r.print_counts();
	printf("\nexact prob |0000000000>: %.4f\n", r.prob(0));
	printf("exact prob |0000000011>: %.4f\n", r.prob(3));
	printf("correlation <Z0 Z1>: %+.4f  (ideal Bell = +1)\n",
	       r.expectation_z({0, 1}));
	printf("\nbackend: %s | live qubits: %d/%d | memory: %.2f MB | %.1f ms\n",
	       r.stats.backend.c_str(), r.stats.qubits_live, r.stats.qubits_total,
	       r.stats.memory_peak_bytes / 1048576.0, r.stats.time_ms);
	return 0;
}
