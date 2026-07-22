/*
 * The paper's central figure: fidelity vs memory vs time as a function
 * of the compression budget, on the echo circuit (ideal P(0) = 1, so
 * the measured value IS the end-to-end fidelity at any size).
 *
 * CSV: qubits,budget,time_ms,peak_mb,echo_fidelity,err_D,bound,estimate
 *
 * Usage: curve [qubits] — sweeps budgets from exact to unconstrained.
 * Build: nvcc -O2 -std=c++17 -arch=sm_86 -DQUBIT_CUDA -I include \
 *        bench/curve.cpp src/qubit_gpu.cu -o curve
 */
#include <qubit/qubit.h>
#include <cstdio>
#include <cstdlib>

namespace qubit {
std::shared_ptr<Backend> make_blocks_gpu(int block_shift, double l2_budget);
double blocks_gpu_error(const Backend* be);
double blocks_gpu_fidelity_bound(const Backend* be);
double blocks_gpu_fidelity_estimate(const Backend* be);
}

static qubit::Circuit echo(int n)
{
	qubit::Circuit c(n);
	for (int q = 0; q < n; q++) c.h(q);
	for (int q = 0; q < n; q++) c.rz(q, 0.31 + 0.07 * q);
	for (int q = 0; q < n; q++) c.rz(q, -(0.31 + 0.07 * q));
	for (int q = 0; q < n; q++) c.h(q);
	return c;
}

int main(int argc, char** argv)
{
	int n = (argc > 1) ? std::atoi(argv[1]) : 26;
	const double budgets[] = { 0.0, 1e-3, 1e-2, 0.1, 0.5, 2.0, 100.0 };

	auto c = echo(n);
	printf("qubits,budget,time_ms,peak_mb,echo_fidelity,err_D,bound,estimate\n");
	for (double budget : budgets) {
		auto blocks = qubit::make_blocks_gpu(16, budget);
		qubit::RunOptions o;
		o.shots = 1;
		o.fuse = false;
		o.custom_backend = [&] { return blocks; };
		try {
			auto r = qubit::run(c, o);
			printf("%d,%g,%.0f,%.1f,%.6f,%.6f,%.6f,%.6f\n",
			       n, budget, r.stats.time_ms,
			       r.stats.memory_peak_bytes / 1048576.0,
			       r.prob(0),
			       qubit::blocks_gpu_error(blocks.get()),
			       qubit::blocks_gpu_fidelity_bound(blocks.get()),
			       qubit::blocks_gpu_fidelity_estimate(blocks.get()));
		} catch (const qubit::Error& e) {
			printf("%d,%g,OOM,,,,,\n", n, budget);
		}
		fflush(stdout);
	}
	return 0;
}
