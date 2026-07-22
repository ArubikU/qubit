/*
 * COMPRESSED-tier-at-scale benchmark: Loschmidt echo.
 *
 * Circuit: H^n . RZ(th_q) . RZ(-th_q) . H^n  == identity, so the ideal
 * final state is |0...0> and P(0) = 1 exactly. The intermediate state
 * is uniform superposition — every block nonzero, worst case for the
 * ZERO tier, real work for the COMPRESSED tier. Any quantization loss
 * shows up directly as P(0) < 1, measurable from ONE amplitude at any
 * size — no exponential reference state needed.
 *
 * 28 qubits: dense-gpu still fits (2 GB) -> direct time comparison.
 * 30 qubits: dense needs 8 GB, does not fit the card. Blocks-gpu holds
 * the uniform state in int16 (4 GB budget-compressed) and completes.
 *
 * Build: nvcc -O2 -std=c++17 -arch=sm_86 -DQUBIT_CUDA -I include \
 *        bench/blocks30.cpp src/blocks_gpu.cu src/backend_gpu.cu -o blocks30
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
	int n = (argc > 1) ? std::atoi(argv[1]) : 28;
	const char* mode = (argc > 2) ? argv[2] : "blocks";
	double budget = (argc > 3) ? std::atof(argv[3]) : 0.5;

	auto c = echo(n);
	qubit::RunOptions o;
	o.shots = 1;
	o.fuse = false;		/* keep the gate stream honest for blocks */

	std::shared_ptr<qubit::Backend> blocks;
	if (std::string(mode) == "gpu") {
		o.backend = qubit::BackendSel::DenseGPU_;
	} else {
		blocks = qubit::make_blocks_gpu(16, budget);
		o.custom_backend = [&] { return blocks; };
	}

	try {
		auto r = qubit::run(c, o);
		double p0 = r.prob(0);
		printf("echo n=%d backend=%s gates=%zu\n", n,
		       r.stats.backend.c_str(), c.gates().size());
		printf("  time        : %.0f ms\n", r.stats.time_ms);
		printf("  peak state  : %.0f MB (dense f32: %.0f MB)\n",
		       r.stats.memory_peak_bytes / 1048576.0,
		       std::ldexp(8.0, n) / 1048576.0);
		printf("  P(|0...0>)  : %.6f   (ideal 1.0 -> echo fidelity)\n", p0);
		if (blocks) {
			printf("  err used D  : %.6f of budget %g\n",
			       qubit::blocks_gpu_error(blocks.get()), budget);
			printf("  formal bound: F >= %.6f (worst case)\n",
			       qubit::blocks_gpu_fidelity_bound(blocks.get()));
			printf("  estimate    : F ~= %.6f (random-walk model)\n",
			       qubit::blocks_gpu_fidelity_estimate(blocks.get()));
		}
	} catch (const qubit::Error& e) {
		printf("echo n=%d %s FAILED:\n%s\n", n, mode, e.what());
		return 1;
	}
	return 0;
}
