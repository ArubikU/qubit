/*
 * GPU vs CPU differential test. Same seed drives both backends, so
 * measurement outcomes must match bit-exactly and amplitudes to float
 * tolerance. Any divergence is a kernel bug.
 *
 * Build: nvcc -O2 -std=c++17 -arch=sm_86 -DQUBIT_CUDA -I include \
 *        tests/difftest_gpu.cpp src/backend_gpu.cu -o difftest_gpu
 */
#include <qubit/qubit.h>
#include <cstdio>
#include <cstdlib>

static qubit::Circuit random_circuit(std::mt19937_64& rng, int n, int depth)
{
	qubit::Circuit c(n);
	std::uniform_int_distribution<int> q(0, n - 1), kind(0, 9);
	std::uniform_real_distribution<float> ang(0.0f, 6.2832f);
	for (int d = 0; d < depth; d++) {
		int a = q(rng), b = q(rng);
		switch (kind(rng)) {
		case 0: c.h(a); break;
		case 1: c.x(a); break;
		case 2: c.t(a); break;
		case 3: c.rx(a, ang(rng)); break;
		case 4: c.rz(a, ang(rng)); break;
		case 5: case 6:
			if (a != b) c.cnot(a, b);
			else c.h(a);
			break;
		case 7:
			if (a != b) c.cz(a, b);
			else c.z(a);
			break;
		case 8: c.measure(a); break;
		case 9: c.reset(a); break;
		}
	}
	return c;
}

int main(int argc, char** argv)
{
	int trials = (argc > 1) ? std::atoi(argv[1]) : 50;
	std::mt19937_64 meta(20260721);
	int failures = 0;

	for (int t = 0; t < trials; t++) {
		int n = 3 + int(meta() % 10);		/* 3..12 qubits */
		int depth = 20 + int(meta() % 60);
		auto c = random_circuit(meta, n, depth);

		qubit::RunOptions oc, og;
		oc.shots = og.shots = 128;
		oc.seed = og.seed = 555 + t;
		oc.backend = qubit::BackendSel::DenseCPU_;
		og.backend = qubit::BackendSel::DenseGPU_;

		if (c.has_measurements()) {
			auto rc = qubit::run(c, oc);
			auto rg = qubit::run(c, og);
			if (rc.counts() != rg.counts()) {
				printf("FAIL trial %d: counts differ cpu vs gpu\n", t);
				failures++;
			}
		} else {
			auto rc = qubit::run(c, oc);
			auto rg = qubit::run(c, og);
			for (uint64_t i = 0; i < (uint64_t(1) << n); i++) {
				float d = std::abs(rc.amplitude(i) - rg.amplitude(i));
				if (d > 1e-4f) {
					printf("FAIL trial %d: amp %llu differs %g\n",
					       t, (unsigned long long)i, d);
					failures++;
					break;
				}
			}
		}
	}

	if (failures) {
		printf("%d/%d trials FAILED\n", failures, trials);
		return 1;
	}
	printf("OK: %d trials, dense-cpu == dense-gpu (counts and amplitudes)\n", trials);
	return 0;
}
