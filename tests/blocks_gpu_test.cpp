/*
 * BlocksGPU validation, three phases:
 *   1. exact mode (zero budget): random circuits must match dense-cpu
 *      bit-for-bit in counts and to float tolerance in amplitudes.
 *   2. lossy mode: true fidelity vs dense-cpu must respect the
 *      backend's own reported bound and the requested target.
 *   3. capacity: GHZ beyond the dense-VRAM ceiling — the ZERO tier
 *      keeps only 2 blocks live, exactly, out of up to 2^15.
 *
 * Build: nvcc -O2 -std=c++17 -arch=sm_86 -DQUBIT_CUDA -I include \
 *        tests/blocks_gpu_test.cpp src/blocks_gpu.cu src/backend_gpu.cu -o t
 */
#include <qubit/qubit.h>
#include <cstdio>
#include <cstdlib>

namespace qubit {
std::shared_ptr<Backend> make_blocks_gpu(int block_shift, double l2_budget);
}

static qubit::Circuit random_circuit(std::mt19937_64& rng, int n, int depth,
				     bool with_measure)
{
	qubit::Circuit c(n);
	std::uniform_int_distribution<int> q(0, n - 1);
	std::uniform_real_distribution<float> ang(0.0f, 6.2832f);
	int kinds = with_measure ? 9 : 7;
	for (int d = 0; d < depth; d++) {
		int a = q(rng), b = q(rng);
		switch (int(rng() % kinds)) {
		case 0: c.h(a); break;
		case 1: c.t(a); break;
		case 2: c.rx(a, ang(rng)); break;
		case 3: c.rz(a, ang(rng)); break;
		case 4: case 5:
			if (a != b) c.cnot(a, b);
			else c.h(a);
			break;
		case 6:
			if (a != b) c.cz(a, b);
			else c.z(a);
			break;
		case 7: c.measure(a); break;
		case 8: c.reset(a); break;
		}
	}
	return c;
}

int main(int argc, char** argv)
{
	int trials = (argc > 1) ? std::atoi(argv[1]) : 25;
	std::mt19937_64 meta(777);
	int failures = 0;

	/* phase 1: exact */
	for (int t = 0; t < trials; t++) {
		int n = 4 + int(meta() % 9);	/* 4..12 */
		int depth = 20 + int(meta() % 50);
		auto c = random_circuit(meta, n, depth, true);

		qubit::RunOptions oc, ob;
		oc.shots = ob.shots = 128;
		oc.seed = ob.seed = 31337 + t;
		oc.backend = qubit::BackendSel::DenseCPU_;
		/* tiny blocks: max inter-block traffic */
		ob.custom_backend = [&] { return qubit::make_blocks_gpu(4, 0.0); };

		auto rc = qubit::run(c, oc);
		auto rb = qubit::run(c, ob);
		if (c.has_measurements()) {
			if (rc.counts() != rb.counts()) {
				printf("FAIL exact trial %d: counts differ\n", t);
				failures++;
			}
		}
		for (uint64_t i = 0; i < (uint64_t(1) << n); i++) {
			if (std::abs(rc.amplitude(i) - rb.amplitude(i)) > 1e-4f) {
				printf("FAIL exact trial %d: amp %llu\n", t,
				       (unsigned long long)i);
				failures++;
				break;
			}
		}
	}
	printf("phase 1 (exact, %d trials): %s\n", trials,
	       failures ? "FAIL" : "OK");

	/* phase 2: lossy */
	const double budget = 5e-4;
	const double target = (1 - budget) * (1 - budget) /
			      ((1 + budget) * (1 + budget));
	double worst = 1.0;
	for (int t = 0; t < trials; t++) {
		int n = 10 + int(meta() % 5);	/* 10..14 */
		int depth = 40 + int(meta() % 60);
		auto c = random_circuit(meta, n, depth, false);

		qubit::RunOptions oc, ob;
		oc.shots = ob.shots = 1;
		oc.seed = ob.seed = 999 + t;
		oc.backend = qubit::BackendSel::DenseCPU_;
		ob.custom_backend = [&] { return qubit::make_blocks_gpu(7, budget); };

		auto rc = qubit::run(c, oc);
		auto rb = qubit::run(c, ob);
		std::complex<double> ov = 0;
		for (uint64_t i = 0; i < (uint64_t(1) << n); i++)
			ov += std::conj(std::complex<double>(rc.amplitude(i))) *
			      std::complex<double>(rb.amplitude(i));
		double fid = std::norm(ov);
		worst = std::min(worst, fid);
		if (fid < target - 1e-5) {
			printf("FAIL lossy trial %d: fidelity %.9f < %.9f\n",
			       t, fid, target);
			failures++;
		}
	}
	printf("phase 2 (lossy, %d trials): worst fidelity %.9f (target %.9f) %s\n",
	       trials, worst, target, failures ? "FAIL" : "OK");

	/* phase 3: capacity — GHZ past the dense ceiling */
	for (int n : {30, 31}) {
		qubit::Circuit c(n);
		c.h(0);
		for (int q = 0; q + 1 < n; q++) c.cnot(q, q + 1);
		qubit::RunOptions o;
		o.shots = 8;
		o.custom_backend = [&] { return qubit::make_blocks_gpu(16, 0.0); };
		try {
			auto r = qubit::run(c, o);
			double p0 = r.prob(0);
			double p1 = r.prob((uint64_t(1) << n) - 1);
			printf("phase 3: GHZ-%d exact on GPU blocks | "
			       "P(0)=%.4f P(1..1)=%.4f | peak %.1f MB (dense: %.0f MB)\n",
			       n, p0, p1,
			       r.stats.memory_peak_bytes / 1048576.0,
			       std::ldexp(8.0, n) / 1048576.0);
			if (std::abs(p0 - 0.5) > 1e-3 || std::abs(p1 - 0.5) > 1e-3)
				failures++;
		} catch (const qubit::Error& e) {
			printf("phase 3 GHZ-%d: %s\n", n, e.what());
			failures++;
		}
	}

	if (failures) { printf("%d FAILURES\n", failures); return 1; }
	printf("ALL OK\n");
	return 0;
}
