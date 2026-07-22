/*
 * Differential test: run random circuits on two backends with the same
 * seed and require identical counts and matching amplitudes. The RNG
 * lives in the runtime and uniforms are handed to backends, so any
 * divergence is a backend bug, not sampling noise.
 *
 * Circuits mix single-qubit gates, entangling gates, and mid-circuit
 * measurement to force merges and splits in the grouped backend.
 */
#include <qubit/qubit.h>
#include <qubit/blocks.h>
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

static int compare(const qubit::Result& r1, const qubit::Result& r2, int trial)
{
	if (r1.counts() != r2.counts()) {
		printf("FAIL trial %d: counts differ (%s vs %s)\n",
		       trial, r1.stats.backend.c_str(), r2.stats.backend.c_str());
		return 1;
	}
	return 0;
}

int main(int argc, char** argv)
{
	int trials = (argc > 1) ? std::atoi(argv[1]) : 50;
	std::mt19937_64 meta(12345);
	int failures = 0;

	for (int t = 0; t < trials; t++) {
		int n = 2 + int(meta() % 7);		/* 2..8 qubits */
		int depth = 10 + int(meta() % 40);
		auto c = random_circuit(meta, n, depth);

		qubit::RunOptions o1, o2, o3;
		o1.shots = o2.shots = o3.shots = 256;
		o1.seed = o2.seed = o3.seed = 999 + t;
		o1.backend = qubit::BackendSel::DenseCPU_;
		o2.backend = qubit::BackendSel::GroupsCPU_;
		/* tiny blocks (2^3) force heavy inter-block traffic */
		o3.custom_backend = [] { return std::make_shared<qubit::BlocksCPU>(3); };

		/*
		 * Counts are bit-exact only for circuits with measurement
		 * gates: there both backends consume the same uniform per
		 * Measure/Reset. Measurement-free counts go through each
		 * backend's own sampler (global CDF vs per-factor), which
		 * agrees in distribution but not per-draw; those circuits
		 * are covered by the amplitude check below instead.
		 */
		if (c.has_measurements()) {
			auto r1 = qubit::run(c, o1);
			auto r2 = qubit::run(c, o2);
			auto r3 = qubit::run(c, o3);
			failures += compare(r1, r2, t);
			failures += compare(r1, r3, t);
		}

		/* measurement-free amplitude check on a fresh circuit */
		auto c2 = [&] {
			qubit::Circuit cc(n);
			std::mt19937_64 g(777 + t);
			std::uniform_int_distribution<int> q(0, n - 1);
			std::uniform_real_distribution<float> ang(0.0f, 6.2832f);
			for (int d = 0; d < depth; d++) {
				int a = q(g), b = q(g);
				int k = int(g() % 5);
				if (k == 0) cc.h(a);
				else if (k == 1) cc.t(a);
				else if (k == 2) cc.ry(a, ang(g));
				else if (a != b) cc.cnot(a, b);
				else cc.h(a);
			}
			return cc;
		}();
		auto s1 = qubit::run(c2, o1);
		auto s2 = qubit::run(c2, o2);
		auto s3 = qubit::run(c2, o3);
		for (uint64_t i = 0; i < (uint64_t(1) << n); i++) {
			float d2 = std::abs(s1.amplitude(i) - s2.amplitude(i));
			float d3 = std::abs(s1.amplitude(i) - s3.amplitude(i));
			if (d2 > 1e-4f || d3 > 1e-4f) {
				printf("FAIL trial %d: amplitude %llu differs (groups %g, blocks %g)\n",
				       t, (unsigned long long)i, d2, d3);
				failures++;
				break;
			}
		}
	}

	if (failures) {
		printf("%d/%d trials FAILED\n", failures, trials);
		return 1;
	}
	printf("OK: %d trials, dense-cpu == groups-cpu == blocks-cpu (counts and amplitudes)\n", trials);
	return 0;
}
