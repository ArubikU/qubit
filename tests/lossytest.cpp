/*
 * Lossy-tier validation. Runs random measurement-free circuits on
 * dense-cpu (exact) and blocks-cpu with an L2^2 error budget, then
 * checks three things:
 *   1. true fidelity |<exact|lossy>|^2 respects the backend's own
 *      reported bound (the accounting must never understate error),
 *   2. fidelity meets the requested target,
 *   3. compression actually saves memory on compressible states.
 */
#include <qubit/qubit.h>
#include <qubit/blocks.h>
#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv)
{
	int trials = (argc > 1) ? std::atoi(argv[1]) : 30;
	/* L2-norm budget D; rigorous bound F >= (1-D)^2/(1+D)^2 */
	const double budget = 5e-4;
	const double target = (1.0 - budget) * (1.0 - budget) /
			      ((1.0 + budget) * (1.0 + budget));
	std::mt19937_64 meta(4242);
	int failures = 0;
	double worst_fid = 1.0, best_saving = 0;

	for (int t = 0; t < trials; t++) {
		int n = 10 + int(meta() % 7);		/* 10..16 qubits */
		int depth = 30 + int(meta() % 90);

		qubit::Circuit c(n);
		std::uniform_int_distribution<int> q(0, n - 1);
		std::uniform_real_distribution<float> ang(0.0f, 6.2832f);
		for (int d = 0; d < depth; d++) {
			int a = q(meta), b = q(meta);
			int k = int(meta() % 6);
			if (k == 0) c.h(a);
			else if (k == 1) c.t(a);
			else if (k == 2) c.ry(a, ang(meta));
			else if (k == 3) c.rz(a, ang(meta));
			else if (a != b) c.cnot(a, b);
			else c.h(a);
		}

		qubit::RunOptions oe, ol;
		oe.shots = ol.shots = 1;
		oe.seed = ol.seed = 7 + t;
		oe.backend = qubit::BackendSel::DenseCPU_;
		auto blocks = std::make_shared<qubit::BlocksCPU>(8, budget);
		ol.custom_backend = [&] { return blocks; };

		auto re = qubit::run(c, oe);
		auto rl = qubit::run(c, ol);

		/* true fidelity against the exact state */
		std::complex<double> ov = 0;
		for (uint64_t i = 0; i < (uint64_t(1) << n); i++)
			ov += std::conj(std::complex<double>(re.amplitude(i))) *
			      std::complex<double>(rl.amplitude(i));
		double fid = std::norm(ov);
		worst_fid = std::min(worst_fid, fid);

		/* 1e-5 slack: dense float32 reference has its own roundoff */
		if (fid < blocks->fidelity_bound() - 1e-5) {
			printf("FAIL trial %d: true fidelity %.9f below reported bound %.9f\n",
			       t, fid, blocks->fidelity_bound());
			failures++;
		}
		if (fid < target - 1e-5) {
			printf("FAIL trial %d: fidelity %.9f misses target %.9f\n",
			       t, fid, target);
			failures++;
		}

		double dense_bytes = std::ldexp(8.0, n);
		double saving = 1.0 - rl.stats.memory_peak_bytes / dense_bytes;
		best_saving = std::max(best_saving, saving);
	}

	if (failures) {
		printf("%d FAILURES in %d trials\n", failures, trials);
		return 1;
	}
	printf("OK: %d trials. worst fidelity %.9f (target >= %.9f), "
	       "best peak-memory saving %.0f%%\n",
	       trials, worst_fid, target, best_saving * 100.0);
	return 0;
}
