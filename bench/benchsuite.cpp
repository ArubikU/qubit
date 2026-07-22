/*
 * Benchmark suite. Same circuit families as bench/bench_qiskit.py so
 * results are directly comparable. Output is CSV on stdout:
 *
 *   circuit,qubits,backend,time_ms,peak_mem_bytes,check
 *
 * `check` is a circuit-specific correctness probe (probability or
 * expectation value), so a fast-but-wrong run is visible immediately.
 *
 * Usage: benchsuite [max_qubits] [backend] [reps]
 *   backend: auto | dense | groups | blocks | gpu   (default auto)
 *   reps: runs per case; the MEDIAN time is reported (default 3).
 *   One warmup run per case is discarded — laptop thermal jitter makes
 *   single-shot timings differ by 2x on identical binaries.
 */
#include <qubit/qubit.h>
#include <qubit/blocks.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

/* Quantum Fourier Transform: the dense worst case, gates on every pair. */
static qubit::Circuit qft(int n)
{
	qubit::Circuit c(n);
	for (int q = 0; q < n; q++) c.h(q);	/* nontrivial input */
	for (int i = n - 1; i >= 0; i--) {
		c.h(i);
		for (int j = i - 1; j >= 0; j--) {
			float theta = 3.14159265f / float(1 << (i - j));
			/* controlled phase via controlled-U */
			c.controlled(j, i, {1,0},{0,0},{0,0},
				     {std::cos(theta), std::sin(theta)});
		}
	}
	return c;
}

/* GHZ chain: maximal entanglement, two nonzero amplitudes. */
static qubit::Circuit ghz(int n)
{
	qubit::Circuit c(n);
	c.h(0);
	for (int q = 0; q + 1 < n; q++) c.cnot(q, q + 1);
	return c;
}

/* QAOA-style: alternating entangler + mixer layers, p rounds. */
static qubit::Circuit qaoa(int n, int p)
{
	qubit::Circuit c(n);
	for (int q = 0; q < n; q++) c.h(q);
	for (int r = 0; r < p; r++) {
		for (int q = 0; q + 1 < n; q++) {	/* ring ZZ coupling */
			c.cnot(q, q + 1);
			c.rz(q + 1, 0.4f + 0.1f * r);
			c.cnot(q, q + 1);
		}
		for (int q = 0; q < n; q++)		/* mixer */
			c.rx(q, 0.7f - 0.05f * r);
	}
	return c;
}

/* Random dense circuit: supremacy-style stress, no structure to exploit. */
static qubit::Circuit random_dense(int n, int depth, uint64_t seed)
{
	qubit::Circuit c(n);
	std::mt19937_64 g(seed);
	std::uniform_int_distribution<int> q(0, n - 1);
	std::uniform_real_distribution<float> ang(0.0f, 6.2832f);
	for (int d = 0; d < depth; d++) {
		for (int k = 0; k < n; k++) {
			int sel = int(g() % 3);
			if (sel == 0) c.h(k);
			else if (sel == 1) c.t(k);
			else c.ry(k, ang(g));
		}
		for (int k = 0; k + 1 < n; k += 2) {
			int a = q(g), b = q(g);
			if (a != b) c.cnot(a, b);
		}
	}
	return c;
}

/* Independent Bell pairs: the grouped backend's best case. */
static qubit::Circuit pairs(int n)
{
	qubit::Circuit c(n);
	for (int q = 0; q + 1 < n; q += 2) {
		c.h(q);
		c.cnot(q, q + 1);
	}
	return c;
}

struct Case {
	const char* name;
	qubit::Circuit circuit;
	double check;		/* filled after run */
};

int main(int argc, char** argv)
{
	int maxq = (argc > 1) ? std::atoi(argv[1]) : 24;
	const char* bsel = (argc > 2) ? argv[2] : "auto";
	int reps = (argc > 3) ? std::atoi(argv[3]) : 3;
	if (reps < 1) reps = 1;

	qubit::RunOptions opt;
	opt.shots = 64;
	opt.seed = 1;
	opt.cpu_mem_budget = 6.0 * 1024 * 1024 * 1024;
	if (!strcmp(bsel, "dense"))  opt.backend = qubit::BackendSel::DenseCPU_;
	if (!strcmp(bsel, "groups")) opt.backend = qubit::BackendSel::GroupsCPU_;
	if (!strcmp(bsel, "gpu"))    opt.backend = qubit::BackendSel::DenseGPU_;
	if (!strcmp(bsel, "blocks"))
		opt.custom_backend = [] { return std::make_shared<qubit::BlocksCPU>(12); };

	printf("circuit,qubits,backend,time_ms,peak_mem_bytes,check\n");

	for (int n = 8; n <= maxq; n += 4) {
		struct { const char* name; qubit::Circuit c; } cases[] = {
			{ "qft",    qft(n) },
			{ "ghz",    ghz(n) },
			{ "qaoa4",  qaoa(n, 4) },
			{ "random", random_dense(n, 10, 42) },
			{ "pairs",  pairs(n) },
		};
		for (auto& k : cases) {
			try {
				qubit::run(k.c, opt);	/* warmup, discarded */
				std::vector<double> times;
				qubit::Result last = qubit::run(k.c, opt);
				times.push_back(last.stats.time_ms);
				for (int rep = 1; rep < reps; rep++) {
					auto r = qubit::run(k.c, opt);
					times.push_back(r.stats.time_ms);
					last = std::move(r);
				}
				std::sort(times.begin(), times.end());
				double med = times[times.size() / 2];
				/*
				 * checks: ghz -> P(all zeros) = 0.5;
				 * pairs -> <Z0 Z1> = 1; others -> <Z0 Z1>.
				 */
				double check;
				if (!strcmp(k.name, "ghz"))
					check = last.prob(0);
				else
					check = last.expectation_z({0, 1});
				printf("%s,%d,%s,%.2f,%.0f,%.6f\n",
				       k.name, n, last.stats.backend.c_str(),
				       med, last.stats.memory_peak_bytes,
				       check);
				fflush(stdout);
			} catch (const qubit::Error& e) {
				printf("%s,%d,SKIP,,,\n", k.name, n);
				fflush(stdout);
			}
		}
	}
	return 0;
}
