/*
 * Grover search over 3 qubits: amplify |101> out of 8 candidates.
 * Two iterations (~pi/4 * sqrt(8)) take the target probability from
 * 0.125 to 0.9453; the exact value checks the whole gate pipeline.
 */
#include <qubit/qubit.h>
#include <cstdio>

/* Flip the sign of |101>. X on the zero positions, then CCZ, then undo. */
static void oracle(qubit::Circuit& c)
{
	c.x(1);
	c.h(2); c.toffoli(0, 1, 2); c.h(2);	/* CCZ = H CCX H on target */
	c.x(1);
}

/* Reflect all amplitudes about their mean. */
static void diffuser(qubit::Circuit& c)
{
	for (int q = 0; q < 3; q++) c.h(q);
	for (int q = 0; q < 3; q++) c.x(q);
	c.h(2); c.toffoli(0, 1, 2); c.h(2);
	for (int q = 0; q < 3; q++) c.x(q);
	for (int q = 0; q < 3; q++) c.h(q);
}

int main()
{
	qubit::Circuit c(3);
	for (int q = 0; q < 3; q++) c.h(q);

	for (int iter = 0; iter < 2; iter++) {
		oracle(c);
		diffuser(c);
	}

	qubit::RunOptions opt;
	opt.shots = 4096;
	auto r = qubit::run(c, opt);

	printf("Grover searching |101> (index 5):\n\n");
	r.print_counts();
	printf("\nprob |101>: %.4f  (start: 0.125, theory after 2 iters: 0.9453)\n",
	       r.prob(5));
	printf("backend: %s | %.1f ms\n", r.stats.backend.c_str(), r.stats.time_ms);
	return 0;
}
