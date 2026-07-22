/*
 * Kernel-only throughput: time gate application on dense-gpu with CUDA
 * events, excluding init and state read-back, for a fair kernel-vs-kernel
 * comparison against Aer's internal result.time_taken. Fusion off so the
 * gate stream matches the reference circuits.
 *
 * Build: nvcc -O2 -std=c++17 -arch=sm_86 -DQUBIT_CUDA -Xcompiler "/EHsc" \
 *        -I include bench/kernel_bench.cu src/qubit_gpu.cu -o kernel_bench
 */
#include <qubit/qubit.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
#include <thread>
#include <chrono>

using namespace qubit;

static Circuit qft(int n) {
	Circuit c(n);
	for (int q = 0; q < n; q++) c.h(q);
	for (int i = n - 1; i >= 0; i--) {
		c.h(i);
		for (int j = i - 1; j >= 0; j--) {
			double t = 3.14159265358979 / double(1 << (i - j));
			c.controlled(j, i, {1,0},{0,0},{0,0},{std::cos(t), std::sin(t)});
		}
	}
	return c;
}
static Circuit qaoa(int n, int p) {
	Circuit c(n);
	for (int q = 0; q < n; q++) c.h(q);
	for (int r = 0; r < p; r++) {
		for (int q = 0; q + 1 < n; q++) { c.cnot(q, q+1); c.rz(q+1, 0.4+0.1*r); c.cnot(q, q+1); }
		for (int q = 0; q < n; q++) c.rx(q, 0.7-0.05*r);
	}
	return c;
}
static Circuit randc(int n, int depth, uint64_t seed) {
	Circuit c(n); std::mt19937_64 g(seed);
	std::uniform_int_distribution<int> q(0, n-1); std::uniform_real_distribution<float> a(0, 6.2832f);
	for (int d = 0; d < depth; d++) {
		for (int k = 0; k < n; k++) { int s = int(g()%3); if (s==0) c.h(k); else if (s==1) c.t(k); else c.ry(k, a(g)); }
		for (int k = 0; k+1 < n; k += 2) { int x=q(g), y=q(g); if (x!=y) c.cnot(x,y); }
	}
	return c;
}

static double one_shot_ms(const Circuit& c) {
	auto be = make_dense_gpu();
	be->init(c.num_qubits());
	cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
	cudaDeviceSynchronize();
	cudaEventRecord(s);
	for (const auto& g : c.gates()) be->apply(g);   /* gates only, no read-back */
	cudaEventRecord(e);
	cudaEventSynchronize(e);
	float ms = 0; cudaEventElapsedTime(&ms, s, e);
	cudaEventDestroy(s); cudaEventDestroy(e);
	return ms;
}

/*
 * Thermal-throttling defense without clock control (which needs admin):
 * run many reps with a cooldown between each and report the MINIMUM.
 * Throttling only ever slows a run down, so the fastest observed time is
 * the one least affected by it -- the closest estimate of unthrottled
 * kernel speed. Median and max show the spread.
 */
static void bench(const char* name, const Circuit& c, int reps, int cooldown_ms) {
	one_shot_ms(c);   /* warm up, discarded */
	std::vector<double> t;
	for (int r = 0; r < reps; r++) {
		t.push_back(one_shot_ms(c));
		std::this_thread::sleep_for(std::chrono::milliseconds(cooldown_ms));
	}
	std::sort(t.begin(), t.end());
	printf("%s,%d,%.2f,%.2f,%.2f\n", name, c.num_qubits(),
	       t.front(), t[t.size()/2], t.back());
}

int main(int argc, char** argv) {
	int n = 28;
	int reps = (argc > 1) ? std::atoi(argv[1]) : 15;
	int cd = (argc > 2) ? std::atoi(argv[2]) : 1500;   /* ms cooldown */
	printf("circuit,qubits,min_ms,median_ms,max_ms\n");
	bench("qft", qft(n), reps, cd);
	bench("qaoa4", qaoa(n, 4), reps, cd);
	bench("random", randc(n, 10, 42), reps, cd);
	return 0;
}
