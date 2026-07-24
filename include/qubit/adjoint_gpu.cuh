/*
 * qubit/adjoint_gpu.cuh — CUDA adjoint executors, part of the qubit library.
 *
 * The CPU adjoint (qubit/adjoint.h) on the GPU, in complex64 so a 6 GB card
 * holds a larger state. The four state ops become CUDA kernels; the backward
 * pass keeps only the two trajectories (phi, lambda) resident, so the dense
 * ceiling is 2 * 8 * 2^n bytes. GPUCircuitQ stores them as int16 (4 B/amp)
 * and round-trips through the int16 transform on device (Phase 3's bound).
 *
 * Circuit data model from qubit/circuit.h. This header is CUDA-only (no
 * qubit.h, no pybind), so it compiles into the qtrain binding module
 * (qubit_gpu_native) via nvcc. Include it from a .cu compiled with -arch.
 */
#include <thrust/complex.h>
#include <cuda_runtime.h>
#include <vector>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <cmath>

#include "qubit/circuit.h"   // qubit::AGate, Gen, Term, Ham, CircuitBuilder (no qubit.h)

using qubit::AGate; using qubit::Ham; using qubit::Term;
using qubit::GEN_NONE; using qubit::GEN_X; using qubit::GEN_Y; using qubit::GEN_Z;

using cf = thrust::complex<float>;

#define CUDA_OK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) \
	throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(e_)); } while (0)

static const int TPB = 256;
static inline int blocks(uint64_t N) { return int((N + TPB - 1) / TPB); }

/* ---- kernels ---- */
__global__ void k_apply2x2(cf* s, uint64_t N, uint64_t bit, uint64_t cmask,
                           cf m0, cf m1, cf m2, cf m3) {
	uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= N || (i & bit)) return;
	if ((i & cmask) != cmask) return;
	uint64_t j = i | bit;
	cf a = s[i], b = s[j];
	s[i] = m0 * a + m1 * b;
	s[j] = m2 * a + m3 * b;
}

__global__ void k_generator(cf* s, uint64_t N, uint64_t bit, int gen) {
	uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= N || (i & bit)) return;
	uint64_t j = i | bit;
	cf a = s[i], b = s[j];
	if (gen == GEN_X) { s[i] = b; s[j] = a; }
	else if (gen == GEN_Y) { s[i] = cf(0, -1) * b; s[j] = cf(0, 1) * a; }
	else { s[j] = -b; }
}

/* out += c * in */
__global__ void k_axpy(cf* out, const cf* in, float c, uint64_t N) {
	uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) out[i] += c * in[i];
}
__global__ void k_copy(cf* out, const cf* in, uint64_t N) {
	uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) out[i] = in[i];
}
__global__ void k_setzero(cf* s, uint64_t N) {
	uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) s[i] = cf(0, 0);
}

/* Re<a|b> accumulation into a double */
__global__ void k_redot(const cf* a, const cf* b, uint64_t N, double* acc) {
	uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= N) return;
	cf v = thrust::conj(a[i]) * b[i];
	atomicAdd(acc, (double)v.real());
}

/* Im<lambda|G|phi> over disjoint pairs -> grad term */
__global__ void k_gradterm(const cf* L, const cf* P, uint64_t N, uint64_t bit,
                           int gen, double* acc) {
	uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= N || (i & bit)) return;
	uint64_t j = i | bit;
	cf t;
	if (gen == GEN_X)      t = thrust::conj(L[i]) * P[j] + thrust::conj(L[j]) * P[i];
	else if (gen == GEN_Y) t = thrust::conj(L[i]) * (cf(0,-1) * P[j]) + thrust::conj(L[j]) * (cf(0,1) * P[i]);
	else                   t = thrust::conj(L[i]) * P[i] + thrust::conj(L[j]) * (-P[j]);
	atomicAdd(acc, (double)t.imag());
}

/* quantization: pass 1 max|.|, pass 2 round + accumulate err^2 */
__global__ void k_absmax(const cf* s, uint64_t N, unsigned* umax) {
	uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= N) return;
	float m = fmaxf(fabsf(s[i].real()), fabsf(s[i].imag()));
	atomicMax(umax, __float_as_uint(m));   /* nonneg floats: uint order == float order */
}
__global__ void k_quant(cf* s, uint64_t N, float scale, double* err2) {
	uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= N) return;
	float re = rintf(s[i].real() / scale) * scale;
	float im = rintf(s[i].imag() / scale) * scale;
	float dr = s[i].real() - re, di = s[i].imag() - im;
	atomicAdd(err2, (double)(dr * dr + di * di));
	s[i] = cf(re, im);
}

/* =====================================================================
 * int16 storage path — the compression that saves real device memory.
 * State stored as short2 (int16 real, int16 imag) + a scalar scale. A
 * normalized state has |amp|<=1 so scale=1/32767 covers phi; lambda=H|phi>
 * has |amp|<=sum|coeff| so its scale is set a priori. Each gate dequantizes
 * a pair to float, applies the 2x2, and requantizes — that per-gate
 * round-trip IS the injected compression (tracked into D). Paulis are exact
 * on int16 (permute/negate), so generators inject nothing.
 * Residency: 2 trajectories * 4 B/amp = 8 B/amp (half of complex64).
 * ===================================================================== */
#define Q16 32767.0f

__device__ __forceinline__ cf dq(short2 v, float s) { return cf(v.x * s, v.y * s); }
__device__ __forceinline__ short2 qz(cf z, float s) {
	float r = rintf(z.real() / s), i = rintf(z.imag() / s);
	r = fmaxf(-Q16, fminf(Q16, r)); i = fmaxf(-Q16, fminf(Q16, i));
	return make_short2((short)r, (short)i);
}

__global__ void k16_apply2x2(short2* s, uint64_t N, uint64_t bit, uint64_t cmask,
                             cf m0, cf m1, cf m2, cf m3, float scale, double* err2) {
	uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= N || (i & bit)) return;
	if ((i & cmask) != cmask) return;
	uint64_t j = i | bit;
	cf a = dq(s[i], scale), b = dq(s[j], scale);
	cf na = m0 * a + m1 * b, nb = m2 * a + m3 * b;
	short2 qa = qz(na, scale), qb = qz(nb, scale);
	cf da = na - dq(qa, scale), db = nb - dq(qb, scale);
	atomicAdd(err2, (double)(thrust::norm(da) + thrust::norm(db)));
	s[i] = qa; s[j] = qb;
}
__global__ void k16_generator(short2* s, uint64_t N, uint64_t bit, int gen) {
	uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= N || (i & bit)) return;
	uint64_t j = i | bit; short2 a = s[i], b = s[j];
	if (gen == GEN_X) { s[i] = b; s[j] = a; }
	else if (gen == GEN_Y) { s[i] = make_short2(b.y, (short)-b.x); s[j] = make_short2((short)-a.y, a.x); }
	else { s[j] = make_short2((short)-b.x, (short)-b.y); }
}
__global__ void k16_setzero(short2* s, uint64_t N) {
	uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) s[i] = make_short2(0, 0);
}
/* lambda += coeff * phi  (phi at sphi, lambda at slam) */
__global__ void k16_axpy(short2* lam, const short2* phi, float coeff,
                         float sphi, float slam, uint64_t N) {
	uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= N) return;
	cf L = dq(lam[i], slam) + coeff * dq(phi[i], sphi);
	lam[i] = qz(L, slam);
}
__global__ void k16_redot(const short2* a, const short2* b, float sa, float sb,
                          uint64_t N, double* acc) {
	uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= N) return;
	cf v = thrust::conj(dq(a[i], sa)) * dq(b[i], sb);
	atomicAdd(acc, (double)v.real());
}
__global__ void k16_gradterm(const short2* L, const short2* P, uint64_t N, uint64_t bit,
                             int gen, float sL, float sP, double* acc) {
	uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= N || (i & bit)) return;
	uint64_t j = i | bit;
	cf Li = dq(L[i], sL), Lj = dq(L[j], sL), Pi = dq(P[i], sP), Pj = dq(P[j], sP);
	cf t;
	if (gen == GEN_X)      t = thrust::conj(Li) * Pj + thrust::conj(Lj) * Pi;
	else if (gen == GEN_Y) t = thrust::conj(Li) * (cf(0,-1) * Pj) + thrust::conj(Lj) * (cf(0,1) * Pi);
	else                   t = thrust::conj(Li) * Pi + thrust::conj(Lj) * (-Pj);
	atomicAdd(acc, (double)t.imag());
}

/* ---- host helpers ---- */
static void rot_cf(int gen, double th, cf m[4]) {
	float c = (float)std::cos(th / 2), s = (float)std::sin(th / 2);
	if (gen == GEN_Z)      { m[0] = cf(c, -s); m[1] = 0; m[2] = 0; m[3] = cf(c, s); }
	else if (gen == GEN_Y) { m[0] = cf(c, 0); m[1] = cf(-s, 0); m[2] = cf(s, 0); m[3] = cf(c, 0); }
	else                   { m[0] = cf(c, 0); m[1] = cf(0, -s); m[2] = cf(0, -s); m[3] = cf(c, 0); }
}
static void mat_cf(const AGate& g, cf m[4]) {
	for (int i = 0; i < 4; i++) m[i] = cf((float)g.m[i].real(), (float)g.m[i].imag());
}
static void dagger_cf(const cf in[4], cf out[4]) {
	out[0] = thrust::conj(in[0]); out[1] = thrust::conj(in[2]);
	out[2] = thrust::conj(in[1]); out[3] = thrust::conj(in[3]);
}

struct DevAccum {
	double* d = nullptr;
	DevAccum() { CUDA_OK(cudaMalloc(&d, sizeof(double))); }
	~DevAccum() { cudaFree(d); }
	void zero() { CUDA_OK(cudaMemset(d, 0, sizeof(double))); }
	double get() { double h; CUDA_OK(cudaMemcpy(&h, d, sizeof(double), cudaMemcpyDeviceToHost)); return h; }
};

class GPUCircuit : public qubit::CircuitBuilder {
public:
	using qubit::CircuitBuilder::CircuitBuilder;

	/* levels<=0 => exact; else compress phi/lambda each boundary.
	   returns (value, grad, D). */
	std::tuple<double, std::vector<double>, double>
	run(const Ham& H, int levels) {
		const uint64_t N = uint64_t(1) << n_;
		const int B = blocks(N);
		cf *phi, *lambda;
		CUDA_OK(cudaMalloc(&phi, N * sizeof(cf)));
		CUDA_OK(cudaMalloc(&lambda, N * sizeof(cf)));

		/* forward: phi = U|0> */
		k_setzero<<<B, TPB>>>(phi, N);
		{ cf one(1, 0); CUDA_OK(cudaMemcpy(phi, &one, sizeof(cf), cudaMemcpyHostToDevice)); }
		for (auto& g : gates_) apply_gate(phi, g, N);

		DevAccum acc;
		/* lambda = H|phi>, built WITHOUT a third buffer: apply each term's
		   Pauli string to phi in place, axpy into lambda, then re-apply to
		   restore phi (Paulis are involutions). Peak stays at 2 states, so
		   the dense ceiling is 2*8*2^n bytes. */
		k_setzero<<<B, TPB>>>(lambda, N);
		for (auto& t : H) {
			for (auto& op : t.ops) k_generator<<<B, TPB>>>(phi, N, uint64_t(1) << op.first, op.second);
			k_axpy<<<B, TPB>>>(lambda, phi, (float)t.coeff, N);
			for (auto& op : t.ops) k_generator<<<B, TPB>>>(phi, N, uint64_t(1) << op.first, op.second);
		}
		/* value = <phi|H|phi> = Re<phi|lambda> */
		acc.zero();
		k_redot<<<B, TPB>>>(phi, lambda, N, acc.d);
		double value = acc.get();

		std::vector<double> grad(nparams_, 0.0);
		double D = 0;
		for (int k = int(gates_.size()) - 1; k >= 0; k--) {
			const AGate& g = gates_[k];
			if (g.param && g.pidx >= 0) {
				acc.zero();
				k_gradterm<<<B, TPB>>>(lambda, phi, N, uint64_t(1) << g.q, g.gen, acc.d);
				grad[g.pidx] = acc.get();
			}
			apply_gate_inv(phi, g, N);
			apply_gate_inv(lambda, g, N);
			if (levels > 0) { D += quantize(phi, N, levels); D += quantize(lambda, N, levels); }
		}
		CUDA_OK(cudaFree(phi));
		CUDA_OK(cudaFree(lambda));
		return {value, grad, D};
	}

private:
	void apply_gate(cf* s, const AGate& g, uint64_t N) {
		cf m[4];
		if (g.gen != GEN_NONE) rot_cf(g.gen, g.theta, m); else mat_cf(g, m);
		uint64_t cmask = 0; for (int c : g.ctrl) cmask |= uint64_t(1) << c;
		k_apply2x2<<<blocks(N), TPB>>>(s, N, uint64_t(1) << g.q, cmask, m[0], m[1], m[2], m[3]);
	}
	void apply_gate_inv(cf* s, const AGate& g, uint64_t N) {
		cf m[4];
		if (g.gen != GEN_NONE) rot_cf(g.gen, -g.theta, m);
		else { cf tmp[4]; mat_cf(g, tmp); dagger_cf(tmp, m); }
		uint64_t cmask = 0; for (int c : g.ctrl) cmask |= uint64_t(1) << c;
		k_apply2x2<<<blocks(N), TPB>>>(s, N, uint64_t(1) << g.q, cmask, m[0], m[1], m[2], m[3]);
	}
	double quantize(cf* s, uint64_t N, int levels) {
		unsigned* umax; CUDA_OK(cudaMalloc(&umax, sizeof(unsigned)));
		CUDA_OK(cudaMemset(umax, 0, sizeof(unsigned)));
		k_absmax<<<blocks(N), TPB>>>(s, N, umax);
		unsigned uh; CUDA_OK(cudaMemcpy(&uh, umax, sizeof(unsigned), cudaMemcpyDeviceToHost));
		CUDA_OK(cudaFree(umax));
		float mx = __uint_as_float_host(uh);
		if (mx == 0) return 0;
		float scale = mx / levels;
		DevAccum err; err.zero();
		k_quant<<<blocks(N), TPB>>>(s, N, scale, err.d);
		return std::sqrt(err.get());
	}
	/* host reinterpret of the uint bits produced by __float_as_uint */
	static float __uint_as_float_host(unsigned u) { float f; std::memcpy(&f, &u, 4); return f; }
};

/* int16-storage executor: same builder base, half the resident bytes. */
class GPUCircuitQ : public qubit::CircuitBuilder {
public:
	using qubit::CircuitBuilder::CircuitBuilder;

	std::tuple<double, std::vector<double>, double> run(const Ham& H) {
		const uint64_t N = uint64_t(1) << n_;
		const int B = blocks(N);
		const float sphi = 1.0f / Q16;                 /* |phi| <= 1 */
		double Cabs = 0; for (auto& t : H) Cabs += std::fabs(t.coeff);
		const float slam = (float)(Cabs > 0 ? Cabs : 1.0) / Q16;  /* |lambda| <= sum|coeff| */

		short2 *phi, *lambda;
		CUDA_OK(cudaMalloc(&phi, N * sizeof(short2)));
		CUDA_OK(cudaMalloc(&lambda, N * sizeof(short2)));
		DevAccum err;

		/* forward on int16 phi (scale sphi) */
		k16_setzero<<<B, TPB>>>(phi, N);
		{ short2 one = make_short2((short)Q16, 0);
		  CUDA_OK(cudaMemcpy(phi, &one, sizeof(short2), cudaMemcpyHostToDevice)); }
		for (auto& g : gates_) apply16(phi, g, N, sphi, err.d);

		/* lambda = H|phi> in int16 (scale slam), no float buffer */
		k16_setzero<<<B, TPB>>>(lambda, N);
		for (auto& t : H) {
			for (auto& op : t.ops) k16_generator<<<B, TPB>>>(phi, N, uint64_t(1) << op.first, op.second);
			k16_axpy<<<B, TPB>>>(lambda, phi, (float)t.coeff, sphi, slam, N);
			for (auto& op : t.ops) k16_generator<<<B, TPB>>>(phi, N, uint64_t(1) << op.first, op.second);
		}
		DevAccum acc; acc.zero();
		k16_redot<<<B, TPB>>>(phi, lambda, sphi, slam, N, acc.d);
		double value = acc.get();

		std::vector<double> grad(nparams_, 0.0);
		err.zero();
		for (int k = int(gates_.size()) - 1; k >= 0; k--) {
			const AGate& g = gates_[k];
			if (g.param && g.pidx >= 0) {
				acc.zero();
				k16_gradterm<<<B, TPB>>>(lambda, phi, N, uint64_t(1) << g.q, g.gen, slam, sphi, acc.d);
				grad[g.pidx] = acc.get();
			}
			apply16_inv(phi, g, N, sphi, err.d);
			apply16_inv(lambda, g, N, slam, err.d);
		}
		double D = std::sqrt(err.get());
		CUDA_OK(cudaFree(phi)); CUDA_OK(cudaFree(lambda));
		return {value, grad, D};
	}

private:
	void apply16(short2* s, const AGate& g, uint64_t N, float scale, double* err) {
		cf m[4];
		if (g.gen != GEN_NONE) rot_cf(g.gen, g.theta, m); else mat_cf(g, m);
		uint64_t cmask = 0; for (int c : g.ctrl) cmask |= uint64_t(1) << c;
		k16_apply2x2<<<blocks(N), TPB>>>(s, N, uint64_t(1) << g.q, cmask, m[0], m[1], m[2], m[3], scale, err);
	}
	void apply16_inv(short2* s, const AGate& g, uint64_t N, float scale, double* err) {
		cf m[4];
		if (g.gen != GEN_NONE) rot_cf(g.gen, -g.theta, m);
		else { cf tmp[4]; mat_cf(g, tmp); dagger_cf(tmp, m); }
		uint64_t cmask = 0; for (int c : g.ctrl) cmask |= uint64_t(1) << c;
		k16_apply2x2<<<blocks(N), TPB>>>(s, N, uint64_t(1) << g.q, cmask, m[0], m[1], m[2], m[3], scale, err);
	}
};

/* GPUCircuit (dense complex64) and GPUCircuitQ (int16 storage) are the
 * library's CUDA adjoint executors. The Python module that exposes them
 * (qubit_gpu_native) is the implementation layer and lives in qtrain
 * (bindings/qubit_gpu.cu), which includes this header. */
