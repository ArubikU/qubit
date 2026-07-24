/*
 * adjoint.h — adjoint (reverse-mode) differentiation for the variational
 * training loop, running ON the qubit engine. All parameter gradients of
 * <psi|H|psi> in one forward + one backward pass (Jones & Gacon 2020), vs
 * 2P forward passes for parameter-shift.
 *
 * The unitary evolution (forward gates and their inverses) is delegated to
 * qubit's DenseCPUT backend and its optimized apply() kernel — qtrain no
 * longer carries a second gate kernel. This executor only adds the
 * training-specific arithmetic on the raw amplitude buffers: applying the
 * observable (H|phi>), the bra/ket gradient inner product, and the optional
 * int16 compression round-trip that Phase 3's bound is stated over.
 *
 * The `buffer()` accessor this relies on was added to qubit.h for exactly
 * this use. The circuit data model lives in circuit.h (shared with the GPU
 * executor, which keeps its own CUDA kernels).
 */
#pragma once
#include <vector>
#include <complex>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <tuple>

#include "qubit/circuit.h"
#include "qubit/qubit.h"

namespace qubit {

/* ---- buffer arithmetic (the parts that are NOT gate evolution) ---- */

/* Pauli generator G (no phase, no controls) applied to a raw buffer. */
inline void apply_generator(Vec& s, int q, int gen) {
	const uint64_t bit = 1ull << q;
	const long long N = (long long)s.size();
	cd* p = s.data();
	#pragma omp parallel for schedule(static)
	for (long long i = 0; i < N; i++) {
		if (i & bit) continue;
		uint64_t j = i | bit; cd a = p[i], b = p[j];
		if (gen == GEN_X) { p[i] = b; p[j] = a; }
		else if (gen == GEN_Y) { p[i] = cd(0, -1) * b; p[j] = cd(0, 1) * a; }
		else { p[j] = -b; }   /* Z */
	}
}

/* fused gradient term: g_k = Im <lambda|G|phi> (= 2 Re((-i/2)<lambda|G|phi>)),
   over disjoint pairs, no state copy. */
inline double grad_term(const Vec& lambda, const Vec& phi, int q, int gen) {
	const uint64_t bit = 1ull << q;
	const long long N = (long long)phi.size();
	const cd* L = lambda.data();
	const cd* P = phi.data();
	double im = 0;
	#pragma omp parallel for schedule(static) reduction(+:im)
	for (long long i = 0; i < N; i++) {
		if (i & bit) continue;
		uint64_t j = i | bit;
		cd acc;
		if (gen == GEN_X)      acc = std::conj(L[i]) * P[j] + std::conj(L[j]) * P[i];
		else if (gen == GEN_Y) acc = std::conj(L[i]) * (cd(0,-1) * P[j]) + std::conj(L[j]) * (cd(0,1) * P[i]);
		else                   acc = std::conj(L[i]) * P[i] + std::conj(L[j]) * (-P[j]);
		im += acc.imag();
	}
	return im;
}

/* lambda = H|phi> on raw buffers */
inline Vec apply_ham(const Vec& phi, const Ham& H) {
	Vec r(phi.size(), cd(0, 0));
	for (auto& t : H) {
		Vec ts = phi;
		for (auto& op : t.ops) apply_generator(ts, op.first, op.second);
		const long long N = (long long)r.size();
		const double c = t.coeff; cd* o = r.data(); const cd* s = ts.data();
		#pragma omp parallel for schedule(static)
		for (long long i = 0; i < N; i++) o[i] += c * s[i];
	}
	return r;
}

/* value = <phi|H|phi> = Re<phi|lambda> with lambda = H|phi> */
inline double redot(const Vec& a, const Vec& b) {
	const long long N = (long long)a.size();
	const cd* pa = a.data(); const cd* pb = b.data();
	double re = 0;
	#pragma omp parallel for schedule(static) reduction(+:re)
	for (long long i = 0; i < N; i++) re += (std::conj(pa[i]) * pb[i]).real();
	return re;
}

/* int16 block-scaled quantization round-trip; returns injected L2 norm. */
inline double quantize_roundtrip(Vec& s, int levels) {
	double mx = 0;
	for (auto& z : s) { mx = std::max(mx, std::fabs(z.real())); mx = std::max(mx, std::fabs(z.imag())); }
	if (mx == 0) return 0;
	const double scale = mx / levels;
	const long long N = (long long)s.size();
	cd* p = s.data();
	double err2 = 0;
	#pragma omp parallel for schedule(static) reduction(+:err2)
	for (long long i = 0; i < N; i++) {
		double re = std::round(p[i].real() / scale) * scale;
		double im = std::round(p[i].imag() / scale) * scale;
		double dr = p[i].real() - re, di = p[i].imag() - im;
		err2 += dr * dr + di * di;
		p[i] = cd(re, im);
	}
	return std::sqrt(err2);
}

/* ---- CPU executor on qubit's dense backend ---- */
class ACircuit : public CircuitBuilder {
public:
	using CircuitBuilder::CircuitBuilder;

	Vec forward() const {
		qubit::DenseCPUT<double> be;
		be.init(n_);
		for (auto& g : gates_) be.apply(to_gate(g, false));
		return be.buffer();
	}

	/* (value, grad) — exact adjoint */
	std::pair<double, std::vector<double>> value_and_grad(const Ham& H) const {
		double v, D; std::vector<double> g;
		std::tie(v, g, D) = run(H, 0);
		return {v, g};
	}

	/* (value, grad, D) — trajectories round-tripped through int16 each gate
	   boundary when levels>0; D is the injected budget. */
	std::tuple<double, std::vector<double>, double>
	value_and_grad_q(const Ham& H, int levels) const { return run(H, levels); }

private:
	static qubit::Gate to_gate(const AGate& g, bool inverse) {
		qubit::Gate q;
		q.target = g.q;
		q.controls = g.ctrl;
		if (g.gen != GEN_NONE) {
			cd m[4]; rot_matrix(g.gen, inverse ? -g.theta : g.theta, m);
			for (int i = 0; i < 4; i++) q.m[i] = m[i];
		} else if (inverse) {
			cd m[4]; dagger(g.m, m);
			for (int i = 0; i < 4; i++) q.m[i] = m[i];
		} else {
			for (int i = 0; i < 4; i++) q.m[i] = g.m[i];
		}
		return q;
	}

	std::tuple<double, std::vector<double>, double>
	run(const Ham& H, int levels) const {
		qubit::DenseCPUT<double> phi_be, lambda_be;
		phi_be.init(n_);
		for (auto& g : gates_) phi_be.apply(to_gate(g, false));

		Vec& phi = phi_be.buffer();
		Vec lam = apply_ham(phi, H);
		double value = redot(phi, lam);

		lambda_be.init(n_);
		lambda_be.buffer() = lam;              /* seed lambda backend with H|phi> */
		Vec& lambda = lambda_be.buffer();

		std::vector<double> grad(nparams_, 0.0);
		double D = 0;
		for (int k = int(gates_.size()) - 1; k >= 0; k--) {
			const AGate& g = gates_[k];
			if (g.param && g.pidx >= 0)
				grad[g.pidx] = grad_term(lambda, phi, g.q, g.gen);
			qubit::Gate ig = to_gate(g, true);
			phi_be.apply(ig);
			lambda_be.apply(ig);
			if (levels > 0) {
				D += quantize_roundtrip(phi, levels);
				D += quantize_roundtrip(lambda, levels);
			}
		}
		return {value, grad, D};
	}
};

} // namespace qubit
