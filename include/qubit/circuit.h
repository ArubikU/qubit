/*
 * circuit.h — the parametric circuit data model shared by the CPU and GPU
 * adjoint executors. Pure data + the builder API driven from Python; no
 * dependency on the qubit engine or on CUDA, so both adjoint.h (CPU, which
 * runs on qubit's DenseCPUT backend) and adjoint_gpu.cu (own CUDA kernels)
 * include it without pulling each other's toolchain in.
 */
#pragma once
#include <vector>
#include <complex>
#include <utility>

namespace qubit {

using cd = std::complex<double>;
using Vec = std::vector<cd>;

/* generator of a parametric single-qubit rotation exp(-i theta G / 2) */
enum Gen { GEN_NONE = 0, GEN_X = 1, GEN_Y = 2, GEN_Z = 3 };

struct AGate {
	int q = 0;                 /* target qubit */
	std::vector<int> ctrl;     /* control qubits (fixed gates only) */
	cd m[4] = {};              /* fixed-gate 2x2 (row-major) when gen==NONE */
	int gen = GEN_NONE;        /* parametric rotation generator */
	double theta = 0.0;        /* parametric angle */
	bool param = false;        /* trainable rotation contributing a gradient */
	int pidx = -1;             /* gradient slot */
};

/* Hamiltonian: weighted sum of Pauli strings. pauli code per wire: GEN_X/Y/Z */
struct Term { double coeff; std::vector<std::pair<int, int>> ops; };
using Ham = std::vector<Term>;

/* 2x2 matrix of exp(-i theta G / 2) */
inline void rot_matrix(int gen, double th, cd m[4]) {
	double c = std::cos(th / 2), s = std::sin(th / 2);
	if (gen == GEN_Z)      { m[0] = cd(c, -s); m[1] = 0; m[2] = 0; m[3] = cd(c, s); }
	else if (gen == GEN_Y) { m[0] = c; m[1] = -s; m[2] = s; m[3] = c; }
	else                   { m[0] = c; m[1] = cd(0, -s); m[2] = cd(0, -s); m[3] = c; } /* X */
}
inline void dagger(const cd in[4], cd out[4]) {
	out[0] = std::conj(in[0]); out[1] = std::conj(in[2]);
	out[2] = std::conj(in[1]); out[3] = std::conj(in[3]);
}

/*
 * Builder: records the gate list and the trainable-parameter count. One
 * place owns the circuit; executors (CPU ACircuit, GPU GPUCircuit/
 * GPUCircuitQ) inherit it and add only their run logic.
 */
class CircuitBuilder {
public:
	explicit CircuitBuilder(int n) : n_(n) {}
	int num_qubits() const { return n_; }
	int num_params() const { return nparams_; }

	/* parametric single-qubit rotation; slot<0 means fixed (non-trainable) */
	void rot(int gen, int q, double theta, bool trainable, int slot) {
		AGate g; g.gen = gen; g.q = q; g.theta = theta;
		g.param = trainable; g.pidx = trainable ? slot : -1;
		if (trainable && slot + 1 > nparams_) nparams_ = slot + 1;
		gates_.push_back(g);
	}
	void fixed(int q, cd m00, cd m01, cd m10, cd m11) {
		AGate g; g.q = q; g.m[0] = m00; g.m[1] = m01; g.m[2] = m10; g.m[3] = m11;
		gates_.push_back(g);
	}
	void cfixed(std::vector<int> ctrl, int q, cd m00, cd m01, cd m10, cd m11) {
		AGate g; g.q = q; g.ctrl = std::move(ctrl);
		g.m[0] = m00; g.m[1] = m01; g.m[2] = m10; g.m[3] = m11;
		gates_.push_back(g);
	}
	const std::vector<AGate>& gates() const { return gates_; }

protected:
	int n_;
	int nparams_ = 0;
	std::vector<AGate> gates_;
};

} // namespace qubit
