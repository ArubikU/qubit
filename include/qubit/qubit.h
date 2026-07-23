/*
 * qubit - GPU-accelerated quantum state-vector simulator.
 *
 * See DESIGN.md for architecture. Short version: Circuit records gates,
 * run() analyzes the whole circuit, picks a backend, executes.
 */

#pragma once
#include <complex>
#include <vector>
#include <string>
#include <map>
#include <random>
#include <memory>
#include <stdexcept>
#include <algorithm>
#include <chrono>
#include <functional>
#include <cstdint>
#include <cstdio>
#include <cmath>

namespace qubit {

using cf = std::complex<float>;
using cd = std::complex<double>;

inline cf to_cf(const cd& z) { return cf(float(z.real()), float(z.imag())); }

struct Error : std::runtime_error {
	explicit Error(const std::string& msg) : std::runtime_error(msg) {}
};

/*
 * Every operation reduces to one of three primitives. Single-qubit gates
 * are a 2x2 matrix; multi-qubit gates are the same matrix plus control
 * qubits. Keeping one representation means backends implement exactly
 * one gate kernel.
 */
struct Gate {
	enum class Op { U1, U2, Measure, Reset };
	Op op = Op::U1;
	int target = 0;
	int target2 = -1;		/* U2 only: the matrix's high qubit */
	std::vector<int> controls;
	cd m[4] = {};			/* m00 m01 m10 m11, row-major; backends narrow */
	cd m4[16] = {};			/* U2 only: 4x4, basis index = hi*2 + lo */
	int cbit = -1;			/* Measure: classical bit written */
	int cond_bit = -1;		/* execute only if cbits[cond_bit] == cond_val */
	int cond_val = 1;
	char label = '?';		/* dump() only */
};

/*
 * Circuit is a recorder, not an executor. Execution is deferred until
 * run() so the analyzer can see the full gate list before committing
 * to a state representation.
 */
class Circuit {
public:
	explicit Circuit(int num_qubits) : n_(num_qubits) {
		if (num_qubits < 1 || num_qubits > 1024)
			throw Error("Circuit: num_qubits must be in [1, 1024]");
	}

	int num_qubits() const { return n_; }
	const std::vector<Gate>& gates() const { return gates_; }
	int num_cbits() const { return next_cbit_; }

	void h(int q) { double s = 1.0 / std::sqrt(2.0); u1(q, {s,0},{s,0},{s,0},{-s,0}, 'H'); }
	void x(int q) { u1(q, {0,0},{1,0},{1,0},{0,0}, 'X'); }
	void y(int q) { u1(q, {0,0},{0,-1},{0,1},{0,0}, 'Y'); }
	void z(int q) { u1(q, {1,0},{0,0},{0,0},{-1,0}, 'Z'); }
	void s(int q) { u1(q, {1,0},{0,0},{0,0},{0,1}, 'S'); }
	void t(int q) {
		double c = std::cos(3.14159265358979 / 4), sn = std::sin(3.14159265358979 / 4);
		u1(q, {1,0},{0,0},{0,0},{c,sn}, 'T');
	}

	void rx(int q, double th) {
		double c = std::cos(th/2), sn = std::sin(th/2);
		u1(q, {c,0},{0,-sn},{0,-sn},{c,0}, 'x');
	}
	void ry(int q, double th) {
		double c = std::cos(th/2), sn = std::sin(th/2);
		u1(q, {c,0},{-sn,0},{sn,0},{c,0}, 'y');
	}
	void rz(int q, double th) {
		double c = std::cos(th/2), sn = std::sin(th/2);
		u1(q, {c,-sn},{0,0},{0,0},{c,sn}, 'z');
	}
	void phase(int q, double phi) {
		u1(q, {1,0},{0,0},{0,0},{std::cos(phi),std::sin(phi)}, 'P');
	}

	void unitary(int q, cd m00, cd m01, cd m10, cd m11) { u1(q, m00,m01,m10,m11, 'U'); }

	void cnot(int ctrl, int tgt) { u1c({ctrl}, tgt, {0,0},{1,0},{1,0},{0,0}, 'X'); }
	void cz(int ctrl, int tgt)   { u1c({ctrl}, tgt, {1,0},{0,0},{0,0},{-1,0}, 'Z'); }
	void toffoli(int c1, int c2, int tgt) { u1c({c1,c2}, tgt, {0,0},{1,0},{1,0},{0,0}, 'X'); }
	void controlled(int ctrl, int q, cd m00, cd m01, cd m10, cd m11) {
		u1c({ctrl}, q, m00,m01,m10,m11, 'U');
	}
	void swap(int a, int b) { cnot(a,b); cnot(b,a); cnot(a,b); }

	/* Mid-circuit measurement. Returns classical bit id for if_result(). */
	int measure(int q) {
		check(q);
		Gate g; g.op = Gate::Op::Measure; g.target = q; g.cbit = next_cbit_++;
		g.cond_bit = cur_cond_; g.label = 'M';
		gates_.push_back(g);
		return g.cbit;
	}

	/* Measure and force |0>. Frees the qubit for reuse. */
	void reset(int q) {
		check(q);
		Gate g; g.op = Gate::Op::Reset; g.target = q;
		g.cond_bit = cur_cond_; g.label = 'R';
		gates_.push_back(g);
	}

	/* Gates recorded inside body execute only when cbit measured 1. */
	template <class F>
	void if_result(int cbit, F body) {
		int prev = cur_cond_;
		cur_cond_ = cbit;
		body();
		cur_cond_ = prev;
	}

	bool has_measurements() const {
		for (auto& g : gates_)
			if (g.op != Gate::Op::U1) return true;
		return false;
	}

	void dump() const {
		std::vector<std::string> row(n_);
		for (auto& g : gates_) {
			int lo = g.target, hi = g.target;
			for (int c : g.controls) { lo = std::min(lo, c); hi = std::max(hi, c); }
			for (int q = 0; q < n_; q++) {
				bool is_ctrl = std::find(g.controls.begin(), g.controls.end(), q) != g.controls.end();
				if (q == g.target)		row[q] += std::string("-") + g.label + "-";
				else if (is_ctrl)		row[q] += "-o-";
				else if (q > lo && q < hi)	row[q] += "-|-";
				else				row[q] += "---";
			}
		}
		for (int q = 0; q < n_; q++)
			printf("q%-2d: %s\n", q, row[q].c_str());
	}

private:
	void check(int q) const {
		if (q < 0 || q >= n_)
			throw Error("qubit " + std::to_string(q) + " out of range");
	}
	void u1(int q, cd a, cd b, cd c, cd d, char label) { u1c({}, q, a, b, c, d, label); }
	void u1c(std::vector<int> ctrls, int q, cd a, cd b, cd c, cd d, char label) {
		check(q);
		for (int ct : ctrls) {
			check(ct);
			if (ct == q)
				throw Error("control equals target");
		}
		Gate g; g.target = q; g.controls = std::move(ctrls);
		g.m[0]=a; g.m[1]=b; g.m[2]=c; g.m[3]=d;
		g.cond_bit = cur_cond_; g.label = label;
		gates_.push_back(g);
	}

	int n_;
	int next_cbit_ = 0;
	int cur_cond_ = -1;
	std::vector<Gate> gates_;
};

/*
 * Static circuit analysis. Costs nothing to run and bounds the cost of
 * every backend before any state memory is allocated.
 *
 * The chi bound uses the fact that a two-qubit gate crossing a cut can
 * at most double the Schmidt rank across it, so chi <= 2^crossings,
 * further capped by 2^min(side sizes).
 */
struct AnalyzeReport {
	int n = 0;
	std::vector<int> sleeping;		/* never touched by any gate */
	std::vector<std::vector<int>> groups;	/* interaction graph components */
	std::vector<int> cut_crossings;		/* 2q gates crossing cut (q, q+1) */
	int max_chi_log2 = 0;
	double mem_dense = 0;
	double mem_groups = 0;
	double mem_mps_bound = 0;
	int split_gates = 0;	/* superposition-creating gates: bounds
				 * nonzero amplitudes by 2^split_gates */

	void print() const {
		printf("=== qubit::analyze ===\n");
		printf("qubits: %d  (sleeping: %zu)\n", n, sleeping.size());
		printf("entanglement groups: %zu ->", groups.size());
		for (auto& g : groups) printf(" [%zu]", g.size());
		printf("\n");
		printf("worst cut: chi <= 2^%d\n", max_chi_log2);
		printf("splitting gates: %d (nonzero amps <= 2^%d)\n",
		       split_gates, split_gates);
		printf("estimated memory:\n");
		printf("  dense : %.1f MB\n", mem_dense  / 1048576.0);
		printf("  groups: %.1f MB\n", mem_groups / 1048576.0);
		printf("  MPS   : %.1f MB (worst-case bound)\n", mem_mps_bound / 1048576.0);
	}
};

inline AnalyzeReport analyze(const Circuit& c) {
	const int n = c.num_qubits();
	AnalyzeReport r; r.n = n;

	std::vector<bool> touched(n, false);
	std::vector<int> parent(n);
	for (int i = 0; i < n; i++) parent[i] = i;
	std::function<int(int)> find = [&](int a) {
		while (parent[a] != a) a = parent[a] = parent[parent[a]];
		return a;
	};

	r.cut_crossings.assign(n > 1 ? n - 1 : 0, 0);
	for (auto& g : c.gates()) {
		touched[g.target] = true;
		for (int ct : g.controls) {
			touched[ct] = true;
			parent[find(ct)] = find(g.target);
			int lo = std::min(ct, g.target), hi = std::max(ct, g.target);
			for (int cut = lo; cut < hi; cut++)
				r.cut_crossings[cut]++;
		}
		/*
		 * A gate can only increase the count of nonzero amplitudes if
		 * it is a genuine superposition (neither diagonal nor
		 * anti-diagonal): H and RX/RY split, while X/Y/Z/S/T/RZ/phase
		 * and controlled permutations (CNOT/CZ) map basis to basis or
		 * only rephase. Each split at most doubles the support, so the
		 * nonzero-amplitude count is bounded by 2^split_gates.
		 */
		if (g.op == Gate::Op::U1) {
			bool diag = g.m[1] == cd(0,0) && g.m[2] == cd(0,0);
			bool anti = g.m[0] == cd(0,0) && g.m[3] == cd(0,0);
			if (!diag && !anti && r.split_gates < 62)
				r.split_gates++;
		}
	}

	int awake = 0;
	for (int q = 0; q < n; q++) {
		if (!touched[q]) r.sleeping.push_back(q);
		else awake++;
	}

	std::map<int, std::vector<int>> comp;
	for (int q = 0; q < n; q++)
		if (touched[q]) comp[find(q)].push_back(q);
	for (auto& [root, members] : comp) r.groups.push_back(members);

	for (int cut = 0; cut + 1 < n; cut++) {
		int bound = std::min({r.cut_crossings[cut], cut + 1, n - cut - 1, 40});
		r.max_chi_log2 = std::max(r.max_chi_log2, bound);
	}

	/* 2^k overflows double at k ~ 1024; saturate for reporting */
	r.mem_dense = awake < 500 ? std::ldexp(8.0, awake) : 1e300;
	r.mem_groups = 0;
	for (auto& g : r.groups)
		r.mem_groups += g.size() < 500 ? std::ldexp(8.0, int(g.size())) : 1e300;
	double chi = std::ldexp(1.0, r.max_chi_log2);
	r.mem_mps_bound = double(awake) * 2.0 * chi * chi * 8.0;
	return r;
}

/*
 * Gate fusion for dense backends. Dense gate cost is one full pass over
 * 2^n amplitudes regardless of the gate, so fewer, fatter gates win:
 * consecutive operations confined to the same one or two qubits collapse
 * into a single 4x4 (or 2x2) matrix. CNOT-RZ-CNOT plus neighboring
 * single-qubit rotations — the QAOA staple — becomes one pass instead
 * of five.
 *
 * Only unconditional U1 gates with at most one control fuse. Measure,
 * Reset, conditionals and multi-control gates act as barriers. Gates on
 * disjoint qubits commute, so flushing out of program order is safe.
 */
inline void mat4_mul(cd* r, const cd* x, const cd* y)
{
	for (int i = 0; i < 4; i++)
		for (int j = 0; j < 4; j++) {
			cd s = 0;
			for (int k = 0; k < 4; k++)
				s += x[i*4 + k] * y[k*4 + j];
			r[i*4 + j] = s;
		}
}

inline std::vector<Gate> fuse_for_dense(const std::vector<Gate>& in, int n)
{
	std::vector<Gate> out;
	out.reserve(in.size());

	struct Pend { bool has = false; cd m[4]; };
	struct Grp {
		bool live = false;
		int hi = -1, lo = -1;
		int ngates = 0;		/* U2 pays only when it replaces >= 2 gates */
		Gate first;		/* emitted verbatim if the group stays at 1 */
		cd M[16];
	};
	std::vector<Pend> pend(n);
	std::vector<Grp> grp;
	std::vector<int> owner(n, -1);

	auto flush_pend = [&](int q) {
		if (!pend[q].has) return;
		Gate g; g.target = q; g.label = 'F';
		std::copy(pend[q].m, pend[q].m + 4, g.m);
		out.push_back(g);
		pend[q].has = false;
	};
	auto flush_grp = [&](int gi) {
		Grp& g = grp[gi];
		if (!g.live) return;
		if (g.ngates == 1) {
			/* a lone controlled gate is cheaper unfused: its pair
			 * kernel touches half the amplitudes a 4x4 pass does */
			out.push_back(g.first);
		} else {
			Gate e; e.op = Gate::Op::U2; e.label = 'F';
			e.target = g.lo; e.target2 = g.hi;
			std::copy(g.M, g.M + 16, e.m4);
			out.push_back(e);
		}
		owner[g.hi] = owner[g.lo] = -1;
		g.live = false;
	};
	auto flush_qubit = [&](int q) {
		if (owner[q] >= 0) flush_grp(owner[q]);
		flush_pend(q);
	};
	auto flush_all = [&] {
		for (int q = 0; q < n; q++) flush_qubit(q);
	};

	/* left-multiply group by (m on qubit q) ⊗ (identity on the other) */
	auto fold1 = [&](Grp& g, int q, const cd* m) {
		cd K[16] = {};
		if (q == g.hi) {
			for (int i = 0; i < 2; i++)
				for (int j = 0; j < 2; j++)
					for (int k = 0; k < 2; k++)
						K[(i*2 + k)*4 + (j*2 + k)] = m[i*2 + j];
		} else {
			for (int i = 0; i < 2; i++)
				for (int j = 0; j < 2; j++)
					for (int k = 0; k < 2; k++)
						K[(k*2 + i)*4 + (k*2 + j)] = m[i*2 + j];
		}
		cd R[16];
		mat4_mul(R, K, g.M);
		std::copy(R, R + 16, g.M);
	};

	/* right-multiply: pending gates ran BEFORE the group, M = M * K */
	auto fold_right = [&](Grp& g, int q, const cd* m) {
		cd K[16] = {};
		if (q == g.hi) {
			for (int i = 0; i < 2; i++)
				for (int j = 0; j < 2; j++)
					for (int k = 0; k < 2; k++)
						K[(i*2 + k)*4 + (j*2 + k)] = m[i*2 + j];
		} else {
			for (int i = 0; i < 2; i++)
				for (int j = 0; j < 2; j++)
					for (int k = 0; k < 2; k++)
						K[(k*2 + i)*4 + (k*2 + j)] = m[i*2 + j];
		}
		cd R[16];
		mat4_mul(R, g.M, K);
		std::copy(R, R + 16, g.M);
	};

	/* controlled-U as a 4x4 in the group's (hi, lo) basis */
	auto build_cu = [](cd* C, int hi, int ctrl, const cd* U) {
		for (int h = 0; h < 2; h++)
			for (int l = 0; l < 2; l++) {
				int idx = h*2 + l;
				int cb = (ctrl == hi) ? h : l;
				if (!cb) { C[idx*4 + idx] = cd(1, 0); continue; }
				for (int tv = 0; tv < 2; tv++) {
					int h2 = (ctrl == hi) ? h : tv;
					int l2 = (ctrl == hi) ? tv : l;
					int src = (ctrl == hi) ? l : h;
					C[(h2*2 + l2)*4 + idx] = U[tv*2 + src];
				}
			}
	};

	for (const Gate& g : in) {
		const bool plain1 = g.op == Gate::Op::U1 && g.controls.empty() &&
				    g.cond_bit < 0;
		const bool plain2 = g.op == Gate::Op::U1 && g.controls.size() == 1 &&
				    g.cond_bit < 0;

		if (plain1) {
			int q = g.target;
			if (owner[q] >= 0) {
				fold1(grp[owner[q]], q, g.m);
				grp[owner[q]].ngates++;
			} else if (pend[q].has) {
				const cd* o = pend[q].m;
				cd r[4] = {
					g.m[0]*o[0] + g.m[1]*o[2], g.m[0]*o[1] + g.m[1]*o[3],
					g.m[2]*o[0] + g.m[3]*o[2], g.m[2]*o[1] + g.m[3]*o[3],
				};
				std::copy(r, r + 4, pend[q].m);
			} else {
				pend[q].has = true;
				std::copy(g.m, g.m + 4, pend[q].m);
			}
			continue;
		}

		if (plain2) {
			int a = g.controls[0], b = g.target;
			int gi = owner[a];
			if (gi >= 0 && gi == owner[b]) {
				cd C[16] = {}, R[16];
				build_cu(C, grp[gi].hi, a, g.m);
				mat4_mul(R, C, grp[gi].M);
				std::copy(R, R + 16, grp[gi].M);
				grp[gi].ngates++;
			} else {
				if (owner[a] >= 0) flush_grp(owner[a]);
				if (owner[b] >= 0) flush_grp(owner[b]);
				Grp ng; ng.live = true; ng.hi = a; ng.lo = b;
				ng.ngates = 1; ng.first = g;
				cd C[16] = {};
				build_cu(C, ng.hi, a, g.m);
				std::copy(C, C + 16, ng.M);
				/* earlier 1q gates on a/b apply first: M = C * (Pa ⊗ Pb) */
				if (pend[a].has) { fold_right(ng, a, pend[a].m); pend[a].has = false; ng.ngates++; }
				if (pend[b].has) { fold_right(ng, b, pend[b].m); pend[b].has = false; ng.ngates++; }
				grp.push_back(ng);
				owner[a] = owner[b] = int(grp.size()) - 1;
			}
			continue;
		}

		/* barrier: order against measurement/conditionals must hold */
		flush_all();
		out.push_back(g);
	}
	flush_all();
	return out;
}

/*
 * Backend contract. Any state representation (dense, grouped, MPS)
 * implements init/apply/measure/reset; the query methods have generic
 * state()-based defaults that structured backends override with
 * representation-aware versions (a grouped state answers amplitude
 * queries as a product over factors without ever materializing 2^n).
 *
 * The RNG lives in the runtime and uniforms are passed in, so identical
 * seeds give identical outcomes on every backend. That property is what
 * makes cross-backend differential testing work.
 */
struct Backend {
	virtual ~Backend() = default;
	virtual const char* name() const = 0;
	virtual void init(int num_qubits) = 0;
	virtual void apply(const Gate& g) = 0;
	virtual int measure(int q, float random_u) = 0;
	virtual void reset(int q, float random_u) = 0;
	virtual double memory_bytes() const = 0;

	/* Full amplitude vector. May be refused if materialization is absurd. */
	virtual std::vector<cf> state() const = 0;

	virtual cf amplitude(uint64_t idx) const { return state()[idx]; }

	/* <Z...Z> for the qubits set in mask. */
	virtual double expectation_z(uint64_t mask) const {
		auto st = state();
		double e = 0;
		for (uint64_t i = 0; i < st.size(); i++) {
			int pc = 0;
			uint64_t x = i & mask;
			while (x) { pc ^= 1; x &= x - 1; }
			e += (pc ? -1.0 : 1.0) * std::norm(st[i]);
		}
		return e;
	}

	/* Draw `shots` basis states from |psi|^2. */
	virtual std::vector<uint64_t> sample(std::mt19937_64& rng, int shots) const {
		auto st = state();
		std::uniform_real_distribution<float> uni(0.0f, 1.0f);
		std::vector<float> us(shots);
		for (auto& u : us) u = uni(rng);
		std::sort(us.begin(), us.end());
		std::vector<uint64_t> out;
		out.reserve(shots);
		double cum = 0;
		size_t ui = 0;
		for (uint64_t i = 0; i < st.size() && ui < us.size(); i++) {
			cum += std::norm(st[i]);
			while (ui < us.size() && us[ui] < cum) {
				out.push_back(i);
				ui++;
			}
		}
		/* float roundoff can leave cum slightly below 1 */
		while (out.size() < size_t(shots))
			out.push_back(st.size() - 1);
		return out;
	}
};

/*
 * Dense CPU backend, templated on amplitude precision (float = 8 B/amp,
 * double = 16 B/amp). Correctness reference for everything else.
 *
 * Gate application enumerates WORKING PAIRS instead of scanning all 2^n
 * indices: a gate with c controls touches 2^(n-1-c) pairs, and the pair
 * index expands to a state index by inserting the target bit (0) and
 * control bits (1) at their positions. No wasted iterations, no branch
 * misses, and the loop parallelizes cleanly with OpenMP.
 */
template <typename R>
class DenseCPUT final : public Backend {
	using C = std::complex<R>;

public:
	const char* name() const override {
		return sizeof(R) == 4 ? "dense-cpu" : "dense-cpu-f64";
	}

	void init(int nq) override {
		if (nq > 30)
			throw Error("dense-cpu: " + std::to_string(nq) + " qubits exceeds host RAM");
		n_ = nq; N_ = uint64_t(1) << nq;
		v_.assign(N_, C(0, 0));
		v_[0] = C(1, 0);
	}

	void apply(const Gate& g) override {
		if (g.op == Gate::Op::U2) {
			if (is_diag4(g.m4)) apply_diag2(g);
			else apply_u2(g);
			return;
		}
		if (g.m[1] == cd(0, 0) && g.m[2] == cd(0, 0)) {
			apply_diag1(g);
			return;
		}
		/* fixed bit positions, ascending: target contributes 0, controls 1 */
		int pos[8], bit[8], np = 0;
		pos[np] = g.target; bit[np++] = 0;
		for (int c : g.controls) { pos[np] = c; bit[np++] = 1; }
		for (int a = 1; a < np; a++)	/* tiny insertion sort */
			for (int b = a; b > 0 && pos[b] < pos[b-1]; b--) {
				std::swap(pos[b], pos[b-1]);
				std::swap(bit[b], bit[b-1]);
			}

		const uint64_t tbit = uint64_t(1) << g.target;
		const int work = int(N_ >> np);
		const C m00(g.m[0]), m01(g.m[1]), m10(g.m[2]), m11(g.m[3]);
		C* v = v_.data();

#pragma omp parallel for if(work > 4096)
		for (int t = 0; t < work; t++) {
			uint64_t i = uint64_t(t);
			for (int k = 0; k < np; k++) {
				uint64_t low = (uint64_t(1) << pos[k]) - 1;
				i = ((i & ~low) << 1) | (uint64_t(bit[k]) << pos[k]) | (i & low);
			}
			uint64_t j = i | tbit;
			C a = v[i], b = v[j];
			v[i] = m00 * a + m01 * b;
			v[j] = m10 * a + m11 * b;
		}
	}

	int measure(int q, float u) override {
		const uint64_t bit = uint64_t(1) << q;
		const int N = int(N_);
		const C* v = v_.data();
		double p1 = 0;
#pragma omp parallel for reduction(+:p1) if(N > 8192)
		for (int i = 0; i < N; i++)
			if (uint64_t(i) & bit) p1 += std::norm(v[i]);
		int outcome = (u < p1) ? 1 : 0;
		collapse(q, outcome, outcome ? p1 : 1.0 - p1);
		return outcome;
	}

	void reset(int q, float u) override {
		if (measure(q, u) == 1) {
			Gate x; x.target = q;
			x.m[0]={0,0}; x.m[1]={1,0}; x.m[2]={1,0}; x.m[3]={0,0};
			apply(x);
		}
	}

	std::vector<cf> state() const override {
		std::vector<cf> out(N_);
		for (uint64_t i = 0; i < N_; i++)
			out[i] = cf(float(v_[i].real()), float(v_[i].imag()));
		return out;
	}
	cf amplitude(uint64_t idx) const override {
		return cf(float(v_[idx].real()), float(v_[idx].imag()));
	}
	double memory_bytes() const override { return double(N_) * sizeof(C); }

	/*
	 * Raw dense amplitude buffer, at the backend's own precision (no
	 * float narrowing, unlike state()). For algorithms that drive the
	 * backend directly — e.g. qtrain's adjoint differentiation, which
	 * evolves the state with apply() and reads bra/ket amplitudes here
	 * instead of maintaining a second gate kernel. Dense backend only.
	 */
	std::vector<C>& buffer() { return v_; }
	const std::vector<C>& buffer() const { return v_; }

private:
	static bool is_diag4(const cd* m) {
		for (int i = 0; i < 4; i++)
			for (int j = 0; j < 4; j++)
				if (i != j && m[i*4 + j] != cd(0, 0))
					return false;
		return true;
	}

	/*
	 * Diagonal gates (Z, S, T, RZ, phase and their fusions) touch no
	 * pair partner: every amplitude just picks up a phase chosen by
	 * its own bits. Purely sequential traffic, no gather.
	 */
	void apply_diag1(const Gate& g) {
		const uint64_t tbit = uint64_t(1) << g.target;
		uint64_t cmask = 0;
		for (int c : g.controls) cmask |= uint64_t(1) << c;
		const C d0(g.m[0]), d1(g.m[3]);
		const int N = int(N_);
		C* v = v_.data();
#pragma omp parallel for if(N > 8192)
		for (int i = 0; i < N; i++) {
			if ((uint64_t(i) & cmask) != cmask) continue;
			v[i] *= (uint64_t(i) & tbit) ? d1 : d0;
		}
	}

	void apply_diag2(const Gate& g) {
		const uint64_t hbit = uint64_t(1) << g.target2;
		const uint64_t lbit = uint64_t(1) << g.target;
		const C d[4] = { C(g.m4[0]), C(g.m4[5]), C(g.m4[10]), C(g.m4[15]) };
		const int N = int(N_);
		C* v = v_.data();
#pragma omp parallel for if(N > 8192)
		for (int i = 0; i < N; i++) {
			int loc = (int((uint64_t(i) & hbit) != 0) << 1) |
				  int((uint64_t(i) & lbit) != 0);
			v[i] *= d[loc];
		}
	}

	/* fused 4x4 over qubits (hi=target2, lo=target); one pass, 4 amps/thread */
	void apply_u2(const Gate& g) {
		const int hi = g.target2, lo = g.target;
		const int p0 = std::min(hi, lo), p1 = std::max(hi, lo);
		const uint64_t hbit = uint64_t(1) << hi, lbit = uint64_t(1) << lo;
		const int work = int(N_ >> 2);
		C M[16];
		for (int k = 0; k < 16; k++) M[k] = C(g.m4[k]);
		C* v = v_.data();

#pragma omp parallel for if(work > 2048)
		for (int t = 0; t < work; t++) {
			uint64_t i = uint64_t(t);
			uint64_t low0 = (uint64_t(1) << p0) - 1;
			i = ((i & ~low0) << 1) | (i & low0);
			uint64_t low1 = (uint64_t(1) << p1) - 1;
			i = ((i & ~low1) << 1) | (i & low1);
			const uint64_t i00 = i;
			const uint64_t i01 = i00 | lbit;
			const uint64_t i10 = i00 | hbit;
			const uint64_t i11 = i00 | hbit | lbit;
			C x0 = v[i00], x1 = v[i01], x2 = v[i10], x3 = v[i11];
			v[i00] = M[0]*x0  + M[1]*x1  + M[2]*x2  + M[3]*x3;
			v[i01] = M[4]*x0  + M[5]*x1  + M[6]*x2  + M[7]*x3;
			v[i10] = M[8]*x0  + M[9]*x1  + M[10]*x2 + M[11]*x3;
			v[i11] = M[12]*x0 + M[13]*x1 + M[14]*x2 + M[15]*x3;
		}
	}

	void collapse(int q, int outcome, double p) {
		const uint64_t bit = uint64_t(1) << q;
		const R inv = R(1) / R(std::sqrt(std::max(p, 1e-30)));
		const int N = int(N_);
		C* v = v_.data();
#pragma omp parallel for if(N > 8192)
		for (int i = 0; i < N; i++) {
			bool is1 = (uint64_t(i) & bit) != 0;
			if (is1 != (outcome == 1)) v[i] = C(0, 0);
			else v[i] *= inv;
		}
	}

	int n_ = 0;
	uint64_t N_ = 0;
	std::vector<C> v_;
};

using DenseCPU = DenseCPUT<float>;
using DenseCPU64 = DenseCPUT<double>;

/*
 * Grouped (factorized) CPU backend.
 *
 * The state is a product of independent factors, each a small dense
 * vector over the qubits entangled so far. Memory is the SUM of factor
 * sizes, not 2^n. Factors merge when a multi-qubit gate spans them and
 * split when measurement disentangles a qubit. A 1000-qubit circuit
 * costs whatever its largest entangled cluster costs.
 *
 * Splitting is done only on measurement, where it is exact and O(2^k).
 * Detecting that a factor happens to be separable after a unitary would
 * need an SVD rank check per gate; deliberately not done here.
 */
class GroupsCPU final : public Backend {
	struct Factor {
		std::vector<int> qubits;	/* position in list = local bit */
		std::vector<cf> v;		/* size 1 << qubits.size() */
		bool dead = false;
	};

public:
	const char* name() const override { return "groups-cpu"; }

	void init(int nq) override {
		n_ = nq;
		factors_.clear();
		factors_.reserve(nq);
		where_.resize(nq);
		for (int q = 0; q < nq; q++) {
			Factor f;
			f.qubits = {q};
			f.v = {cf(1,0), cf(0,0)};
			where_[q] = int(factors_.size());
			factors_.push_back(std::move(f));
		}
	}

	void apply(const Gate& g) override {
		/*
		 * Controls sitting in a definite basis state are classical:
		 * |0> makes the gate a no-op, |1> just drops the control.
		 * This keeps Toffoli chains over ancillas from inflating
		 * factors that never actually entangle.
		 */
		std::vector<int> ctrls;
		for (int c : g.controls) {
			const Factor& f = factors_[where_[c]];
			if (f.qubits.size() == 1) {
				float p0 = std::norm(f.v[0]);
				if (p0 > 1.0f - 1e-6f) return;
				if (p0 < 1e-6f) continue;
			}
			ctrls.push_back(c);
		}

		std::vector<int> involved = ctrls;
		involved.push_back(g.target);
		int fi = merge_all(involved);
		Factor& f = factors_[fi];

		const uint64_t tbit = uint64_t(1) << local_pos(f, g.target);
		uint64_t cmask = 0;
		for (int c : ctrls) cmask |= uint64_t(1) << local_pos(f, c);

		const cf m00 = to_cf(g.m[0]), m01 = to_cf(g.m[1]);
		const cf m10 = to_cf(g.m[2]), m11 = to_cf(g.m[3]);
		const uint64_t N = f.v.size();
		for (uint64_t i = 0; i < N; i++) {
			if (i & tbit) continue;
			if ((i & cmask) != cmask) continue;
			uint64_t j = i | tbit;
			cf a = f.v[i], b = f.v[j];
			f.v[i] = m00 * a + m01 * b;
			f.v[j] = m10 * a + m11 * b;
		}
	}

	/*
	 * Measurement disentangles: collapse inside the factor, then peel
	 * the measured qubit off into its own singleton. The factor halves.
	 */
	int measure(int q, float u) override {
		int fi = where_[q];
		Factor& f = factors_[fi];
		int pos = local_pos(f, q);
		const uint64_t bit = uint64_t(1) << pos;

		double p1 = 0;
		for (uint64_t i = 0; i < f.v.size(); i++)
			if (i & bit) p1 += std::norm(f.v[i]);
		int outcome = (u < p1) ? 1 : 0;
		double p = outcome ? p1 : 1.0 - p1;
		float inv = 1.0f / std::sqrt(float(std::max(p, 1e-30)));

		if (f.qubits.size() == 1) {
			f.v[outcome] = cf(1, 0);
			f.v[1 - outcome] = cf(0, 0);
			return outcome;
		}

		/* project on outcome, drop bit `pos` from the index space */
		std::vector<cf> nv(f.v.size() >> 1);
		const uint64_t low = bit - 1;
		for (uint64_t i = 0; i < nv.size(); i++) {
			uint64_t src = ((i & ~low) << 1) | (i & low) | (outcome ? bit : 0);
			nv[i] = f.v[src] * inv;
		}
		f.v = std::move(nv);
		f.qubits.erase(f.qubits.begin() + pos);

		Factor s;
		s.qubits = {q};
		s.v = {cf(outcome ? 0.f : 1.f, 0), cf(outcome ? 1.f : 0.f, 0)};
		where_[q] = int(factors_.size());
		factors_.push_back(std::move(s));
		return outcome;
	}

	void reset(int q, float u) override {
		measure(q, u);
		Factor& f = factors_[where_[q]];
		f.v = {cf(1,0), cf(0,0)};
	}

	std::vector<cf> state() const override {
		if (n_ > 26)
			throw Error("groups-cpu: refusing to materialize 2^" +
				    std::to_string(n_) + " amplitudes; use counts/prob/expectation");
		std::vector<cf> full(uint64_t(1) << n_);
		for (uint64_t i = 0; i < full.size(); i++)
			full[i] = amplitude(i);
		return full;
	}

	/* Product over factors: each contributes its local amplitude. */
	cf amplitude(uint64_t idx) const override {
		cf a(1, 0);
		for (const Factor& f : factors_) {
			if (f.dead) continue;
			uint64_t local = 0;
			for (size_t p = 0; p < f.qubits.size(); p++)
				if ((idx >> f.qubits[p]) & 1)
					local |= uint64_t(1) << p;
			a *= f.v[local];
			if (a == cf(0, 0)) return a;
		}
		return a;
	}

	double expectation_z(uint64_t mask) const override {
		double e = 1.0;
		for (const Factor& f : factors_) {
			if (f.dead) continue;
			uint64_t lmask = 0;
			for (size_t p = 0; p < f.qubits.size(); p++)
				if ((mask >> f.qubits[p]) & 1)
					lmask |= uint64_t(1) << p;
			if (!lmask) continue;
			double fe = 0;
			for (uint64_t i = 0; i < f.v.size(); i++) {
				int pc = 0;
				uint64_t x = i & lmask;
				while (x) { pc ^= 1; x &= x - 1; }
				fe += (pc ? -1.0 : 1.0) * std::norm(f.v[i]);
			}
			e *= fe;
		}
		return e;
	}

	/* Factors are independent: sample each one and OR the bits. */
	std::vector<uint64_t> sample(std::mt19937_64& rng, int shots) const override {
		std::uniform_real_distribution<float> uni(0.0f, 1.0f);
		std::vector<uint64_t> out(shots, 0);
		for (const Factor& f : factors_) {
			if (f.dead) continue;
			for (int s = 0; s < shots; s++) {
				float u = uni(rng);
				double cum = 0;
				uint64_t local = f.v.size() - 1;
				for (uint64_t i = 0; i < f.v.size(); i++) {
					cum += std::norm(f.v[i]);
					if (u < cum) { local = i; break; }
				}
				for (size_t p = 0; p < f.qubits.size(); p++)
					if ((local >> p) & 1)
						out[s] |= uint64_t(1) << f.qubits[p];
			}
		}
		return out;
	}

	double memory_bytes() const override {
		double b = 0;
		for (const Factor& f : factors_)
			if (!f.dead) b += double(f.v.size()) * sizeof(cf);
		return b;
	}

	/* Largest live factor, in qubits. This is what run() reports. */
	int peak_factor_qubits() const {
		int m = 0;
		for (const Factor& f : factors_)
			if (!f.dead) m = std::max(m, int(f.qubits.size()));
		return m;
	}

private:
	static int local_pos(const Factor& f, int q) {
		for (size_t p = 0; p < f.qubits.size(); p++)
			if (f.qubits[p] == q) return int(p);
		throw Error("internal: qubit not in factor");
	}

	/* Tensor all factors containing `qs` into one. Order-preserving. */
	int merge_all(const std::vector<int>& qs) {
		int fa = where_[qs[0]];
		for (size_t k = 1; k < qs.size(); k++) {
			int fb = where_[qs[k]];
			if (fb == fa) continue;
			fa = merge(fa, fb);
		}
		return fa;
	}

	int merge(int ia, int ib) {
		Factor& A = factors_[ia];
		Factor& B = factors_[ib];
		if (A.qubits.size() + B.qubits.size() > 40)
			throw Error(
				"qubit: entangled cluster would exceed 40 qubits "
				"(2^40 amplitudes); circuit entangles too much for groups-cpu.\n"
				"  options: dense GPU backend | future blocks/MPS backends");
		Factor C;
		C.qubits = A.qubits;
		C.qubits.insert(C.qubits.end(), B.qubits.begin(), B.qubits.end());
		C.v.resize(A.v.size() * B.v.size());
		const int ka = int(A.qubits.size());
		for (uint64_t ib2 = 0; ib2 < B.v.size(); ib2++)
			for (uint64_t ia2 = 0; ia2 < A.v.size(); ia2++)
				C.v[(ib2 << ka) | ia2] = A.v[ia2] * B.v[ib2];
		A.dead = true; A.v.clear(); A.v.shrink_to_fit(); A.qubits.clear();
		B.dead = true; B.v.clear(); B.v.shrink_to_fit(); B.qubits.clear();
		for (int q : C.qubits) where_[q] = int(factors_.size());
		factors_.push_back(std::move(C));
		return int(factors_.size()) - 1;
	}

	int n_ = 0;
	std::vector<Factor> factors_;
	std::vector<int> where_;
};

/*
 * Tiered-block CPU backend: correctness prototype for the GPU version.
 *
 * The dense vector is split into fixed-size blocks. Each block is either
 * ZERO (no storage) or FULL (dense complex64). Blocks demote to ZERO
 * when their norm falls below a threshold — this catches both never-
 * written regions and destructive interference (Grover's diffuser zeroes
 * large regions mid-circuit). The planned COMPRESSED tier (block-scaled
 * short mantissas, decompressed in registers) only pays off on GPU and
 * is not implemented here.
 *
 * Gates on qubit k with 2^k < blocksize stay inside blocks; higher
 * qubits pair block b with block b ^ (1 << (k - bshift)). A ZERO/FULL
 * pair can turn both blocks nonzero, so the ZERO side materializes
 * before the update and both sides re-tier after.
 */


class BlocksCPU final : public Backend {
	/* norm^2 below this is indistinguishable from float roundoff */
	static constexpr double kZeroEps = 1e-24;

	/*
	 * Three tiers per block:
	 *   ZERO       - no storage
	 *   COMPRESSED - block-scaled int16 pairs, 4 B/amplitude
	 *   FULL       - complex64, 8 B/amplitude
	 * A block is COMPRESSED only if the quantization error fits the
	 * remaining global error budget.
	 *
	 * The budget is an L2 NORM, not a squared norm: an error vector
	 * delta_i injected at step i is carried through every later
	 * unitary with its norm intact, so injections compose by the
	 * triangle inequality, ||delta_total|| <= sum ||delta_i||.
	 * Accounting in squared norms understates the worst case by up to
	 * the number of injections (measured: ~600x on random circuits).
	 */
	struct Block {
		std::vector<cf> v;		/* FULL storage */
		std::vector<int16_t> q;		/* COMPRESSED storage: re,im pairs */
		float scale = 0;
		bool zero() const { return v.empty() && q.empty(); }
		bool compressed() const { return !q.empty(); }
	};

public:
	explicit BlocksCPU(int block_shift = 12, double l2_budget = 0.0)
		: bshift_(block_shift), budget_(l2_budget) {}

	const char* name() const override { return "blocks-cpu"; }

	void init(int nq) override {
		if (nq > 40)
			throw Error("blocks-cpu: " + std::to_string(nq) + " qubits is beyond any RAM");
		n_ = nq;
		if (bshift_ > n_) bshift_ = n_;
		B_ = uint64_t(1) << bshift_;
		blocks_.assign(uint64_t(1) << (n_ - bshift_), {});
		blocks_[0].v.assign(B_, cf(0, 0));
		blocks_[0].v[0] = cf(1, 0);
		err_used_ = 0;
	}

	/* Accumulated L2-norm error bound D = sum of injection norms. */
	double error_l2() const { return err_used_; }

	/*
	 * Rigorous fidelity lower bound. With ||delta|| <= D < 1:
	 * <psi|psi~> >= (1 - D) / (1 + D) after renormalization, hence
	 * F = |<psi|psi~>|^2 >= (1-D)^2 / (1+D)^2.
	 */
	double fidelity_bound() const {
		if (err_used_ >= 1.0) return 0.0;
		double r = (1.0 - err_used_) / (1.0 + err_used_);
		return r * r;
	}

	void apply(const Gate& g) override {
		uint64_t cmask = 0;
		for (int c : g.controls) cmask |= uint64_t(1) << c;
		if (g.target < bshift_)
			apply_intra(g, cmask);
		else
			apply_inter(g, cmask);
	}

	int measure(int q, float u) override {
		const uint64_t bit = uint64_t(1) << q;
		double p1 = 0;
		for (uint64_t b = 0; b < blocks_.size(); b++) {
			if (blocks_[b].zero()) continue;
			promote(blocks_[b]);
			uint64_t base = b << bshift_;
			for (uint64_t i = 0; i < B_; i++)
				if ((base | i) & bit)
					p1 += std::norm(blocks_[b].v[i]);
		}
		int outcome = (u < p1) ? 1 : 0;
		double p = outcome ? p1 : 1.0 - p1;
		float inv = 1.0f / std::sqrt(float(std::max(p, 1e-30)));
		for (uint64_t b = 0; b < blocks_.size(); b++) {
			if (blocks_[b].zero()) continue;
			uint64_t base = b << bshift_;
			for (uint64_t i = 0; i < B_; i++) {
				bool is1 = ((base | i) & bit) != 0;
				if (is1 != (outcome == 1)) blocks_[b].v[i] = cf(0, 0);
				else blocks_[b].v[i] *= inv;
			}
			retier(b);
		}
		return outcome;
	}

	void reset(int q, float u) override {
		if (measure(q, u) == 1) {
			Gate x; x.target = q;
			x.m[0]={0,0}; x.m[1]={1,0}; x.m[2]={1,0}; x.m[3]={0,0};
			apply(x);
		}
	}

	std::vector<cf> state() const override {
		if (n_ > 26)
			throw Error("blocks-cpu: refusing to materialize 2^" +
				    std::to_string(n_) + " amplitudes");
		std::vector<cf> full(uint64_t(1) << n_, cf(0, 0));
		for (uint64_t b = 0; b < blocks_.size(); b++) {
			if (blocks_[b].zero()) continue;
			for (uint64_t i = 0; i < B_; i++)
				full[(b << bshift_) | i] = read(blocks_[b], i);
		}
		return full;
	}

	cf amplitude(uint64_t idx) const override {
		const Block& b = blocks_[idx >> bshift_];
		if (b.zero()) return cf(0, 0);
		return read(b, idx & (B_ - 1));
	}

	double memory_bytes() const override {
		double bytes = 0;
		for (auto& b : blocks_) {
			if (b.compressed()) bytes += double(B_) * 2 * sizeof(int16_t);
			else if (!b.zero()) bytes += double(B_) * sizeof(cf);
		}
		return bytes;
	}

	int full_blocks() const {
		int k = 0;
		for (auto& b : blocks_) if (!b.zero()) k++;
		return k;
	}
	int total_blocks() const { return int(blocks_.size()); }

private:
	void apply_intra(const Gate& g, uint64_t cmask) {
		const uint64_t tbit = uint64_t(1) << g.target;
		for (uint64_t b = 0; b < blocks_.size(); b++) {
			Block& blk = blocks_[b];
			if (blk.zero()) continue;	/* unitary on zeros is zeros */
			const uint64_t base = b << bshift_;
			/* controls in the high bits are constant per block */
			if ((base & cmask) != (cmask & ~(B_ - 1))) continue;
			promote(blk);
			const cf m00 = to_cf(g.m[0]), m01 = to_cf(g.m[1]);
			const cf m10 = to_cf(g.m[2]), m11 = to_cf(g.m[3]);
			const uint64_t lmask = cmask & (B_ - 1);
			for (uint64_t i = 0; i < B_; i++) {
				if (i & tbit) continue;
				if ((i & lmask) != lmask) continue;
				uint64_t j = i | tbit;
				cf a = blk.v[i], bv = blk.v[j];
				blk.v[i] = m00 * a + m01 * bv;
				blk.v[j] = m10 * a + m11 * bv;
			}
			retier(b);
		}
	}

	void apply_inter(const Gate& g, uint64_t cmask) {
		const uint64_t bbit = uint64_t(1) << (g.target - bshift_);
		for (uint64_t b = 0; b < blocks_.size(); b++) {
			if (b & bbit) continue;		/* owner: target block-bit 0 */
			Block& b0 = blocks_[b];
			Block& b1 = blocks_[b | bbit];
			if (b0.zero() && b1.zero()) continue;
			materialize(b0);
			materialize(b1);
			promote(b0);
			promote(b1);
			const cf m00 = to_cf(g.m[0]), m01 = to_cf(g.m[1]);
			const cf m10 = to_cf(g.m[2]), m11 = to_cf(g.m[3]);
			const uint64_t base0 = b << bshift_;
			const uint64_t base1 = (b | bbit) << bshift_;
			for (uint64_t i = 0; i < B_; i++) {
				if (((base0 | i) & cmask) != cmask) continue;
				cf a = b0.v[i], bv = b1.v[i];
				b0.v[i] = m00 * a + m01 * bv;
				b1.v[i] = m10 * a + m11 * bv;
			}
			(void)base1;
			retier(b);
			retier(b | bbit);
		}
	}

	void materialize(Block& b) {
		if (b.zero()) b.v.assign(B_, cf(0, 0));
	}

	static cf read(const Block& b, uint64_t i) {
		if (b.compressed())
			return cf(b.q[2*i] * b.scale, b.q[2*i + 1] * b.scale);
		return b.v[i];
	}

	void promote(Block& b) {
		if (!b.compressed()) return;
		b.v.resize(B_);
		for (uint64_t i = 0; i < B_; i++)
			b.v[i] = cf(b.q[2*i] * b.scale, b.q[2*i + 1] * b.scale);
		b.q.clear();
		b.q.shrink_to_fit();
	}

	/*
	 * After a block is touched: dead blocks demote to ZERO; live ones
	 * demote to COMPRESSED only when the exactly-measured quantization
	 * error still fits the global budget. Every requantization spends
	 * budget, so a hot block that keeps getting promoted naturally
	 * ends up staying FULL once the budget runs dry — no heuristics.
	 */
	void retier(uint64_t bi) {
		Block& b = blocks_[bi];
		if (b.zero() || b.compressed()) return;
		double norm = 0;
		float mx = 0;
		for (auto& a : b.v) {
			norm += std::norm(a);
			mx = std::max({mx, std::abs(a.real()), std::abs(a.imag())});
		}
		if (norm < kZeroEps) {
			b.v.clear();
			b.v.shrink_to_fit();
			return;
		}
		if (budget_ <= 0 || mx == 0)
			return;
		const float scale = mx / 32767.0f;
		double err = 0;
		std::vector<int16_t> q(2 * B_);
		for (uint64_t i = 0; i < B_; i++) {
			float re = b.v[i].real(), im = b.v[i].imag();
			int16_t qr = int16_t(std::lround(re / scale));
			int16_t qi = int16_t(std::lround(im / scale));
			q[2*i] = qr; q[2*i + 1] = qi;
			float dre = re - qr * scale, dim = im - qi * scale;
			err += double(dre) * dre + double(dim) * dim;
		}
		double err_norm = std::sqrt(err);
		if (err_used_ + err_norm > budget_)
			return;		/* budget exhausted: stay FULL, stay exact */
		err_used_ += err_norm;
		b.q = std::move(q);
		b.scale = scale;
		b.v.clear();
		b.v.shrink_to_fit();
	}

	int n_ = 0;
	int bshift_;
	double budget_;
	double err_used_ = 0;
	uint64_t B_ = 0;
	std::vector<Block> blocks_;
};

#ifdef QUBIT_CUDA
/* Implemented in src/qubit_gpu.cu; requires nvcc and -DQUBIT_CUDA. */
std::unique_ptr<Backend> make_dense_gpu();
std::shared_ptr<Backend> make_blocks_gpu(int block_shift, double l2_budget);
double gpu_free_vram_bytes();
#endif

/*
 * The only device knob a user needs: Auto prefers the GPU when the
 * library was built with CUDA and falls back to the CPU otherwise.
 * WHICH state representation runs (dense, grouped, tiered blocks) is
 * the planner's job, not the user's.
 */
enum class Device { Auto, CPU, GPU };

/* Testing/benchmark override: pin one concrete backend. Not part of
 * the normal API surface — prefer Device. */
enum class BackendSel { Auto, DenseCPU_, DenseGPU_, GroupsCPU_ };

/*
 * Invert the fidelity bound F >= (1-D)^2/(1+D)^2 to get the largest
 * compression budget D that still guarantees the user's target.
 */
inline double budget_from_fidelity(double f) {
	if (f >= 1.0) return 0.0;
	double s = std::sqrt(std::max(f, 0.0));
	return (1.0 - s) / (1.0 + s);
}

/* F32: 8 B/amplitude, GPU-friendly. F64: 16 B/amplitude, one fewer qubit
 * per budget, for when accumulated float roundoff matters. */
enum class Precision { F32, F64 };

/*
 * Everything a normal user touches. Declare qubits, add gates, run:
 *
 *   qubit::Circuit c(30);
 *   c.h(0); c.cnot(0, 1);
 *   auto r = qubit::run(c);
 *
 * `device` picks the hardware; `fidelity` < 1 lets the planner trade
 * exactly-bounded precision for capacity. The rest have sane defaults.
 * Fields below the "testing" line pin implementation details and exist
 * for tests and benchmarks, not applications.
 */
struct RunOptions {
	Device device = Device::Auto;
	double fidelity = 1.0;		/* guaranteed lower bound when < 1 */
	int shots = 1024;
	uint64_t seed = 0xC0FFEE;
	Precision precision = Precision::F32;
	double cpu_mem_budget = 2.0 * 1024 * 1024 * 1024;

	/* --- testing / benchmarking only --- */
	BackendSel backend = BackendSel::Auto;	/* pin a concrete backend */
	bool fuse = true;			/* gate fusion on dense backends */
	std::function<std::shared_ptr<Backend>()> custom_backend;
};

struct RunStats {
	std::string backend;
	double memory_peak_bytes = 0;
	double time_ms = 0;
	int qubits_total = 0, qubits_live = 0;
};

class Result {
public:
	const std::map<std::string, int>& counts() const { return counts_; }

	void print_counts() const {
		for (auto& [k, v] : counts_)
			printf("|%s>  x%d  (%.3f)\n", k.c_str(), v, double(v) / shots_);
	}

	/* idx addresses the ORIGINAL qubit numbering, sleeping qubits included. */
	double prob(uint64_t idx) const {
		return double(std::norm(amplitude(idx)));
	}

	/*
	 * Exact only for measurement-free circuits; with mid-circuit
	 * measurement this queries the collapsed state of the last shot.
	 */
	cf amplitude(uint64_t idx) const {
		uint64_t compact = 0;
		for (int q = 0; q < stats.qubits_total; q++) {
			bool bit = (idx >> q) & 1;
			if (live_map_[q] < 0) {
				/* sleeping qubit is |0> by construction */
				if (bit) return cf(0, 0);
				continue;
			}
			if (bit) compact |= uint64_t(1) << live_map_[q];
		}
		return be_->amplitude(compact);
	}

	/* <Z_q0 Z_q1 ...>. Sleeping qubits contribute +1 (Z|0> = |0>). */
	double expectation_z(const std::vector<int>& qubits) const {
		uint64_t mask = 0;
		for (int q : qubits) {
			if (live_map_[q] < 0) continue;
			mask |= uint64_t(1) << live_map_[q];
		}
		if (!mask) return 1.0;
		return be_->expectation_z(mask);
	}

	RunStats stats;

private:
	friend Result run(const Circuit&, const RunOptions&);
	std::map<std::string, int> counts_;
	std::shared_ptr<Backend> be_;	/* the final state, in whatever representation */
	std::vector<int> live_map_;	/* original qubit -> compact index, -1 = sleeping */
	int shots_ = 1;
};

inline Result run(const Circuit& c, const RunOptions& opt = {}) {
	auto t0 = std::chrono::steady_clock::now();
	AnalyzeReport rep = analyze(c);
	const int n = c.num_qubits();

	/*
	 * Sleeping qubits stay factored out as |0>: simulate only touched
	 * qubits in a compact index space and translate at the API surface.
	 * Memory cost is 2^live, not 2^declared.
	 */
	std::vector<int> live_map(n, -1);
	int live = 0;
	for (int q = 0; q < n; q++) {
		bool asleep = std::find(rep.sleeping.begin(), rep.sleeping.end(), q) != rep.sleeping.end();
		if (!asleep) live_map[q] = live++;
	}
	if (live == 0) live_map[0] = 0, live = 1;

	const bool f64 = opt.precision == Precision::F64;
	const double amp_bytes = f64 ? 16.0 : 8.0;
	double need_dense = live < 500 ? std::ldexp(amp_bytes, live) : 1e300;

	auto make_dense_cpu = [&]() -> std::shared_ptr<Backend> {
		if (f64) return std::make_shared<DenseCPU64>();
		return std::make_shared<DenseCPU>();
	};

	std::shared_ptr<Backend> be;
	if (opt.custom_backend)
		be = opt.custom_backend();
	else switch (opt.backend) {
	case BackendSel::DenseCPU_:
		be = make_dense_cpu();
		break;
	case BackendSel::GroupsCPU_:
		be = std::make_shared<GroupsCPU>();
		break;
	case BackendSel::DenseGPU_:
#ifdef QUBIT_CUDA
		if (f64)
			throw Error("qubit: F64 on GPU not implemented yet; use F32 or dense-cpu");
		be = std::shared_ptr<Backend>(make_dense_gpu().release());
		break;
#else
		throw Error("qubit: GPU backend requested but not compiled in (nvcc -DQUBIT_CUDA with src/backend_gpu.cu)");
#endif
	case BackendSel::Auto: {
		/*
		 * The planner's ladder, per device. Dense wins whenever it
		 * fits — fastest exact path, no bookkeeping. Groups turns
		 * "does not fit" into "fits if the entangled clusters are
		 * small" (the analyzer's component sum bounds its peak
		 * exactly). Tiered blocks are the last resort: exact via the
		 * ZERO tier when the state is block-sparse, and lossy-with-
		 * guarantee when the user granted fidelity headroom — the
		 * compression budget is derived by inverting the fidelity
		 * bound, so the contract holds by construction.
		 */
		const double budget = budget_from_fidelity(opt.fidelity);
		const bool want_gpu = opt.device != Device::CPU;
		const bool want_cpu = opt.device != Device::GPU;

		/*
		 * Factorization fast path, device-independent: if the
		 * interaction graph splits into small disjoint groups, the
		 * factorized state is far cheaper than any dense pass on any
		 * device (n/2 Bell pairs cost 2n amplitudes, not 2^n). Take it
		 * whenever the group footprint is a small fraction of dense and
		 * fits comfortably in host memory. A single large entangled
		 * group (GHZ chain, QFT) does not trigger this and falls
		 * through to the GPU/dense ladder below.
		 */
		if (rep.mem_groups * 8.0 <= need_dense &&
		    rep.mem_groups <= opt.cpu_mem_budget * 0.5) {
			be = std::make_shared<GroupsCPU>();
			break;
		}
#ifndef QUBIT_CUDA
		if (opt.device == Device::GPU)
			throw Error("qubit: Device::GPU requested but the library was "
				    "built without CUDA (compile src/qubit_gpu.cu with "
				    "nvcc -DQUBIT_CUDA)");
#else
		if (want_gpu && !f64) {
			/*
			 * Provably-sparse fast path: if the nonzero-amplitude
			 * bound (2^split_gates) makes the tiered representation
			 * far smaller than dense, route to blocks-gpu even when
			 * dense fits. The ZERO tier then skips the empty blocks,
			 * turning e.g. a GHZ chain from a full 2^n pass into a
			 * few live blocks. Worst-case block memory is bounded
			 * below, so this never risks an out-of-memory blowup:
			 * circuits that are not actually sparse (QFT, random)
			 * hit the cap and fall through to dense.
			 */
			const int kBshift = 16;
			if (live > kBshift) {
				int nz_blocks_log2 = std::min(rep.split_gates, live - kBshift);
				double sparse_bytes = std::ldexp(8.0, nz_blocks_log2 + kBshift);
				if (sparse_bytes * 8.0 <= need_dense &&
				    sparse_bytes <= gpu_free_vram_bytes() * 0.5) {
					be = make_blocks_gpu(kBshift, budget);
					break;
				}
			}
			if (need_dense <= gpu_free_vram_bytes() * 0.9) {
				be = std::shared_ptr<Backend>(make_dense_gpu().release());
				break;
			}
			/* dense does not fit VRAM: tiered blocks, ZERO-exact or
			 * budget-compressed, still entirely GPU-resident */
			if (opt.device == Device::GPU || !want_cpu ||
			    rep.mem_groups > opt.cpu_mem_budget) {
				be = make_blocks_gpu(16, budget);
				break;
			}
		}
#endif
		if (want_cpu) {
			if (need_dense <= opt.cpu_mem_budget) {
				be = make_dense_cpu();
				break;
			}
			if (rep.mem_groups <= opt.cpu_mem_budget) {
				be = std::make_shared<GroupsCPU>();
				break;
			}
			/* last resort: tiered blocks — exact via ZERO tier on
			 * block-sparse states, compressed when budget > 0.
			 * Guarded so a truly dense huge state errors instead of
			 * thrashing host RAM. */
			if (budget > 0 || live <= 32) {
				be = std::make_shared<BlocksCPU>(12, budget);
				break;
			}
		}
		throw Error(
			"qubit: no backend fits.\n"
			"  live qubits: " + std::to_string(live) +
			" -> dense needs " + std::to_string(need_dense / 1048576.0) + " MB,"
			" groups worst-case " + std::to_string(rep.mem_groups / 1048576.0) + " MB,"
			" budget " + std::to_string(opt.cpu_mem_budget / 1048576.0) + " MB.\n"
			"  analyzer: groups=" + std::to_string(rep.groups.size()) +
			", worst chi<=2^" + std::to_string(rep.max_chi_log2) + ".\n"
			"  options: raise cpu_mem_budget | GPU build (nvcc -DQUBIT_CUDA)\n"
			"           | grant fidelity headroom (fidelity < 1 enables\n"
			"             compressed tiers) | future MPS backend (chi bound\n"
			"             suggests " + std::to_string(rep.mem_mps_bound / 1048576.0) + " MB)");
	}
	}

	std::mt19937_64 rng(opt.seed);
	std::uniform_real_distribution<float> uni(0.0f, 1.0f);

	/*
	 * Measurement-free circuits are deterministic: evolve once, sample
	 * the final distribution. Mid-circuit measurement branches the
	 * evolution, so each shot needs its own run.
	 */
	const bool per_shot = c.has_measurements();
	const int runs = per_shot ? opt.shots : 1;

	Result out;
	out.live_map_ = live_map;
	out.shots_ = opt.shots;

	/* fusion pays only where a gate costs a full 2^n pass */
	const bool dense_backend = std::string(be->name()).rfind("dense", 0) == 0;
	std::vector<Gate> fused;
	const std::vector<Gate>* stream = &c.gates();
	if (opt.fuse && dense_backend) {
		fused = fuse_for_dense(c.gates(), n);
		stream = &fused;
	}

	double mem_peak = 0;
	for (int shot = 0; shot < runs; shot++) {
		be->init(live);
		std::vector<int> cbits(c.num_cbits(), 0);
		for (auto& g : *stream) {
			if (g.cond_bit >= 0 && cbits[g.cond_bit] != g.cond_val)
				continue;
			Gate gg = g;
			gg.target = live_map[g.target];
			if (gg.target2 >= 0) gg.target2 = live_map[g.target2];
			for (auto& ct : gg.controls) ct = live_map[ct];
			switch (gg.op) {
			case Gate::Op::U1:
			case Gate::Op::U2:
				be->apply(gg);
				break;
			case Gate::Op::Measure:
				cbits[gg.cbit] = be->measure(gg.target, uni(rng));
				break;
			case Gate::Op::Reset:
				be->reset(gg.target, uni(rng));
				break;
			}
			mem_peak = std::max(mem_peak, be->memory_bytes());
		}
		if (per_shot) {
			std::string key;
			for (int b = int(cbits.size()) - 1; b >= 0; b--)
				key += char('0' + cbits[b]);
			out.counts_[key]++;
		}
	}

	if (!per_shot) {
		auto samples = be->sample(rng, opt.shots);
		for (uint64_t sidx : samples) {
			std::string key;
			for (int q = n - 1; q >= 0; q--) {
				int lq = live_map[q];
				key += (lq >= 0 && ((sidx >> lq) & 1)) ? '1' : '0';
			}
			out.counts_[key]++;
		}
	}

	auto t1 = std::chrono::steady_clock::now();
	out.stats.backend = be->name();
	out.stats.memory_peak_bytes = mem_peak;
	out.stats.time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
	out.stats.qubits_total = n;
	out.stats.qubits_live = live;
	out.be_ = std::move(be);
	return out;
}

} /* namespace qubit */
