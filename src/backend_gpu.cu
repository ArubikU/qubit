/*
 * Dense GPU backend. Same pair-update algorithm as DenseCPU, one thread
 * per index. The state vector lives in VRAM for the whole circuit; the
 * only host round-trips are measurement scalars and the final state().
 *
 * Build: nvcc -O2 -std=c++17 -arch=sm_86 -DQUBIT_CUDA -I include \
 *        examples/foo.cpp src/backend_gpu.cu -o foo
 */

#include <qubit/qubit.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cstdio>

namespace qubit {

#define QUBIT_CUDA_CHECK(call)						\
	do {								\
		cudaError_t err__ = (call);				\
		if (err__ != cudaSuccess)				\
			throw Error(std::string("CUDA error: ") +	\
				    cudaGetErrorString(err__));		\
	} while (0)

/*
 * One thread per WORKING pair. The pair index expands to a state index
 * by inserting the target bit (0) and control bits (1) at their sorted
 * positions, passed packed: pos[k] in bits 0..5 of byte k of `poss`,
 * fixed bit value in `bits` bit k. A gate with c controls launches
 * 2^(n-1-c) threads — none idle, no divergence.
 */
__global__ void k_apply(cuFloatComplex* v, uint64_t npairs,
			uint64_t poss, uint32_t bits, int np, uint64_t tbit,
			cuFloatComplex m00, cuFloatComplex m01,
			cuFloatComplex m10, cuFloatComplex m11)
{
	uint64_t t = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
	if (t >= npairs) return;
	uint64_t i = t;
	for (int k = 0; k < np; k++) {
		int p = int((poss >> (8 * k)) & 0x3f);
		uint64_t low = (uint64_t(1) << p) - 1;
		i = ((i & ~low) << 1) | (uint64_t((bits >> k) & 1) << p) | (i & low);
	}
	uint64_t j = i | tbit;
	cuFloatComplex a = v[i], b = v[j];
	v[i] = cuCaddf(cuCmulf(m00, a), cuCmulf(m01, b));
	v[j] = cuCaddf(cuCmulf(m10, a), cuCmulf(m11, b));
}

/* Diagonal gates: no pair partner, each amplitude picks up a phase
 * chosen by its own bits. Perfectly coalesced, no gather. */
__global__ void k_diag1(cuFloatComplex* v, uint64_t N, uint64_t tbit, uint64_t cmask,
			cuFloatComplex d0, cuFloatComplex d1)
{
	uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
	if (i >= N) return;
	if ((i & cmask) != cmask) return;
	v[i] = cuCmulf(v[i], (i & tbit) ? d1 : d0);
}

__global__ void k_diag2(cuFloatComplex* v, uint64_t N, uint64_t hbit, uint64_t lbit,
			cuFloatComplex d0, cuFloatComplex d1,
			cuFloatComplex d2, cuFloatComplex d3)
{
	uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
	if (i >= N) return;
	int loc = (int((i & hbit) != 0) << 1) | int((i & lbit) != 0);
	cuFloatComplex d = loc == 0 ? d0 : loc == 1 ? d1 : loc == 2 ? d2 : d3;
	v[i] = cuCmulf(v[i], d);
}

/* Fused 4x4 over (hi, lo): one thread per quartet, one pass over VRAM.
 * The matrix travels by value in the kernel parameter buffer. */
struct M16 { cuFloatComplex m[16]; };

__global__ void k_apply2(cuFloatComplex* v, uint64_t nquads,
			 int phi, int plo, M16 Mv)
{
	const cuFloatComplex* M = Mv.m;
	uint64_t t = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
	if (t >= nquads) return;
	const int p0 = phi < plo ? phi : plo;
	const int p1 = phi < plo ? plo : phi;
	uint64_t i = t;
	uint64_t low0 = (uint64_t(1) << p0) - 1;
	i = ((i & ~low0) << 1) | (i & low0);
	uint64_t low1 = (uint64_t(1) << p1) - 1;
	i = ((i & ~low1) << 1) | (i & low1);
	const uint64_t hbit = uint64_t(1) << phi, lbit = uint64_t(1) << plo;
	const uint64_t i00 = i, i01 = i | lbit, i10 = i | hbit, i11 = i | hbit | lbit;
	cuFloatComplex x0 = v[i00], x1 = v[i01], x2 = v[i10], x3 = v[i11];
	auto row = [&](int r) {
		cuFloatComplex s = cuCmulf(M[r*4+0], x0);
		s = cuCaddf(s, cuCmulf(M[r*4+1], x1));
		s = cuCaddf(s, cuCmulf(M[r*4+2], x2));
		return cuCaddf(s, cuCmulf(M[r*4+3], x3));
	};
	v[i00] = row(0); v[i01] = row(1); v[i10] = row(2); v[i11] = row(3);
}

/*
 * P(bit q = 1). Block-level shared-memory reduction, one atomicAdd per
 * block. Accumulate in double: 2^29 float squares summed naively lose
 * precision that measurement decisions actually need.
 */
__global__ void k_prob1(const cuFloatComplex* v, uint64_t N, uint64_t bit, double* out)
{
	__shared__ double sh[256];
	uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
	double p = 0;
	if (i < N && (i & bit))
		p = double(v[i].x) * v[i].x + double(v[i].y) * v[i].y;
	sh[threadIdx.x] = p;
	__syncthreads();
	for (int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
		__syncthreads();
	}
	if (threadIdx.x == 0) atomicAdd(out, sh[0]);
}

/* Post-measurement collapse: zero the dead branch, rescale the survivor. */
__global__ void k_collapse(cuFloatComplex* v, uint64_t N, uint64_t bit,
			   int outcome, float inv_norm)
{
	uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
	if (i >= N) return;
	bool is1 = (i & bit) != 0;
	if (is1 != (outcome == 1)) v[i] = make_cuFloatComplex(0, 0);
	else v[i] = make_cuFloatComplex(v[i].x * inv_norm, v[i].y * inv_norm);
}

class DenseGPU final : public Backend {
public:
	const char* name() const override { return "dense-gpu"; }

	~DenseGPU() override {
		if (d_v_) cudaFree(d_v_);
		if (d_scratch_) cudaFree(d_scratch_);
	}

	void init(int nq) override {
		n_ = nq; N_ = uint64_t(1) << nq;
		if (d_v_) { cudaFree(d_v_); d_v_ = nullptr; }
		size_t bytes = N_ * sizeof(cuFloatComplex);
		cudaError_t e = cudaMalloc(&d_v_, bytes);
		if (e != cudaSuccess) {
			size_t free_b = 0, total_b = 0;
			cudaMemGetInfo(&free_b, &total_b);
			throw Error(
				"qubit: not enough VRAM for dense-gpu backend.\n"
				"  live qubits: " + std::to_string(nq) + " -> " +
				std::to_string(bytes / 1048576.0) + " MB needed, free VRAM: " +
				std::to_string(free_b / 1048576.0) + " MB.\n"
				"  options: fewer qubits | wait for groups/blocks/MPS backends");
		}
		if (!d_scratch_)
			QUBIT_CUDA_CHECK(cudaMalloc(&d_scratch_, sizeof(double)));
		QUBIT_CUDA_CHECK(cudaMemset(d_v_, 0, bytes));
		cuFloatComplex one = make_cuFloatComplex(1.0f, 0.0f);
		QUBIT_CUDA_CHECK(cudaMemcpy(d_v_, &one, sizeof(one), cudaMemcpyHostToDevice));
	}

	void apply(const Gate& g) override {
		auto cc = [](cd z) {
			return make_cuFloatComplex(float(z.real()), float(z.imag()));
		};
		if (g.op == Gate::Op::U2) {
			bool diag = true;
			for (int i = 0; i < 4 && diag; i++)
				for (int j = 0; j < 4; j++)
					if (i != j && g.m4[i*4 + j] != cd(0, 0)) {
						diag = false;
						break;
					}
			if (diag) {
				k_diag2<<<grid(N_), 256>>>(d_v_, N_,
					uint64_t(1) << g.target2, uint64_t(1) << g.target,
					cc(g.m4[0]), cc(g.m4[5]), cc(g.m4[10]), cc(g.m4[15]));
				QUBIT_CUDA_CHECK(cudaGetLastError());
				return;
			}
			M16 M;
			for (int k = 0; k < 16; k++)
				M.m[k] = make_cuFloatComplex(float(g.m4[k].real()),
							     float(g.m4[k].imag()));
			const uint64_t nquads = N_ >> 2;
			k_apply2<<<grid(nquads), 256>>>(d_v_, nquads,
							g.target2, g.target, M);
			QUBIT_CUDA_CHECK(cudaGetLastError());
			return;
		}
		if (g.m[1] == cd(0, 0) && g.m[2] == cd(0, 0)) {
			uint64_t cmask = 0;
			for (int c : g.controls) cmask |= uint64_t(1) << c;
			k_diag1<<<grid(N_), 256>>>(d_v_, N_, uint64_t(1) << g.target,
						   cmask, cc(g.m[0]), cc(g.m[3]));
			QUBIT_CUDA_CHECK(cudaGetLastError());
			return;
		}
		if (g.controls.size() > 7)
			throw Error("dense-gpu: more than 7 controls unsupported");
		int pos[8], bit[8], np = 0;
		pos[np] = g.target; bit[np++] = 0;
		for (int c : g.controls) { pos[np] = c; bit[np++] = 1; }
		for (int a = 1; a < np; a++)
			for (int b = a; b > 0 && pos[b] < pos[b-1]; b--) {
				std::swap(pos[b], pos[b-1]);
				std::swap(bit[b], bit[b-1]);
			}
		uint64_t poss = 0; uint32_t bits = 0;
		for (int k = 0; k < np; k++) {
			poss |= uint64_t(pos[k]) << (8 * k);
			bits |= uint32_t(bit[k]) << k;
		}
		const uint64_t npairs = N_ >> np;
		k_apply<<<grid(npairs), 256>>>(d_v_, npairs, poss, bits, np,
					       uint64_t(1) << g.target,
					       cc(g.m[0]), cc(g.m[1]), cc(g.m[2]), cc(g.m[3]));
		QUBIT_CUDA_CHECK(cudaGetLastError());
	}

	int measure(int q, float u) override {
		double p1 = prob1(q);
		int outcome = (u < p1) ? 1 : 0;
		double p = outcome ? p1 : 1.0 - p1;
		float inv = 1.0f / std::sqrt(float(p > 1e-30 ? p : 1e-30));
		k_collapse<<<grid(N_), 256>>>(d_v_, N_, uint64_t(1) << q, outcome, inv);
		QUBIT_CUDA_CHECK(cudaGetLastError());
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
		QUBIT_CUDA_CHECK(cudaDeviceSynchronize());
		std::vector<cf> h(N_);
		QUBIT_CUDA_CHECK(cudaMemcpy(h.data(), d_v_, N_ * sizeof(cuFloatComplex),
					   cudaMemcpyDeviceToHost));
		return h;
	}

	double memory_bytes() const override { return double(N_) * sizeof(cuFloatComplex); }

	/* Device state pointer for on-device algorithms (qtrain GPU adjoint). */
	void* device_state() override { return d_v_; }

private:
	static unsigned grid(uint64_t work) { return unsigned((work + 255) / 256); }

	double prob1(int q) {
		QUBIT_CUDA_CHECK(cudaMemset(d_scratch_, 0, sizeof(double)));
		k_prob1<<<grid(N_), 256>>>(d_v_, N_, uint64_t(1) << q, d_scratch_);
		QUBIT_CUDA_CHECK(cudaGetLastError());
		double p = 0;
		QUBIT_CUDA_CHECK(cudaMemcpy(&p, d_scratch_, sizeof(double), cudaMemcpyDeviceToHost));
		return p;
	}

	int n_ = 0;
	uint64_t N_ = 0;
	cuFloatComplex* d_v_ = nullptr;
	double* d_scratch_ = nullptr;
};

std::unique_ptr<Backend> make_dense_gpu() { return std::make_unique<DenseGPU>(); }

double gpu_free_vram_bytes()
{
	size_t free_b = 0, total_b = 0;
	if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess)
		return 0;
	return double(free_b);
}

} /* namespace qubit */
