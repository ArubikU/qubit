/*
 * Tiered-block GPU backend: the VRAM-resident compressed state vector.
 *
 * The state is split into fixed-size blocks living entirely in VRAM,
 * each ZERO (no storage), COMPRESSED (block-scaled int16 pairs, 4 B/amp)
 * or FULL (complex64, 8 B/amp). Tier metadata lives on the host; per
 * gate the host builds a worklist of participating blocks and launches
 * one batched kernel over it, so kernel-launch count is O(gates), not
 * O(gates x blocks).
 *
 * Unlike streaming designs (BMQSim et al.) the compressed state never
 * leaves the GPU: promotion and demotion are kernels, and the only
 * host traffic is worklist metadata and measurement scalars.
 *
 * Error accounting matches blocks.h: quantization residuals are summed
 * exactly (atomicAdd double), injections compose in L2 NORM by the
 * triangle inequality, and a block compresses only when the remaining
 * budget allows. Budget exhausted -> blocks stay FULL -> exact.
 */

#include <qubit/qubit.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cstdio>

namespace qubit {

#define BG_CHECK(call)							\
	do {								\
		cudaError_t err__ = (call);				\
		if (err__ != cudaSuccess)				\
			throw Error(std::string("CUDA error: ") +	\
				    cudaGetErrorString(err__));		\
	} while (0)

/* ------------------------------------------------------------------ */
/* kernels                                                             */
/* ------------------------------------------------------------------ */

/* atomicMax for non-negative floats: their bit patterns order as ints */
__device__ inline void atomic_max_pos(float* addr, float v)
{
	atomicMax(reinterpret_cast<int*>(addr), __float_as_int(v));
}

__global__ void k_decompress(const int16_t* q, float scale, cuFloatComplex* out,
			     uint64_t B)
{
	uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
	if (i >= B) return;
	out[i] = make_cuFloatComplex(q[2*i] * scale, q[2*i + 1] * scale);
}

/*
 * Batched stats: one launch covers a whole chunk of state blocks.
 * Thread t works on state block t/B; the shared-memory reduction stays
 * within one state block because B is a multiple of the CUDA block
 * size, so slot indices never straddle.
 */
__global__ void k_stats_batch(cuFloatComplex* const* ptrs, int nblk, uint64_t B,
			      float* d_maxs, double* d_norms)
{
	__shared__ float smax[256];
	__shared__ double snorm[256];
	uint64_t t = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
	int b = int(t / B);
	float m = 0; double p = 0;
	if (t < uint64_t(nblk) * B) {
		cuFloatComplex a = ptrs[b][t % B];
		m = fmaxf(fabsf(a.x), fabsf(a.y));
		p = double(a.x) * a.x + double(a.y) * a.y;
	}
	/*
	 * The shared reduction is only valid when one CUDA block maps to a
	 * single state block (B >= blockDim, both powers of two). Tiny
	 * blocks (tests use B=16) fall back to per-thread atomics.
	 */
	if (B < blockDim.x) {
		if (t < uint64_t(nblk) * B) {
			atomic_max_pos(&d_maxs[b], m);
			atomicAdd(&d_norms[b], p);
		}
		return;
	}
	smax[threadIdx.x] = m;
	snorm[threadIdx.x] = p;
	__syncthreads();
	for (int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (threadIdx.x < s) {
			smax[threadIdx.x] = fmaxf(smax[threadIdx.x], smax[threadIdx.x + s]);
			snorm[threadIdx.x] += snorm[threadIdx.x + s];
		}
		__syncthreads();
	}
	if (threadIdx.x == 0 && blockIdx.x * blockDim.x < uint64_t(nblk) * B) {
		atomic_max_pos(&d_maxs[b], smax[0]);
		atomicAdd(&d_norms[b], snorm[0]);
	}
}

/* Batched quantization: chunk of blocks, per-block scale and error slot. */
__global__ void k_compress_batch(cuFloatComplex* const* srcs, int16_t* const* dsts,
				 const float* scales, int nblk, uint64_t B,
				 double* d_errs)
{
	__shared__ double serr[256];
	uint64_t t = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
	int b = int(t / B);
	double e = 0;
	if (t < uint64_t(nblk) * B) {
		uint64_t i = t % B;
		cuFloatComplex a = srcs[b][i];
		float scale = scales[b];
		int qr = __float2int_rn(a.x / scale);
		int qi = __float2int_rn(a.y / scale);
		dsts[b][2*i]     = int16_t(qr);
		dsts[b][2*i + 1] = int16_t(qi);
		float dre = a.x - qr * scale, dim = a.y - qi * scale;
		e = double(dre) * dre + double(dim) * dim;
	}
	if (B < blockDim.x) {		/* same boundary issue as k_stats_batch */
		if (t < uint64_t(nblk) * B)
			atomicAdd(&d_errs[b], e);
		return;
	}
	serr[threadIdx.x] = e;
	__syncthreads();
	for (int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (threadIdx.x < s) serr[threadIdx.x] += serr[threadIdx.x + s];
		__syncthreads();
	}
	if (threadIdx.x == 0 && blockIdx.x * blockDim.x < uint64_t(nblk) * B)
		atomicAdd(&d_errs[b], serr[0]);
}

/*
 * Intra-block gate (target < bshift): worklist of FULL blocks. One
 * thread per pair per block; the block's global base index resolves
 * the high control bits.
 */
__global__ void k_intra(cuFloatComplex* const* ptrs, const uint64_t* bases,
			int nblk, uint64_t pairs_per_blk, int bshift,
			uint64_t tbit, uint64_t cmask,
			cuFloatComplex m00, cuFloatComplex m01,
			cuFloatComplex m10, cuFloatComplex m11)
{
	uint64_t t = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
	if (t >= uint64_t(nblk) * pairs_per_blk) return;
	int b = int(t / pairs_per_blk);
	uint64_t p = t % pairs_per_blk;

	/* expand pair index inside the block around the target bit */
	uint64_t low = tbit - 1;
	uint64_t i = ((p & ~low) << 1) | (p & low);
	uint64_t j = i | tbit;

	uint64_t gi = bases[b] | i;
	if ((gi & cmask) != cmask) return;

	cuFloatComplex* v = ptrs[b];
	cuFloatComplex a = v[i], c = v[j];
	v[i] = cuCaddf(cuCmulf(m00, a), cuCmulf(m01, c));
	v[j] = cuCaddf(cuCmulf(m10, a), cuCmulf(m11, c));
}

/*
 * Inter-block gate (target >= bshift): worklist of block PAIRS, both
 * sides materialized FULL. One thread per element.
 */
__global__ void k_inter(cuFloatComplex* const* ptrA, cuFloatComplex* const* ptrB,
			const uint64_t* baseA, int npairs, uint64_t B,
			uint64_t cmask,
			cuFloatComplex m00, cuFloatComplex m01,
			cuFloatComplex m10, cuFloatComplex m11)
{
	uint64_t t = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
	if (t >= uint64_t(npairs) * B) return;
	int p = int(t / B);
	uint64_t i = t % B;

	uint64_t gi = baseA[p] | i;
	if ((gi & cmask) != cmask) return;

	cuFloatComplex a = ptrA[p][i], c = ptrB[p][i];
	ptrA[p][i] = cuCaddf(cuCmulf(m00, a), cuCmulf(m01, c));
	ptrB[p][i] = cuCaddf(cuCmulf(m10, a), cuCmulf(m11, c));
}

/* P(bit=1) contribution of a worklist of FULL blocks */
__global__ void k_prob1_blocks(cuFloatComplex* const* ptrs, const uint64_t* bases,
			       int nblk, uint64_t B, uint64_t bit, double* out)
{
	__shared__ double sh[256];
	uint64_t t = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
	double p = 0;
	if (t < uint64_t(nblk) * B) {
		int b = int(t / B);
		uint64_t i = t % B;
		if ((bases[b] | i) & bit) {
			cuFloatComplex a = ptrs[b][i];
			p = double(a.x) * a.x + double(a.y) * a.y;
		}
	}
	sh[threadIdx.x] = p;
	__syncthreads();
	for (int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
		__syncthreads();
	}
	if (threadIdx.x == 0) atomicAdd(out, sh[0]);
}

__global__ void k_collapse_blocks(cuFloatComplex* const* ptrs, const uint64_t* bases,
				  int nblk, uint64_t B, uint64_t bit,
				  int outcome, float inv)
{
	uint64_t t = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
	if (t >= uint64_t(nblk) * B) return;
	int b = int(t / B);
	uint64_t i = t % B;
	bool is1 = ((bases[b] | i) & bit) != 0;
	cuFloatComplex* v = ptrs[b];
	if (is1 != (outcome == 1)) v[i] = make_cuFloatComplex(0, 0);
	else v[i] = make_cuFloatComplex(v[i].x * inv, v[i].y * inv);
}

/* ------------------------------------------------------------------ */
/* backend                                                             */
/* ------------------------------------------------------------------ */

class BlocksGPU final : public Backend {
	static constexpr double kZeroEps = 1e-24;

	enum class Tier { ZERO, COMP, FULL };
	struct Meta {
		Tier tier = Tier::ZERO;
		void* dev = nullptr;	/* cuFloatComplex* (FULL) or int16_t* (COMP) */
		float scale = 0;
		double norm = 0;	/* fresh after every retier */
	};

public:
	explicit BlocksGPU(int block_shift = 16, double l2_budget = 0.0)
		: bshift_(block_shift), budget_(l2_budget) {}

	~BlocksGPU() override { release(); }

	const char* name() const override { return "blocks-gpu"; }

	void init(int nq) override {
		release();
		n_ = nq;
		if (bshift_ > n_) bshift_ = n_;
		B_ = uint64_t(1) << bshift_;
		blk_.assign(size_t(1) << (n_ - bshift_), Meta{});
		err_used_ = 0;
		err_sq_used_ = 0;

		const int cap = int(blk_.size());
		BG_CHECK(cudaMalloc(&d_ptrA_, cap * sizeof(void*)));
		BG_CHECK(cudaMalloc(&d_ptrB_, cap * sizeof(void*)));
		BG_CHECK(cudaMalloc(&d_base_, cap * sizeof(uint64_t)));
		BG_CHECK(cudaMalloc(&d_scalar_, 2 * sizeof(double)));
		BG_CHECK(cudaMalloc(&d_max_, sizeof(float)));
		/* chunk-retier scratch: one slot per block in a chunk */
		const int cc = int(2 * kChunk);
		BG_CHECK(cudaMalloc(&d_cptr_, cc * sizeof(void*)));
		BG_CHECK(cudaMalloc(&d_cdst_, cc * sizeof(void*)));
		BG_CHECK(cudaMalloc(&d_cmax_, cc * sizeof(float)));
		BG_CHECK(cudaMalloc(&d_cscale_, cc * sizeof(float)));
		BG_CHECK(cudaMalloc(&d_cnorm_, cc * sizeof(double)));
		BG_CHECK(cudaMalloc(&d_cerr_, cc * sizeof(double)));

		/* |00...0>: single FULL block, amplitude 1 at 0 */
		alloc_full(0);
		cuFloatComplex one = make_cuFloatComplex(1, 0);
		BG_CHECK(cudaMemcpy(blk_[0].dev, &one, sizeof(one), cudaMemcpyHostToDevice));
		blk_[0].norm = 1.0;
	}

	void apply(const Gate& g) override {
		if (g.op == Gate::Op::U2)
			throw Error("blocks-gpu: fused U2 gates not supported (fusion is dense-only)");
		uint64_t cmask = 0;
		for (int c : g.controls) cmask |= uint64_t(1) << c;
		auto cc = [](cd z) {
			return make_cuFloatComplex(float(z.real()), float(z.imag()));
		};
		if (g.target < bshift_)
			apply_intra(g, cmask, cc);
		else
			apply_inter(g, cmask, cc);
	}

	int measure(int q, float u) override {
		auto work = live_full_worklist();	/* promotes COMP */
		upload_worklist(work);
		BG_CHECK(cudaMemset(d_scalar_, 0, sizeof(double)));
		uint64_t total = uint64_t(work.size()) * B_;
		if (total)
			k_prob1_blocks<<<grid(total), 256>>>(
				(cuFloatComplex**)d_ptrA_, d_base_,
				int(work.size()), B_, uint64_t(1) << q, d_scalar_);
		double p1 = 0;
		BG_CHECK(cudaMemcpy(&p1, d_scalar_, sizeof(double), cudaMemcpyDeviceToHost));

		int outcome = (u < p1) ? 1 : 0;
		double p = outcome ? p1 : 1.0 - p1;
		float inv = 1.0f / std::sqrt(float(p > 1e-30 ? p : 1e-30));
		if (total)
			k_collapse_blocks<<<grid(total), 256>>>(
				(cuFloatComplex**)d_ptrA_, d_base_,
				int(work.size()), B_, uint64_t(1) << q, outcome, inv);
		BG_CHECK(cudaGetLastError());
		for (size_t at = 0; at < work.size(); at += 2 * kChunk) {
			size_t end = std::min(at + 2 * kChunk, work.size());
			retier_chunk(std::vector<uint64_t>(work.begin() + at,
							   work.begin() + end));
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
			throw Error("blocks-gpu: refusing to materialize 2^" +
				    std::to_string(n_) + " amplitudes");
		std::vector<cf> full(uint64_t(1) << n_, cf(0, 0));
		std::vector<cuFloatComplex> tmp(B_);
		for (uint64_t b = 0; b < blk_.size(); b++) {
			const Meta& m = blk_[b];
			if (m.tier == Tier::ZERO) continue;
			if (m.tier == Tier::FULL) {
				BG_CHECK(cudaMemcpy(tmp.data(), m.dev,
						    B_ * sizeof(cuFloatComplex),
						    cudaMemcpyDeviceToHost));
			} else {
				std::vector<int16_t> q(2 * B_);
				BG_CHECK(cudaMemcpy(q.data(), m.dev,
						    q.size() * sizeof(int16_t),
						    cudaMemcpyDeviceToHost));
				for (uint64_t i = 0; i < B_; i++)
					tmp[i] = make_cuFloatComplex(q[2*i] * m.scale,
								     q[2*i+1] * m.scale);
			}
			for (uint64_t i = 0; i < B_; i++)
				full[(b << bshift_) | i] = cf(tmp[i].x, tmp[i].y);
		}
		return full;
	}

	cf amplitude(uint64_t idx) const override {
		const Meta& m = blk_[idx >> bshift_];
		uint64_t i = idx & (B_ - 1);
		if (m.tier == Tier::ZERO) return cf(0, 0);
		if (m.tier == Tier::FULL) {
			cuFloatComplex a;
			BG_CHECK(cudaMemcpy(&a, (cuFloatComplex*)m.dev + i,
					    sizeof(a), cudaMemcpyDeviceToHost));
			return cf(a.x, a.y);
		}
		int16_t q[2];
		BG_CHECK(cudaMemcpy(q, (int16_t*)m.dev + 2*i, sizeof(q),
				    cudaMemcpyDeviceToHost));
		return cf(q[0] * m.scale, q[1] * m.scale);
	}

	/* two-level sampling: block by cached norm, element within block */
	std::vector<uint64_t> sample(std::mt19937_64& rng, int shots) const override {
		std::uniform_real_distribution<float> uni(0.0f, 1.0f);
		double total = 0;
		for (auto& m : blk_) total += m.norm;
		std::vector<uint64_t> out;
		out.reserve(shots);
		std::vector<cf> cache;
		uint64_t cached = ~uint64_t(0);
		for (int s = 0; s < shots; s++) {
			double u = uni(rng) * total, cum = 0;
			uint64_t b = blk_.size() - 1;
			for (uint64_t k = 0; k < blk_.size(); k++) {
				cum += blk_[k].norm;
				if (u < cum) { b = k; break; }
			}
			if (b != cached) { cache = block_host(b); cached = b; }
			float u2 = uni(rng);
			double c2 = 0;
			uint64_t pick = B_ - 1;
			double bn = blk_[b].norm > 0 ? blk_[b].norm : 1.0;
			for (uint64_t i = 0; i < B_; i++) {
				c2 += std::norm(cache[i]) / bn;
				if (u2 < c2) { pick = i; break; }
			}
			out.push_back((b << bshift_) | pick);
		}
		return out;
	}

	double memory_bytes() const override {
		double bytes = 0;
		for (auto& m : blk_) {
			if (m.tier == Tier::FULL) bytes += double(B_) * 8;
			else if (m.tier == Tier::COMP) bytes += double(B_) * 4;
		}
		return bytes;
	}

	double error_l2() const { return err_used_; }

	/* Rigorous worst-case: injections aligned, D = sum of norms. */
	double fidelity_bound() const {
		if (err_used_ >= 1.0) return 0.0;
		double r = (1.0 - err_used_) / (1.0 + err_used_);
		return r * r;
	}

	/*
	 * Probabilistic estimate: quantization residuals have no preferred
	 * direction, so injections add like a random walk and the total
	 * concentrates near D_rms = sqrt(sum ||delta_i||^2) instead of the
	 * adversarial sum. An ESTIMATE under that independence assumption,
	 * not a guarantee — report alongside the bound, never instead.
	 */
	double fidelity_estimate() const {
		double d = std::sqrt(err_sq_used_);
		if (d >= 1.0) return 0.0;
		double r = (1.0 - d) / (1.0 + d);
		return r * r;
	}
	int full_blocks() const {
		int k = 0;
		for (auto& m : blk_) if (m.tier == Tier::FULL) k++;
		return k;
	}
	int comp_blocks() const {
		int k = 0;
		for (auto& m : blk_) if (m.tier == Tier::COMP) k++;
		return k;
	}

private:
	static unsigned grid(uint64_t work) { return unsigned((work + 255) / 256); }

	/*
	 * Buffer pools: tier churn is constant (every touched block cycles
	 * COMP->FULL->COMP per gate), so freeing to the driver would mean
	 * millions of cudaMalloc calls per circuit. Buffers are fixed-size
	 * per init, so a free-list amortizes allocation to near zero.
	 */
	void* pool_pop(std::vector<void*>& pool, size_t bytes) {
		if (!pool.empty()) {
			void* p = pool.back();
			pool.pop_back();
			return p;
		}
		void* p = nullptr;
		BG_CHECK(cudaMalloc(&p, bytes));
		return p;
	}

	void release() {
		for (auto& m : blk_)
			if (m.dev) cudaFree(m.dev);
		for (void* p : pool_full_) cudaFree(p);
		for (void* p : pool_comp_) cudaFree(p);
		pool_full_.clear();
		pool_comp_.clear();
		blk_.clear();
		if (d_ptrA_) { cudaFree(d_ptrA_); d_ptrA_ = nullptr; }
		if (d_ptrB_) { cudaFree(d_ptrB_); d_ptrB_ = nullptr; }
		if (d_base_) { cudaFree(d_base_); d_base_ = nullptr; }
		if (d_scalar_) { cudaFree(d_scalar_); d_scalar_ = nullptr; }
		if (d_max_) { cudaFree(d_max_); d_max_ = nullptr; }
		if (d_cptr_) { cudaFree(d_cptr_); d_cptr_ = nullptr; }
		if (d_cdst_) { cudaFree(d_cdst_); d_cdst_ = nullptr; }
		if (d_cmax_) { cudaFree(d_cmax_); d_cmax_ = nullptr; }
		if (d_cscale_) { cudaFree(d_cscale_); d_cscale_ = nullptr; }
		if (d_cnorm_) { cudaFree(d_cnorm_); d_cnorm_ = nullptr; }
		if (d_cerr_) { cudaFree(d_cerr_); d_cerr_ = nullptr; }
	}

	void alloc_full(uint64_t b) {
		Meta& m = blk_[b];
		m.dev = pool_pop(pool_full_, B_ * sizeof(cuFloatComplex));
		BG_CHECK(cudaMemset(m.dev, 0, B_ * sizeof(cuFloatComplex)));
		m.tier = Tier::FULL;
		m.scale = 0;
	}

	void promote(uint64_t b) {
		Meta& m = blk_[b];
		if (m.tier == Tier::FULL) return;
		if (m.tier == Tier::ZERO) { alloc_full(b); return; }
		void* qdev = m.dev;
		float scale = m.scale;
		cuFloatComplex* full =
			(cuFloatComplex*)pool_pop(pool_full_, B_ * sizeof(cuFloatComplex));
		k_decompress<<<grid(B_), 256>>>((int16_t*)qdev, scale, full, B_);
		BG_CHECK(cudaGetLastError());
		pool_comp_.push_back(qdev);
		m.dev = full;
		m.tier = Tier::FULL;
	}

	/*
	 * Chunk-batched retier: ZERO if a block's norm vanished, COMPRESSED
	 * if the exactly-measured quantization error still fits the L2
	 * budget, FULL otherwise. Same policy as blocks.h, but the whole
	 * chunk shares TWO kernel launches and THREE host copies — the
	 * per-block version cost ~6 synchronizing calls per block and
	 * dominated runtime (measured 710 s for a 30-qubit echo).
	 *
	 * Compression is speculative-batched: every candidate quantizes in
	 * one launch, then the host walks the chunk in order granting
	 * budget; rejected blocks return their buffer to the pool. When
	 * the budget is already dry the compress pass is skipped entirely.
	 */
	void retier_chunk(const std::vector<uint64_t>& w) {
		if (w.empty()) return;
		const int nc = int(w.size());

		std::vector<void*> ptrs(nc);
		for (int k = 0; k < nc; k++) ptrs[k] = blk_[w[k]].dev;
		BG_CHECK(cudaMemcpy(d_cptr_, ptrs.data(), nc * sizeof(void*),
				    cudaMemcpyHostToDevice));
		BG_CHECK(cudaMemset(d_cmax_, 0, nc * sizeof(float)));
		BG_CHECK(cudaMemset(d_cnorm_, 0, nc * sizeof(double)));
		k_stats_batch<<<grid(uint64_t(nc) * B_), 256>>>(
			(cuFloatComplex**)d_cptr_, nc, B_, d_cmax_, d_cnorm_);
		BG_CHECK(cudaGetLastError());

		std::vector<float> maxs(nc);
		std::vector<double> norms(nc);
		BG_CHECK(cudaMemcpy(maxs.data(), d_cmax_, nc * sizeof(float),
				    cudaMemcpyDeviceToHost));
		BG_CHECK(cudaMemcpy(norms.data(), d_cnorm_, nc * sizeof(double),
				    cudaMemcpyDeviceToHost));

		std::vector<int> cand;		/* indices into w */
		for (int k = 0; k < nc; k++) {
			Meta& m = blk_[w[k]];
			m.norm = norms[k];
			if (norms[k] < kZeroEps) {
				pool_full_.push_back(m.dev);
				m.dev = nullptr;
				m.tier = Tier::ZERO;
				m.norm = 0;
			} else if (budget_ > 0 && err_used_ < budget_ && maxs[k] > 0) {
				cand.push_back(k);
			}
		}
		if (cand.empty()) return;

		const int cn = int(cand.size());
		std::vector<void*> srcs(cn), dsts(cn);
		std::vector<float> scales(cn);
		for (int j = 0; j < cn; j++) {
			int k = cand[j];
			srcs[j] = blk_[w[k]].dev;
			dsts[j] = pool_pop(pool_comp_, 2 * B_ * sizeof(int16_t));
			scales[j] = maxs[k] / 32767.0f;
		}
		BG_CHECK(cudaMemcpy(d_cptr_, srcs.data(), cn * sizeof(void*),
				    cudaMemcpyHostToDevice));
		BG_CHECK(cudaMemcpy(d_cdst_, dsts.data(), cn * sizeof(void*),
				    cudaMemcpyHostToDevice));
		BG_CHECK(cudaMemcpy(d_cscale_, scales.data(), cn * sizeof(float),
				    cudaMemcpyHostToDevice));
		BG_CHECK(cudaMemset(d_cerr_, 0, cn * sizeof(double)));
		k_compress_batch<<<grid(uint64_t(cn) * B_), 256>>>(
			(cuFloatComplex**)d_cptr_, (int16_t**)d_cdst_,
			d_cscale_, cn, B_, d_cerr_);
		BG_CHECK(cudaGetLastError());

		std::vector<double> errs(cn);
		BG_CHECK(cudaMemcpy(errs.data(), d_cerr_, cn * sizeof(double),
				    cudaMemcpyDeviceToHost));
		for (int j = 0; j < cn; j++) {
			int k = cand[j];
			Meta& m = blk_[w[k]];
			double err_norm = std::sqrt(errs[j]);
			if (err_used_ + err_norm > budget_) {
				pool_comp_.push_back(dsts[j]);	/* stay FULL, stay exact */
				continue;
			}
			err_used_ += err_norm;
			err_sq_used_ += errs[j];
			pool_full_.push_back(m.dev);
			m.dev = dsts[j];
			m.scale = scales[j];
			m.tier = Tier::COMP;
		}
	}

	std::vector<uint64_t> live_full_worklist() {
		std::vector<uint64_t> w;
		for (uint64_t b = 0; b < blk_.size(); b++) {
			if (blk_[b].tier == Tier::ZERO) continue;
			promote(b);
			w.push_back(b);
		}
		return w;
	}

	void upload_worklist(const std::vector<uint64_t>& w) {
		std::vector<void*> ptrs(w.size());
		std::vector<uint64_t> bases(w.size());
		for (size_t k = 0; k < w.size(); k++) {
			ptrs[k] = blk_[w[k]].dev;
			bases[k] = w[k] << bshift_;
		}
		if (!w.empty()) {
			BG_CHECK(cudaMemcpy(d_ptrA_, ptrs.data(), w.size() * sizeof(void*),
					    cudaMemcpyHostToDevice));
			BG_CHECK(cudaMemcpy(d_base_, bases.data(), w.size() * sizeof(uint64_t),
					    cudaMemcpyHostToDevice));
		}
	}

	/*
	 * Promote-apply-retier runs in CHUNKS so the FULL-tier transient
	 * stays bounded (kChunk blocks) no matter the state size. Blocks
	 * are independent within an intra gate and pairwise independent
	 * within an inter gate, so chunking changes nothing semantically.
	 * Without it, promoting the whole worklist would momentarily need
	 * the entire dense footprint — the exact thing this backend avoids.
	 */
	static constexpr size_t kChunk = 64;

	template <class CC>
	void apply_intra(const Gate& g, uint64_t cmask, CC cc) {
		/* high control bits are constant per block: prefilter on host */
		const uint64_t himask = cmask & ~(B_ - 1);
		std::vector<uint64_t> w;
		for (uint64_t b = 0; b < blk_.size(); b++) {
			if (blk_[b].tier == Tier::ZERO) continue;
			if (((b << bshift_) & himask) != himask) continue;
			w.push_back(b);
		}
		uint64_t pairs = B_ >> 1;
		for (size_t at = 0; at < w.size(); at += kChunk) {
			size_t end = std::min(at + kChunk, w.size());
			std::vector<uint64_t> chunk(w.begin() + at, w.begin() + end);
			for (uint64_t b : chunk) promote(b);
			upload_worklist(chunk);
			k_intra<<<grid(uint64_t(chunk.size()) * pairs), 256>>>(
				(cuFloatComplex**)d_ptrA_, d_base_, int(chunk.size()),
				pairs, bshift_, uint64_t(1) << g.target, cmask & (B_ - 1),
				cc(g.m[0]), cc(g.m[1]), cc(g.m[2]), cc(g.m[3]));
			BG_CHECK(cudaGetLastError());
			retier_chunk(chunk);
		}
	}

	template <class CC>
	void apply_inter(const Gate& g, uint64_t cmask, CC cc) {
		const uint64_t bbit = uint64_t(1) << (g.target - bshift_);
		std::vector<uint64_t> A, Bl;
		for (uint64_t b = 0; b < blk_.size(); b++) {
			if (b & bbit) continue;
			uint64_t b1 = b | bbit;
			if (blk_[b].tier == Tier::ZERO && blk_[b1].tier == Tier::ZERO)
				continue;
			/*
			 * Control bits >= bshift are constant per pair: both
			 * members share every block bit except the target
			 * bit, and the target is never a control.
			 */
			uint64_t himask = cmask & ~(B_ - 1);
			if (((b << bshift_) & himask) != himask)
				continue;
			promote(b);
			promote(b1);
			A.push_back(b);
			Bl.push_back(b1);
		}
		for (size_t at = 0; at < A.size(); at += kChunk / 2) {
			size_t end = std::min(at + kChunk / 2, A.size());
			size_t nc = end - at;
			std::vector<void*> pa(nc), pb(nc);
			std::vector<uint64_t> bases(nc);
			for (size_t k = 0; k < nc; k++) {
				promote(A[at + k]);
				promote(Bl[at + k]);
				pa[k] = blk_[A[at + k]].dev;
				pb[k] = blk_[Bl[at + k]].dev;
				bases[k] = A[at + k] << bshift_;
			}
			BG_CHECK(cudaMemcpy(d_ptrA_, pa.data(), nc * sizeof(void*),
					    cudaMemcpyHostToDevice));
			BG_CHECK(cudaMemcpy(d_ptrB_, pb.data(), nc * sizeof(void*),
					    cudaMemcpyHostToDevice));
			BG_CHECK(cudaMemcpy(d_base_, bases.data(), nc * sizeof(uint64_t),
					    cudaMemcpyHostToDevice));
			k_inter<<<grid(uint64_t(nc) * B_), 256>>>(
				(cuFloatComplex**)d_ptrA_, (cuFloatComplex**)d_ptrB_,
				d_base_, int(nc), B_, cmask,
				cc(g.m[0]), cc(g.m[1]), cc(g.m[2]), cc(g.m[3]));
			BG_CHECK(cudaGetLastError());
			std::vector<uint64_t> touched;
			touched.reserve(2 * nc);
			for (size_t k = 0; k < nc; k++) {
				touched.push_back(A[at + k]);
				touched.push_back(Bl[at + k]);
			}
			retier_chunk(touched);
		}
	}

	std::vector<cf> block_host(uint64_t b) const {
		std::vector<cf> out(B_, cf(0, 0));
		const Meta& m = blk_[b];
		if (m.tier == Tier::ZERO) return out;
		if (m.tier == Tier::FULL) {
			std::vector<cuFloatComplex> tmp(B_);
			BG_CHECK(cudaMemcpy(tmp.data(), m.dev, B_ * sizeof(cuFloatComplex),
					    cudaMemcpyDeviceToHost));
			for (uint64_t i = 0; i < B_; i++) out[i] = cf(tmp[i].x, tmp[i].y);
			return out;
		}
		std::vector<int16_t> q(2 * B_);
		BG_CHECK(cudaMemcpy(q.data(), m.dev, q.size() * sizeof(int16_t),
				    cudaMemcpyDeviceToHost));
		for (uint64_t i = 0; i < B_; i++)
			out[i] = cf(q[2*i] * m.scale, q[2*i+1] * m.scale);
		return out;
	}

	int n_ = 0;
	int bshift_;
	uint64_t B_ = 0;
	double budget_;
	double err_used_ = 0;
	double err_sq_used_ = 0;	/* for the probabilistic estimate */
	std::vector<Meta> blk_;
	std::vector<void*> pool_full_;
	std::vector<void*> pool_comp_;

	void** d_ptrA_ = nullptr;
	void** d_ptrB_ = nullptr;
	uint64_t* d_base_ = nullptr;
	double* d_scalar_ = nullptr;	/* [0]=norm/prob, [1]=err */
	float* d_max_ = nullptr;
	void** d_cptr_ = nullptr;	/* chunk-retier scratch */
	void** d_cdst_ = nullptr;
	float* d_cmax_ = nullptr;
	float* d_cscale_ = nullptr;
	double* d_cnorm_ = nullptr;
	double* d_cerr_ = nullptr;
};

std::shared_ptr<Backend> make_blocks_gpu(int block_shift, double l2_budget)
{
	return std::make_shared<BlocksGPU>(block_shift, l2_budget);
}

/* accounting probes for benchmarks; be must come from make_blocks_gpu */
double blocks_gpu_error(const Backend* be)
{
	return static_cast<const BlocksGPU*>(be)->error_l2();
}
double blocks_gpu_fidelity_bound(const Backend* be)
{
	return static_cast<const BlocksGPU*>(be)->fidelity_bound();
}
double blocks_gpu_fidelity_estimate(const Backend* be)
{
	return static_cast<const BlocksGPU*>(be)->fidelity_estimate();
}

} /* namespace qubit */
