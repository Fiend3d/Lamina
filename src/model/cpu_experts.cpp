#include "lamina/model/cpu_experts.hpp"
#include "strata/artifact/dequant.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#ifdef LAMINA_CPU_QUANT
#include "ggml.h"
#include "ggml-cpu.h"
#endif
#if defined(_M_X64) || defined(__x86_64__)
#include <immintrin.h>
#ifdef _MSC_VER
#include <intrin.h>
#endif
#endif

namespace lamina::model {
namespace {
constexpr int kGateRows = 32;   // rows per gate/up work item
constexpr int kDownRows = 128;  // rows per down work item
constexpr int kSpins = 1 << 16; // roughly a millisecond of polling before sleeping

int64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

void pause() {
#if defined(_M_X64) || defined(__x86_64__)
    _mm_pause();
#else
    std::this_thread::yield();
#endif
}

bool avx2_available() {
#if defined(_MSC_VER) && defined(_M_X64)
    int regs[4]; __cpuid(regs, 1);
    if ((regs[2] & (1 << 27)) == 0 || (_xgetbv(0) & 6) != 6) return false;
    __cpuidex(regs, 7, 0); return (regs[1] & (1 << 5)) != 0;
#elif defined(__x86_64__)
    return __builtin_cpu_supports("avx2");
#else
    return false;
#endif
}
#if defined(_M_X64) || defined(__x86_64__)
#ifndef _MSC_VER
__attribute__((target("avx2")))
#endif
double avx_dot(const float* a, const float* b, int n) {
    __m256d lo = _mm256_setzero_pd(), hi = lo;
    for (int i = 0; i < n; i += 8) {
        lo = _mm256_add_pd(lo, _mm256_mul_pd(_mm256_cvtps_pd(_mm_loadu_ps(a+i)), _mm256_cvtps_pd(_mm_loadu_ps(b+i))));
        hi = _mm256_add_pd(hi, _mm256_mul_pd(_mm256_cvtps_pd(_mm_loadu_ps(a+i+4)), _mm256_cvtps_pd(_mm_loadu_ps(b+i+4))));
    }
    double sums[4]; _mm256_storeu_pd(sums, _mm256_add_pd(lo, hi));
    return (sums[0] + sums[1]) + (sums[2] + sums[3]);
}
#endif
#ifdef LAMINA_CPU_QUANT
// Same ggml-cpu traits and quantizers as Strata's native expert adapter.
void quantize_for(const strata::TensorInfo& t, const float* x, std::vector<uint8_t>& quantized) {
    const auto* wt = ggml_get_type_traits_cpu(static_cast<ggml_type>(t.type));
    const auto* at = ggml_get_type_traits_cpu(wt->vec_dot_type);
    if (!wt->vec_dot || !at->from_float) throw std::runtime_error("CPU quantized expert type unsupported");
    const int width = int(t.shape[0]);
    quantized.resize(ggml_row_size(wt->vec_dot_type, width));
    at->from_float(x, quantized.data(), width);
}

void project_quantized_rows(const strata::TensorInfo& t, const uint8_t* data, int expert,
                            const std::array<const uint8_t*, CpuExperts::kMaxTokens>& quantized,
                            const std::array<float*, CpuExperts::kMaxTokens>& out, int begin, int end) {
    const auto type = static_cast<ggml_type>(t.type);
    const auto* wt = ggml_get_type_traits_cpu(type);
    const int width = int(t.shape[0]), rows = int(t.shape[1]);
    const size_t row_bytes = ggml_row_size(type, width);
    const uint8_t* source = data + size_t(expert) * rows * row_bytes;
    // Row-outer, token-inner: a row read from RAM serves every routed token from L1.
    for (int row = begin; row < end; ++row)
        for (int k = 0; k < CpuExperts::kMaxTokens; ++k)
            if (out[size_t(k)]) wt->vec_dot(width, &out[size_t(k)][row], 0, source + size_t(row) * row_bytes, 0, quantized[size_t(k)], 0, 1);
}

bool same_dot_type(const strata::TensorInfo& a, const strata::TensorInfo& b) {
    return ggml_get_type_traits_cpu(static_cast<ggml_type>(a.type))->vec_dot_type ==
           ggml_get_type_traits_cpu(static_cast<ggml_type>(b.type))->vec_dot_type;
}
#endif

// Dequantizes rows [begin, end) of one expert matrix and dots them with each
// token's x in FP64; the decoded row is reused for every routed token.
void project_rows(const strata::TensorInfo& t, const uint8_t* data, int expert,
                  const std::array<const float*, CpuExperts::kMaxTokens>& x,
                  const std::array<float*, CpuExperts::kMaxTokens>& out, int begin, int end) {
    int block = 0, bytes = 0;
    if (!strata::block_geometry(t.type, block, bytes))
        throw std::invalid_argument("CPU expert projection shape or type");
    const int width = int(t.shape[0]), rows = int(t.shape[1]);
    const size_t row_bytes = size_t(width / block) * bytes;
    const uint8_t* source = data + (size_t(expert) * rows + size_t(begin)) * row_bytes;
    thread_local std::vector<float> decoded;
    decoded.resize(size_t(width));
    static const bool avx = avx2_available();
    for (int row = begin; row < end; ++row) {
        for (int i = 0; i < width; i += block, source += bytes) {
            switch (t.type) {
            case 8: strata::dequantize_q8_0(source, decoded.data()+i); break;
            case 12: strata::dequantize_q4_K(source, decoded.data()+i); break;
            case 13: strata::dequantize_q5_K(source, decoded.data()+i); break;
            case 14: strata::dequantize_q6_K(source, decoded.data()+i); break;
            default: throw std::invalid_argument("unsupported CPU expert quantization");
            }
        }
        for (int k = 0; k < CpuExperts::kMaxTokens; ++k) {
            if (!out[size_t(k)]) continue;
            const float* xk = x[size_t(k)];
            double sum = 0;
#if defined(_M_X64) || defined(__x86_64__)
            if (avx && width % 8 == 0) sum = avx_dot(decoded.data(), xk, width);
            else
#endif
            for (int i = 0; i < width; ++i) sum += double(decoded[i]) * xk[i];
            out[size_t(k)][row] = float(sum);
        }
    }
}
}

CpuExperts::CpuExperts(unsigned workers, bool fast) : fast_(fast) {
#ifdef LAMINA_CPU_QUANT
    if (fast) {
        if (!avx2_available()) throw std::runtime_error("quantized CPU experts require AVX2");
        static std::once_flag init;
        std::call_once(init, [] { ggml_cpu_init(); });
    }
#else
    if (fast) throw std::runtime_error("quantized CPU experts were not built");
#endif
    try {
        for (unsigned i = 0; i < std::max(1u, workers); ++i) workers_.emplace_back([this] { worker_loop(); });
    } catch (...) {
        stopping_.store(true);
        { std::lock_guard lock(mutex_); }
        wake_.notify_all(); for (auto& worker : workers_) worker.join(); throw;
    }
}

CpuExperts::~CpuExperts() {
    stopping_.store(true, std::memory_order_release);
    { std::lock_guard lock(mutex_); }
    wake_.notify_all();
    for (auto& worker : workers_) worker.join();
}

void CpuExperts::worker_loop() {
    uint64_t seen = 0;
    while (true) {
        int spins = 0;
        uint64_t current;
        while ((current = generation_.load(std::memory_order_acquire)) == seen) {
            if (stopping_.load(std::memory_order_acquire)) return;
            if (++spins < kSpins) { pause(); continue; }
            std::unique_lock lock(mutex_);
            wake_.wait(lock, [&] { return stopping_.load() || generation_.load() != seen; });
            spins = 0;
        }
        seen = current;
        work();
    }
}

void CpuExperts::start(const MoeWeights& weights, const std::vector<int>& ids, const float* input,
                       const std::vector<float*>& outputs) {
    std::vector<std::array<float*, kMaxTokens>> paired(outputs.size());
    for (size_t i = 0; i < outputs.size(); ++i) paired[i] = {outputs[i], nullptr};
    start(weights, ids, {input, nullptr}, 1, paired);
}

void CpuExperts::start(const MoeWeights& weights, const std::vector<int>& ids,
                       const std::array<const float*, kMaxTokens>& inputs, int tokens,
                       const std::vector<std::array<float*, kMaxTokens>>& outputs) {
    if (ids.empty() || ids.size() != outputs.size() || tokens < 1 || tokens > kMaxTokens)
        throw std::invalid_argument("CPU expert batch");
    for (int k = 0; k < tokens; ++k) if (!inputs[size_t(k)]) throw std::invalid_argument("CPU expert batch input");
    for (const auto& out : outputs) {
        bool any = false;
        for (int k = 0; k < kMaxTokens; ++k) {
            if (out[size_t(k)] && k >= tokens) throw std::invalid_argument("CPU expert output for a missing token");
            any |= out[size_t(k)] != nullptr;
        }
        if (!any) throw std::invalid_argument("CPU expert without a routed token");
    }
    const auto& g = *weights.gate;
    const auto& u = *weights.up;
    const auto& d = *weights.down;
    if (g.shape.size() != 3 || u.shape != g.shape || d.shape.size() != 3 || d.shape[0] != g.shape[1])
        throw std::invalid_argument("CPU expert projection shape or type");
    const int ff = int(g.shape[1]), hidden = int(d.shape[1]);
    for (const int id : ids)
        if (id < 0 || id >= int(g.shape[2])) throw std::out_of_range("CPU expert index");
    // wait() left next_ at kIdle, so no worker can claim an item while the
    // batch is rebuilt.
    weights_ = weights; ids_ = ids; tokens_ = tokens; inputs_ = inputs; outputs_ = outputs;
#ifdef LAMINA_CPU_QUANT
    if (fast_) {
        for (int k = 0; k < tokens; ++k) {
            quantize_for(g, inputs[size_t(k)], input_q_[size_t(k)]);
            if (!same_dot_type(g, u)) quantize_for(u, inputs[size_t(k)], input_q_up_[size_t(k)]);
            else input_q_up_[size_t(k)].clear();
        }
    }
#endif
    while (experts_.size() < ids.size()) experts_.push_back(std::make_unique<ExpertState>());
    items_.clear();
    for (int e = 0; e < int(ids.size()); ++e) {
        auto& state = *experts_[size_t(e)];
        for (int k = 0; k < tokens; ++k) {
            state.gate[size_t(k)].resize(size_t(ff)); state.up[size_t(k)].resize(size_t(ff));
            state.swiglu[size_t(k)].resize(size_t(ff));
        }
        state.pending.store(2 * ((ff + kGateRows - 1) / kGateRows), std::memory_order_relaxed);
        state.ready.store(false, std::memory_order_relaxed);
        for (int b = 0; b < ff; b += kGateRows) {
            items_.push_back({e, 0, b, std::min(ff, b + kGateRows)});
            items_.push_back({e, 1, b, std::min(ff, b + kGateRows)});
        }
    }
    // Down items follow every gate/up item, so by the time one is claimed all
    // of its expert's gate/up items have already been claimed by running threads.
    for (int e = 0; e < int(ids.size()); ++e)
        for (int b = 0; b < hidden; b += kDownRows) items_.push_back({e, 2, b, std::min(hidden, b + kDownRows)});
    failed_.store(false, std::memory_order_relaxed);
    error_ = nullptr;
    started_ns_.store(now_ns(), std::memory_order_relaxed);
    first_claim_ns_.store(0, std::memory_order_relaxed);
    stats_.experts += ids.size();
    finished_.store(0, std::memory_order_relaxed);
    count_.store(items_.size(), std::memory_order_relaxed);
    next_.store(0, std::memory_order_release);
    generation_.fetch_add(1, std::memory_order_release);
    { std::lock_guard lock(mutex_); }
    wake_.notify_all();
}

void CpuExperts::wait() {
    work();
    const size_t total = count_.load(std::memory_order_acquire);
    while (finished_.load(std::memory_order_acquire) < total) pause();
    const int64_t done = now_ns();  // observed completion; the last worker's own store could still be pending
    next_.store(kIdle, std::memory_order_release);
    const int64_t begin = started_ns_.load(std::memory_order_relaxed);
    stats_.batch_ms += double(done - begin) * 1e-6;
    stats_.first_claim_ms += double(first_claim_ns_.load(std::memory_order_relaxed) - begin) * 1e-6;
    ++stats_.batches;
    if (failed_.load(std::memory_order_acquire)) {
        std::exception_ptr error;
        { std::lock_guard lock(mutex_); error = error_; }
        std::rethrow_exception(error);
    }
}

void CpuExperts::work() {
    while (true) {
        const size_t i = next_.fetch_add(1, std::memory_order_acq_rel);
        if (i >= count_.load(std::memory_order_acquire)) return;
        const Item item = items_[i];
        if (i == 0) first_claim_ns_.store(now_ns(), std::memory_order_relaxed);
        if (!failed_.load(std::memory_order_acquire)) {
            try { run(item); }
            catch (...) {
                std::lock_guard lock(mutex_);
                if (!error_) error_ = std::current_exception();
                failed_.store(true, std::memory_order_release);
            }
        }
        // Always count gate/up chunks down, even after an error, so no down
        // item can wait forever on its expert.
        if (item.matrix < 2 && experts_[size_t(item.expert)]->pending.fetch_sub(1, std::memory_order_acq_rel) == 1)
            finish_gate_up(item.expert);
        finished_.fetch_add(1, std::memory_order_acq_rel);
    }
}

void CpuExperts::run(const Item& item) {
    auto& state = *experts_[size_t(item.expert)];
    const int expert = ids_[size_t(item.expert)];
    const auto& routed = outputs_[size_t(item.expert)];  // non-null for every token that routed this expert
    if (item.matrix == 2) {
        while (!state.ready.load(std::memory_order_acquire)) pause();
        if (failed_.load(std::memory_order_acquire)) return;
#ifdef LAMINA_CPU_QUANT
        if (fast_) {
            std::array<const uint8_t*, kMaxTokens> q{};
            for (int k = 0; k < kMaxTokens; ++k) if (routed[size_t(k)]) q[size_t(k)] = state.swiglu_q[size_t(k)].data();
            project_quantized_rows(*weights_.down, weights_.down_data, expert, q, routed, item.begin, item.end);
            return;
        }
#endif
        std::array<const float*, kMaxTokens> x{};
        for (int k = 0; k < kMaxTokens; ++k) if (routed[size_t(k)]) x[size_t(k)] = state.swiglu[size_t(k)].data();
        project_rows(*weights_.down, weights_.down_data, expert, x, routed, item.begin, item.end);
        return;
    }
    const auto& tensor = item.matrix == 0 ? *weights_.gate : *weights_.up;
    const uint8_t* data = item.matrix == 0 ? weights_.gate_data : weights_.up_data;
    std::array<float*, kMaxTokens> out{};
    for (int k = 0; k < kMaxTokens; ++k)
        if (routed[size_t(k)]) out[size_t(k)] = (item.matrix == 0 ? state.gate : state.up)[size_t(k)].data();
#ifdef LAMINA_CPU_QUANT
    if (fast_) {
        std::array<const uint8_t*, kMaxTokens> q{};
        for (int k = 0; k < kMaxTokens; ++k) {
            if (!routed[size_t(k)]) continue;
            const auto& quantized = item.matrix == 1 && !input_q_up_[size_t(k)].empty() ? input_q_up_[size_t(k)] : input_q_[size_t(k)];
            q[size_t(k)] = quantized.data();
        }
        project_quantized_rows(tensor, data, expert, q, out, item.begin, item.end);
        return;
    }
#endif
    project_rows(tensor, data, expert, inputs_, out, item.begin, item.end);
}

void CpuExperts::finish_gate_up(int expert) {
    auto& state = *experts_[size_t(expert)];
    const auto& routed = outputs_[size_t(expert)];
    if (!failed_.load(std::memory_order_acquire)) {
        try {
            for (int k = 0; k < kMaxTokens; ++k) {
                if (!routed[size_t(k)]) continue;
                auto& gate = state.gate[size_t(k)];
                auto& up = state.up[size_t(k)];
                auto& swiglu = state.swiglu[size_t(k)];
                for (size_t i = 0; i < gate.size(); ++i)
                    swiglu[i] = (gate[i] / (1.0f + std::exp(-gate[i]))) * up[i];
#ifdef LAMINA_CPU_QUANT
                if (fast_) quantize_for(*weights_.down, swiglu.data(), state.swiglu_q[size_t(k)]);
#endif
            }
        } catch (...) {
            std::lock_guard lock(mutex_);
            if (!error_) error_ = std::current_exception();
            failed_.store(true, std::memory_order_release);
        }
    }
    state.ready.store(true, std::memory_order_release);
}
}
