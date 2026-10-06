#include "lamina/model/cpu_experts.hpp"
#include "strata/artifact/dequant.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#if defined(_M_X64) || defined(__x86_64__)
#include <immintrin.h>
#ifdef _MSC_VER
#include <intrin.h>
#endif
#endif

namespace lamina::model {
namespace {
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
std::vector<float> project(const strata::TensorInfo& t, const uint8_t* data,
                           int expert, const std::vector<float>& x) {
    int block = 0, bytes = 0;
    if (!strata::block_geometry(t.type, block, bytes) || x.size() != t.shape[0])
        throw std::invalid_argument("CPU expert projection shape or type");
    const int width = int(t.shape[0]), rows = int(t.shape[1]);
    const size_t row_bytes = size_t(width / block) * bytes;
    const uint8_t* source = data + size_t(expert) * rows * row_bytes;
    std::vector<float> decoded(width), out(rows);
    static const bool avx = avx2_available();
    for (int row = 0; row < rows; ++row) {
        for (int i = 0; i < width; i += block, source += bytes) {
            switch (t.type) {
            case 8: strata::dequantize_q8_0(source, decoded.data()+i); break;
            case 12: strata::dequantize_q4_K(source, decoded.data()+i); break;
            case 13: strata::dequantize_q5_K(source, decoded.data()+i); break;
            case 14: strata::dequantize_q6_K(source, decoded.data()+i); break;
            default: throw std::invalid_argument("unsupported CPU expert quantization");
            }
        }
        double sum = 0;
#if defined(_M_X64) || defined(__x86_64__)
        if (avx && width % 8 == 0) sum = avx_dot(decoded.data(), x.data(), width);
        else
#endif
        for (int i = 0; i < width; ++i) sum += double(decoded[i]) * x[i];
        out[row] = float(sum);
    }
    return out;
}
}
CpuExperts::CpuExperts(unsigned workers) {
    try {
        for (unsigned i = 0; i < std::max(1u, workers); ++i) workers_.emplace_back([this] {
            while (true) {
                std::function<void()> job;
                {
                    std::unique_lock lock(mutex_);
                    ready_.wait(lock, [this] { return stopping_ || !jobs_.empty(); });
                    if (stopping_ && jobs_.empty()) return;
                    job = std::move(jobs_.front()); jobs_.pop();
                }
                job();
            }
        });
    } catch (...) {
        { std::lock_guard lock(mutex_); stopping_ = true; }
        ready_.notify_all(); for (auto& worker : workers_) worker.join(); throw;
    }
}
CpuExperts::~CpuExperts() {
    { std::lock_guard lock(mutex_); stopping_ = true; }
    ready_.notify_all(); for (auto& worker : workers_) worker.join();
}
std::future<std::vector<float>> CpuExperts::submit(MoeWeights weights, int expert,
                                                 std::shared_ptr<const std::vector<float>> input) {
    auto task = std::make_shared<std::packaged_task<std::vector<float>()>>([weights, expert, input] {
        auto gate = project(*weights.gate, weights.gate_data, expert, *input);
        auto up = project(*weights.up, weights.up_data, expert, *input);
        for (size_t i = 0; i < gate.size(); ++i) gate[i] = (gate[i] / (1.0f + std::exp(-gate[i]))) * up[i];
        return project(*weights.down, weights.down_data, expert, gate);
    });
    auto result = task->get_future();
    { std::lock_guard lock(mutex_); jobs_.emplace([task] { (*task)(); }); }
    ready_.notify_one(); return result;
}
}
