// Two-column Q8 table MMVQ (native_mmvq_q8_grouped2) against two single-column
// launches on real model weights. Speculative verification relies on every
// column being bitwise equal to the single-token decode kernel.
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
}

bool compare(const strata::GgufFile& file, const std::string& name, int64_t expert, cudaStream_t stream) {
    const auto* t = file.find(name);
    if (!t) throw std::runtime_error("missing tensor: " + name);
    const int n_in = int(t->shape.at(0)), n_out = int(t->shape.at(1));
    const size_t bytes = strata::kernels::native_mmvq_weight_bytes(t->type, n_in, n_out);
    const uint8_t* source = file.tensor_data(*t) + (expert < 0 ? 0 : size_t(expert) * bytes);
    void* weight = nullptr;
    check(cudaMalloc(&weight, bytes), "weight");
    check(cudaMemcpy(weight, source, bytes, cudaMemcpyHostToDevice), "upload weight");
    std::mt19937 random(uint32_t(n_in * 31 + n_out));
    std::normal_distribution<float> normal(0.f, 1.f);
    std::vector<float> x(size_t(2) * n_in);
    for (auto& v : x) v = normal(random);
    float* x_dev = nullptr;
    check(cudaMalloc(&x_dev, x.size() * sizeof(float)), "x");
    check(cudaMemcpy(x_dev, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice), "upload x");
    const size_t q_bytes = strata::kernels::native_q8_1_bytes(n_in);
    uint8_t* q = nullptr;
    check(cudaMalloc(&q, 2 * q_bytes), "q");
    strata::kernels::native_quantize_q8_1(x_dev, q, n_in, 2, stream);  // two consecutive columns
    float* y = nullptr;
    check(cudaMalloc(&y, size_t(6) * n_out * sizeof(float)), "y");
    for (int c = 0; c < 2; ++c) {  // reference: one single-column launch per column
        strata::kernels::NativeF32Grouped args{};
        args.count = 1; args.weights[0] = weight; args.n_outs[0] = n_out;
        args.inputs[0] = reinterpret_cast<const float*>(q + c * q_bytes);
        args.outputs[0] = y + size_t(c) * n_out;
        strata::kernels::native_mmvq_q8_grouped(t->type, args, n_out, n_in, stream);
    }
    strata::kernels::NativeF32Grouped args{};
    args.count = 1; args.weights[0] = weight; args.n_outs[0] = n_out;
    args.inputs[0] = reinterpret_cast<const float*>(q);
    args.outputs[0] = y + size_t(2) * n_out;
    strata::kernels::native_mmvq_q8_grouped2(t->type, args, n_out, n_in, int(q_bytes / 36), n_out, stream);
    strata::kernels::NativeF32Pairs pairs{};  // explicit per-column pointers, columns written out of order
    pairs.count = 1; pairs.weights[0] = weight; pairs.n_outs[0] = n_out;
    pairs.inputs[0][0] = reinterpret_cast<const float*>(q); pairs.inputs[1][0] = reinterpret_cast<const float*>(q + q_bytes);
    pairs.outputs[0][0] = y + size_t(4) * n_out; pairs.outputs[1][0] = y + size_t(5) * n_out;
    strata::kernels::native_mmvq_q8_pairs(t->type, pairs, n_out, n_in, stream);
    check(cudaStreamSynchronize(stream), "run");
    // Timing: two single-column launches against one pair launch.
    cudaEvent_t e0, e1, e2;
    cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventCreate(&e2);
    constexpr int kRuns = 50;
    strata::kernels::NativeF32Grouped one{};
    one.count = 1; one.weights[0] = weight; one.n_outs[0] = n_out;
    cudaEventRecord(e0, stream);
    for (int r = 0; r < kRuns; ++r)
        for (int c = 0; c < 2; ++c) {
            one.inputs[0] = reinterpret_cast<const float*>(q + c * q_bytes); one.outputs[0] = y + size_t(c) * n_out;
            strata::kernels::native_mmvq_q8_grouped(t->type, one, n_out, n_in, stream);
        }
    cudaEventRecord(e1, stream);
    for (int r = 0; r < kRuns; ++r) strata::kernels::native_mmvq_q8_pairs(t->type, pairs, n_out, n_in, stream);
    cudaEventRecord(e2, stream);
    check(cudaEventSynchronize(e2), "time");
    float single_ms = 0, pair_ms = 0;
    cudaEventElapsedTime(&single_ms, e0, e1); cudaEventElapsedTime(&pair_ms, e1, e2);
    std::printf("%-32s two single launches %.1f us, one pair launch %.1f us\n", name.c_str(), 1000 * single_ms / kRuns, 1000 * pair_ms / kRuns);
    std::vector<float> out(size_t(6) * n_out);
    check(cudaMemcpy(out.data(), y, out.size() * sizeof(float), cudaMemcpyDeviceToHost), "download");
    const bool equal = std::memcmp(out.data(), out.data() + size_t(2) * n_out, size_t(2) * n_out * sizeof(float)) == 0 &&
                       std::memcmp(out.data(), out.data() + size_t(4) * n_out, size_t(2) * n_out * sizeof(float)) == 0;
    std::printf("%-32s type=%u %dx%d expert=%lld two-column and pair launches bitwise %s\n", name.c_str(), unsigned(t->type), n_in, n_out,
                static_cast<long long>(expert), equal ? "equal" : "DIFFERENT");
    cudaFree(weight); cudaFree(x_dev); cudaFree(q); cudaFree(y);
    return equal;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) { std::fprintf(stderr, "usage: lamina-cuda-ncols-check MODEL\n"); return 2; }
    try {
        strata::GgufFile file(argv[1]);
        cudaStream_t stream = nullptr;
        check(cudaStreamCreate(&stream), "stream");
        bool ok = true;
        ok &= compare(file, "blk.0.attn_qkv.weight", -1, stream);        // Q8_0, 2048 wide: four-warp kernel
        ok &= compare(file, "blk.0.ssm_out.weight", -1, stream);         // Q8_0, 4096 wide
        ok &= compare(file, "blk.0.ffn_gate_exps.weight", 3, stream);    // Q4_K, warp-per-row kernel
        ok &= compare(file, "blk.0.ffn_down_exps.weight", 3, stream);    // Q5_K, 512 wide
        ok &= compare(file, "blk.34.ffn_down_exps.weight", 3, stream);   // Q6_K, 512 wide
        ok &= compare(file, "blk.0.ffn_down_shexp.weight", -1, stream);  // Q8_0, 512 wide
        std::puts(ok ? "two-column checks passed" : "two-column checks FAILED");
        return ok ? 0 : 1;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "lamina-cuda-ncols-check: %s\n", error.what());
        return 1;
    }
}
