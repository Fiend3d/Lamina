// Isolated numeric check for Lamina's elementwise CUDA kernels against a host
// reference. GPU + toolkit only.

#include "lamina/model/cuda_kernels.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

int main() {
    const int n = 512;
    std::vector<float> gate(n), up(n), out(n, 0.0f), ref(n, 0.0f), base(n, 0.0f);
    for (int i = 0; i < n; ++i) {
        gate[i] = 0.15f * i - 30.0f;
        up[i] = std::sin(0.3f * i) + 0.01f * i;
        ref[i] = (gate[i] / (1.0f + std::exp(-gate[i]))) * up[i];
        base[i] = 0.5f * i - 20.0f;
    }
    float *dg = nullptr, *du = nullptr, *dout = nullptr;
    cudaMalloc(&dg, n * sizeof(float));
    cudaMalloc(&du, n * sizeof(float));
    cudaMalloc(&dout, n * sizeof(float));
    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);
    cudaMemcpy(dg, gate.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(du, up.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    lamina::model::cuda::swiglu(dg, du, dout, n, stream);
    cudaStreamSynchronize(stream);
    cudaMemcpy(out.data(), dout, n * sizeof(float), cudaMemcpyDeviceToHost);
    double max_swiglu = 0.0;
    for (int i = 0; i < n; ++i) max_swiglu = std::max(max_swiglu, std::abs(double(out[i]) - ref[i]));

    cudaMemcpy(dout, base.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    lamina::model::cuda::accumulate(dout, dg, 0.25f, n, stream);
    cudaStreamSynchronize(stream);
    cudaMemcpy(out.data(), dout, n * sizeof(float), cudaMemcpyDeviceToHost);
    double max_accum = 0.0;
    for (int i = 0; i < n; ++i) {
        const double want = double(base[i]) + 0.25 * double(gate[i]);
        max_accum = std::max(max_accum, std::abs(double(out[i]) - want));
    }

    // The first rotary pair has frequency exactly one. Its largest argument
    // can be checked directly against double-precision sin/cos without an
    // implementation-dependent rounding of the remaining FP32 frequencies.
    double max_long_rope = 0.0;
    std::vector<float> rotary(n, 0.0f), gamma(n, 0.0f);
    rotary[0] = 3.0f; rotary[32] = 4.0f;
    gamma[0] = gamma[32] = 1.0f;
    for (int i = 256; i < n; ++i) rotary[i] = float(i) * 0.01f;
    const double norm = 1.0 / std::sqrt(25.0 / 256.0 + 1e-6);
    for (int position : {0, 32767, 65536, 131071}) {
        cudaMemcpy(dg, rotary.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(du, gamma.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        lamina::model::cuda::attn_norm_mrope(dg, 1, 512, 256, 64, du, 1e-6f,
                                           10000000.0f, position, position, position, stream);
        cudaStreamSynchronize(stream);
        cudaMemcpy(out.data(), dg, n * sizeof(float), cudaMemcpyDeviceToHost);
        for (int i = 0; i < n; ++i) {
            double want = i >= 256 ? rotary[i] : 0.0;
            if (i == 0) want = norm * (3.0 * std::cos(double(position)) - 4.0 * std::sin(double(position)));
            if (i == 32) want = norm * (3.0 * std::sin(double(position)) + 4.0 * std::cos(double(position)));
            max_long_rope = std::max(max_long_rope, std::abs(double(out[i]) - want));
        }
    }

    cudaFree(dg);
    cudaFree(du);
    cudaFree(dout);
    cudaStreamDestroy(stream);
    std::printf("swiglu max_abs_diff=%.6g accumulate max_abs_diff=%.6g long_rope_first_pair max_abs_diff=%.6g\n",
                max_swiglu, max_accum, max_long_rope);
    return (max_swiglu < 1e-5 && max_accum < 1e-5 && max_long_rope < 1e-5) ? 0 : 1;
}
