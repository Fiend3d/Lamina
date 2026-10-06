// Isolated numeric check for Lamina's elementwise CUDA kernels against a host
// reference. GPU + toolkit only.

#include "lamina/model/cuda_kernels.hpp"

#include <cuda_runtime.h>

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

    cudaFree(dg);
    cudaFree(du);
    cudaFree(dout);
    cudaStreamDestroy(stream);
    std::printf("swiglu max_abs_diff=%.6g accumulate max_abs_diff=%.6g\n", max_swiglu, max_accum);
    return (max_swiglu < 1e-5 && max_accum < 1e-5) ? 0 : 1;
}
