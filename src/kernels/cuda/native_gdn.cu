// Arithmetic adapted from gated_delta_net.cu at pinned llama.cpp
// 3cf03257f219afbe7334045ff7c6a06ac68c627d. Only state addressing differs:
// each warp owns one Strata (head,column) and retains its four rows in registers.
// Compile with --use_fast_math to match the pinned CUDA implementation.
//
// MIT License
// Copyright (c) 2023-2026 The ggml authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#include "strata/kernels/native_gdn.hpp"
#include <cuda_runtime.h>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace strata::kernels {
namespace {
std::atomic<bool> enabled{false};
constexpr int S = 128;

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        value += __shfl_xor_sync(0xffffffff, value, offset, 32);
    return value;
}

template<bool Multi>
__global__ void __launch_bounds__(128, 2)
step(float* __restrict__ state, const float* __restrict__ q, const float* __restrict__ k,
     const float* __restrict__ v, const float* __restrict__ gate, const float* __restrict__ beta,
     float* __restrict__ output, int h_k, int h_v, float scale, int columns) {
    const int head = blockIdx.x;
    const int lane = threadIdx.x;
    const int col = blockIdx.z * blockDim.y + threadIdx.y;
    const int q_head = head % h_k;
    float s_shard[4], k_reg[4], q_reg[4];
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        const int i = r * 32 + lane;
        s_shard[r] = state[(size_t(i) * h_v + head) * S + col];
    }
    const int count = Multi ? columns : 1;
    for (int token = 0; token < count; ++token) {
    const size_t qkv_offset = size_t(token) * 8192;
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        const int i = r * 32 + lane;
        k_reg[r] = k[qkv_offset + q_head * S + i];
        q_reg[r] = q[qkv_offset + q_head * S + i];
    }
    const float g_val = expf(gate[size_t(token) * h_v + head]);
    float kv_shard = 0.0f;
#pragma unroll
    for (int r = 0; r < 4; ++r) kv_shard += s_shard[r] * k_reg[r];
    const float kv_col = warp_sum(kv_shard);
    const float delta_col = (v[qkv_offset + head * S + col] - g_val * kv_col) * beta[size_t(token) * h_v + head];
    float attn_partial = 0.0f;
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        s_shard[r] = g_val * s_shard[r] + k_reg[r] * delta_col;
        attn_partial += s_shard[r] * q_reg[r];
    }
    const float attn_col = warp_sum(attn_partial);
    if (lane == 0) output[(size_t(token) * h_v + head) * S + col] = attn_col * scale;
    }
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        const int i = r * 32 + lane;
        state[(size_t(i) * h_v + head) * S + col] = s_shard[r];
    }
}

// Decode variant of step<false> with coalesced state traffic. In step<false>
// the 32 lanes of a warp read rows i = r * 32 + lane of one column, addresses
// h_v * S floats (16 KiB) apart, so every load touches its own 32-byte sector.
// Here a block stages a [128 rows x 32 columns] tile of one head through shared
// memory with 128-byte row loads and stores. Each warp then runs the same
// per-(head, column) computation as step<false>, with the same lane-to-row
// mapping, the same sums and the same warp reductions, so results are equal.
constexpr int kTileCols = 32;
__global__ void __launch_bounds__(256)
step_tiled(float* __restrict__ state, const float* __restrict__ q, const float* __restrict__ k,
           const float* __restrict__ v, const float* __restrict__ gate, const float* __restrict__ beta,
           float* __restrict__ output, int h_k, int h_v, float scale) {
    __shared__ float tile[S][kTileCols + 1];  // +1 column: conflict-free column reads
    const int head = blockIdx.x, col0 = blockIdx.y * kTileCols;
    const int lane = threadIdx.x, tid = threadIdx.y * 32 + threadIdx.x;
    const int threads = blockDim.x * blockDim.y;
    for (int index = tid; index < S * kTileCols; index += threads) {
        const int i = index / kTileCols, c = index % kTileCols;
        tile[i][c] = state[(size_t(i) * h_v + head) * S + col0 + c];
    }
    __syncthreads();
    const int q_head = head % h_k;
    float k_reg[4], q_reg[4];
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        const int i = r * 32 + lane;
        k_reg[r] = k[q_head * S + i];
        q_reg[r] = q[q_head * S + i];
    }
    const float g_val = expf(gate[head]);
    for (int cc = threadIdx.y; cc < kTileCols; cc += blockDim.y) {
        const int col = col0 + cc;
        float s_shard[4];
#pragma unroll
        for (int r = 0; r < 4; ++r) s_shard[r] = tile[r * 32 + lane][cc];
        float kv_shard = 0.0f;
#pragma unroll
        for (int r = 0; r < 4; ++r) kv_shard += s_shard[r] * k_reg[r];
        const float kv_col = warp_sum(kv_shard);
        const float delta_col = (v[head * S + col] - g_val * kv_col) * beta[head];
        float attn_partial = 0.0f;
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            s_shard[r] = g_val * s_shard[r] + k_reg[r] * delta_col;
            attn_partial += s_shard[r] * q_reg[r];
        }
        const float attn_col = warp_sum(attn_partial);
        if (lane == 0) output[head * S + col] = attn_col * scale;
#pragma unroll
        for (int r = 0; r < 4; ++r) tile[r * 32 + lane][cc] = s_shard[r];
    }
    __syncthreads();
    for (int index = tid; index < S * kTileCols; index += threads) {
        const int i = index / kTileCols, c = index % kTileCols;
        state[(size_t(i) * h_v + head) * S + col0 + c] = tile[i][c];
    }
}

bool valid_span(const void* pointer, size_t bytes) {
    const auto address = reinterpret_cast<uintptr_t>(pointer);
    return pointer && address % sizeof(float) == 0 && bytes <= UINTPTR_MAX - address;
}
bool overlap(const void* a, size_t an, const void* b, size_t bn) {
    const auto ap = reinterpret_cast<uintptr_t>(a), bp = reinterpret_cast<uintptr_t>(b);
    return ap < bp + bn && bp < ap + an;
}
}

void native_gdn_set_enabled(bool value) { enabled.store(value, std::memory_order_relaxed); }
bool native_gdn_enabled() { return enabled.load(std::memory_order_relaxed); }

void native_gdn_step(float* state, const float* q, const float* k, const float* v,
                     const float* gate, const float* beta, float* output,
                     const GdnShapes& shape, void* stream) {
    if (!stream || shape.S != S || shape.h_k <= 0 || shape.h_v <= 0 ||
        shape.h_v > 65535 || shape.h_v % shape.h_k != 0)
        throw std::invalid_argument("native GDN requires a stream, S=128 and positive divisible head counts <=65535");
    const size_t state_bytes = size_t(S) * S * size_t(shape.h_v) * sizeof(float);
    const size_t qk_bytes = size_t(S) * size_t(shape.h_k) * sizeof(float);
    const size_t output_bytes = size_t(S) * size_t(shape.h_v) * sizeof(float);
    const size_t head_bytes = size_t(shape.h_v) * sizeof(float);
    if (!valid_span(state, state_bytes) || !valid_span(output, output_bytes) ||
        overlap(state, state_bytes, output, output_bytes))
        throw std::invalid_argument("native GDN requires aligned, disjoint state and output spans");
    const void* inputs[] = {q, k, v, gate, beta};
    const size_t bytes[] = {qk_bytes, qk_bytes, output_bytes, head_bytes, head_bytes};
    for (int i = 0; i < 5; ++i) {
        if (!valid_span(inputs[i], bytes[i]) || overlap(state, state_bytes, inputs[i], bytes[i]) ||
            overlap(output, output_bytes, inputs[i], bytes[i]))
            throw std::invalid_argument("native GDN requires aligned input spans disjoint from state and output");
    }
    const float scale = 1.0f / sqrtf(float(S));
    step_tiled<<<dim3(unsigned(shape.h_v), S / kTileCols), dim3(32, 8), 0, static_cast<cudaStream_t>(stream)>>>(
        state, q, k, v, gate, beta, output, int(shape.h_k), int(shape.h_v), scale);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

void native_gdn_step_columns(float* state, const float* qkv, const float* gate,
                              const float* beta, float* output, int columns, void* stream) {
    if (!stream || columns < 1 || columns > 2048 || !state || !qkv || !gate || !beta || !output)
        throw std::invalid_argument("invalid GDN prefill spans");
    step<true><<<dim3(32, 1, S / 4), dim3(32, 4), 0, static_cast<cudaStream_t>(stream)>>>(
        state, qkv, qkv + 2048, qkv + 4096, gate, beta, output, 16, 32, 1.0f / sqrtf(float(S)), columns);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
} // namespace strata::kernels
