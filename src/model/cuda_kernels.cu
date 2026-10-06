// Elementwise CUDA kernels for Lamina's device-resident graph. Compiled without
// --use_fast_math (see CMakeLists.txt: lamina_cuda_ops) so expf is the precise
// libdevice function and the 40-layer parity gate keeps its tolerance.

#include "lamina/model/cuda_kernels.hpp"

#include <cuda_runtime.h>

namespace lamina::model::cuda {
namespace {

constexpr int kThreads = 256;

__device__ __forceinline__ float sigmoid_precise(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__global__ void swiglu_kernel(const float* __restrict__ gate, const float* __restrict__ up,
                              float* __restrict__ out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float g = gate[i];
    out[i] = (g * sigmoid_precise(g)) * up[i];
}

__global__ void accumulate_kernel(float* __restrict__ out, const float* __restrict__ source,
                                  float weight, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] += weight * source[i];
}

__global__ void rms_norm_kernel(float* __restrict__ x, const float* __restrict__ weight,
                                int n, float epsilon) {
    __shared__ float partial[kThreads];
    float sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) sum += x[i] * x[i];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    const float scale = rsqrtf(partial[0] / static_cast<float>(n) + epsilon);
    for (int i = threadIdx.x; i < n; i += blockDim.x) x[i] = (x[i] * scale) * weight[i];
}

__global__ void add_inplace_kernel(float* __restrict__ x, const float* __restrict__ y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] += y[i];
}

__global__ void gdn_out_norm_silu_kernel(const float* __restrict__ x, const float* __restrict__ z,
                                         const float* __restrict__ gamma, float* __restrict__ out,
                                         float epsilon) {
    const int col = threadIdx.x;
    const size_t offset = static_cast<size_t>(blockIdx.x) * 128;
    __shared__ float sums[kThreads];
    const float value = col < 128 ? x[offset + col] : 0.0f;
    sums[col] = col < 128 ? value * value : 0.0f;
    __syncthreads();
    for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
        if (col < stride) sums[col] += sums[col + stride];
        __syncthreads();
    }
    if (col < 128) {
        const float scale = rsqrtf(sums[0] / 128.0f + epsilon);
        const float g = z[offset + col];
        const float silu = g / (1.0f + expf(-g));
        out[offset + col] = (value * scale) * gamma[col] * silu;
    }
}

__global__ void moe_combine_kernel(const float* __restrict__ down,
                                   const float* __restrict__ scales, int slots, int hidden,
                                   float* __restrict__ out) {
    const int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= hidden) return;
    float sum = 0.0f;
    for (int s = 0; s < slots; ++s) sum += scales[s] * down[static_cast<size_t>(s) * hidden + o];
    out[o] = sum;
}

__global__ void attn_norm_rope_kernel(float* __restrict__ x, int stride, int n, int n_rot,
                                      const float* __restrict__ gamma, float epsilon, float rope_base,
                                      int position) {
    float* head = x + static_cast<size_t>(blockIdx.x) * stride;
    const int j = threadIdx.x;
    __shared__ float sums[256];
    sums[j] = head[j] * head[j];
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (j < s) sums[j] += sums[j + s];
        __syncthreads();
    }
    const float scale = rsqrtf(sums[0] / static_cast<float>(n) + epsilon);
    head[j] = (head[j] * scale) * gamma[j];
    __syncthreads();
    if (j < n_rot / 2) {
        const float theta = position / powf(rope_base, 2.0f * j / n_rot);
        const float c = cosf(theta), s = sinf(theta);
        const float a = head[j], b = head[j + n_rot / 2];
        head[j] = a * c - b * s;
        head[j + n_rot / 2] = a * s + b * c;
    }
}

__global__ void attn_decode_kernel(const float* __restrict__ q, const float* __restrict__ k_cache,
                                   const float* __restrict__ v_cache, int position, int heads,
                                   int kv_heads, int head_dim, float scale, float* __restrict__ out) {
    const int h = blockIdx.x;
    const int j = threadIdx.x;
    const int kvh = h / (heads / kv_heads);
    const float* qh = q + static_cast<size_t>(h) * 2 * head_dim;
    const float qj = qh[j];
    const int cache_stride = kv_heads * head_dim;
    __shared__ float red[256];
    float run_max = -INFINITY, run_sum = 0.0f, acc = 0.0f;
    for (int t = 0; t <= position; ++t) {
        const float* key = k_cache + static_cast<size_t>(t) * cache_stride + kvh * head_dim;
        red[j] = qj * key[j];
        __syncthreads();
        for (int s = head_dim / 2; s > 0; s >>= 1) {
            if (j < s) red[j] += red[j + s];
            __syncthreads();
        }
        const float score = red[0] * scale;
        const float new_max = fmaxf(run_max, score);
        const float alpha = expf(run_max - new_max);
        const float beta = expf(score - new_max);
        run_sum = run_sum * alpha + beta;
        const float* value = v_cache + static_cast<size_t>(t) * cache_stride + kvh * head_dim;
        acc = acc * alpha + beta * value[j];
        run_max = new_max;
        __syncthreads();
    }
    const float gate = qh[head_dim + j];
    out[static_cast<size_t>(h) * head_dim + j] =
        (acc / run_sum) * (1.0f / (1.0f + expf(-gate)));
}

__global__ void router_topk_kernel(const float* __restrict__ logits, int n, int k,
                                   int* __restrict__ ids, float* __restrict__ weights) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    bool used[256];
    for (int i = 0; i < n; ++i) used[i] = false;
    int chosen[16];
    for (int s = 0; s < k; ++s) {
        int best = -1;
        float best_value = -INFINITY;
        for (int i = 0; i < n; ++i)
            if (!used[i] && logits[i] > best_value) {
                best_value = logits[i];
                best = i;
            }
        used[best] = true;
        chosen[s] = best;
    }
    const float maximum = logits[chosen[0]];
    float sum = 0.0f;
    for (int s = 0; s < k; ++s) sum += expf(logits[chosen[s]] - maximum);
    for (int s = 0; s < k; ++s) {
        ids[s] = chosen[s];
        weights[s] = expf(logits[chosen[s]] - maximum) / sum;
    }
}

__global__ void dot_sigmoid_kernel(const float* __restrict__ a, const float* __restrict__ b,
                                   int n, float* __restrict__ out) {
    __shared__ float sums[kThreads];
    float partial = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) partial += a[i] * b[i];
    sums[threadIdx.x] = partial;
    __syncthreads();
    for (int s = kThreads / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sums[threadIdx.x] += sums[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[0] = 1.0f / (1.0f + expf(-sums[0]));
}

__global__ void doorbell_signal_kernel(int* flag, int value) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        __threadfence_system();
        atomicExch(flag, value);
    }
}

__global__ void gemv_f32_kernel(const float* __restrict__ w, const float* __restrict__ x,
                                float* __restrict__ y, int n_in, int n_out) {
    __shared__ float partial[kThreads];
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    float sum = 0.0f;
    for (int i = tid; i < n_in; i += blockDim.x) sum += w[static_cast<size_t>(row) * n_in + i] * x[i];
    partial[tid] = sum;
    __syncthreads();
    for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        __syncthreads();
    }
    if (tid == 0) y[row] = partial[0];
}

int blocks(int n) { return (n + kThreads - 1) / kThreads; }

}  // namespace

void swiglu(const float* gate, const float* up, float* out, int n, void* stream) {
    swiglu_kernel<<<blocks(n), kThreads, 0, static_cast<cudaStream_t>(stream)>>>(gate, up, out, n);
}

void accumulate(float* out, const float* source, float weight, int n, void* stream) {
    accumulate_kernel<<<blocks(n), kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
        out, source, weight, n);
}

void rms_norm(float* x, const float* weight, int n, float epsilon, void* stream) {
    rms_norm_kernel<<<1, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(x, weight, n, epsilon);
}

void add_inplace(float* x, const float* y, int n, void* stream) {
    add_inplace_kernel<<<blocks(n), kThreads, 0, static_cast<cudaStream_t>(stream)>>>(x, y, n);
}

void gemv_f32(const float* weights, const float* x, float* y, int n_in, int n_out, void* stream) {
    gemv_f32_kernel<<<static_cast<unsigned>(n_out), kThreads, 0,
                      static_cast<cudaStream_t>(stream)>>>(weights, x, y, n_in, n_out);
}

void gdn_out_norm_silu(const float* x, const float* z, const float* gamma, float* out, int heads,
                       float epsilon, void* stream) {
    gdn_out_norm_silu_kernel<<<static_cast<unsigned>(heads), kThreads, 0,
                               static_cast<cudaStream_t>(stream)>>>(x, z, gamma, out, epsilon);
}

void moe_combine(const float* down, const float* scales, int slots, int hidden, float* out,
                 void* stream) {
    moe_combine_kernel<<<blocks(hidden), kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
        down, scales, slots, hidden, out);
}

void attn_norm_rope(float* x, int heads, int stride, int n, int n_rot, const float* gamma,
                    float epsilon, float rope_base, int position, void* stream) {
    attn_norm_rope_kernel<<<static_cast<unsigned>(heads), static_cast<unsigned>(n), 0,
                            static_cast<cudaStream_t>(stream)>>>(x, stride, n, n_rot, gamma,
                                                                 epsilon, rope_base, position);
}

void attn_decode(const float* q, const float* k_cache, const float* v_cache, int position,
                 int heads, int kv_heads, int head_dim, float scale, float* out, void* stream) {
    attn_decode_kernel<<<static_cast<unsigned>(heads), static_cast<unsigned>(head_dim), 0,
                         static_cast<cudaStream_t>(stream)>>>(q, k_cache, v_cache, position, heads,
                                                              kv_heads, head_dim, scale, out);
}

void router_topk(const float* logits, int n, int k, int* ids, float* weights, void* stream) {
    router_topk_kernel<<<1, 1, 0, static_cast<cudaStream_t>(stream)>>>(logits, n, k, ids, weights);
}

void dot_sigmoid(const float* a, const float* b, int n, float* out, void* stream) {
    dot_sigmoid_kernel<<<1, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(a, b, n, out);
}

void doorbell_signal(int* flag, int value, void* stream) {
    doorbell_signal_kernel<<<1, 1, 0, static_cast<cudaStream_t>(stream)>>>(flag, value);
}

}  // namespace lamina::model::cuda
