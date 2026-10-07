// Elementwise CUDA kernels for Lamina's device-resident graph. Compiled without
// --use_fast_math (see CMakeLists.txt: lamina_cuda_ops) so expf is the precise
// libdevice function and the 40-layer parity gate keeps its tolerance.

#include "lamina/model/cuda_kernels.hpp"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace lamina::model::cuda {
namespace {

constexpr int kThreads = 256;

__global__ void to_bf16_kernel(const float* source, __nv_bfloat16* destination, int n) {
    int i = int(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) destination[i] = __float2bfloat16_rn(source[i]);
}


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
    x += size_t(blockIdx.x) * n;
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
                                      int position, int pos_h, int pos_w, const int* positions = nullptr, int heads = 0) {
    if (positions) { position=positions[blockIdx.y*3]; pos_h=positions[blockIdx.y*3+1]; pos_w=positions[blockIdx.y*3+2]; }
    float* head = x + (size_t(blockIdx.y)*heads + blockIdx.x) * stride;
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
        const int pos = j % 3 == 1 && j < 33 ? pos_h : j % 3 == 2 && j < 30 ? pos_w : position;
        const float theta = pos / powf(rope_base, 2.0f * j / n_rot);
        const float c = cosf(theta), s = sinf(theta);
        const float a = head[j], b = head[j + n_rot / 2];
        head[j] = a * c - b * s;
        head[j + n_rot / 2] = a * s + b * c;
    }
}

// One warp per query head; all eight heads share each KV load. Keep the
// original 128-key reduction and value accumulation order for FP32 parity.
template<class T>
__global__ void attn_gqa_partial_kernel(const float* q, const T* keys, const T* values,
    int count, int tile_offset, int total_tiles, float* partial) {
    const int lane = threadIdx.x & 31, head = threadIdx.x >> 5;
    const int kv = blockIdx.y, h = kv * 8 + head, start = blockIdx.x * 128;
    __shared__ float tile[16 * 256], scores[8][128], reduction[8][256];
    float query[8], acc[8]{};
    for (int j=0; j<8; ++j) query[j] = q[h*512+lane+j*32];
    for (int begin=0; begin<128; begin+=16) {
        for (int i=threadIdx.x; i<16*256; i+=256) {
            const int t=start+begin+i/256;
            tile[i]=t<count ? float(keys[size_t(t)*512+kv*256+i%256]) : 0.0f;
        }
        __syncthreads();
        for (int t=0; t<16; ++t) {
            float dot=0;
            for (int j=0; j<8; ++j) dot += query[j]*tile[t*256+lane+j*32];
            for (int offset=16; offset; offset>>=1) dot += __shfl_down_sync(0xffffffffu,dot,offset);
            if (!lane) scores[head][begin+t]=start+begin+t<count ? dot*0.0625f : -INFINITY;
        }
        __syncthreads();
    }
    for (int i=lane; i<256; i+=32) reduction[head][i]=i<128 ? scores[head][i] : -INFINITY;
    __syncthreads();
    for (int step=128; step; step>>=1) {
        for (int i=lane; i<step; i+=32) reduction[head][i]=fmaxf(reduction[head][i],reduction[head][i+step]);
        __syncthreads();
    }
    const float maximum=reduction[head][0];
    for (int i=lane; i<128; i+=32) scores[head][i]=isfinite(maximum) ? expf(scores[head][i]-maximum) : 0;
    __syncthreads();
    for (int i=lane; i<256; i+=32) reduction[head][i]=i<128 ? scores[head][i] : 0;
    __syncthreads();
    for (int step=128; step; step>>=1) {
        for (int i=lane; i<step; i+=32) reduction[head][i] += reduction[head][i+step];
        __syncthreads();
    }
    for (int begin=0; begin<128; begin+=16) {
        for (int i=threadIdx.x; i<16*256; i+=256) {
            const int t=start+begin+i/256;
            tile[i]=t<count ? float(values[size_t(t)*512+kv*256+i%256]) : 0;
        }
        __syncthreads();
        for (int t=0; t<16 && start+begin+t<count; ++t)
            for (int j=0; j<8; ++j) acc[j] += scores[head][begin+t]*tile[t*256+lane+j*32];
        __syncthreads();
    }
    float* dst=partial+(size_t(h)*total_tiles+tile_offset+blockIdx.x)*258;
    if (!lane) { dst[0]=maximum; dst[1]=reduction[head][0]; }
    for (int j=0; j<8; ++j) dst[2+lane+j*32]=acc[j];
}

template<class T>
__global__ void attn_partial_kernel(const float* q, const T* keys, const T* values,
                                    int count, int tile_offset, int total_tiles, int heads,
                                    int kv_heads, int head_dim, float scale, float* partial,
                                    int query_start = -1, int key_start = 0) {
    const int h = blockIdx.y, tid = threadIdx.x;
    const int lane = tid & 31, warp = tid >> 5;
    const int start = blockIdx.x * 128, kvh = h / (heads / kv_heads);
    const int col = blockIdx.z;
    const int valid_count = query_start < 0 ? count : min(count, max(0, query_start + col + 1 - key_start));
    const float* query = q + (size_t(col) * heads + h) * 2 * head_dim;
    __shared__ float scores[128], reduction[256];
    for (int t = warp; t < 128; t += 8) {
        float score = 0.0f;
        if (start + t < valid_count) {
            const T* key = keys + size_t(start + t) * kv_heads * head_dim + kvh * head_dim;
            for (int j = lane; j < head_dim; j += 32) score += query[j] * float(key[j]);
            for (int offset = 16; offset; offset >>= 1)
                score += __shfl_down_sync(0xffffffffu, score, offset);
        }
        if (!lane) scores[t] = start + t < valid_count ? score * scale : -INFINITY;
    }
    __syncthreads();
    reduction[tid] = tid < 128 ? scores[tid] : -INFINITY;
    __syncthreads();
    for (int s = 128; s; s >>= 1) {
        if (tid < s) reduction[tid] = fmaxf(reduction[tid], reduction[tid + s]);
        __syncthreads();
    }
    const float maximum = reduction[0];
    __syncthreads();
    if (tid < 128) scores[tid] = isfinite(maximum) ? expf(scores[tid] - maximum) : 0.0f;
    __syncthreads();
    reduction[tid] = tid < 128 ? scores[tid] : 0.0f;
    __syncthreads();
    for (int s = 128; s; s >>= 1) {
        if (tid < s) reduction[tid] += reduction[tid + s];
        __syncthreads();
    }
    float* row = partial + ((size_t(col) * heads + h) * total_tiles + tile_offset + blockIdx.x) * (head_dim + 2);
    if (!tid) { row[0] = maximum; row[1] = reduction[0]; }
    if (tid < head_dim) {
        float acc = 0.0f;
        for (int t = 0; t < 128 && start + t < valid_count; ++t)
            acc += scores[t] * float(values[size_t(start + t) * kv_heads * head_dim + kvh * head_dim + tid]);
        row[tid + 2] = acc;
    }
}

// Eight query warps reuse a KV head in shared memory. Keep the reference
// 128-key dot/value/reduction order, but merge directly into the accumulator:
// no [query,head,key-tile,258] intermediate or second merge launch.
template<class T>
__global__ void attn_fused_columns_kernel(const float* q, const T* keys, const T* values,
    int count, int key_start, int query_start, int columns, float* accumulator, bool first) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int h = blockIdx.x, col = blockIdx.y * 8 + warp, kvh = h / 8;
    const bool active = col < columns;
    const int valid = active ? min(count, max(0, query_start + col + 1 - key_start)) : 0;
    __shared__ float tile[16 * 256];
    __shared__ float scores[8][128];
    float query[8], acc[8];
    float* dst = active ? accumulator + (size_t(col) * 16 + h) * 258 : nullptr;
    float maximum = first || !active ? -INFINITY : dst[0];
    float denominator = first || !active ? 0.0f : dst[1];
    for (int j = 0; j < 8; ++j) {
        query[j] = active ? q[(size_t(col) * 16 + h) * 512 + lane + j * 32] : 0.0f;
        acc[j] = first || !active ? 0.0f : dst[lane + j * 32 + 2];
    }
    for (int begin = 0; begin < count; begin += 128) {
        for (int sub = 0; sub < 128; sub += 16) {
            for (int i = threadIdx.x; i < 16 * 256; i += 256) {
                const int t = begin + sub + i / 256;
                tile[i] = t < count ? float(keys[size_t(t) * 512 + kvh * 256 + i % 256]) : 0.0f;
            }
            __syncthreads();
            for (int t = 0; t < 16; ++t) {
                float dot = 0.0f;
                if (begin + sub + t < valid)
                    for (int j = 0; j < 8; ++j) dot += query[j] * tile[t * 256 + lane + j * 32];
                for (int s = 16; s; s >>= 1) dot += __shfl_down_sync(0xffffffffu, dot, s);
                if (!lane) scores[warp][sub + t] = begin + sub + t < valid ? dot / 16.0f : -INFINITY;
            }
            __syncthreads();
        }
        float m = fmaxf(fmaxf(scores[warp][lane], scores[warp][lane + 64]),
                       fmaxf(scores[warp][lane + 32], scores[warp][lane + 96]));
        for (int s = 16; s; s >>= 1) m = fmaxf(m, __shfl_down_sync(0xffffffffu, m, s));
        m = __shfl_sync(0xffffffffu, m, 0);
        for (int t = lane; t < 128; t += 32)
            scores[warp][t] = isfinite(m) ? expf(scores[warp][t] - m) : 0.0f;
        __syncwarp();
        float d = (scores[warp][lane] + scores[warp][lane + 64]) +
                  (scores[warp][lane + 32] + scores[warp][lane + 96]);
        for (int s = 16; s; s >>= 1) d += __shfl_down_sync(0xffffffffu, d, s);
        d = __shfl_sync(0xffffffffu, d, 0);
        float sum[8] = {};
        for (int sub = 0; sub < 128; sub += 16) {
            for (int i = threadIdx.x; i < 16 * 256; i += 256) {
                const int t = begin + sub + i / 256;
                tile[i] = t < count ? float(values[size_t(t) * 512 + kvh * 256 + i % 256]) : 0.0f;
            }
            __syncthreads();
            for (int t = 0; t < 16 && begin + sub + t < valid; ++t)
                for (int j = 0; j < 8; ++j)
                    sum[j] += scores[warp][sub + t] * tile[t * 256 + lane + j * 32];
            __syncthreads();
        }
        if (d != 0.0f) {
            const float next = fmaxf(maximum, m);
            const float a = expf(maximum - next), b = expf(m - next);
            denominator = denominator * a + d * b;
            for (int j = 0; j < 8; ++j) acc[j] = acc[j] * a + sum[j] * b;
            maximum = next;
        }
    }
    if (active) {
        if (!lane) { dst[0] = maximum; dst[1] = denominator; }
        for (int j = 0; j < 8; ++j) dst[lane + j * 32 + 2] = acc[j];
    }
}

__global__ void kv_pack_kernel(const float* source, __half* destination, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) destination[i] = __float2half_rn(source[i]);
}

__global__ void attn_columns_accumulate_kernel(const float* partial, int tiles,
                                               float* accumulator, bool first) {
    const int h = blockIdx.x, col = blockIdx.y, j = threadIdx.x;
    float* dst = accumulator + (size_t(col) * 16 + h) * 258;
    float maximum = first ? -INFINITY : dst[0], denominator = first ? 0.0f : dst[1];
    float acc = first ? 0.0f : dst[j + 2];
    for (int t = 0; t < tiles; ++t) {
        const float* row = partial + ((size_t(col) * 16 + h) * tiles + t) * 258;
        if (row[1] == 0.0f) continue;
        const float next = fmaxf(maximum, row[0]);
        const float a = expf(maximum - next), b = expf(row[0] - next);
        denominator = denominator * a + row[1] * b;
        acc = acc * a + row[j + 2] * b;
        maximum = next;
    }
    // All threads finish reading shared row metadata before it is replaced.
    __syncthreads();
    if (!j) { dst[0] = maximum; dst[1] = denominator; }
    dst[j + 2] = acc;
}

template<typename T>
__global__ void attn_matrix_queries_kernel(const float* q, int columns, T* packed) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(16) * columns * 256) return;
    int d = i % 256, c = (i / 256) % columns, h = i / (size_t(columns) * 256);
    packed[i] = T(q[size_t(c) * 8192 + h * 512 + d]);
}
template<typename S, typename T>
__global__ void attn_matrix_kv_kernel(const S* k, const S* v, int count, T* pk, T* pv) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(2) * count * 256) return;
    int d = i % 256, c = (i / 256) % count, h = i / (size_t(count) * 256);
    size_t from = size_t(c) * 512 + h * 256 + d;
    pk[i] = T(float(k[from])); pv[i] = T(float(v[from]));
}
__global__ void attn_matrix_softmax_kernel(float* scores, int count, int columns,
    int key_start, int query_start, float* metadata, __nv_bfloat16* probability) {
    int c = blockIdx.x % columns, h = blockIdx.x / columns, lane = threadIdx.x;
    float* row = scores + size_t(blockIdx.x) * count;
    __shared__ float reduce[256];
    float maximum = -INFINITY;
    for (int k = lane; k < count; k += 256) {
        float score = key_start + k <= query_start + c ? row[k] : -INFINITY;
        row[k] = score;
        maximum = fmaxf(maximum, score);
    }
    reduce[lane] = maximum; __syncthreads();
    for (int d = 128; d; d >>= 1) { if (lane < d) reduce[lane] = fmaxf(reduce[lane], reduce[lane+d]); __syncthreads(); }
    maximum = reduce[0];
    float denominator = 0.f;
    for (int k = lane; k < count; k += 256) {
        float weight = isfinite(maximum) ? expf(row[k] - maximum) : 0.f;
        row[k] = weight;
        denominator += weight;
        if (probability) probability[size_t(blockIdx.x) * count + k] = __float2bfloat16_rn(weight);
    }
    reduce[lane] = denominator; __syncthreads();
    for (int d = 128; d; d >>= 1) { if (lane < d) reduce[lane] += reduce[lane+d]; __syncthreads(); }
    if (!lane) { metadata[2 * blockIdx.x] = maximum; metadata[2 * blockIdx.x + 1] = reduce[0]; }
}
__global__ void attn_matrix_merge_kernel(const float* output, const float* metadata, int columns,
    float* accumulator, bool first) {
    int c = blockIdx.x % columns, h = blockIdx.x / columns, d = threadIdx.x;
    float* row = accumulator + (size_t(c) * 16 + h) * 258;
    float tile_max = metadata[2 * blockIdx.x], tile_den = metadata[2 * blockIdx.x + 1];
    float previous_max = first ? -INFINITY : row[0], previous_den = first ? 0.f : row[1];
    float previous = first ? 0.f : row[d+2];
    float maximum = fmaxf(previous_max, tile_max);
    float a = previous_den ? expf(previous_max - maximum) : 0.f;
    float b = tile_den ? expf(tile_max - maximum) : 0.f;
    float numerator = previous * a + output[size_t(blockIdx.x) * 256 + d] * b;
    __syncthreads();
    if (!d) { row[0] = maximum; row[1] = previous_den * a + tile_den * b; }
    row[d+2] = numerator;
}

__global__ void attn_columns_finish_kernel(const float* q, const float* accumulator, float* out) {
    const size_t h = size_t(blockIdx.y) * 16 + blockIdx.x;
    const int j = threadIdx.x;
    const float* row = accumulator + h * 258;
    out[h * 256 + j] = row[j + 2] / row[1] * sigmoid_precise(q[h * 512 + 256 + j]);
}

__global__ void attn_merge_kernel(const float* q, const float* partial, int tiles,
                                  int head_dim, float* out) {
    const int h = blockIdx.x, j = threadIdx.x;
    float maximum = -INFINITY, denominator = 0.0f, acc = 0.0f;
    for (int t = 0; t < tiles; ++t) {
        const float* row = partial + (size_t(h) * tiles + t) * (head_dim + 2);
        const float next = fmaxf(maximum, row[0]);
        const float alpha = expf(maximum - next), beta = expf(row[0] - next);
        denominator = denominator * alpha + row[1] * beta;
        acc = acc * alpha + row[j + 2] * beta;
        maximum = next;
    }
    out[h * head_dim + j] = (acc / denominator) * sigmoid_precise(q[h * 2 * head_dim + head_dim + j]);
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
    if (threadIdx.x != 0) return;
    logits += size_t(blockIdx.x) * n;
    ids += size_t(blockIdx.x) * k;
    weights += size_t(blockIdx.x) * (k + 1);
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
                                   int n, float* __restrict__ out, int stride = 1) {
    b += size_t(blockIdx.x) * n;
    out += size_t(blockIdx.x) * stride;
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
__global__ void gather_expert_inputs_kernel(const float* x, const int* map, int count,
                                            int hidden, float* selected) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count*hidden) selected[i] = x[size_t(map[i/hidden]/9)*hidden+i%hidden];
}

__global__ void scatter_expert_outputs_kernel(const float* down, const int* map, int count,
                                              int hidden, float* slots) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count*hidden) slots[size_t(map[i/hidden])*hidden+i%hidden] = down[i];
}

__global__ void moe_combine_columns_kernel(const float* down, const float* scales,
                                           int columns, int hidden, float* output) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= columns*hidden) return;
    const int col = i/hidden, j = i%hidden;
    float sum = 0;
    for (int s = 0; s < 9; ++s) sum += scales[col*9+s] * down[(size_t(col)*9+s)*hidden+j];
    output[i] = sum;
}
__global__ void gemv_f32_columns_kernel(const float* weights, const float* x, float* y,
                                        int n_in, int n_out, int columns) {
    __shared__ float partial[8][kThreads];
    const int row = blockIdx.x, tid = threadIdx.x, first = blockIdx.y * 8;
    float sums[8] = {};
    for (int i = tid; i < n_in; i += kThreads) {
        const float value = weights[size_t(row) * n_in + i];
#pragma unroll
        for (int c = 0; c < 8; ++c)
            if (first + c < columns) sums[c] += value * x[size_t(first+c) * n_in + i];
    }
#pragma unroll
    for (int c = 0; c < 8; ++c) partial[c][tid] = sums[c];
    __syncthreads();
    for (int stride = kThreads/2; stride; stride >>= 1) {
        if (tid < stride) {
#pragma unroll
            for (int c = 0; c < 8; ++c) partial[c][tid] += partial[c][tid+stride];
        }
        __syncthreads();
    }
    if (!tid) {
#pragma unroll
        for (int c = 0; c < 8; ++c)
            if (first+c < columns) y[size_t(first+c)*n_out+row] = partial[c][0];
    }
}

}  // namespace

void to_bf16(const float* source, void* destination, int n, void* stream) {
    to_bf16_kernel<<<(n + 255) / 256, 256, 0, static_cast<cudaStream_t>(stream)>>>(source, static_cast<__nv_bfloat16*>(destination), n);
}

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
                                                                 epsilon, rope_base, position, position, position);
}

void rms_norm_columns(float* x, const float* weight, int width, int columns, float epsilon, void* stream) {
    rms_norm_kernel<<<columns, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(x, weight, width, epsilon);
}

void gemv_f32_columns(const float* weights, const float* x, float* y, int n_in,
                       int n_out, int columns, void* stream) {
    gemv_f32_columns_kernel<<<dim3(n_out, (columns+7)/8), kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
        weights, x, y, n_in, n_out, columns);
}

void attn_norm_mrope(float* x, int heads, int stride, int n, int n_rot, const float* gamma,
                     float epsilon, float rope_base, int t, int h, int w, void* stream) {
    attn_norm_rope_kernel<<<heads, n, 0, static_cast<cudaStream_t>(stream)>>>(
        x, stride, n, n_rot, gamma, epsilon, rope_base, t, h, w);
}

void attn_norm_mrope_columns(float* x, int heads, int stride, int columns,
    const float* gamma, float epsilon, float rope_base, const int* positions, void* stream) {
    attn_norm_rope_kernel<<<dim3(heads,columns),256,0,static_cast<cudaStream_t>(stream)>>>(
        x,stride,256,64,gamma,epsilon,rope_base,0,0,0,positions,heads);
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

void router_topk_columns(const float* logits, int columns, int* ids, float* scales, void* stream) {
    router_topk_kernel<<<columns, 1, 0, static_cast<cudaStream_t>(stream)>>>(logits, 256, 8, ids, scales);
}

void dot_sigmoid_columns(const float* weights, const float* x, int columns, float* scales, void* stream) {
    dot_sigmoid_kernel<<<columns, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(weights, x, 2048, scales+8, 9);
}

void gather_expert_inputs(const float* x, const int* map, int count, int hidden, float* selected, void* stream) {
    gather_expert_inputs_kernel<<<blocks(count*hidden), kThreads, 0, static_cast<cudaStream_t>(stream)>>>(x, map, count, hidden, selected);
}

void scatter_expert_outputs(const float* down, const int* map, int count, int hidden, float* slots, void* stream) {
    scatter_expert_outputs_kernel<<<blocks(count*hidden), kThreads, 0, static_cast<cudaStream_t>(stream)>>>(down, map, count, hidden, slots);
}

void moe_combine_columns(const float* down, const float* scales, int columns, int hidden, float* output, void* stream) {
    moe_combine_columns_kernel<<<blocks(columns*hidden), kThreads, 0, static_cast<cudaStream_t>(stream)>>>(down, scales, columns, hidden, output);
}

void attn_partials(const float* q, const float* k, const float* v, int count,
                   int offset, int tiles, int heads, int kv_heads, int dim,
                   float scale, float* partial, void* stream) {
    if (count >= 2048 && heads == 16 && kv_heads == 2 && dim == 256 && scale == 0.0625f) {
        attn_gqa_partial_kernel<<<dim3((count + 127) / 128, 2), 256, 0, static_cast<cudaStream_t>(stream)>>>(q, k, v, count, offset, tiles, partial);
        return;
    }
    attn_partial_kernel<<<dim3((count + 127) / 128, heads), 256, 0,
                           static_cast<cudaStream_t>(stream)>>>(q, k, v, count, offset,
                                                               tiles, heads, kv_heads, dim, scale, partial);
}

void attn_merge(const float* q, const float* partial, int tiles, int heads, int dim,
                float* out, void* stream) {
    attn_merge_kernel<<<heads, dim, 0, static_cast<cudaStream_t>(stream)>>>(q, partial, tiles, dim, out);
}

void attn_columns_tile(const float* q, const float* k, const float* v, int count,
                       int key_start, int query_start, int columns, float* partial,
                       float* accumulator, bool first, void* stream) {
    const int tiles = (count + 127) / 128;
    auto s = static_cast<cudaStream_t>(stream);
    attn_partial_kernel<<<dim3(tiles, 16, columns), 256, 0, s>>>(
        q, k, v, count, 0, tiles, 16, 2, 256, 1.0f / 16.0f, partial, query_start, key_start);
    attn_columns_accumulate_kernel<<<dim3(16, columns), 256, 0, s>>>(partial, tiles, accumulator, first);
}

void attn_columns_finish(const float* q, const float* accumulator, int columns,
                         float* out, void* stream) {
    attn_columns_finish_kernel<<<dim3(16, columns), 256, 0, static_cast<cudaStream_t>(stream)>>>(q, accumulator, out);
}

void attn_fused_columns(const float* q, const void* k, const void* v, int count,
    int key_start, int query_start, int columns, float* accumulator, bool first, bool half, void* stream) {
    const dim3 grid(16, (columns + 7) / 8);
    auto s = static_cast<cudaStream_t>(stream);
    if (half) attn_fused_columns_kernel<<<grid, 256, 0, s>>>(q, static_cast<const __half*>(k),
        static_cast<const __half*>(v), count, key_start, query_start, columns, accumulator, first);
    else attn_fused_columns_kernel<<<grid, 256, 0, s>>>(q, static_cast<const float*>(k),
        static_cast<const float*>(v), count, key_start, query_start, columns, accumulator, first);
}

void attn_half_partials(const float* q, const void* k, const void* v, int count,
    int offset, int tiles, float* partial, void* stream) {
    if (count >= 2048) attn_gqa_partial_kernel<<<dim3((count + 127) / 128, 2), 256, 0, static_cast<cudaStream_t>(stream)>>>(
        q, static_cast<const __half*>(k), static_cast<const __half*>(v), count, offset, tiles, partial);
    else attn_partial_kernel<<<dim3((count+127)/128,16),256,0,static_cast<cudaStream_t>(stream)>>>(
        q,static_cast<const __half*>(k),static_cast<const __half*>(v),count,offset,tiles,16,2,256,.0625f,partial);
}

void attn_matrix_queries(const float* q, int columns, void* packed, bool bf16, void* stream) {
    auto s = static_cast<cudaStream_t>(stream); int blocks = 16 * columns;
    if (bf16) attn_matrix_queries_kernel<<<blocks, 256, 0, s>>>(q, columns, static_cast<__nv_bfloat16*>(packed));
    else attn_matrix_queries_kernel<<<blocks, 256, 0, s>>>(q, columns, static_cast<float*>(packed));
}
void attn_matrix_kv(const void* k, const void* v, int count, bool half, void* pk, void* pv, bool bf16, void* stream) {
    auto s = static_cast<cudaStream_t>(stream); int blocks = 2 * count;
    if (bf16 && half) attn_matrix_kv_kernel<<<blocks, 256, 0, s>>>(static_cast<const __half*>(k), static_cast<const __half*>(v), count, static_cast<__nv_bfloat16*>(pk), static_cast<__nv_bfloat16*>(pv));
    else if (bf16) attn_matrix_kv_kernel<<<blocks, 256, 0, s>>>(static_cast<const float*>(k), static_cast<const float*>(v), count, static_cast<__nv_bfloat16*>(pk), static_cast<__nv_bfloat16*>(pv));
    else if (half) attn_matrix_kv_kernel<<<blocks, 256, 0, s>>>(static_cast<const __half*>(k), static_cast<const __half*>(v), count, static_cast<float*>(pk), static_cast<float*>(pv));
    else attn_matrix_kv_kernel<<<blocks, 256, 0, s>>>(static_cast<const float*>(k), static_cast<const float*>(v), count, static_cast<float*>(pk), static_cast<float*>(pv));
}
void attn_matrix_softmax(float* scores, int count, int columns, int ks, int qs, float* metadata, void* probability, bool bf16, void* stream) {
    attn_matrix_softmax_kernel<<<16 * columns, 256, 0, static_cast<cudaStream_t>(stream)>>>(scores, count, columns, ks, qs, metadata, bf16 ? static_cast<__nv_bfloat16*>(probability) : nullptr);
}
void attn_matrix_merge(const float* output, const float* metadata, int columns, float* accumulator, bool first, void* stream) {
    attn_matrix_merge_kernel<<<16 * columns, 256, 0, static_cast<cudaStream_t>(stream)>>>(output, metadata, columns, accumulator, first);
}

void kv_store(const float* source, void* destination, int elements, bool half, void* stream) {
    auto s = static_cast<cudaStream_t>(stream);
    if (half) kv_pack_kernel<<<(elements + 255) / 256, 256, 0, s>>>(source, static_cast<__half*>(destination), elements);
    else cudaMemcpyAsync(destination, source, size_t(elements) * sizeof(float), cudaMemcpyDeviceToDevice, s);
}

void dot_sigmoid(const float* a, const float* b, int n, float* out, void* stream) {
    dot_sigmoid_kernel<<<1, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(a, b, n, out);
}

void doorbell_signal(int* flag, int value, void* stream) {
    doorbell_signal_kernel<<<1, 1, 0, static_cast<cudaStream_t>(stream)>>>(flag, value);
}

}  // namespace lamina::model::cuda
