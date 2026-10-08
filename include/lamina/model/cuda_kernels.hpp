#pragma once

// Small elementwise CUDA kernels used by Lamina's device-resident graph.
// These are compiled without --use_fast_math so their arithmetic matches the
// scalar reference closely enough for the 40-layer parity gate. All pointers
// are device pointers and all calls enqueue on the explicit non-null stream.

namespace lamina::model::cuda {
void to_bf16(const float* source, void* destination, int elements, void* stream);

void swiglu(const float* gate, const float* up, float* out, int n, void* stream);

void accumulate(float* out, const float* source, float weight, int n, void* stream);

void rms_norm(float* x, const float* weight, int n, float epsilon, void* stream);

void add_inplace(float* x, const float* y, int n, void* stream);

// Dense FP32 matrix-vector product for weights in GGUF row-major layout
// (element (row, i) at row * n_in + i).
void gemv_f32(const float* weights, const float* x, float* y, int n_in, int n_out,
               void* stream);
void gemv_f32_columns(const float* weights, const float* x, float* y, int n_in,
                       int n_out, int columns, void* stream);
void rms_norm_columns(float* x, const float* weight, int width, int columns,
                       float epsilon, void* stream);

// Qwen3.6 DeltaNet closing norm: out[h,j] = rms_norm(x[h,:])[j] * gamma[j] * silu(z[h,j]).
// One block per head; width is fixed at 128.
void gdn_out_norm_silu(const float* x, const float* z, const float* gamma, float* out,
                       int heads, float epsilon, void* stream);

// Weighted sum of `slots` expert rows: out[o] = sum_s scales[s] * down[s*hidden + o],
// accumulated in slot order (routed experts then the shared expert).
void moe_combine(const float* down, const float* scales, int slots, int hidden, float* out,
                 void* stream);

// Per-head weighted RMS norm followed by partial RoPE over the first n_rot
// coordinates. One block of `n` threads per head; rows are `stride` apart.
void attn_norm_rope(float* x, int heads, int stride, int n, int n_rot, const float* gamma,
                    float epsilon, float rope_base, int position, void* stream);
void attn_norm_mrope(float* x, int heads, int stride, int n, int n_rot, const float* gamma,
                     float epsilon, float rope_base, int t, int h, int w, void* stream);

// Device positions are [columns,3]; all heads/columns share one launch.
void attn_norm_mrope_columns(float* x, int heads, int stride, int columns,
    const float* gamma, float epsilon, float rope_base, const int* positions, void* stream);

// Single-token GQA attention: q is [heads, 2*head_dim] (query then gate),
// k_cache/v_cache are [context, kv_heads*head_dim], one row per position.
// out is [heads, head_dim].
void attn_decode(const float* q, const float* k_cache, const float* v_cache, int position,
                 int heads, int kv_heads, int head_dim, float scale, float* out, void* stream);

// Parallel FP32 attention tiles. Partial rows contain max, denominator and the
// unnormalised value sum; merging retains all tokens, including host KV tiles.
void attn_partials(const float* q, const float* keys, const float* values, int count,
                   int tile_offset, int total_tiles, int heads, int kv_heads,
                   int head_dim, float scale, float* partial, void* stream);
void attn_merge(const float* q, const float* partial, int total_tiles, int heads,
                int head_dim, float* out, void* stream);
// Causal query columns share each staged KV tile. Accumulator rows hold
// (maximum, denominator, unnormalized output), preserving all earlier tiles.
void attn_columns_tile(const float* q, const float* k, const float* v, int count,
                       int key_start, int query_start, int columns, float* partial,
                       float* accumulator, bool first, void* stream);
void attn_columns_finish(const float* q, const float* accumulator, int columns,
                         float* out, void* stream);
// FP32 accumulation, FP32 or FP16 storage. Fused path reuses KV across queries.
void attn_fused_columns(const float* q, const void* k, const void* v, int count,
                        int key_start, int query_start, int columns,
                        float* accumulator, bool first, bool half, void* stream);
void attn_half_partials(const float* q, const void* k, const void* v, int count,
                        int offset, int tiles, float* partial, void* stream);
// Bounded matrix attention adapters; head-major matrices for strided cuBLAS.
void attn_matrix_queries(const float* q, int columns, void* packed, bool bf16, void* stream);
void attn_matrix_kv(const void* k, const void* v, int count, bool half, void* packed_k, void* packed_v, bool bf16, void* stream);
void attn_matrix_softmax(float* scores, int count, int columns, int key_start, int query_start, float* metadata, void* probability, bool bf16, void* stream);
void attn_matrix_merge(const float* output, const float* metadata, int columns, float* accumulator, bool first, void* stream);
void kv_store(const float* source, void* destination, int elements, bool half, void* stream);

// Stable top-k over `n` logits with renormalised softmax weights; single block.
void router_topk(const float* logits, int n, int k, int* ids, float* weights, void* stream);
// Decode router tail fused into one launch: sigmoid(gate_weight . x) into
// shared_out, top-k ids/weights, optional copy of x into publish (mapped host
// memory for CPU experts; may be null), then *flag = value as the doorbell.
void router_finish(const float* logits, int n, int k, int* ids, float* weights,
                   const float* gate_weight, const float* x, int n_in, float* shared_out,
                   float* publish, int* flag, int value, void* stream);
void router_topk_columns(const float* logits, int columns, int* ids, float* scales, void* stream);
void dot_sigmoid_columns(const float* weights, const float* x, int columns, float* scales, void* stream);
void gather_expert_inputs(const float* x, const int* slot_map, int count, int hidden,
                           float* selected, void* stream);
void scatter_expert_outputs(const float* down, const int* slot_map, int count, int hidden,
                             float* slots, void* stream);
void moe_combine_columns(const float* down, const float* scales, int columns,
                          int hidden, float* output, void* stream);

// sigmoid(sum(a[i]*b[i])) into out[0], single block.
void dot_sigmoid(const float* a, const float* b, int n, float* out, void* stream);

// Publishes `value` to a mapped pinned flag after a system-scope fence, so a
// host spin loop sees the router's writes to mapped memory. One thread.
void doorbell_signal(int* flag, int value, void* stream);

// Per-layer expert pointer tables and combine scales, passed by value as kernel
// arguments. One small launch replaces two cudaMemcpyAsync calls, which cost
// considerably more host time under WDDM. Fits the 4 KiB parameter limit.
constexpr int kPublishItems = 24, kPublishScales = 33;
struct ExpertPublish {
    int counts[4];
    const void* weights[4][kPublishItems];
    const float* inputs[4][kPublishItems];
    float* outputs[4][kPublishItems];
    int n_outs[4][kPublishItems];
    float scales[kPublishScales];
    int scale_count;
};
// tables: four consecutive strata::kernels::NativeF32Grouped in device memory.
void publish_expert_tables(const ExpertPublish& args, void* tables, float* scales, void* stream);

// Greedy token selection: result[0] = index of the first maximum (as
// std::max_element), result[1] = 1 if any value is not finite.
void argmax_f32(const float* x, int n, int* result, void* stream);

// Element copy by the SMs. With a mapped host pointer on either side it moves
// small data over PCIe without queueing behind copy-engine DMA.
void copy_f32(const float* source, float* destination, int n, void* stream);

}  // namespace lamina::model::cuda
