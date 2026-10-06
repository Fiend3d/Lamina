#pragma once

#include "strata/artifact/gguf_reader.hpp"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace lamina::model {

// A quantized expert projection set: gate (hidden x ff), up (hidden x ff) and
// down (ff x hidden), each with the base pointer of its mapped GGUF payload.
struct MoeWeights {
    const strata::TensorInfo* gate = nullptr;
    const uint8_t* gate_data = nullptr;
    const strata::TensorInfo* up = nullptr;
    const uint8_t* up_data = nullptr;
    const strata::TensorInfo* down = nullptr;
    const uint8_t* down_data = nullptr;
};

// The quantized and F32 weights of one Qwen3.6 DeltaNet layer, each with the
// base pointer of its mapped GGUF payload.
struct GdnWeights {
    const strata::TensorInfo* qkv = nullptr;
    const uint8_t* qkv_data = nullptr;
    const strata::TensorInfo* gate = nullptr;
    const uint8_t* gate_data = nullptr;
    const strata::TensorInfo* alpha = nullptr;
    const uint8_t* alpha_data = nullptr;
    const strata::TensorInfo* beta = nullptr;
    const uint8_t* beta_data = nullptr;
    const strata::TensorInfo* a = nullptr;
    const uint8_t* a_data = nullptr;
    const strata::TensorInfo* dt = nullptr;
    const uint8_t* dt_data = nullptr;
    const strata::TensorInfo* conv = nullptr;
    const uint8_t* conv_data = nullptr;
    const strata::TensorInfo* norm = nullptr;
    const uint8_t* norm_data = nullptr;
    const strata::TensorInfo* out = nullptr;
    const uint8_t* out_data = nullptr;
};

// The weights of one Qwen3.6 full-attention layer.
struct AttnWeights {
    const strata::TensorInfo* q = nullptr;
    const uint8_t* q_data = nullptr;
    const strata::TensorInfo* k = nullptr;
    const uint8_t* k_data = nullptr;
    const strata::TensorInfo* v = nullptr;
    const uint8_t* v_data = nullptr;
    const strata::TensorInfo* q_norm = nullptr;
    const uint8_t* q_norm_data = nullptr;
    const strata::TensorInfo* k_norm = nullptr;
    const uint8_t* k_norm_data = nullptr;
    const strata::TensorInfo* out = nullptr;
    const uint8_t* out_data = nullptr;
};

// Optional hybrid backend: GGUF matrix-vector projections run on CUDA with
// full-precision activations; Qwen3.6 state and routing remain on the CPU.
class CudaProjection {
public:
    CudaProjection();
    ~CudaProjection();
    CudaProjection(const CudaProjection&) = delete;
    CudaProjection& operator=(const CudaProjection&) = delete;

    bool supports(uint32_t ggml_type) const;
    std::vector<float> matvec(const strata::TensorInfo& tensor, const uint8_t* data,
                              const std::vector<float>& x, int64_t expert);
    std::vector<std::vector<float>> matvec_many(
        const std::vector<const strata::TensorInfo*>& tensors,
        const std::vector<const uint8_t*>& data, const std::vector<int64_t>& experts,
        const std::vector<float>& x);

    // Dense FP32 2-D projection (GGUF F32 weights), e.g. the MoE router. The
    // weight is cached on the device like the quantized projections.
    std::vector<float> matvec_f32(const strata::TensorInfo& tensor, const uint8_t* data,
                                  const std::vector<float>& x);

    // Runs the whole Qwen3.6 DeltaNet layer on the device: qkv/gate/alpha/beta
    // projections, causal conv + SiLU, L2 normalization, decay/learning gates,
    // the gated delta-rule recurrence, the closing norm with SiLU and the ssm_out
    // projection. `layer` selects the carried convolution history and recurrent
    // state. Only the 2048-wide result returns to the host.
    std::vector<float> delta_net(int layer, const std::vector<float>& x, const GdnWeights& w);

    bool supports_gdn(const GdnWeights& weights) const;

    // Runs the whole routed and shared expert block on the device with a single
    // host synchronization: gate/up projections, SwiGLU, down projections and
    // the weighted combination. `experts` and `weights` are the routed top-k in
    // selection order; `shared_weight` scales the shared expert output.
    std::vector<float> moe(const std::vector<float>& x, const MoeWeights& routed,
                           const std::vector<int>& experts, const std::vector<float>& weights,
                           const MoeWeights& shared, float shared_weight);

    // True when every required projection is a supported quantized type and the
    // shapes form a valid Qwen3.6 expert block.
    bool supports_moe(const MoeWeights& routed, const MoeWeights& shared) const;

    // ---- Device-resident token chain -------------------------------------
    // The hidden state and every intermediate tensor stay in VRAM across a token;
    // only the router logits and the final result touch the host.
    void set_context(int max_context);
    void hidden_upload(const std::vector<float>& x);
    std::vector<float> hidden_download();
    void mix_upload(const std::vector<float>& x);
    void hidden_rms(const strata::TensorInfo& gamma, const uint8_t* data, float epsilon);
    void add_hidden_mix();
    void delta_net_into_mix(int layer, const GdnWeights& w);
    void moe_into_mix(const strata::TensorInfo& router, const uint8_t* router_data,
                      const strata::TensorInfo& shared_gate, const uint8_t* shared_gate_data,
                      const MoeWeights& routed, const MoeWeights& shared);
    void attention_into_mix(int layer, const AttnWeights& w, int position, float rope_base);
    bool supports_attention(const AttnWeights& w) const;

    // Runs a whole DeltaNet layer's pre-MoE sequence (input RMS norm, recurrence,
    // residual, post-attention RMS norm) and captures it into a per-layer CUDA
    // graph after the first call, so later tokens replay one graph launch instead
    // of ~16 small kernels.
    void delta_layer_graph(int layer, const GdnWeights& w, const strata::TensorInfo& input_norm,
                           const uint8_t* input_norm_data, const strata::TensorInfo& post_norm,
                           const uint8_t* post_norm_data, float epsilon);

    // The internal kernel stream (for diagnostics).
    void* cuda_stream() const;

    // CUDA-event timing across the device chain (diagnostics).
    void mark_begin();
    double mark_end_ms();

    struct Stats {
        uint64_t hits = 0;
        uint64_t misses = 0;
        size_t uploaded_bytes = 0;
        size_t evicted_bytes = 0;
        size_t resident_bytes = 0;
        size_t cache_limit = 0;
    };
    Stats stats() const;

private:
    void delta_net_core(int layer, const float* x_dev, float* out_dev, const GdnWeights& w);
    void moe_core(const float* x_dev, float* out_dev, const MoeWeights& routed,
                  const std::vector<int>& experts, const std::vector<float>& weights,
                  const MoeWeights& shared, float shared_weight);
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace lamina::model
