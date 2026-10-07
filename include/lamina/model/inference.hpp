#pragma once

#include "lamina/model/cuda_projection.hpp"
#include "strata/artifact/gguf_reader.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace lamina::model {


// Scalar, streaming reference implementation. Its state is per conversation;
// GGUF weights remain memory mapped and only selected expert rows are touched.
class Inference {
public:
    explicit Inference(const std::string& path, int context = 32768, int layers = 40,
                       bool cuda = false, const std::string& kv_cache = "auto",
                       size_t vram_limit_mb = 0, const std::string& kv_type = "f32", const std::string& compute_mode = "f32");
    ~Inference();
    std::vector<float> step(int token);
    // Development diagnostic: return the residual stream before final norm.
    std::vector<float> step_hidden(int token);
    std::vector<float> prefill_hidden(const std::vector<int>& tokens);
    std::vector<float> prefill_long(const std::vector<int>& tokens, int chunk = 2048);
    std::vector<float> prefill_embeddings(const std::vector<float>& embeddings,
                                          const std::vector<std::array<int, 3>>& positions);
    std::vector<float> logits(std::vector<float> hidden);
    std::vector<float> step_hidden_embedding(const std::vector<float>& embedding,
                                            const std::array<int, 3>& positions);
    void reset();
    int position() const { return position_; }
    int rope_position() const { return rope_position_; }

private:
    struct LinearState {
        std::vector<float> conv;       // [8192, 3], oldest first
        std::vector<float> recurrent;  // [32, 128, 128]
    };
    struct AttentionState {
        std::vector<float> keys;       // [context, 2, 256]
        std::vector<float> values;
    };

    strata::GgufFile file_;
    int context_;
    int layers_;
    int position_ = 0;
    int rope_position_ = 0;
    std::array<int, 3> rope_positions_{};
    unsigned cpu_workers_ = 1;
    std::array<LinearState, 40> linear_;
    std::array<AttentionState, 40> attention_;
    std::unique_ptr<CudaProjection> cuda_;

    const strata::TensorInfo& tensor(const std::string& name) const;
    void row(const strata::TensorInfo& t, int64_t index, float* out) const;
    std::vector<float> matvec(const strata::TensorInfo& t, const std::vector<float>& x,
                              int64_t expert = -1) const;
    std::vector<std::vector<float>> matvec_many(
        const std::vector<const strata::TensorInfo*>& tensors, const std::vector<float>& x,
        int64_t expert = -1) const;
    std::vector<float> vec(const strata::TensorInfo& t) const;
    std::vector<float> mixer(int layer, const std::vector<float>& x);
    std::vector<float> delta_net(int layer, const std::vector<float>& x);
    std::vector<float> full_attention(int layer, const std::vector<float>& x);
    std::vector<float> moe(int layer, const std::vector<float>& x);
    std::vector<float> moe_columns(int layer, const std::vector<float>& x, int columns);

    // Device-resident token chain helpers.
    bool device_chain_supported();
    std::vector<float> step_hidden_device(const std::vector<float>& embedding);
    std::vector<float> forward_hidden(std::vector<float> embedding);
    GdnWeights gdn_weights(int layer) const;
    AttnWeights attention_weights(int layer) const;
    void moe_weights(int layer, MoeWeights& routed, MoeWeights& shared) const;
    bool device_chain_ = false;
    bool device_chain_checked_ = false;
};

}  // namespace lamina::model
