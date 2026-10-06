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
                       bool cuda = false);
    ~Inference();
    std::vector<float> step(int token);
    // Development diagnostic: return the residual stream before final norm.
    std::vector<float> step_hidden(int token);
    int position() const { return position_; }

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

    // Device-resident token chain helpers.
    bool device_chain_supported();
    std::vector<float> step_hidden_device(int token_id);
    GdnWeights gdn_weights(int layer) const;
    AttnWeights attention_weights(int layer) const;
    void moe_weights(int layer, MoeWeights& routed, MoeWeights& shared) const;
    bool device_chain_ = false;
    bool device_chain_checked_ = false;
};

}  // namespace lamina::model
