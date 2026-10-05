#pragma once

#include "strata/artifact/gguf_reader.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace lamina::model {

// Scalar, streaming reference implementation. Its state is per conversation;
// GGUF weights remain memory mapped and only selected expert rows are touched.
class Inference {
public:
    explicit Inference(const std::string& path, int context = 32768);
    std::vector<float> step(int token);
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
    int position_ = 0;
    std::array<LinearState, 40> linear_;
    std::array<AttentionState, 40> attention_;

    const strata::TensorInfo& tensor(const std::string& name) const;
    void row(const strata::TensorInfo& t, int64_t index, float* out) const;
    std::vector<float> matvec(const strata::TensorInfo& t, const std::vector<float>& x,
                              int64_t expert = -1) const;
    std::vector<float> vec(const strata::TensorInfo& t) const;
    std::vector<float> mixer(int layer, const std::vector<float>& x);
    std::vector<float> delta_net(int layer, const std::vector<float>& x);
    std::vector<float> full_attention(int layer, const std::vector<float>& x);
    std::vector<float> moe(int layer, const std::vector<float>& x);
};

}  // namespace lamina::model
