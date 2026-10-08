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
    // Greedy next token without downloading the logits: the first maximum, as
    // std::max_element over logits(hidden). Returns -1 when the GPU path is
    // unavailable or a logit is not finite; callers then use logits().
    int greedy(std::vector<float> hidden);
    std::vector<float> step_hidden_embedding(const std::vector<float>& embedding,
                                            const std::array<int, 3>& positions);
    void reset();
    // One immutable prompt-prefix checkpoint. RESET invalidates it; generated
    // suffixes only append KV and never overwrite its prefix. No weight pointers
    // or captured graph addresses are stored in the checkpoint.
    void cache_prefix();
    void restore_prefix();
    // Multi-token prediction draft head (tools/lamina_mtp.py pack). After the
    // main model produced hidden state h_t and greedy token x_{t+1}, mtp_draft
    // runs the head's single layer and returns its greedy guess for x_{t+2}.
    // The head attends only to its own drafts since reset (no prompt context).
    // rope_position is h_t's position; the default numbers drafts from zero.
    void load_mtp(const std::string& path);
    bool has_mtp() const { return mtp_file_ != nullptr; }
    int mtp_draft(int next_token, const std::vector<float>& hidden, int rope_position = -1);

    // Greedy generation of `count` tokens. `token` is the last emitted token,
    // not yet fed; `hidden` is the hidden state that predicted it and is
    // updated to the one that predicted the last returned token. With an MTP
    // head on the CUDA fast path, each step drafts one token and verifies it
    // together with `token` in a single two-token pass (greedy speculation);
    // otherwise it steps one token at a time.
    struct SpecStats { uint64_t steps = 0, drafted = 0, accepted = 0, backoffs = 0; double draft_ms = 0, verify_ms = 0; };
    std::vector<int> generate_greedy(int token, std::vector<float>& hidden, int count);
    const SpecStats& spec_stats() const { return spec_stats_; }
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
    std::array<LinearState, 40> prefix_linear_;
    std::array<AttentionState, 40> prefix_attention_;
    int prefix_position_ = -1;
    int prefix_rope_position_ = 0;
    std::array<int, 3> prefix_rope_positions_{};
    std::unique_ptr<CudaProjection> cuda_;
    std::unique_ptr<strata::GgufFile> mtp_file_;
    int mtp_position_ = 0;
    SpecStats spec_stats_;
    double spec_rate_ = 0.87;  // running acceptance (exponential average)
    int spec_backoff_ = 0;     // single steps left before speculation is probed again
    bool pair_supported();
    // Two-token pass over `first` and `second` at the next two positions.
    // Returns both hidden columns; commit_pair advances by one or two tokens.
    std::vector<float> step_pair_hidden(int first, int second);
    void commit_pair(bool keep_second);
    int greedy_token(const std::vector<float>& hidden);

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
