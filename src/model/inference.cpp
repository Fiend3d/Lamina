#include "lamina/model/inference.hpp"
#include "lamina/model/cuda_projection.hpp"
#include "lamina/model/qwen36.hpp"
#include "strata/artifact/dequant.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <exception>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <thread>

namespace lamina::model {
namespace {
constexpr int HIDDEN = 2048;
constexpr int KEY_HEADS = 16;
constexpr int VALUE_HEADS = 32;
constexpr int HEAD_DIM = 128;
constexpr int CONV_DIM = 8192;
constexpr float EPS = 1e-6f;

float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
float silu(float x) { return x * sigmoid(x); }
float softplus(float x) { return std::max(x, 0.0f) + std::log1p(std::exp(-std::abs(x))); }
float dot(const float* a, const float* b, int n) {
    double sum = 0;
    for (int i = 0; i < n; ++i) sum += static_cast<double>(a[i]) * b[i];
    return static_cast<float>(sum);
}
void rms(float* x, const float* weight, int n) {
    const float scale = 1.0f / std::sqrt(dot(x, x, n) / n + EPS);
    for (int i = 0; i < n; ++i) x[i] *= scale * weight[i];
}
void l2(float* x, int n) {
    const float scale = 1.0f / std::sqrt(dot(x, x, n) + EPS);
    for (int i = 0; i < n; ++i) x[i] *= scale;
}
void rope(float* x, int position, float base) {
    // GGUF's text RoPE has 64 rotary coordinates in each 256-wide head.
    for (int i = 0; i < 32; ++i) {
        const float theta = position / std::pow(base, 2.0f * i / 64.0f);
        const float c = std::cos(theta), s = std::sin(theta);
        const float a = x[i], b = x[i + 32];
        x[i] = a * c - b * s;
        x[i + 32] = a * s + b * c;
    }
}
std::string layer_name(int layer, const char* suffix) {
    return "blk." + std::to_string(layer) + "." + suffix;
}
}  // namespace

Inference::Inference(const std::string& path, int context, int layers, bool cuda)
    : file_(path), context_(context), layers_(layers) {
    if (context < 1 || context > 32768) throw std::invalid_argument("context must be 1..32768");
    if (layers < 1 || layers > 40) throw std::invalid_argument("layers must be 1..40");
    if (layers == 40 && file_.file_size() != 22134528992ULL)
        throw std::runtime_error("GGUF size differs from the pinned Qwen3.6 UD-Q4_K_M artifact");
    const auto arch_error = strata::check_architecture(file_);
    if (!arch_error.empty()) throw std::runtime_error(arch_error);
    const auto tensor_error = check_qwen36_tensors(file_);
    if (!tensor_error.empty()) throw std::runtime_error(tensor_error);
    // A cropped GGUF is useful for checking a layer prefix. Check every
    // accessed payload before mmap reads; the directory alone is not enough.
    for (const auto& t : file_.tensors()) {
        if (t.name.rfind("blk.", 0) == 0) {
            const size_t dot = t.name.find('.', 4);
            const int layer = std::stoi(t.name.substr(4, dot - 4));
            if (layer >= layers) continue;
        }
        int block_elements = 0, block_bytes = 0;
        strata::block_geometry(t.type, block_elements, block_bytes);
        const uint64_t size = t.elements() / block_elements * block_bytes;
        if (t.offset > file_.file_size() - file_.data_start() ||
            size > file_.file_size() - file_.data_start() - t.offset)
            throw std::runtime_error("GGUF lacks payload for " + t.name);
    }
    for (int layer = 0; layer < layers_; ++layer) {
        if (layer % 4 == 3) {
            attention_[layer].keys.reserve(static_cast<size_t>(std::min(context, 1024)) * 512);
            attention_[layer].values.reserve(static_cast<size_t>(std::min(context, 1024)) * 512);
        } else {
            linear_[layer].conv.resize(CONV_DIM * 3);
            linear_[layer].recurrent.resize(VALUE_HEADS * HEAD_DIM * HEAD_DIM);
        }
    }
    if (cuda) cuda_ = std::make_unique<CudaProjection>();
}

Inference::~Inference() = default;

const strata::TensorInfo& Inference::tensor(const std::string& name) const {
    const auto* t = file_.find(name);
    if (!t) throw std::runtime_error("missing tensor: " + name);
    return *t;
}

void Inference::row(const strata::TensorInfo& t, int64_t index, float* out) const {
    const int64_t width = static_cast<int64_t>(t.shape.at(0));
    const int64_t rows = static_cast<int64_t>(t.elements()) / width;
    if (index < 0 || index >= rows) throw std::out_of_range(t.name + " row");
    int block_elements = 0, block_bytes = 0;
    if (!strata::block_geometry(t.type, block_elements, block_bytes) || width % block_elements)
        throw std::runtime_error("unsupported row encoding: " + t.name);
    const uint8_t* source = file_.tensor_data(t) + index * (width / block_elements) * block_bytes;
    if (t.type == 0) { strata::dequantize_f32(source, out, static_cast<int>(width)); return; }
    if (t.type == 1) { strata::dequantize_f16(source, out, static_cast<int>(width)); return; }
    if (t.type == 30) { strata::dequantize_bf16(source, out, static_cast<int>(width)); return; }
    for (int64_t i = 0; i < width; i += block_elements, source += block_bytes) {
        switch (t.type) {
        case 8: strata::dequantize_q8_0(source, out + i); break;
        case 12: strata::dequantize_q4_K(source, out + i); break;
        case 13: strata::dequantize_q5_K(source, out + i); break;
        case 14: strata::dequantize_q6_K(source, out + i); break;
        default: throw std::runtime_error("no inference dequantizer for " + t.name + " (" + t.type_name() + ")");
        }
    }
}

std::vector<float> Inference::vec(const strata::TensorInfo& t) const {
    if (t.shape.size() != 1) throw std::runtime_error(t.name + " is not a vector");
    std::vector<float> result(t.shape[0]);
    row(t, 0, result.data());
    return result;
}

std::vector<float> Inference::matvec(const strata::TensorInfo& t, const std::vector<float>& x,
                                      int64_t expert) const {
    if (t.shape.size() != (expert < 0 ? 2u : 3u) || t.shape[0] != x.size())
        throw std::runtime_error("matrix dimensions differ for " + t.name);
    const int64_t rows = static_cast<int64_t>(t.shape[1]);
    if (expert >= 0 && expert >= static_cast<int64_t>(t.shape[2]))
        throw std::out_of_range("expert index");
    if (cuda_ && cuda_->supports(t.type))
        return cuda_->matvec(t, file_.tensor_data(t), x, expert);
    std::vector<float> out(rows);
    const int64_t first = expert < 0 ? 0 : expert * rows;
    const unsigned available = std::thread::hardware_concurrency();
    const unsigned workers = static_cast<uint64_t>(rows) * x.size() >= 4'000'000
        ? std::min(4u, available ? available : 1u) : 1u;
    std::vector<std::exception_ptr> failures(workers);
    auto work = [&](unsigned worker) {
        try {
            std::vector<float> weights(x.size());
            const int64_t begin = rows * worker / workers;
            const int64_t end = rows * (worker + 1) / workers;
            for (int64_t r = begin; r < end; ++r) {
                row(t, first + r, weights.data());
                out[r] = dot(weights.data(), x.data(), static_cast<int>(x.size()));
            }
        } catch (...) {
            failures[worker] = std::current_exception();
        }
    };
    std::vector<std::jthread> threads;
    threads.reserve(workers - 1);
    for (unsigned i = 1; i < workers; ++i) threads.emplace_back(work, i);
    work(0);
    for (auto& thread : threads) thread.join();
    for (const auto& failure : failures) if (failure) std::rethrow_exception(failure);
    return out;
}

std::vector<float> Inference::delta_net(int layer, const std::vector<float>& x) {
    const auto get = [&](const char* suffix) -> const strata::TensorInfo& { return tensor(layer_name(layer, suffix)); };
    auto qkv = matvec(get("attn_qkv.weight"), x);
    auto z = matvec(get("attn_gate.weight"), x);
    auto alpha = matvec(get("ssm_alpha.weight"), x);
    auto beta = matvec(get("ssm_beta.weight"), x);
    const auto a = vec(get("ssm_a")), dt = vec(get("ssm_dt.bias"));
    const auto conv_weight = &get("ssm_conv1d.weight");
    const auto norm_weight = vec(get("ssm_norm.weight"));
    auto& state = linear_[layer];
    std::vector<float> convolved(CONV_DIM), kernel(4);
    for (int c = 0; c < CONV_DIM; ++c) {
        row(*conv_weight, c, kernel.data());
        const float* history = state.conv.data() + c * 3;
        convolved[c] = silu(history[0] * kernel[0] + history[1] * kernel[1] +
                            history[2] * kernel[2] + qkv[c] * kernel[3]);
        float* mutable_history = state.conv.data() + c * 3;
        mutable_history[0] = mutable_history[1];
        mutable_history[1] = mutable_history[2];
        mutable_history[2] = qkv[c];
    }
    std::vector<float> out(4096);
    for (int head = 0; head < VALUE_HEADS; ++head) {
        // llama.cpp's GGUF converter tiles V heads, gates, decay and output
        // columns: [K0v0, K1v0, ..., K0v1, K1v1, ...].
        const int kh = head % KEY_HEADS;
        float q[HEAD_DIM], k[HEAD_DIM];
        std::copy_n(convolved.data() + kh * HEAD_DIM, HEAD_DIM, q);
        std::copy_n(convolved.data() + 2048 + kh * HEAD_DIM, HEAD_DIM, k);
        l2(q, HEAD_DIM); l2(k, HEAD_DIM);
        const float decay = std::exp(a[head] * softplus(alpha[head] + dt[head]));
        const float learning = sigmoid(beta[head]);
        float* memory = state.recurrent.data() + head * HEAD_DIM * HEAD_DIM;
        const float* value = convolved.data() + 4096 + head * HEAD_DIM;
        float delta[HEAD_DIM];
        for (int j = 0; j < HEAD_DIM; ++j) {
            float prediction = 0;
            for (int i = 0; i < HEAD_DIM; ++i) prediction += memory[i * HEAD_DIM + j] * decay * k[i];
            delta[j] = (value[j] - prediction) * learning;
        }
        for (int i = 0; i < HEAD_DIM; ++i) {
            for (int j = 0; j < HEAD_DIM; ++j)
                memory[i * HEAD_DIM + j] = memory[i * HEAD_DIM + j] * decay + k[i] * delta[j];
        }
        float* result = out.data() + head * HEAD_DIM;
        for (int j = 0; j < HEAD_DIM; ++j) {
            float sum = 0;
            for (int i = 0; i < HEAD_DIM; ++i) sum += memory[i * HEAD_DIM + j] * q[i];
            result[j] = sum / std::sqrt(static_cast<float>(HEAD_DIM));
        }
        rms(result, norm_weight.data(), HEAD_DIM);
        for (int j = 0; j < HEAD_DIM; ++j) result[j] *= silu(z[head * HEAD_DIM + j]);
    }
    return matvec(get("ssm_out.weight"), out);
}

std::vector<float> Inference::full_attention(int layer, const std::vector<float>& x) {
    const auto get = [&](const char* suffix) -> const strata::TensorInfo& { return tensor(layer_name(layer, suffix)); };
    auto q = matvec(get("attn_q.weight"), x);
    auto k = matvec(get("attn_k.weight"), x);
    auto v = matvec(get("attn_v.weight"), x);
    const auto q_norm = vec(get("attn_q_norm.weight"));
    const auto k_norm = vec(get("attn_k_norm.weight"));
    float rope_base = 10000000.0f;
    if (const auto* meta = file_.get("qwen35moe.rope.freq_base")) rope_base = static_cast<float>(meta->num());
    for (int h = 0; h < 16; ++h) {
        float* query = q.data() + h * 512;
        rms(query, q_norm.data(), 256);
        rope(query, position_, rope_base);
    }
    for (int h = 0; h < 2; ++h) {
        float* key = k.data() + h * 256;
        rms(key, k_norm.data(), 256);
        rope(key, position_, rope_base);
    }
    auto& cache = attention_[layer];
    cache.keys.insert(cache.keys.end(), k.begin(), k.end());
    cache.values.insert(cache.values.end(), v.begin(), v.end());
    std::vector<float> output(4096), scores(position_ + 1);
    for (int h = 0; h < 16; ++h) {
        const float* query = q.data() + h * 512;
        const int kv_head = h / 8;
        float maximum = -std::numeric_limits<float>::infinity();
        for (int t = 0; t <= position_; ++t) {
            scores[t] = dot(query, cache.keys.data() + t * 512 + kv_head * 256, 256) / 16.0f;
            maximum = std::max(maximum, scores[t]);
        }
        float denominator = 0;
        for (float& score : scores) { score = std::exp(score - maximum); denominator += score; }
        float* destination = output.data() + h * 256;
        for (int t = 0; t <= position_; ++t) {
            const float* value = cache.values.data() + t * 512 + kv_head * 256;
            const float probability = scores[t] / denominator;
            for (int j = 0; j < 256; ++j) destination[j] += probability * value[j];
        }
        for (int j = 0; j < 256; ++j) destination[j] *= sigmoid(query[256 + j]);
    }
    return matvec(get("attn_output.weight"), output);
}

std::vector<float> Inference::mixer(int layer, const std::vector<float>& x) {
    return layer % 4 == 3 ? full_attention(layer, x) : delta_net(layer, x);
}

std::vector<float> Inference::moe(int layer, const std::vector<float>& x) {
    const auto get = [&](const char* suffix) -> const strata::TensorInfo& { return tensor(layer_name(layer, suffix)); };
    auto logits = matvec(get("ffn_gate_inp.weight"), x);
    std::array<int, 256> order{};
    std::iota(order.begin(), order.end(), 0);
    std::partial_sort(order.begin(), order.begin() + 8, order.end(),
                      [&](int a, int b) { return logits[a] > logits[b]; });
    const float max_logit = logits[order[0]];
    float denominator = 0;
    for (int i = 0; i < 8; ++i) denominator += std::exp(logits[order[i]] - max_logit);
    std::vector<float> output(HIDDEN);
    for (int i = 0; i < 8; ++i) {
        const int expert = order[i];
        auto gate = matvec(get("ffn_gate_exps.weight"), x, expert);
        auto up = matvec(get("ffn_up_exps.weight"), x, expert);
        for (int j = 0; j < 512; ++j) gate[j] = silu(gate[j]) * up[j];
        auto down = matvec(get("ffn_down_exps.weight"), gate, expert);
        const float weight = std::exp(logits[expert] - max_logit) / denominator;
        for (int j = 0; j < HIDDEN; ++j) output[j] += weight * down[j];
    }
    auto shared_gate = vec(get("ffn_gate_inp_shexp.weight"));
    const float shared_weight = sigmoid(dot(shared_gate.data(), x.data(), HIDDEN));
    auto gate = matvec(get("ffn_gate_shexp.weight"), x);
    auto up = matvec(get("ffn_up_shexp.weight"), x);
    for (int j = 0; j < 512; ++j) gate[j] = silu(gate[j]) * up[j];
    auto shared = matvec(get("ffn_down_shexp.weight"), gate);
    for (int j = 0; j < HIDDEN; ++j) output[j] += shared_weight * shared[j];
    return output;
}

std::vector<float> Inference::step_hidden(int token_id) {
    if (position_ >= context_) throw std::out_of_range("context length exceeded");
    if (token_id < 0 || token_id >= 248320) throw std::out_of_range("token id");
    std::vector<float> x(HIDDEN);
    row(tensor("token_embd.weight"), token_id, x.data());
    for (int layer = 0; layer < layers_; ++layer) {
        auto normalized = x;
        const auto input_norm = vec(tensor(layer_name(layer, "attn_norm.weight")));
        rms(normalized.data(), input_norm.data(), HIDDEN);
        auto mixed = mixer(layer, normalized);
        for (int j = 0; j < HIDDEN; ++j) x[j] += mixed[j];
        normalized = x;
        const auto post_norm = vec(tensor(layer_name(layer, "post_attention_norm.weight")));
        rms(normalized.data(), post_norm.data(), HIDDEN);
        auto feed_forward = moe(layer, normalized);
        for (int j = 0; j < HIDDEN; ++j) x[j] += feed_forward[j];
    }
    ++position_;
    return x;
}

std::vector<float> Inference::step(int token_id) {
    auto x = step_hidden(token_id);
    const auto output_norm = vec(tensor("output_norm.weight"));
    rms(x.data(), output_norm.data(), HIDDEN);
    return matvec(tensor("output.weight"), x);
}

}  // namespace lamina::model
