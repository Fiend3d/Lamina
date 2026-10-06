#include "lamina/model/inference.hpp"
#include "lamina/model/cuda_projection.hpp"
#include "lamina/model/qwen36.hpp"
#include "strata/artifact/dequant.hpp"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <tuple>

namespace lamina::model {
namespace {
constexpr int HIDDEN = 2048;
constexpr int KEY_HEADS = 16;
constexpr int VALUE_HEADS = 32;
constexpr int HEAD_DIM = 128;
constexpr int CONV_DIM = 8192;
constexpr float EPS = 1e-6f;
thread_local double g_cpu_matvec_ms = 0.0;
thread_local double g_cuda_matvec_ms = 0.0;
thread_local double g_deltanet_ms = 0.0;
thread_local double g_attention_ms = 0.0;
thread_local double g_moe_ms = 0.0;
thread_local double g_dn_conv_ms = 0.0;
thread_local double g_dn_recur_ms = 0.0;
thread_local double g_attn_scores_ms = 0.0;
thread_local double g_device_moe_ms = 0.0;

struct ScopedTimer {
    double* target;
    std::chrono::steady_clock::time_point start;
    bool on;
    ScopedTimer(double* t, bool enabled)
        : target(t), start(enabled ? std::chrono::steady_clock::now()
                                   : std::chrono::steady_clock::time_point{}), on(enabled) {}
    ~ScopedTimer() {
        if (on)
            *target += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count();
    }
};

bool env_enabled(const char* name) {
#ifdef _MSC_VER
    char* raw_setting = nullptr;
    size_t length = 0;
    if (_dupenv_s(&raw_setting, &length, name)) return false;
    const std::unique_ptr<char, decltype(&std::free)> storage(raw_setting, &std::free);
    return storage && storage.get()[0] == '1';
#else
    const char* setting = std::getenv(name);
    return setting && setting[0] == '1';
#endif
}

bool profile_enabled() {
    static const bool enabled = env_enabled("LAMINA_PROFILE");
    return enabled;
}

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
    const unsigned available = std::thread::hardware_concurrency();
    cpu_workers_ = std::min(8u, available ? available : 1u);
#ifdef _MSC_VER
    char* raw_setting = nullptr;
    size_t setting_length = 0;
    if (_dupenv_s(&raw_setting, &setting_length, "LAMINA_CPU_THREADS"))
        throw std::runtime_error("cannot read LAMINA_CPU_THREADS");
    const std::unique_ptr<char, decltype(&std::free)> setting_storage(raw_setting, &std::free);
    const char* setting = setting_storage.get();
#else
    const char* setting = std::getenv("LAMINA_CPU_THREADS");
#endif
    if (setting) {
        const char* end = setting + std::char_traits<char>::length(setting);
        const auto parsed = std::from_chars(setting, end, cpu_workers_);
        if (parsed.ec != std::errc{} || parsed.ptr != end || cpu_workers_ < 1 || cpu_workers_ > 64)
            throw std::invalid_argument("LAMINA_CPU_THREADS must be 1..64");
    }
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
    auto cpu_matvec = [&]() {
        std::vector<float> out(rows);
        const int64_t first = expert < 0 ? 0 : expert * rows;
        const unsigned workers = static_cast<uint64_t>(rows) * x.size() >= 4'000'000
            ? std::min(cpu_workers_, static_cast<unsigned>(rows)) : 1u;
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
    };
    if (cuda_ && t.type == 0 && expert < 0 && t.shape.size() == 2 &&
        rows * static_cast<int64_t>(x.size()) >= 200000) {
        const bool profile = profile_enabled();
        const auto start = profile ? std::chrono::steady_clock::now()
                                   : std::chrono::steady_clock::time_point{};
        auto output = cuda_->matvec_f32(t, file_.tensor_data(t), x);
        if (profile)
            g_cuda_matvec_ms += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count();
        return output;
    }
    if (cuda_ && cuda_->supports(t.type)) {
        const bool profile = profile_enabled();
        const auto start = profile ? std::chrono::steady_clock::now()
                                   : std::chrono::steady_clock::time_point{};
        auto output = cuda_->matvec(t, file_.tensor_data(t), x, expert);
        if (profile)
            g_cuda_matvec_ms += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count();
        return output;
    }
    const bool profile = profile_enabled();
    const auto start = profile ? std::chrono::steady_clock::now()
                               : std::chrono::steady_clock::time_point{};
    auto output = cpu_matvec();
    if (profile)
        g_cpu_matvec_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
    return output;
}

std::vector<std::vector<float>> Inference::matvec_many(
    const std::vector<const strata::TensorInfo*>& tensors, const std::vector<float>& x,
    int64_t expert) const {
    bool compatible = cuda_ && tensors.size() >= 2 && tensors.size() <= 3;
    if (compatible) {
        const uint32_t type = tensors.front()->type;
        for (const auto* tensor : tensors)
            compatible = compatible && tensor && tensor->shape.size() == (expert < 0 ? 2u : 3u) &&
                         tensor->shape[0] == x.size() && tensor->type == type && cuda_->supports(type);
    }
    if (compatible) {
        std::vector<const uint8_t*> data;
        std::vector<int64_t> experts(tensors.size(), expert);
        data.reserve(tensors.size());
        for (const auto* tensor : tensors) data.push_back(file_.tensor_data(*tensor));
        const bool profile = profile_enabled();
        const auto start = profile ? std::chrono::steady_clock::now()
                                   : std::chrono::steady_clock::time_point{};
        auto outputs = cuda_->matvec_many(tensors, data, experts, x);
        if (profile)
            g_cuda_matvec_ms += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count();
        return outputs;
    }
    std::vector<std::vector<float>> outputs;
    outputs.reserve(tensors.size());
    for (const auto* tensor : tensors) outputs.push_back(matvec(*tensor, x, expert));
    return outputs;
}

std::vector<float> Inference::delta_net(int layer, const std::vector<float>& x) {
    const bool profile = profile_enabled();
    ScopedTimer total(&g_deltanet_ms, profile);
    const auto get = [&](const char* suffix) -> const strata::TensorInfo& { return tensor(layer_name(layer, suffix)); };
    if (cuda_) {
        const GdnWeights weights{
            &get("attn_qkv.weight"), file_.tensor_data(get("attn_qkv.weight")),
            &get("attn_gate.weight"), file_.tensor_data(get("attn_gate.weight")),
            &get("ssm_alpha.weight"), file_.tensor_data(get("ssm_alpha.weight")),
            &get("ssm_beta.weight"), file_.tensor_data(get("ssm_beta.weight")),
            &get("ssm_a"), file_.tensor_data(get("ssm_a")),
            &get("ssm_dt.bias"), file_.tensor_data(get("ssm_dt.bias")),
            &get("ssm_conv1d.weight"), file_.tensor_data(get("ssm_conv1d.weight")),
            &get("ssm_norm.weight"), file_.tensor_data(get("ssm_norm.weight")),
            &get("ssm_out.weight"), file_.tensor_data(get("ssm_out.weight"))};
        if (cuda_->supports_gdn(weights)) return cuda_->delta_net(layer, x, weights);
    }
    auto [qkv, z] = [&] {
        auto outputs = matvec_many({&get("attn_qkv.weight"), &get("attn_gate.weight")}, x);
        return std::pair{std::move(outputs[0]), std::move(outputs[1])};
    }();
    auto alpha = matvec(get("ssm_alpha.weight"), x);
    auto beta = matvec(get("ssm_beta.weight"), x);
    const auto a = vec(get("ssm_a")), dt = vec(get("ssm_dt.bias"));
    const auto conv_weight = &get("ssm_conv1d.weight");
    const auto norm_weight = vec(get("ssm_norm.weight"));
    auto& state = linear_[layer];
    std::vector<float> convolved(CONV_DIM), kernel(4);
    {
        ScopedTimer conv_timer(&g_dn_conv_ms, profile);
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
    }
    std::vector<float> out(4096);
    {
    ScopedTimer recur_timer(&g_dn_recur_ms, profile);
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
    }
    return matvec(get("ssm_out.weight"), out);
}

std::vector<float> Inference::full_attention(int layer, const std::vector<float>& x) {
    const bool profile = profile_enabled();
    ScopedTimer total(&g_attention_ms, profile);
    const auto get = [&](const char* suffix) -> const strata::TensorInfo& { return tensor(layer_name(layer, suffix)); };
    auto [q, k, v] = [&] {
        auto outputs = matvec_many({&get("attn_q.weight"), &get("attn_k.weight"), &get("attn_v.weight")}, x);
        return std::tuple{std::move(outputs[0]), std::move(outputs[1]), std::move(outputs[2])};
    }();
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
    {
    ScopedTimer scores_timer(&g_attn_scores_ms, profile);
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
    }
    return matvec(get("attn_output.weight"), output);
}

std::vector<float> Inference::mixer(int layer, const std::vector<float>& x) {
    return layer % 4 == 3 ? full_attention(layer, x) : delta_net(layer, x);
}

std::vector<float> Inference::moe(int layer, const std::vector<float>& x) {
    const bool profile = profile_enabled();
    ScopedTimer total(&g_moe_ms, profile);
    const auto get = [&](const char* suffix) -> const strata::TensorInfo& { return tensor(layer_name(layer, suffix)); };
    auto logits = matvec(get("ffn_gate_inp.weight"), x);
    std::array<int, 256> order{};
    std::iota(order.begin(), order.end(), 0);
    std::partial_sort(order.begin(), order.begin() + 8, order.end(),
                      [&](int a, int b) { return logits[a] > logits[b]; });
    const float max_logit = logits[order[0]];
    float denominator = 0;
    for (int i = 0; i < 8; ++i) denominator += std::exp(logits[order[i]] - max_logit);
    auto shared_gate = vec(get("ffn_gate_inp_shexp.weight"));
    const float shared_weight = sigmoid(dot(shared_gate.data(), x.data(), HIDDEN));
    const auto cpu_path = [&]() {
        std::vector<float> output(HIDDEN);
        for (int i = 0; i < 8; ++i) {
            const int expert = order[i];
            auto [gate, up] = [&] {
                auto outputs = matvec_many({&get("ffn_gate_exps.weight"), &get("ffn_up_exps.weight")}, x, expert);
                return std::pair{std::move(outputs[0]), std::move(outputs[1])};
            }();
            for (int j = 0; j < 512; ++j) gate[j] = silu(gate[j]) * up[j];
            auto down = matvec(get("ffn_down_exps.weight"), gate, expert);
            const float weight = std::exp(logits[expert] - max_logit) / denominator;
            for (int j = 0; j < HIDDEN; ++j) output[j] += weight * down[j];
        }
        auto [gate, up] = [&] {
            auto outputs = matvec_many({&get("ffn_gate_shexp.weight"), &get("ffn_up_shexp.weight")}, x);
            return std::pair{std::move(outputs[0]), std::move(outputs[1])};
        }();
        for (int j = 0; j < 512; ++j) gate[j] = silu(gate[j]) * up[j];
        auto shared = matvec(get("ffn_down_shexp.weight"), gate);
        for (int j = 0; j < HIDDEN; ++j) output[j] += shared_weight * shared[j];
        return output;
    };
    if (cuda_) {
        const MoeWeights routed{&get("ffn_gate_exps.weight"), file_.tensor_data(get("ffn_gate_exps.weight")),
                                &get("ffn_up_exps.weight"), file_.tensor_data(get("ffn_up_exps.weight")),
                                &get("ffn_down_exps.weight"), file_.tensor_data(get("ffn_down_exps.weight"))};
        const MoeWeights shared{&get("ffn_gate_shexp.weight"), file_.tensor_data(get("ffn_gate_shexp.weight")),
                                &get("ffn_up_shexp.weight"), file_.tensor_data(get("ffn_up_shexp.weight")),
                                &get("ffn_down_shexp.weight"), file_.tensor_data(get("ffn_down_shexp.weight"))};
        if (cuda_->supports_moe(routed, shared)) {
            std::vector<int> experts(8);
            std::vector<float> weights(8);
            for (int i = 0; i < 8; ++i) {
                experts[i] = order[i];
                weights[i] = std::exp(logits[order[i]] - max_logit) / denominator;
            }
            const auto device_start = profile ? std::chrono::steady_clock::now()
                                              : std::chrono::steady_clock::time_point{};
            auto device = cuda_->moe(x, routed, experts, weights, shared, shared_weight);
            if (profile)
                g_device_moe_ms += std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - device_start).count();
            return device;
        }
    }
    return cpu_path();
}

GdnWeights Inference::gdn_weights(int layer) const {
    const auto ptr = [&](const char* suffix) {
        const strata::TensorInfo& t = tensor(layer_name(layer, suffix));
        return std::tuple<const strata::TensorInfo*, const uint8_t*>{&t, file_.tensor_data(t)};
    };
    GdnWeights w;
    std::tie(w.qkv, w.qkv_data) = ptr("attn_qkv.weight");
    std::tie(w.gate, w.gate_data) = ptr("attn_gate.weight");
    std::tie(w.alpha, w.alpha_data) = ptr("ssm_alpha.weight");
    std::tie(w.beta, w.beta_data) = ptr("ssm_beta.weight");
    std::tie(w.a, w.a_data) = ptr("ssm_a");
    std::tie(w.dt, w.dt_data) = ptr("ssm_dt.bias");
    std::tie(w.conv, w.conv_data) = ptr("ssm_conv1d.weight");
    std::tie(w.norm, w.norm_data) = ptr("ssm_norm.weight");
    std::tie(w.out, w.out_data) = ptr("ssm_out.weight");
    return w;
}

AttnWeights Inference::attention_weights(int layer) const {
    const auto ptr = [&](const char* suffix) {
        const strata::TensorInfo& t = tensor(layer_name(layer, suffix));
        return std::tuple<const strata::TensorInfo*, const uint8_t*>{&t, file_.tensor_data(t)};
    };
    AttnWeights w;
    std::tie(w.q, w.q_data) = ptr("attn_q.weight");
    std::tie(w.k, w.k_data) = ptr("attn_k.weight");
    std::tie(w.v, w.v_data) = ptr("attn_v.weight");
    std::tie(w.q_norm, w.q_norm_data) = ptr("attn_q_norm.weight");
    std::tie(w.k_norm, w.k_norm_data) = ptr("attn_k_norm.weight");
    std::tie(w.out, w.out_data) = ptr("attn_output.weight");
    return w;
}

void Inference::moe_weights(int layer, MoeWeights& routed, MoeWeights& shared) const {
    const auto ptr = [&](const char* suffix) {
        const strata::TensorInfo& t = tensor(layer_name(layer, suffix));
        return std::tuple<const strata::TensorInfo*, const uint8_t*>{&t, file_.tensor_data(t)};
    };
    std::tie(routed.gate, routed.gate_data) = ptr("ffn_gate_exps.weight");
    std::tie(routed.up, routed.up_data) = ptr("ffn_up_exps.weight");
    std::tie(routed.down, routed.down_data) = ptr("ffn_down_exps.weight");
    std::tie(shared.gate, shared.gate_data) = ptr("ffn_gate_shexp.weight");
    std::tie(shared.up, shared.up_data) = ptr("ffn_up_shexp.weight");
    std::tie(shared.down, shared.down_data) = ptr("ffn_down_shexp.weight");
}

bool Inference::device_chain_supported() {
    if (device_chain_checked_) return device_chain_;
    device_chain_checked_ = true;
    if (!cuda_) return false;
    for (int layer = 0; layer < layers_; ++layer) {
        if (layer % 4 == 3) {
            if (!cuda_->supports_attention(attention_weights(layer))) return false;
        } else if (!cuda_->supports_gdn(gdn_weights(layer))) {
            return false;
        }
        MoeWeights routed;
        MoeWeights shared;
        moe_weights(layer, routed, shared);
        if (!cuda_->supports_moe(routed, shared)) return false;
    }
    cuda_->set_context(context_);
    device_chain_ = true;
    return true;
}

std::vector<float> Inference::step_hidden_device(int token_id) {
    std::vector<float> x(HIDDEN);
    row(tensor("token_embd.weight"), token_id, x.data());
    cuda_->hidden_upload(x);
    float rope_base = 10000000.0f;
    if (const auto* meta = file_.get("qwen35moe.rope.freq_base"))
        rope_base = static_cast<float>(meta->num());
    if (profile_enabled()) cuda_->mark_begin();
    for (int layer = 0; layer < layers_; ++layer) {
        const strata::TensorInfo& input_norm = tensor(layer_name(layer, "attn_norm.weight"));
        const strata::TensorInfo& post_norm = tensor(layer_name(layer, "post_attention_norm.weight"));
        const auto mixer_start = std::chrono::steady_clock::now();
        if (layer % 4 == 3) {
            cuda_->hidden_rms(input_norm, file_.tensor_data(input_norm), EPS);
            cuda_->attention_into_mix(layer, attention_weights(layer), position_, rope_base);
            cuda_->add_hidden_mix();
            cuda_->hidden_rms(post_norm, file_.tensor_data(post_norm), EPS);
        } else {
            cuda_->delta_layer_graph(layer, gdn_weights(layer), input_norm,
                                     file_.tensor_data(input_norm), post_norm,
                                     file_.tensor_data(post_norm), EPS);
        }
        if (profile_enabled()) {
            const double ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - mixer_start).count();
            if (layer % 4 == 3) g_attention_ms += ms; else g_deltanet_ms += ms;
        }
        MoeWeights routed;
        MoeWeights shared;
        moe_weights(layer, routed, shared);
        const strata::TensorInfo& router = tensor(layer_name(layer, "ffn_gate_inp.weight"));
        const strata::TensorInfo& shared_gate = tensor(layer_name(layer, "ffn_gate_inp_shexp.weight"));
        const auto moe_start = std::chrono::steady_clock::now();
        cuda_->moe_into_mix(router, file_.tensor_data(router), shared_gate,
                            file_.tensor_data(shared_gate), routed, shared);
        if (profile_enabled())
            g_device_moe_ms += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - moe_start).count();
        cuda_->add_hidden_mix();
    }
    if (profile_enabled())
        std::fprintf(stderr, "device gpu_busy_ms=%.3f\n", cuda_->mark_end_ms());
    return cuda_->hidden_download();
}

std::vector<float> Inference::step_hidden(int token_id) {
    const bool profile = profile_enabled();
    const auto step_start = std::chrono::steady_clock::now();
    g_cpu_matvec_ms = 0.0;
    g_cuda_matvec_ms = 0.0;
    g_deltanet_ms = 0.0;
    g_attention_ms = 0.0;
    g_moe_ms = 0.0;
    g_dn_conv_ms = 0.0;
    g_dn_recur_ms = 0.0;
    g_attn_scores_ms = 0.0;
    g_device_moe_ms = 0.0;
    if (position_ >= context_) throw std::out_of_range("context length exceeded");
    if (token_id < 0 || token_id >= 248320) throw std::out_of_range("token id");
    std::vector<float> x;
    if (cuda_ && device_chain_supported()) {
        x = step_hidden_device(token_id);
    } else {
        x.assign(HIDDEN, 0.0f);
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
    }
    ++position_;
    if (profile) {
        const double total = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - step_start).count();
        std::fprintf(stderr, "profile token=%d total_ms=%.3f cuda_matvec_ms=%.3f cpu_matvec_ms=%.3f "
                             "graph_other_ms=%.3f deltanet_ms=%.3f attention_ms=%.3f moe_ms=%.3f "
                             "dn_conv_ms=%.3f dn_recur_ms=%.3f attn_scores_ms=%.3f device_moe_ms=%.3f\\n",
                     position_, total, g_cuda_matvec_ms, g_cpu_matvec_ms,
                     std::max(0.0, total - g_cuda_matvec_ms - g_cpu_matvec_ms),
                     g_deltanet_ms, g_attention_ms, g_moe_ms,
                     g_dn_conv_ms, g_dn_recur_ms, g_attn_scores_ms, g_device_moe_ms);
        if (cuda_) {
            const auto stats = cuda_->stats();
            std::fprintf(stderr,
                         "  cache hits=%llu misses=%llu hit_rate=%.3f uploaded_mb=%.1f "
                         "evicted_mb=%.1f resident_mb=%.1f limit_mb=%.1f\n",
                         static_cast<unsigned long long>(stats.hits),
                         static_cast<unsigned long long>(stats.misses),
                         (stats.hits + stats.misses)
                             ? double(stats.hits) / double(stats.hits + stats.misses) : 0.0,
                         double(stats.uploaded_bytes) / 1048576.0,
                         double(stats.evicted_bytes) / 1048576.0,
                         double(stats.resident_bytes) / 1048576.0,
                         double(stats.cache_limit) / 1048576.0);
        }
    }
    return x;
}

std::vector<float> Inference::step(int token_id) {
    auto x = step_hidden(token_id);
    const auto output_norm = vec(tensor("output_norm.weight"));
    rms(x.data(), output_norm.data(), HIDDEN);
    return matvec(tensor("output.weight"), x);
}

}  // namespace lamina::model
