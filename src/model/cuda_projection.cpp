#include "lamina/model/cuda_projection.hpp"

#include <stdexcept>

#ifdef LAMINA_ENABLE_CUDA
#include "lamina/model/cuda_kernels.hpp"
#include "strata/kernels/gdn.hpp"
#include "strata/kernels/native_gdn.hpp"
#include "strata/kernels/native_gdn_preprocess.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

namespace lamina::model {
namespace {
void check(cudaError_t status, const char* action) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("CUDA ") + action + ": " + cudaGetErrorString(status));
}
constexpr size_t MIB = 1024ULL * 1024;
bool device_profile() {
    static const bool enabled = std::getenv("LAMINA_DEV_PROFILE") != nullptr;
    return enabled;
}
}

struct CudaProjection::Impl {
    struct Entry {
        void* device = nullptr;
        size_t bytes = 0;
        uint64_t used = 0;
        bool pinned = false;  // dense and shared weights: never evicted
    };
    std::unordered_map<std::string, Entry> weights;
    // Reuse evicted blocks instead of cudaMalloc/cudaFree per miss: on WDDM the
    // malloc serializes the device and dominated a token once the cache churned.
    std::unordered_map<size_t, std::vector<void*>> free_blocks;
    size_t cached_bytes = 0;
    size_t cache_limit = 0;
    uint64_t clock = 0;
    uint64_t stat_hits = 0;
    uint64_t stat_misses = 0;
    size_t stat_uploaded = 0;
    size_t stat_evicted = 0;
    cudaStream_t stream = nullptr;
    float* input = nullptr;
    float* output = nullptr;
    size_t input_capacity = 0, output_capacity = 0;
    // Device-resident scratch for the MoE block. Slots are one expert wide so
    // every expert plus the shared expert can be enqueued without host waits.
    float* accum_dev = nullptr;
    float* gate_dev = nullptr;
    float* up_dev = nullptr;
    float* swiglu_dev = nullptr;
    float* down_dev = nullptr;
    float* scales_dev = nullptr;
    size_t accum_capacity = 0, gate_capacity = 0, up_capacity = 0,
           swiglu_capacity = 0, down_capacity = 0, scales_capacity = 0;

    // Device-resident hidden state, its normalized copy, and the mixer/MoE result.
    float* hidden_dev = nullptr;
    float* norm_dev = nullptr;
    float* mix_dev = nullptr;
    size_t hidden_cap = 0, norm_cap = 0, mix_cap = 0;
    int max_context = 0;
    // Full-attention KV cache and per-layer scratch.
    struct AttnState {
        float* keys = nullptr;
        float* values = nullptr;
        int capacity = 0;
    };
    std::vector<AttnState> attn_states;
    float* a_q = nullptr;
    float* a_k = nullptr;
    float* a_v = nullptr;
    float* a_ctx = nullptr;
    size_t a_q_cap = 0, a_k_cap = 0, a_v_cap = 0, a_ctx_cap = 0;
    float* r_logits = nullptr;
    int* r_ids = nullptr;
    size_t r_logits_cap = 0, r_ids_cap = 0;
    // Mapped pinned doorbell staging (host pointers and their device aliases).
    static constexpr int kRouteMax = 16;
    int* h_ids_map = nullptr;
    float* h_scales_map = nullptr;
    int* h_flag_map = nullptr;
    int* d_ids_map = nullptr;
    float* d_scales_map = nullptr;
    int* d_flag_map = nullptr;
    unsigned route_seq = 0;
    cudaEvent_t mark_start = nullptr;
    cudaEvent_t mark_stop = nullptr;
    // DeltaNet device state, one entry per layer.
    struct GdnState {
        float* conv = nullptr;
        float* rec = nullptr;
    };
    std::vector<GdnState> gdn_states;
    struct MixerGraph {
        cudaGraphExec_t exec = nullptr;
        bool warmed = false;
    };
    std::vector<MixerGraph> mixer_graphs;
    float* g_qkv = nullptr;
    float* g_z = nullptr;
    float* g_alpha = nullptr;
    float* g_beta = nullptr;
    float* g_gate = nullptr;
    float* g_conv_raw = nullptr;
    float* g_conv_silu = nullptr;
    float* g_outnorm = nullptr;
    float* g_y = nullptr;
    float* g_ssm_out = nullptr;
    size_t g_qkv_cap = 0, g_z_cap = 0, g_alpha_cap = 0, g_beta_cap = 0, g_gate_cap = 0,
           g_conv_raw_cap = 0, g_conv_silu_cap = 0, g_outnorm_cap = 0, g_y_cap = 0,
           g_ssm_out_cap = 0;

    ~Impl() {
        for (auto& [name, entry] : weights) cudaFree(entry.device);
        for (auto& [bytes, blocks] : free_blocks)
            for (void* block : blocks) cudaFree(block);
        for (auto& state : gdn_states) {
            cudaFree(state.conv);
            cudaFree(state.rec);
        }
        for (auto& graph : mixer_graphs)
            if (graph.exec) cudaGraphExecDestroy(graph.exec);
        for (auto& state : attn_states) {
            cudaFree(state.keys);
            cudaFree(state.values);
        }
        cudaFree(hidden_dev);
        cudaFree(norm_dev);
        cudaFree(mix_dev);
        cudaFree(a_q);
        cudaFree(a_k);
        cudaFree(a_v);
        cudaFree(a_ctx);
        cudaFree(r_logits);
        cudaFree(r_ids);
        if (h_ids_map) cudaFreeHost(h_ids_map);
        if (h_scales_map) cudaFreeHost(h_scales_map);
        if (h_flag_map) cudaFreeHost(h_flag_map);
        cudaFree(input);
        cudaFree(output);
        cudaFree(accum_dev);
        cudaFree(gate_dev);
        cudaFree(up_dev);
        cudaFree(swiglu_dev);
        cudaFree(down_dev);
        cudaFree(scales_dev);
        cudaFree(g_qkv);
        cudaFree(g_z);
        cudaFree(g_alpha);
        cudaFree(g_beta);
        cudaFree(g_gate);
        cudaFree(g_conv_raw);
        cudaFree(g_conv_silu);
        cudaFree(g_outnorm);
        cudaFree(g_y);
        cudaFree(g_ssm_out);
        if (stream) cudaStreamDestroy(stream);
    }

    void ensure(float*& pointer, size_t& capacity, size_t elements) {
        if (elements <= capacity) return;
        check(cudaFree(pointer), "free scratch");
        pointer = nullptr;
        check(cudaMalloc(reinterpret_cast<void**>(&pointer), elements * sizeof(float)),
              "allocate scratch");
        capacity = elements;
    }

    void ensure(int*& pointer, size_t& capacity, size_t elements) {
        if (elements <= capacity) return;
        check(cudaFree(pointer), "free scratch");
        pointer = nullptr;
        check(cudaMalloc(reinterpret_cast<void**>(&pointer), elements * sizeof(int)),
              "allocate scratch");
        capacity = elements;
    }

    void reserve(size_t n_in, size_t n_out) {
        if (n_in > input_capacity) {
            check(cudaFree(input), "free input");
            input = nullptr;
            check(cudaMalloc(reinterpret_cast<void**>(&input), n_in * sizeof(float)), "allocate input");
            input_capacity = n_in;
        }
        if (n_out > output_capacity) {
            check(cudaFree(output), "free output");
            output = nullptr;
            check(cudaMalloc(reinterpret_cast<void**>(&output), n_out * sizeof(float)), "allocate output");
            output_capacity = n_out;
        }
        (void)n_in;
    }

    void* projection_weight(const strata::TensorInfo& tensor, const uint8_t* data, int64_t expert) {
        const int n_in = static_cast<int>(tensor.shape.at(0));
        const int n_out = static_cast<int>(tensor.shape.at(1));
        const size_t bytes = strata::kernels::native_mmvq_weight_bytes(tensor.type, n_in, n_out);
        if (expert >= 0 && expert >= static_cast<int64_t>(tensor.shape.at(2)))
            throw std::out_of_range("CUDA projection expert index");
        const auto* slice = data + (expert < 0 ? 0 : static_cast<size_t>(expert) * bytes);
        return weight(tensor, slice, expert, bytes);
    }

    void* device_alloc(size_t bytes) {
        auto found = free_blocks.find(bytes);
        if (found != free_blocks.end() && !found->second.empty()) {
            void* block = found->second.back();
            found->second.pop_back();
            return block;
        }
        void* block = nullptr;
        check(cudaMalloc(&block, bytes), "allocate weight");
        return block;
    }

    void device_release(void* block, size_t bytes) { free_blocks[bytes].push_back(block); }

    bool evict_oldest() {
        auto oldest = weights.end();
        for (auto it = weights.begin(); it != weights.end(); ++it) {
            if (it->second.pinned) continue;
            if (oldest == weights.end() || it->second.used < oldest->second.used) oldest = it;
        }
        if (oldest == weights.end()) return false;
        // MoE enqueues projection kernels asynchronously, so a weight being
        // evicted may still be referenced by work already on the stream.
        check(cudaStreamSynchronize(stream), "wait before eviction");
        device_release(oldest->second.device, oldest->second.bytes);
        cached_bytes -= oldest->second.bytes;
        stat_evicted += oldest->second.bytes;
        weights.erase(oldest);
        return true;
    }

    void* weight(const strata::TensorInfo& tensor, const uint8_t* data,
                 int64_t expert, size_t bytes) {
        const std::string key = tensor.name + "#type=" + std::to_string(tensor.type) +
                                "#expert=" + std::to_string(expert);
        if (auto found = weights.find(key); found != weights.end()) {
            found->second.used = ++clock;
            ++stat_hits;
            return found->second.device;
        }
        ++stat_misses;
        stat_uploaded += bytes;
        while (bytes > cache_limit - std::min(cached_bytes, cache_limit) && evict_oldest()) {}
        void* device = device_alloc(bytes);
        try {
            // Upload on the kernel stream: a synchronous pageable copy on the
            // legacy stream is not ordered against a non-blocking stream and
            // can race the projections already queued there.
            check(cudaMemcpyAsync(device, data, bytes, cudaMemcpyHostToDevice, stream),
                  "upload weight");
            weights.emplace(key, Entry{device, bytes, ++clock, expert < 0});
            cached_bytes += bytes;
            return device;
        } catch (...) {
            device_release(device, bytes);
            throw;
        }
    }
};

CudaProjection::CudaProjection() : impl_(std::make_unique<Impl>()) {
    int count = 0;
    check(cudaGetDeviceCount(&count), "count devices");
    if (count < 1) throw std::runtime_error("CUDA requested but no NVIDIA device is available");
    cudaDeviceProp device{};
    check(cudaGetDeviceProperties(&device, 0), "inspect device");
    if (device.major < 8)
        throw std::runtime_error("Lamina CUDA projections require an Ampere or newer GPU");
    check(cudaSetDevice(0), "select device");
    size_t free_bytes = 0, total_bytes = 0;
    check(cudaMemGetInfo(&free_bytes, &total_bytes), "inspect free memory");
    // Leave a quarter of currently free VRAM for CUDA modules, other work and
    // the activation buffers. The mapped 22 GB GGUF never needs to fit here.
    impl_->cache_limit = free_bytes - free_bytes / 4;
#ifdef _MSC_VER
    char* raw_setting = nullptr;
    size_t setting_length = 0;
    if (_dupenv_s(&raw_setting, &setting_length, "LAMINA_CUDA_CACHE_MB"))
        throw std::runtime_error("cannot read LAMINA_CUDA_CACHE_MB");
    const std::unique_ptr<char, decltype(&std::free)> setting_storage(raw_setting, &std::free);
    const char* setting = setting_storage.get();
#else
    const char* setting = std::getenv("LAMINA_CUDA_CACHE_MB");
#endif
    if (setting) {
        size_t mib = 0;
        const char* end = setting + std::char_traits<char>::length(setting);
        const auto parsed = std::from_chars(setting, end, mib);
        if (parsed.ec != std::errc{} || parsed.ptr != end || mib == 0 ||
            mib > impl_->cache_limit / MIB)
            throw std::invalid_argument("LAMINA_CUDA_CACHE_MB must be 1..75% of free VRAM in MiB");
        impl_->cache_limit = mib * MIB;
    }
    check(cudaStreamCreateWithFlags(&impl_->stream, cudaStreamNonBlocking), "create stream");
    // Mapped pinned staging for the router doorbell: the router writes ids and
    // weights straight into host-visible memory and raises a flag, so the host
    // spin avoids a per-layer cudaStreamSynchronize.
    check(cudaHostAlloc(reinterpret_cast<void**>(&impl_->h_ids_map),
                        impl_->kRouteMax * sizeof(int), cudaHostAllocMapped), "allocate router ids");
    check(cudaHostAlloc(reinterpret_cast<void**>(&impl_->h_scales_map),
                        (impl_->kRouteMax + 1) * sizeof(float), cudaHostAllocMapped),
          "allocate router scales");
    check(cudaHostAlloc(reinterpret_cast<void**>(&impl_->h_flag_map), sizeof(int),
                        cudaHostAllocMapped), "allocate router flag");
    check(cudaHostGetDevicePointer(reinterpret_cast<void**>(&impl_->d_ids_map),
                                   impl_->h_ids_map, 0), "map router ids");
    check(cudaHostGetDevicePointer(reinterpret_cast<void**>(&impl_->d_scales_map),
                                   impl_->h_scales_map, 0), "map router scales");
    check(cudaHostGetDevicePointer(reinterpret_cast<void**>(&impl_->d_flag_map),
                                   impl_->h_flag_map, 0), "map router flag");
}
CudaProjection::~CudaProjection() = default;

bool CudaProjection::supports(uint32_t type) const {
    return type == 8 || type == 12 || type == 13 || type == 14;
}

std::vector<float> CudaProjection::matvec(const strata::TensorInfo& tensor, const uint8_t* data,
                                          const std::vector<float>& x, int64_t expert) {
    if (!supports(tensor.type) || tensor.shape.size() != (expert < 0 ? 2u : 3u) ||
        tensor.shape[0] != x.size())
        throw std::invalid_argument("unsupported CUDA projection: " + tensor.name);
    const int n_in = static_cast<int>(x.size());
    const int n_out = static_cast<int>(tensor.shape[1]);
    const size_t bytes = strata::kernels::native_mmvq_weight_bytes(tensor.type, n_in, n_out);
    if (expert >= 0 && expert >= static_cast<int64_t>(tensor.shape[2]))
        throw std::out_of_range("expert index");
    impl_->reserve(x.size(), static_cast<size_t>(n_out));
    const auto* slice = data + (expert < 0 ? 0 : static_cast<size_t>(expert) * bytes);
    void* device_weight = impl_->weight(tensor, slice, expert, bytes);
    check(cudaMemcpy(impl_->input, x.data(), x.size() * sizeof(float),
                     cudaMemcpyHostToDevice), "upload activation");
    strata::kernels::native_mmvq_f32(tensor.type, device_weight, impl_->input, impl_->output,
                                     n_in, n_out, impl_->stream);
    check(cudaGetLastError(), "launch projection");
    std::vector<float> result(static_cast<size_t>(n_out));
    check(cudaStreamSynchronize(impl_->stream), "finish projection");
    check(cudaMemcpy(result.data(), impl_->output, result.size() * sizeof(float),
                     cudaMemcpyDeviceToHost), "download projection");
    return result;
}

std::vector<std::vector<float>> CudaProjection::matvec_many(
    const std::vector<const strata::TensorInfo*>& tensors,
    const std::vector<const uint8_t*>& data,
    const std::vector<int64_t>& experts, const std::vector<float>& x) {
    const size_t count = tensors.size();
    if (count < 2 || count > 3 || data.size() != count || experts.size() != count)
        throw std::invalid_argument("CUDA projection batch requires 2..3 matched tensors");
    if (!tensors[0]) throw std::invalid_argument("CUDA projection batch contains a null tensor");
    const auto valid = [&](const strata::TensorInfo& tensor, int64_t expert) {
        return supports(tensor.type) && tensor.shape.size() == (expert < 0 ? 2u : 3u) &&
               tensor.shape[0] == x.size() && (expert < 0 || expert < static_cast<int64_t>(tensor.shape[2]));
    };
    const uint32_t type = tensors[0]->type;
    int total_rows = 0;
    std::vector<int> row_counts(count), types(count, static_cast<int>(type));
    std::vector<void*> weights(count);
    std::vector<float*> output_ptrs(count);
    for (size_t i = 0; i < count; ++i) {
        if (!tensors[i] || !data[i] || !valid(*tensors[i], experts[i]) || tensors[i]->type != type)
            throw std::invalid_argument("incompatible CUDA projection batch");
        row_counts[i] = static_cast<int>(tensors[i]->shape[1]);
        total_rows += row_counts[i];
    }
    const int n_in = static_cast<int>(x.size());
    impl_->reserve(x.size(), static_cast<size_t>(total_rows));
    size_t output_offset = 0;
    for (size_t i = 0; i < count; ++i) {
        weights[i] = impl_->projection_weight(*tensors[i], data[i], experts[i]);
        output_ptrs[i] = impl_->output + output_offset;
        output_offset += static_cast<size_t>(row_counts[i]);
    }
    check(cudaMemcpy(impl_->input, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice),
          "upload batched activation");
    strata::kernels::native_mmvq_f32_many(static_cast<int>(count), types.data(), weights.data(),
                                          output_ptrs.data(), row_counts.data(), impl_->input,
                                          n_in, impl_->stream);
    check(cudaGetLastError(), "launch batched projections");
    check(cudaStreamSynchronize(impl_->stream), "finish batched projections");
    check(cudaDeviceSynchronize(), "synchronize batched projections");
    std::vector<float> contiguous_output(static_cast<size_t>(total_rows));
    check(cudaMemcpy(contiguous_output.data(), impl_->output, contiguous_output.size() * sizeof(float),
                     cudaMemcpyDeviceToHost), "download batched projections");
    output_offset = 0;
    std::vector<std::vector<float>> results;
    results.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        results.emplace_back(static_cast<size_t>(row_counts[i]));
        std::copy_n(contiguous_output.data() + output_offset, results.back().size(), results.back().data());
        output_offset += static_cast<size_t>(row_counts[i]);
    }
    return results;
}

std::vector<float> CudaProjection::matvec_f32(const strata::TensorInfo& tensor, const uint8_t* data,
                                              const std::vector<float>& x) {
    if (tensor.type != 0 || tensor.shape.size() != 2 || tensor.shape[0] != x.size())
        throw std::invalid_argument("unsupported CUDA F32 projection: " + tensor.name);
    const int n_in = static_cast<int>(x.size());
    const int n_out = static_cast<int>(tensor.shape[1]);
    const size_t bytes = static_cast<size_t>(n_in) * static_cast<size_t>(n_out) * sizeof(float);
    void* device_weight = impl_->weight(tensor, data, -1, bytes);
    impl_->reserve(x.size(), static_cast<size_t>(n_out));
    check(cudaMemcpyAsync(impl_->input, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice,
                          impl_->stream), "upload F32 activation");
    cuda::gemv_f32(static_cast<const float*>(device_weight), impl_->input, impl_->output, n_in,
                   n_out, impl_->stream);
    check(cudaGetLastError(), "launch F32 projection");
    std::vector<float> result(static_cast<size_t>(n_out));
    check(cudaMemcpyAsync(result.data(), impl_->output, result.size() * sizeof(float),
                          cudaMemcpyDeviceToHost, impl_->stream), "download F32 projection");
    check(cudaStreamSynchronize(impl_->stream), "finish F32 projection");
    return result;
}

bool CudaProjection::supports_gdn(const GdnWeights& w) const {
    const auto present = [](const strata::TensorInfo* t, const uint8_t* d) { return t && d; };
    if (!present(w.qkv, w.qkv_data) || !present(w.gate, w.gate_data) ||
        !present(w.out, w.out_data) || !present(w.alpha, w.alpha_data) ||
        !present(w.beta, w.beta_data) || !present(w.a, w.a_data) || !present(w.dt, w.dt_data) ||
        !present(w.conv, w.conv_data) || !present(w.norm, w.norm_data))
        return false;
    return supports(w.qkv->type) && supports(w.gate->type) && w.qkv->type == w.gate->type &&
           supports(w.out->type) &&
           w.alpha->type == 0 && w.beta->type == 0 && w.a->type == 0 && w.dt->type == 0 &&
           w.conv->type == 0 && w.norm->type == 0 &&
           w.qkv->shape.size() == 2 && w.qkv->shape[0] == 2048 && w.qkv->shape[1] == 8192 &&
           w.gate->shape.size() == 2 && w.gate->shape[0] == 2048 && w.gate->shape[1] == 4096 &&
           w.out->shape.size() == 2 && w.out->shape[0] == 4096 && w.out->shape[1] == 2048 &&
           w.alpha->shape.size() == 2 && w.alpha->shape[0] == 2048 && w.alpha->shape[1] == 32 &&
           w.conv->shape.size() == 2 && w.conv->shape[0] == 4 && w.conv->shape[1] == 8192 &&
           w.norm->shape.size() == 1 && w.norm->shape[0] == 128 &&
           w.a->shape.size() == 1 && w.a->shape[0] == 32 &&
           w.dt->shape.size() == 1 && w.dt->shape[0] == 32;
}

void CudaProjection::delta_net_core(int layer, const float* x_dev, float* out_dev,
                                    const GdnWeights& w) {
    constexpr int kHidden = 2048;
    constexpr int kChannels = 8192;
    constexpr int kHeads = 32;
    constexpr int kS = 128;
    constexpr float kEps = 1e-6f;
    if (!supports_gdn(w)) throw std::invalid_argument("unsupported CUDA DeltaNet layer");
    if (layer < 0) throw std::invalid_argument("CUDA DeltaNet layer index");
    if (static_cast<size_t>(layer) >= impl_->gdn_states.size())
        impl_->gdn_states.resize(static_cast<size_t>(layer) + 1);
    auto& state = impl_->gdn_states[static_cast<size_t>(layer)];
    if (!state.conv) {
        check(cudaMalloc(reinterpret_cast<void**>(&state.conv),
                         static_cast<size_t>(kChannels) * 3 * sizeof(float)), "allocate conv state");
        check(cudaMemset(state.conv, 0, static_cast<size_t>(kChannels) * 3 * sizeof(float)),
              "clear conv state");
    }
    if (!state.rec) {
        check(cudaMalloc(reinterpret_cast<void**>(&state.rec),
                         static_cast<size_t>(kS) * kHeads * kS * sizeof(float)),
              "allocate recurrent state");
        check(cudaMemset(state.rec, 0, static_cast<size_t>(kS) * kHeads * kS * sizeof(float)),
              "clear recurrent state");
    }
    impl_->ensure(impl_->g_qkv, impl_->g_qkv_cap, kChannels);
    impl_->ensure(impl_->g_z, impl_->g_z_cap, 4096);
    impl_->ensure(impl_->g_alpha, impl_->g_alpha_cap, kHeads);
    impl_->ensure(impl_->g_beta, impl_->g_beta_cap, kHeads);
    impl_->ensure(impl_->g_gate, impl_->g_gate_cap, kHeads);
    impl_->ensure(impl_->g_conv_raw, impl_->g_conv_raw_cap, kChannels);
    impl_->ensure(impl_->g_conv_silu, impl_->g_conv_silu_cap, kChannels);
    impl_->ensure(impl_->g_outnorm, impl_->g_outnorm_cap, static_cast<size_t>(kHeads) * kS);
    impl_->ensure(impl_->g_y, impl_->g_y_cap, static_cast<size_t>(kHeads) * kS);
    impl_->ensure(impl_->g_ssm_out, impl_->g_ssm_out_cap, kHidden);

    // qkv and attn_gate share the activation; one grouped MMVQ dispatch.
    const void* qkv_gate_weights[2] = {impl_->projection_weight(*w.qkv, w.qkv_data, -1),
                                       impl_->projection_weight(*w.gate, w.gate_data, -1)};
    const float* qkv_gate_inputs[2] = {x_dev, x_dev};
    float* qkv_gate_outputs[2] = {impl_->g_qkv, impl_->g_z};
    const int qkv_gate_rows[2] = {kChannels, 4096};
    strata::kernels::native_mmvq_f32_grouped(static_cast<int>(w.qkv->type), qkv_gate_weights,
                                             qkv_gate_inputs, qkv_gate_outputs, qkv_gate_rows, 2,
                                             kChannels + 4096, kHidden, impl_->stream);
    // alpha and beta are dense F32 projections.
    void* alpha_weight = impl_->weight(*w.alpha, w.alpha_data, -1,
                                       static_cast<size_t>(kHidden) * kHeads * sizeof(float));
    void* beta_weight = impl_->weight(*w.beta, w.beta_data, -1,
                                      static_cast<size_t>(kHidden) * kHeads * sizeof(float));
    cuda::gemv_f32(static_cast<const float*>(alpha_weight), x_dev, impl_->g_alpha, kHidden,
                   kHeads, impl_->stream);
    cuda::gemv_f32(static_cast<const float*>(beta_weight), x_dev, impl_->g_beta, kHidden,
                   kHeads, impl_->stream);
    // Causal convolution + SiLU, then L2-normalize q and k.
    void* conv_weight = impl_->weight(*w.conv, w.conv_data, -1,
                                      static_cast<size_t>(4) * kChannels * sizeof(float));
    strata::kernels::native_gdn_conv_silu(state.conv, impl_->g_qkv,
                                          static_cast<const float*>(conv_weight), impl_->g_conv_raw,
                                          impl_->g_conv_silu, kChannels, 4, impl_->stream);
    strata::kernels::native_gdn_l2_norm(impl_->g_conv_silu, 16, kS, kEps, impl_->stream);
    strata::kernels::native_gdn_l2_norm(impl_->g_conv_silu + 2048, 16, kS, kEps, impl_->stream);
    // decay = exp(ssm_a * softplus(alpha + dt)); learning = sigmoid(beta).
    void* a_weight = impl_->weight(*w.a, w.a_data, -1, static_cast<size_t>(kHeads) * sizeof(float));
    void* dt_weight = impl_->weight(*w.dt, w.dt_data, -1, static_cast<size_t>(kHeads) * sizeof(float));
    strata::kernels::native_gdn_beta_gate(impl_->g_beta, kHeads, impl_->stream);
    strata::kernels::native_gdn_gate(impl_->g_alpha, static_cast<const float*>(dt_weight),
                                     static_cast<const float*>(a_weight), impl_->g_gate, kHeads,
                                     impl_->stream);
    const strata::kernels::GdnShapes shapes{kS, 16, kHeads};
    strata::kernels::native_gdn_step(state.rec, impl_->g_conv_silu, impl_->g_conv_silu + 2048,
                                     impl_->g_conv_silu + 4096, impl_->g_gate, impl_->g_beta,
                                     impl_->g_outnorm, shapes, impl_->stream);
    // Closing RMS norm with gamma and SiLU(z), then the ssm_out projection.
    void* norm_weight = impl_->weight(*w.norm, w.norm_data, -1,
                                      static_cast<size_t>(kS) * sizeof(float));
    cuda::gdn_out_norm_silu(impl_->g_outnorm, impl_->g_z, static_cast<const float*>(norm_weight),
                            impl_->g_y, kHeads, kEps, impl_->stream);
    const void* ssm_out_weight = impl_->projection_weight(*w.out, w.out_data, -1);
    const float* ssm_out_input = impl_->g_y;
    float* ssm_out_output = out_dev;
    const int ssm_out_rows = kHidden;
    strata::kernels::native_mmvq_f32_grouped(static_cast<int>(w.out->type), &ssm_out_weight,
                                             &ssm_out_input, &ssm_out_output, &ssm_out_rows, 1,
                                             kHidden, 4096, impl_->stream);
    check(cudaGetLastError(), "launch DeltaNet");
}

std::vector<float> CudaProjection::delta_net(int layer, const std::vector<float>& x,
                                             const GdnWeights& w) {
    if (x.size() != 2048) throw std::invalid_argument("CUDA DeltaNet activation width");
    impl_->ensure(impl_->g_ssm_out, impl_->g_ssm_out_cap, 2048);
    impl_->reserve(x.size(), 2048);
    check(cudaMemcpyAsync(impl_->input, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice,
                          impl_->stream), "upload DeltaNet activation");
    delta_net_core(layer, impl_->input, impl_->g_ssm_out, w);
    std::vector<float> result(2048);
    check(cudaMemcpyAsync(result.data(), impl_->g_ssm_out, result.size() * sizeof(float),
                          cudaMemcpyDeviceToHost, impl_->stream), "download DeltaNet");
    check(cudaStreamSynchronize(impl_->stream), "finish DeltaNet");
    return result;
}

bool CudaProjection::supports_moe(const MoeWeights& routed, const MoeWeights& shared) const {
    const auto valid = [&](const MoeWeights& w, size_t rank) {
        if (!w.gate || !w.up || !w.down || !w.gate_data || !w.up_data || !w.down_data) return false;
        const strata::TensorInfo& g = *w.gate;
        const strata::TensorInfo& u = *w.up;
        const strata::TensorInfo& d = *w.down;
        return supports(g.type) && supports(u.type) && supports(d.type) &&
               g.shape.size() == rank && u.shape.size() == rank && d.shape.size() == rank &&
               g.shape[0] == u.shape[0] && g.shape[1] == u.shape[1] &&
               d.shape[0] == g.shape[1] && d.shape[1] == g.shape[0];
    };
    return valid(routed, 3) && valid(shared, 2);
}

void CudaProjection::moe_core(const float* x_dev, float* out_dev, const MoeWeights& routed,
                              const std::vector<int>& experts, const std::vector<float>& weights,
                              const MoeWeights& shared, float shared_weight) {
    if (!supports_moe(routed, shared))
        throw std::invalid_argument("unsupported CUDA MoE block");
    if (weights.size() != experts.size())
        throw std::invalid_argument("CUDA MoE weight count differs from expert count");
    const int n_in = static_cast<int>(routed.gate->shape[0]);
    const int ff = static_cast<int>(routed.gate->shape[1]);
    const int hidden = static_cast<int>(routed.down->shape[1]);
    const size_t slots = experts.size() + 1;  // routed experts plus the shared expert
    impl_->ensure(impl_->accum_dev, impl_->accum_capacity, static_cast<size_t>(hidden));
    impl_->ensure(impl_->gate_dev, impl_->gate_capacity, slots * ff);
    impl_->ensure(impl_->up_dev, impl_->up_capacity, slots * ff);
    impl_->ensure(impl_->swiglu_dev, impl_->swiglu_capacity, slots * ff);
    impl_->ensure(impl_->down_dev, impl_->down_capacity, slots * hidden);
    impl_->ensure(impl_->scales_dev, impl_->scales_capacity, slots);

    const auto t_scales = std::chrono::steady_clock::now();
    std::vector<float> scales(experts.size());
    for (size_t i = 0; i < experts.size(); ++i) scales[i] = weights[i];
    scales.push_back(shared_weight);
    check(cudaMemcpyAsync(impl_->scales_dev, scales.data(), slots * sizeof(float),
                          cudaMemcpyHostToDevice, impl_->stream), "upload MoE scales");
    const auto t_after_scales = std::chrono::steady_clock::now();

    const auto moe_weights = [&](size_t slot) -> const MoeWeights& {
        return slot < experts.size() ? routed : shared;
    };
    const auto expert_id = [&](size_t slot) -> int64_t {
        return slot < experts.size() ? experts[slot] : -1;
    };
    // Group every same-format matrix into one launch. Routed and shared experts
    // have different formats, so there are four batches: routed/shared gate/up
    // and routed/shared down.
    struct GroupItem {
        const void* weight;
        const float* input;
        float* output;
        int rows;
    };
    const auto launch_group = [&](int type, int batch_n_in, const std::vector<GroupItem>& items) {
        if (items.empty()) return;
        const size_t n = items.size();
        std::vector<const void*> weights(n);
        std::vector<const float*> inputs(n);
        std::vector<float*> outputs(n);
        std::vector<int> rows(n);
        int total_rows = 0;
        for (size_t i = 0; i < n; ++i) {
            weights[i] = items[i].weight;
            inputs[i] = items[i].input;
            outputs[i] = items[i].output;
            rows[i] = items[i].rows;
            total_rows += items[i].rows;
        }
        strata::kernels::native_mmvq_f32_grouped(type, weights.data(), inputs.data(),
                                                 outputs.data(), rows.data(), static_cast<int>(n),
                                                 total_rows, batch_n_in, impl_->stream);
    };
    std::vector<GroupItem> gate_up_routed, gate_up_shared, down_routed, down_shared;
    for (size_t s = 0; s < slots; ++s) {
        const MoeWeights& w = moe_weights(s);
        const int64_t expert = expert_id(s);
        const bool is_routed = s < experts.size();
        GroupItem gate{impl_->projection_weight(*w.gate, w.gate_data, expert), x_dev,
                       impl_->gate_dev + s * ff, ff};
        GroupItem up{impl_->projection_weight(*w.up, w.up_data, expert), x_dev,
                     impl_->up_dev + s * ff, ff};
        GroupItem down{impl_->projection_weight(*w.down, w.down_data, expert),
                       impl_->swiglu_dev + s * ff, impl_->down_dev + s * hidden, hidden};
        if (is_routed) {
            gate_up_routed.push_back(gate);
            gate_up_routed.push_back(up);
            down_routed.push_back(down);
        } else {
            gate_up_shared.push_back(gate);
            gate_up_shared.push_back(up);
            down_shared.push_back(down);
        }
    }
    launch_group(static_cast<int>(routed.gate->type), n_in, gate_up_routed);
    launch_group(static_cast<int>(shared.gate->type), n_in, gate_up_shared);
    const auto t_after_gu = std::chrono::steady_clock::now();
    cuda::swiglu(impl_->gate_dev, impl_->up_dev, impl_->swiglu_dev,
                 static_cast<int>(slots) * ff, impl_->stream);
    const int down_n_in = static_cast<int>(routed.down->shape[0]);
    launch_group(static_cast<int>(routed.down->type), down_n_in, down_routed);
    launch_group(static_cast<int>(shared.down->type), down_n_in, down_shared);
    const auto t_after_down = std::chrono::steady_clock::now();
    cuda::moe_combine(impl_->down_dev, impl_->scales_dev, static_cast<int>(slots), hidden, out_dev,
                      impl_->stream);
    check(cudaGetLastError(), "launch MoE");
    if (device_profile()) {
        const auto t_end = std::chrono::steady_clock::now();
        std::fprintf(stderr, "  core scales=%.3f gu=%.3f down=%.3f combine=%.3f\n",
                     std::chrono::duration<double, std::milli>(t_after_scales - t_scales).count(),
                     std::chrono::duration<double, std::milli>(t_after_gu - t_after_scales).count(),
                     std::chrono::duration<double, std::milli>(t_after_down - t_after_gu).count(),
                     std::chrono::duration<double, std::milli>(t_end - t_after_down).count());
    }
}

std::vector<float> CudaProjection::moe(const std::vector<float>& x, const MoeWeights& routed,
                                       const std::vector<int>& experts,
                                       const std::vector<float>& weights,
                                       const MoeWeights& shared, float shared_weight) {
    if (!supports_moe(routed, shared))
        throw std::invalid_argument("unsupported CUDA MoE block");
    const int n_in = static_cast<int>(routed.gate->shape[0]);
    const int hidden = static_cast<int>(routed.down->shape[1]);
    if (x.size() != static_cast<size_t>(n_in))
        throw std::invalid_argument("CUDA MoE activation width");
    impl_->reserve(static_cast<size_t>(n_in), static_cast<size_t>(hidden));
    check(cudaMemcpyAsync(impl_->input, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice,
                          impl_->stream), "upload MoE activation");
    moe_core(impl_->input, impl_->accum_dev, routed, experts, weights, shared, shared_weight);
    std::vector<float> result(static_cast<size_t>(hidden));
    check(cudaMemcpyAsync(result.data(), impl_->accum_dev, result.size() * sizeof(float),
                          cudaMemcpyDeviceToHost, impl_->stream), "download MoE");
    check(cudaStreamSynchronize(impl_->stream), "finish MoE");
    return result;
}

void CudaProjection::set_context(int max_context) { impl_->max_context = max_context; }

void CudaProjection::hidden_upload(const std::vector<float>& x) {
    impl_->ensure(impl_->hidden_dev, impl_->hidden_cap, x.size());
    check(cudaMemcpyAsync(impl_->hidden_dev, x.data(), x.size() * sizeof(float),
                          cudaMemcpyHostToDevice, impl_->stream), "upload hidden");
}

std::vector<float> CudaProjection::hidden_download() {
    std::vector<float> result(static_cast<size_t>(impl_->hidden_cap));
    check(cudaMemcpyAsync(result.data(), impl_->hidden_dev, result.size() * sizeof(float),
                          cudaMemcpyDeviceToHost, impl_->stream), "download hidden");
    check(cudaStreamSynchronize(impl_->stream), "finish hidden");
    return result;
}

void CudaProjection::mix_upload(const std::vector<float>& x) {
    impl_->ensure(impl_->mix_dev, impl_->mix_cap, x.size());
    check(cudaMemcpyAsync(impl_->mix_dev, x.data(), x.size() * sizeof(float),
                          cudaMemcpyHostToDevice, impl_->stream), "upload mix");
}

void CudaProjection::hidden_rms(const strata::TensorInfo& gamma, const uint8_t* data, float epsilon) {
    const int n = static_cast<int>(impl_->hidden_cap);
    impl_->ensure(impl_->norm_dev, impl_->norm_cap, static_cast<size_t>(n));
    check(cudaMemcpyAsync(impl_->norm_dev, impl_->hidden_dev, static_cast<size_t>(n) * sizeof(float),
                          cudaMemcpyDeviceToDevice, impl_->stream), "copy hidden");
    void* weight = impl_->weight(gamma, data, -1, static_cast<size_t>(gamma.shape[0]) * sizeof(float));
    cuda::rms_norm(impl_->norm_dev, static_cast<const float*>(weight), n, epsilon, impl_->stream);
}

void CudaProjection::add_hidden_mix() {
    cuda::add_inplace(impl_->hidden_dev, impl_->mix_dev, static_cast<int>(impl_->hidden_cap),
                      impl_->stream);
}

void CudaProjection::delta_net_into_mix(int layer, const GdnWeights& w) {
    impl_->ensure(impl_->mix_dev, impl_->mix_cap, 2048);
    delta_net_core(layer, impl_->norm_dev, impl_->mix_dev, w);
}

void CudaProjection::delta_layer_graph(int layer, const GdnWeights& w,
                                       const strata::TensorInfo& input_norm,
                                       const uint8_t* input_norm_data,
                                       const strata::TensorInfo& post_norm,
                                       const uint8_t* post_norm_data, float epsilon) {
    if (layer < 0) throw std::invalid_argument("CUDA DeltaNet layer index");
    if (static_cast<size_t>(layer) >= impl_->mixer_graphs.size())
        impl_->mixer_graphs.resize(static_cast<size_t>(layer) + 1);
    auto& graph = impl_->mixer_graphs[static_cast<size_t>(layer)];
    if (graph.exec) {
        check(cudaGraphLaunch(graph.exec, impl_->stream), "launch DeltaNet graph");
        return;
    }
    if (!graph.warmed) {
        graph.warmed = true;
        hidden_rms(input_norm, input_norm_data, epsilon);
        delta_net_into_mix(layer, w);
        add_hidden_mix();
        hidden_rms(post_norm, post_norm_data, epsilon);
        return;
    }
    // Everything is allocated and cached, so capture performs no allocations.
    check(cudaStreamSynchronize(impl_->stream), "flush before capture");
    check(cudaStreamBeginCapture(impl_->stream, cudaStreamCaptureModeThreadLocal), "begin capture");
    hidden_rms(input_norm, input_norm_data, epsilon);
    delta_net_into_mix(layer, w);
    add_hidden_mix();
    hidden_rms(post_norm, post_norm_data, epsilon);
    cudaGraph_t captured = nullptr;
    check(cudaStreamEndCapture(impl_->stream, &captured), "end capture");
    check(cudaGraphInstantiate(&graph.exec, captured, nullptr, nullptr, 0), "instantiate graph");
    cudaGraphDestroy(captured);
    check(cudaGraphLaunch(graph.exec, impl_->stream), "launch DeltaNet graph");
}

void CudaProjection::moe_into_mix(const strata::TensorInfo& router, const uint8_t* router_data,
                                  const strata::TensorInfo& shared_gate,
                                  const uint8_t* shared_gate_data, const MoeWeights& routed,
                                  const MoeWeights& shared) {
    if (!supports_moe(routed, shared)) throw std::invalid_argument("unsupported CUDA MoE block");
    const int n_in = static_cast<int>(routed.gate->shape[0]);
    const int experts_count = static_cast<int>(router.shape[1]);
    const int top_k = 8;
    impl_->ensure(impl_->r_logits, impl_->r_logits_cap, static_cast<size_t>(experts_count));
    void* router_weight = impl_->weight(router, router_data, -1,
                                        static_cast<size_t>(n_in) * experts_count * sizeof(float));
    void* shared_gate_weight = impl_->weight(shared_gate, shared_gate_data, -1,
                                             static_cast<size_t>(n_in) * sizeof(float));
    const auto route_start = std::chrono::steady_clock::now();
    // Router -> top-k -> doorbell, all writing straight into mapped pinned memory.
    cuda::gemv_f32(static_cast<const float*>(router_weight), impl_->norm_dev, impl_->r_logits,
                   n_in, experts_count, impl_->stream);
    cuda::dot_sigmoid(static_cast<const float*>(shared_gate_weight), impl_->norm_dev, n_in,
                      impl_->d_scales_map + top_k, impl_->stream);
    cuda::router_topk(impl_->r_logits, experts_count, top_k, impl_->d_ids_map, impl_->d_scales_map,
                      impl_->stream);
    const unsigned seq = ++impl_->route_seq;
    cuda::doorbell_signal(impl_->d_flag_map, static_cast<int>(seq), impl_->stream);
    volatile const int* flag = impl_->h_flag_map;
    while (*flag != static_cast<int>(seq)) cudaStreamQuery(impl_->stream);
    const auto route_end = std::chrono::steady_clock::now();
    std::vector<int> ids(static_cast<size_t>(top_k));
    std::vector<float> weights(static_cast<size_t>(top_k));
    float shared_weight = 0.0f;
    std::memcpy(ids.data(), impl_->h_ids_map, ids.size() * sizeof(int));
    std::memcpy(weights.data(), impl_->h_scales_map, weights.size() * sizeof(float));
    std::memcpy(&shared_weight, impl_->h_scales_map + top_k, sizeof(float));
    impl_->ensure(impl_->mix_dev, impl_->mix_cap, static_cast<size_t>(n_in));
    moe_core(impl_->norm_dev, impl_->mix_dev, routed, ids, weights, shared, shared_weight);
    if (device_profile()) {
        const auto core_end = std::chrono::steady_clock::now();
        std::fprintf(stderr, "devmoe route_ms=%.3f core_ms=%.3f\n",
                     std::chrono::duration<double, std::milli>(route_end - route_start).count(),
                     std::chrono::duration<double, std::milli>(core_end - route_end).count());
    }
}

bool CudaProjection::supports_attention(const AttnWeights& w) const {
    const auto present = [](const strata::TensorInfo* t, const uint8_t* d) { return t && d; };
    if (!present(w.q, w.q_data) || !present(w.k, w.k_data) || !present(w.v, w.v_data) ||
        !present(w.out, w.out_data) || !present(w.q_norm, w.q_norm_data) ||
        !present(w.k_norm, w.k_norm_data))
        return false;
    return supports(w.q->type) && supports(w.k->type) && supports(w.v->type) &&
           w.q->type == w.k->type && w.q->type == w.v->type && supports(w.out->type) &&
           w.q_norm->type == 0 && w.k_norm->type == 0 &&
           w.q->shape.size() == 2 && w.q->shape[0] == 2048 && w.q->shape[1] == 8192 &&
           w.k->shape.size() == 2 && w.k->shape[0] == 2048 && w.k->shape[1] == 512 &&
           w.v->shape.size() == 2 && w.v->shape[0] == 2048 && w.v->shape[1] == 512 &&
           w.out->shape.size() == 2 && w.out->shape[0] == 4096 && w.out->shape[1] == 2048 &&
           w.q_norm->shape.size() == 1 && w.q_norm->shape[0] == 256 &&
           w.k_norm->shape.size() == 1 && w.k_norm->shape[0] == 256;
}

void CudaProjection::attention_into_mix(int layer, const AttnWeights& w, int position,
                                        float rope_base) {
    constexpr int kHidden = 2048;
    constexpr int kHeads = 16;
    constexpr int kKvHeads = 2;
    constexpr int kHeadDim = 256;
    constexpr int kRot = 64;
    constexpr int kQ = kHeads * 2 * kHeadDim;  // query + gate
    constexpr int kKv = kKvHeads * kHeadDim;
    constexpr float kEps = 1e-6f;
    if (!supports_attention(w)) throw std::invalid_argument("unsupported CUDA attention layer");
    if (layer < 0 || position < 0) throw std::invalid_argument("CUDA attention index");
    impl_->ensure(impl_->a_q, impl_->a_q_cap, kQ);
    impl_->ensure(impl_->a_k, impl_->a_k_cap, kKv);
    impl_->ensure(impl_->a_v, impl_->a_v_cap, kKv);
    impl_->ensure(impl_->a_ctx, impl_->a_ctx_cap, kHeads * kHeadDim);
    impl_->ensure(impl_->mix_dev, impl_->mix_cap, kHidden);

    void* q_weight = impl_->projection_weight(*w.q, w.q_data, -1);
    void* k_weight = impl_->projection_weight(*w.k, w.k_data, -1);
    void* v_weight = impl_->projection_weight(*w.v, w.v_data, -1);
    const void* proj_weights[3] = {q_weight, k_weight, v_weight};
    const float* proj_inputs[3] = {impl_->norm_dev, impl_->norm_dev, impl_->norm_dev};
    float* proj_outputs[3] = {impl_->a_q, impl_->a_k, impl_->a_v};
    const int proj_rows[3] = {kQ, kKv, kKv};
    strata::kernels::native_mmvq_f32_grouped(static_cast<int>(w.q->type), proj_weights, proj_inputs,
                                             proj_outputs, proj_rows, 3, kQ + 2 * kKv, kHidden,
                                             impl_->stream);
    void* q_norm = impl_->weight(*w.q_norm, w.q_norm_data, -1,
                                 static_cast<size_t>(kHeadDim) * sizeof(float));
    void* k_norm = impl_->weight(*w.k_norm, w.k_norm_data, -1,
                                 static_cast<size_t>(kHeadDim) * sizeof(float));
    cuda::attn_norm_rope(impl_->a_q, kHeads, 2 * kHeadDim, kHeadDim, kRot,
                         static_cast<const float*>(q_norm), kEps, rope_base, position, impl_->stream);
    cuda::attn_norm_rope(impl_->a_k, kKvHeads, kHeadDim, kHeadDim, kRot,
                         static_cast<const float*>(k_norm), kEps, rope_base, position, impl_->stream);
    if (static_cast<size_t>(layer) >= impl_->attn_states.size())
        impl_->attn_states.resize(static_cast<size_t>(layer) + 1);
    auto& state = impl_->attn_states[static_cast<size_t>(layer)];
    const int needed = position + 1;
    if (needed > impl_->max_context) throw std::out_of_range("CUDA attention context exceeded");
    if (state.capacity < needed) {
        int capacity = state.capacity ? state.capacity : std::min(2048, std::max(needed, 1));
        while (capacity < needed) capacity *= 2;
        if (capacity > impl_->max_context) capacity = impl_->max_context;
        if (capacity < needed) throw std::out_of_range("CUDA attention context exceeded");
        float* keys = nullptr;
        float* values = nullptr;
        check(cudaMalloc(reinterpret_cast<void**>(&keys),
                         static_cast<size_t>(capacity) * kKv * sizeof(float)), "allocate KV keys");
        check(cudaMalloc(reinterpret_cast<void**>(&values),
                         static_cast<size_t>(capacity) * kKv * sizeof(float)), "allocate KV values");
        if (state.keys) {
            const size_t bytes = static_cast<size_t>(state.capacity) * kKv * sizeof(float);
            check(cudaMemcpy(keys, state.keys, bytes, cudaMemcpyDeviceToDevice), "grow KV keys");
            check(cudaMemcpy(values, state.values, bytes, cudaMemcpyDeviceToDevice), "grow KV values");
            cudaFree(state.keys);
            cudaFree(state.values);
        }
        state.keys = keys;
        state.values = values;
        state.capacity = capacity;
    }
    check(cudaMemcpyAsync(state.keys + static_cast<size_t>(position) * kKv, impl_->a_k,
                          static_cast<size_t>(kKv) * sizeof(float), cudaMemcpyDeviceToDevice,
                          impl_->stream), "append keys");
    check(cudaMemcpyAsync(state.values + static_cast<size_t>(position) * kKv, impl_->a_v,
                          static_cast<size_t>(kKv) * sizeof(float), cudaMemcpyDeviceToDevice,
                          impl_->stream), "append values");
    cuda::attn_decode(impl_->a_q, state.keys, state.values, position, kHeads, kKvHeads, kHeadDim,
                      1.0f / 16.0f, impl_->a_ctx, impl_->stream);
    const void* out_weight = impl_->projection_weight(*w.out, w.out_data, -1);
    const float* out_input = impl_->a_ctx;
    float* out_output = impl_->mix_dev;
    const int out_rows = kHidden;
    strata::kernels::native_mmvq_f32_grouped(static_cast<int>(w.out->type), &out_weight, &out_input,
                                             &out_output, &out_rows, 1, kHidden, kHeads * kHeadDim,
                                             impl_->stream);
    check(cudaGetLastError(), "launch attention");
}

void* CudaProjection::cuda_stream() const { return impl_->stream; }

void CudaProjection::mark_begin() {
    check(cudaEventCreateWithFlags(&impl_->mark_start, cudaEventDefault), "create event");
    check(cudaEventCreateWithFlags(&impl_->mark_stop, cudaEventDefault), "create event");
    check(cudaEventRecord(impl_->mark_start, impl_->stream), "record start");
}

double CudaProjection::mark_end_ms() {
    check(cudaEventRecord(impl_->mark_stop, impl_->stream), "record stop");
    check(cudaEventSynchronize(impl_->mark_stop), "sync stop");
    float ms = 0.0f;
    check(cudaEventElapsedTime(&ms, impl_->mark_start, impl_->mark_stop), "elapsed");
    cudaEventDestroy(impl_->mark_start);
    cudaEventDestroy(impl_->mark_stop);
    impl_->mark_start = nullptr;
    impl_->mark_stop = nullptr;
    return static_cast<double>(ms);
}

CudaProjection::Stats CudaProjection::stats() const {
    Stats stats;
    stats.hits = impl_->stat_hits;
    stats.misses = impl_->stat_misses;
    stats.uploaded_bytes = impl_->stat_uploaded;
    stats.evicted_bytes = impl_->stat_evicted;
    stats.resident_bytes = impl_->cached_bytes;
    stats.cache_limit = impl_->cache_limit;
    return stats;
}

}  // namespace lamina::model
#else
namespace lamina::model {
struct CudaProjection::Impl {};
CudaProjection::CudaProjection() { throw std::runtime_error("Lamina was built without CUDA"); }
CudaProjection::~CudaProjection() = default;
bool CudaProjection::supports(uint32_t) const { return false; }
std::vector<float> CudaProjection::matvec(const strata::TensorInfo&, const uint8_t*,
                                          const std::vector<float>&, int64_t) {
    throw std::runtime_error("Lamina was built without CUDA");
}
std::vector<std::vector<float>> CudaProjection::matvec_many(
    const std::vector<const strata::TensorInfo*>&, const std::vector<const uint8_t*>&,
    const std::vector<int64_t>&, const std::vector<float>&) {
    throw std::runtime_error("Lamina was built without CUDA");
}
std::vector<float> CudaProjection::matvec_f32(const strata::TensorInfo&, const uint8_t*,
                                              const std::vector<float>&) {
    throw std::runtime_error("Lamina was built without CUDA");
}
std::vector<float> CudaProjection::delta_net(int, const std::vector<float>&, const GdnWeights&) {
    throw std::runtime_error("Lamina was built without CUDA");
}
bool CudaProjection::supports_gdn(const GdnWeights&) const { return false; }
bool CudaProjection::supports_moe(const MoeWeights&, const MoeWeights&) const { return false; }
std::vector<float> CudaProjection::moe(const std::vector<float>&, const MoeWeights&,
                                       const std::vector<int>&, const std::vector<float>&,
                                       const MoeWeights&, float) {
    throw std::runtime_error("Lamina was built without CUDA");
}
void CudaProjection::set_context(int) { throw std::runtime_error("Lamina was built without CUDA"); }
void CudaProjection::hidden_upload(const std::vector<float>&) {
    throw std::runtime_error("Lamina was built without CUDA");
}
std::vector<float> CudaProjection::hidden_download() {
    throw std::runtime_error("Lamina was built without CUDA");
}
void CudaProjection::mix_upload(const std::vector<float>&) {
    throw std::runtime_error("Lamina was built without CUDA");
}
void CudaProjection::hidden_rms(const strata::TensorInfo&, const uint8_t*, float) {
    throw std::runtime_error("Lamina was built without CUDA");
}
void CudaProjection::add_hidden_mix() { throw std::runtime_error("Lamina was built without CUDA"); }
void CudaProjection::delta_net_into_mix(int, const GdnWeights&) {
    throw std::runtime_error("Lamina was built without CUDA");
}
void CudaProjection::delta_layer_graph(int, const GdnWeights&, const strata::TensorInfo&,
                                       const uint8_t*, const strata::TensorInfo&, const uint8_t*,
                                       float) {
    throw std::runtime_error("Lamina was built without CUDA");
}
void CudaProjection::moe_into_mix(const strata::TensorInfo&, const uint8_t*,
                                  const strata::TensorInfo&, const uint8_t*, const MoeWeights&,
                                  const MoeWeights&) {
    throw std::runtime_error("Lamina was built without CUDA");
}
void CudaProjection::attention_into_mix(int, const AttnWeights&, int, float) {
    throw std::runtime_error("Lamina was built without CUDA");
}
bool CudaProjection::supports_attention(const AttnWeights&) const { return false; }
void CudaProjection::delta_net_core(int, const float*, float*, const GdnWeights&) {}
void CudaProjection::moe_core(const float*, float*, const MoeWeights&, const std::vector<int>&,
                              const std::vector<float>&, const MoeWeights&, float) {}
CudaProjection::Stats CudaProjection::stats() const { return Stats{}; }
void* CudaProjection::cuda_stream() const { return nullptr; }
void CudaProjection::mark_begin() {}
double CudaProjection::mark_end_ms() { return 0.0; }
}  // namespace lamina::model
#endif
