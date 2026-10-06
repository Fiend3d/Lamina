#include "lamina/model/cuda_projection.hpp"

#include <stdexcept>

#ifdef LAMINA_ENABLE_CUDA
#include "lamina/model/cuda_kernels.hpp"
#include "lamina/model/cpu_experts.hpp"
#include "strata/kernels/gdn.hpp"
#include "strata/kernels/native_gdn.hpp"
#include "strata/kernels/native_gdn_preprocess.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>
#ifdef LAMINA_PREFILL_BLAS
#include <cublas_v2.h>
#endif

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <list>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace lamina::model {
namespace {
void check(cudaError_t status, const char* action) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("CUDA ") + action + ": " + cudaGetErrorString(status));
}
constexpr size_t MIB = 1024ULL * 1024;
#ifdef LAMINA_PREFILL_BLAS
void check_blas(cublasStatus_t status, const char* action) {
    if (status != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string("cuBLAS ") + action + ": " + cublasGetStatusString(status));
}
#endif
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
        std::list<std::string>::iterator lru;
    };
    std::unordered_map<std::string, Entry> weights;
    std::list<std::string> lru;
    static constexpr int kUploadSlots = 8;
    static constexpr size_t kUploadBytes = 4 * 1024 * 1024;
    std::array<uint8_t*, kUploadSlots> upload_host{};
    std::array<cudaEvent_t, kUploadSlots> upload_done{};
    unsigned upload_next = 0;
    // Reuse evicted blocks instead of cudaMalloc/cudaFree per miss: on WDDM the
    // malloc serializes the device and dominated a token once the cache churned.
    std::unordered_map<size_t, std::vector<void*>> free_blocks;
    size_t cached_bytes = 0;
    size_t cache_limit = 0;
    size_t memory_limit = 0, allocated_bytes = 0;
    std::unordered_map<void*, size_t> allocations;
    bool host_kv = false, half_kv = false, lease_active = false;
    std::unordered_set<void*> leases;
    std::unique_ptr<CpuExperts> cpu_experts;
    bool cpu_misses = false;
    float* cpu_down_host = nullptr;
    uint64_t clock = 0;
    uint64_t stat_hits = 0;
    uint64_t stat_misses = 0;
    uint64_t stat_cpu_experts = 0;
    size_t stat_uploaded = 0;
    size_t stat_evicted = 0;
    cudaStream_t stream = nullptr;
    float* input = nullptr;
    float* output = nullptr;
    size_t input_capacity = 0, output_capacity = 0;
#ifdef LAMINA_PREFILL_BLAS
    cublasHandle_t blas = nullptr;
    void* blas_workspace = nullptr;
    float* blas_weights = nullptr;
    size_t blas_weights_cap = 0;
#endif
    float* batch_gdn = nullptr;
    size_t batch_gdn_cap = 0;
    float *b_g_qkv = nullptr, *b_g_z = nullptr, *b_g_alpha = nullptr, *b_g_beta = nullptr,
          *b_g_gate = nullptr, *b_g_conv = nullptr, *b_g_y = nullptr;
    size_t b_g_qkv_cap = 0, b_g_z_cap = 0, b_g_alpha_cap = 0, b_g_beta_cap = 0,
           b_g_gate_cap = 0, b_g_conv_cap = 0, b_g_y_cap = 0;
    float *batch_q = nullptr, *batch_k = nullptr, *batch_v = nullptr,
          *batch_partial = nullptr, *batch_accum = nullptr, *batch_ctx = nullptr;
    size_t batch_q_cap = 0, batch_k_cap = 0, batch_v_cap = 0,
           batch_partial_cap = 0, batch_accum_cap = 0, batch_ctx_cap = 0;
    float *bm_x = nullptr, *bm_router = nullptr, *bm_scales = nullptr, *bm_input = nullptr,
          *bm_gate = nullptr, *bm_up = nullptr, *bm_swiglu = nullptr, *bm_down = nullptr, *bm_slots = nullptr;
    int *bm_ids = nullptr, *bm_map = nullptr;
    size_t bm_x_cap = 0, bm_router_cap = 0, bm_scales_cap = 0, bm_input_cap = 0,
           bm_gate_cap = 0, bm_up_cap = 0, bm_swiglu_cap = 0, bm_down_cap = 0, bm_slots_cap = 0,
           bm_ids_cap = 0, bm_map_cap = 0;
    // Device-resident scratch for the MoE block. Slots are one expert wide so
    // every expert plus the shared expert can be enqueued without host waits.
    float* accum_dev = nullptr;
    float* gate_dev = nullptr;
    float* up_dev = nullptr;
    float* swiglu_dev = nullptr;
    float* down_dev = nullptr;
    float* scales_dev = nullptr;
    float* scales_host = nullptr;
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
        std::vector<float> host_keys, host_values;
        std::vector<uint16_t> half_keys, half_values;
    };
    std::vector<AttnState> attn_states;
    float* a_q = nullptr;
    float* a_k = nullptr;
    float* a_v = nullptr;
    float* a_ctx = nullptr;
    size_t a_q_cap = 0, a_k_cap = 0, a_v_cap = 0, a_ctx_cap = 0;
    float* a_partial = nullptr;
    size_t a_partial_cap = 0;
    static constexpr int kKvTile = 2048;
    struct KvSlot {
        uint8_t* host = nullptr;
        void *keys = nullptr, *values = nullptr;
        cudaEvent_t copied = nullptr, consumed = nullptr;
        bool used = false;
    };
    std::array<KvSlot, 2> kv_slots{};
    cudaStream_t kv_transfer = nullptr;
    size_t kv_element_bytes() const { return half_kv ? sizeof(uint16_t) : sizeof(float); }
    size_t kv_bytes(int tokens) const { return size_t(tokens) * 512 * kv_element_bytes(); }
    void* kv_offset(void* pointer, int tokens) const {
        return static_cast<uint8_t*>(pointer) + kv_bytes(tokens);
    }
    void* host_keys(AttnState& state) { return half_kv ? static_cast<void*>(state.half_keys.data()) : state.host_keys.data(); }
    void* host_values(AttnState& state) { return half_kv ? static_cast<void*>(state.half_values.data()) : state.host_values.data(); }
    void resize_host(AttnState& state, int tokens) {
        if (half_kv) { state.half_keys.resize(size_t(tokens) * 512); state.half_values.resize(size_t(tokens) * 512); }
        else { state.host_keys.resize(size_t(tokens) * 512); state.host_values.resize(size_t(tokens) * 512); }
    }
    void ensure_kv_slots() {
        if (!kv_transfer) check(cudaStreamCreateWithFlags(&kv_transfer, cudaStreamNonBlocking), "create KV transfer stream");
        for (auto& slot : kv_slots) {
            if (!slot.host) check(cudaMallocHost(reinterpret_cast<void**>(&slot.host), 2 * kv_bytes(kKvTile)), "allocate pinned double KV buffer");
            if (!slot.keys) allocate(&slot.keys, kv_bytes(kKvTile));
            if (!slot.values) allocate(&slot.values, kv_bytes(kKvTile));
            if (!slot.copied) check(cudaEventCreateWithFlags(&slot.copied, cudaEventDisableTiming), "create KV copy event");
            if (!slot.consumed) check(cudaEventCreateWithFlags(&slot.consumed, cudaEventDisableTiming), "create KV compute event");
        }
    }
    KvSlot& stage_kv(AttnState& state, int start, int count) {
        auto& slot = kv_slots[(start / kKvTile) % 2];
        if (slot.used) {
            // The CPU may overwrite pinned memory after DMA completes; the
            // transfer stream may overwrite device memory only after attention.
            check(cudaEventSynchronize(slot.copied), "reuse pinned KV slot");
            check(cudaStreamWaitEvent(kv_transfer, slot.consumed, 0), "wait for KV consumer");
        }
        std::memcpy(slot.host, kv_offset(host_keys(state), start), kv_bytes(count));
        std::memcpy(slot.host + kv_bytes(kKvTile), kv_offset(host_values(state), start), kv_bytes(count));
        check(cudaMemcpyAsync(slot.keys, slot.host, kv_bytes(count), cudaMemcpyHostToDevice, kv_transfer), "stage KV keys");
        check(cudaMemcpyAsync(slot.values, slot.host + kv_bytes(kKvTile), kv_bytes(count), cudaMemcpyHostToDevice, kv_transfer), "stage KV values");
        check(cudaEventRecord(slot.copied, kv_transfer), "record KV copy");
        check(cudaStreamWaitEvent(stream, slot.copied, 0), "wait for KV copy");
        slot.used = true;
        return slot;
    }
    void append_host(AttnState& state, const float* keys, const float* values, int position, int columns) {
        resize_host(state, position + columns);
        ensure_kv_slots();
        // Append in bounded pieces. It completes before any history staging.
        check(cudaStreamSynchronize(kv_transfer), "finish previous KV transfers");
        check(cudaStreamSynchronize(stream), "finish previous KV consumers");
        for (int c = 0; c < columns; c += kKvTile) {
            const int n = std::min(kKvTile, columns - c);
            auto& slot = kv_slots[0];
            cuda::kv_store(keys + size_t(c) * 512, slot.keys, n * 512, half_kv, stream);
            cuda::kv_store(values + size_t(c) * 512, slot.values, n * 512, half_kv, stream);
            check(cudaMemcpyAsync(kv_offset(host_keys(state), position + c), slot.keys, kv_bytes(n), cudaMemcpyDeviceToHost, stream), "append host keys");
            check(cudaMemcpyAsync(kv_offset(host_values(state), position + c), slot.values, kv_bytes(n), cudaMemcpyDeviceToHost, stream), "append host values");
            check(cudaStreamSynchronize(stream), "finish KV append");
        }
    }
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
    struct MoeGraph {
        cudaGraphExec_t exec = nullptr;
        uint64_t used = 0;
    };
    std::unordered_map<std::string, MoeGraph> moe_graphs;
    uint64_t moe_graph_hits = 0, moe_graph_misses = 0;
    strata::kernels::NativeF32Grouped *expert_table_host = nullptr, *expert_table_gpu = nullptr;
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
        if (stream) cudaStreamSynchronize(stream);
#ifdef LAMINA_PREFILL_BLAS
        if (blas) cublasDestroy(blas);
        cudaFree(blas_workspace); cudaFree(blas_weights);
#endif
        for (int i = 0; i < kUploadSlots; ++i) {
            if (upload_done[i]) cudaEventDestroy(upload_done[i]);
            if (upload_host[i]) cudaFreeHost(upload_host[i]);
        }
        for (auto& [name, entry] : weights) cudaFree(entry.device);
        for (auto& [bytes, blocks] : free_blocks)
            for (void* block : blocks) cudaFree(block);
        for (auto& state : gdn_states) {
            cudaFree(state.conv);
            cudaFree(state.rec);
        }
        for (auto& graph : mixer_graphs)
            if (graph.exec) cudaGraphExecDestroy(graph.exec);
        for (auto& [key, graph] : moe_graphs)
            if (graph.exec) cudaGraphExecDestroy(graph.exec);
        if (expert_table_host) cudaFreeHost(expert_table_host);
        cudaFree(expert_table_gpu);
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
        cudaFree(a_partial);
        if (kv_transfer) cudaStreamSynchronize(kv_transfer);
        for (auto& slot : kv_slots) {
            cudaFree(slot.keys); cudaFree(slot.values);
            if (slot.host) cudaFreeHost(slot.host);
            if (slot.copied) cudaEventDestroy(slot.copied);
            if (slot.consumed) cudaEventDestroy(slot.consumed);
        }
        if (kv_transfer) cudaStreamDestroy(kv_transfer);
        if (cpu_down_host) cudaFreeHost(cpu_down_host);
        cudaFree(r_logits);
        cudaFree(r_ids);
        if (h_ids_map) cudaFreeHost(h_ids_map);
        if (h_scales_map) cudaFreeHost(h_scales_map);
        if (h_flag_map) cudaFreeHost(h_flag_map);
        cudaFree(input);
        cudaFree(output);
        cudaFree(batch_gdn);
        cudaFree(b_g_qkv); cudaFree(b_g_z); cudaFree(b_g_alpha); cudaFree(b_g_beta);
        cudaFree(b_g_gate); cudaFree(b_g_conv); cudaFree(b_g_y);
        cudaFree(batch_q); cudaFree(batch_k); cudaFree(batch_v);
        cudaFree(batch_partial); cudaFree(batch_accum); cudaFree(batch_ctx);
        cudaFree(bm_x); cudaFree(bm_router); cudaFree(bm_scales); cudaFree(bm_input);
        cudaFree(bm_gate); cudaFree(bm_up); cudaFree(bm_swiglu); cudaFree(bm_down); cudaFree(bm_slots);
        cudaFree(bm_ids); cudaFree(bm_map);
        cudaFree(accum_dev);
        cudaFree(gate_dev);
        cudaFree(up_dev);
        cudaFree(swiglu_dev);
        cudaFree(down_dev);
        cudaFree(scales_dev);
        if (scales_host) cudaFreeHost(scales_host);
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
        if (mark_start) cudaEventDestroy(mark_start);
        if (mark_stop) cudaEventDestroy(mark_stop);
    }

    void release(void* pointer) {
        if (!pointer) return;
        check(cudaFree(pointer), "release device allocation");
        if (auto found = allocations.find(pointer); found != allocations.end()) {
            allocated_bytes -= found->second;
            allocations.erase(found);
        }
    }

    void trim_free() {
        for (auto& [bytes, blocks] : free_blocks) {
            for (void* block : blocks) release(block);
            blocks.clear();
        }
    }

    void allocate(void** pointer, size_t bytes) {
        if (bytes > memory_limit) throw std::runtime_error("allocation exceeds Lamina VRAM budget");
        if (allocated_bytes + bytes > memory_limit) trim_free();
        while (allocated_bytes + bytes > memory_limit) {
            if (!evict_oldest()) throw std::runtime_error("Lamina VRAM budget cannot hold pinned weights and state; use --kv-cache host or a larger budget");
            trim_free();
        }
        size_t available = 0, total = 0;
        check(cudaMemGetInfo(&available, &total), "inspect allocation headroom");
        while (bytes + 512 * MIB > available) {
            trim_free();
            check(cudaMemGetInfo(&available, &total), "inspect allocation headroom");
            if (bytes + 512 * MIB <= available) break;
            if (!evict_oldest()) throw std::runtime_error("not enough free VRAM; stop competing GPU workloads");
        }
        check(cudaMalloc(pointer, bytes), "allocate bounded device memory");
        allocations[*pointer] = bytes;
        allocated_bytes += bytes;
    }

    void ensure(float*& pointer, size_t& capacity, size_t elements) {
        if (elements <= capacity) return;
        release(pointer);
        pointer = nullptr;
        allocate(reinterpret_cast<void**>(&pointer), elements * sizeof(float));
        capacity = elements;
    }

    void ensure(int*& pointer, size_t& capacity, size_t elements) {
        if (elements <= capacity) return;
        release(pointer);
        pointer = nullptr;
        allocate(reinterpret_cast<void**>(&pointer), elements * sizeof(int));
        capacity = elements;
    }

    void reserve(size_t n_in, size_t n_out) {
        ensure(input, input_capacity, n_in);
        ensure(output, output_capacity, n_out);
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

    bool resident(const strata::TensorInfo& tensor, int expert) const {
        return weights.contains(tensor.name + "#type=" + std::to_string(tensor.type) + "#expert=" + std::to_string(expert));
    }

    void* device_alloc(size_t bytes) {
        auto found = free_blocks.find(bytes);
        if (found != free_blocks.end() && !found->second.empty()) {
            void* block = found->second.back();
            found->second.pop_back();
            return block;
        }
        void* block = nullptr;
        allocate(&block, bytes);
        return block;
    }

    void device_release(void* block, size_t bytes) { free_blocks[bytes].push_back(block); }

    bool evict_oldest() {
        auto candidate = lru.begin();
        while (candidate != lru.end() && leases.contains(weights.at(*candidate).device)) ++candidate;
        if (candidate == lru.end()) return false;
        auto oldest = weights.find(*candidate);
        // Reused blocks are uploaded on this same stream, after all earlier
        // readers. cudaFree remains synchronized when a block is actually trimmed.
        device_release(oldest->second.device, oldest->second.bytes);
        cached_bytes -= oldest->second.bytes;
        stat_evicted += oldest->second.bytes;
        weights.erase(oldest);
        lru.erase(candidate);
        return true;
    }

    void upload(void* device, const uint8_t* source, size_t bytes) {
        const char* staging = std::getenv("LAMINA_PINNED_UPLOADS");
        if (staging && staging[0] == '0') {
            check(cudaMemcpyAsync(device, source, bytes, cudaMemcpyHostToDevice, stream), "upload pageable weight");
            return;
        }
        for (size_t offset = 0; offset < bytes; offset += kUploadBytes) {
            const unsigned slot = upload_next++ % kUploadSlots;
            if (!upload_host[slot]) {
                check(cudaMallocHost(reinterpret_cast<void**>(&upload_host[slot]), kUploadBytes), "allocate weight upload staging");
                check(cudaEventCreateWithFlags(&upload_done[slot], cudaEventDisableTiming), "create upload event");
            } else check(cudaEventSynchronize(upload_done[slot]), "reuse upload staging");
            const size_t count = std::min(kUploadBytes, bytes - offset);
            std::memcpy(upload_host[slot], source + offset, count);
            check(cudaMemcpyAsync(static_cast<uint8_t*>(device) + offset, upload_host[slot], count,
                                  cudaMemcpyHostToDevice, stream), "upload staged weight");
            check(cudaEventRecord(upload_done[slot], stream), "record upload completion");
        }
    }

    void* weight(const strata::TensorInfo& tensor, const uint8_t* data,
                 int64_t expert, size_t bytes) {
        const std::string key = tensor.name + "#type=" + std::to_string(tensor.type) +
                                "#expert=" + std::to_string(expert);
        if (auto found = weights.find(key); found != weights.end()) {
            found->second.used = ++clock;
            if (!found->second.pinned) lru.splice(lru.end(), lru, found->second.lru);
            ++stat_hits;
            if (lease_active) leases.insert(found->second.device);
            return found->second.device;
        }
        ++stat_misses;
        stat_uploaded += bytes;
        while (bytes > cache_limit - std::min(cached_bytes, cache_limit) && evict_oldest()) {}
        if (bytes > cache_limit - std::min(cached_bytes, cache_limit))
            throw std::runtime_error("CUDA weight-cache limit is below the pinned/active working set");
        void* device = device_alloc(bytes);
        try {
            // Upload on the kernel stream: a synchronous pageable copy on the
            // legacy stream is not ordered against a non-blocking stream and
            // can race the projections already queued there.
            upload(device, data, bytes);
            auto entry_lru = lru.end();
            if (expert >= 0) entry_lru = lru.insert(lru.end(), key);
            weights.emplace(key, Entry{device, bytes, ++clock, expert < 0, entry_lru});
            cached_bytes += bytes;
            if (lease_active) leases.insert(device);
            return device;
        } catch (...) {
            device_release(device, bytes);
            throw;
        }
    }
};

CudaProjection::CudaProjection(size_t vram_limit_mb, bool host_kv, bool half_kv) : impl_(std::make_unique<Impl>()) {
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
    if (free_bytes <= 512 * MIB) throw std::runtime_error("CUDA needs more than 512 MiB free VRAM");
    impl_->memory_limit = free_bytes - 512 * MIB;
    if (vram_limit_mb) impl_->memory_limit = std::min(impl_->memory_limit, vram_limit_mb * MIB);
    impl_->cache_limit = std::min(impl_->memory_limit, free_bytes * 3 / 4);
    impl_->host_kv = host_kv;
    impl_->half_kv = half_kv;
    if (const char* policy = std::getenv("LAMINA_EXPERT_POLICY")) {
        const std::string value(policy);
        if (value != "stream" && value != "cpu-miss") throw std::invalid_argument("LAMINA_EXPERT_POLICY must be stream or cpu-miss");
        impl_->cpu_misses = value == "cpu-miss";
    }
    if (impl_->cpu_misses) {
        unsigned workers = std::min(8u, std::max(1u, std::thread::hardware_concurrency()));
        if (const char* configured = std::getenv("LAMINA_CPU_THREADS")) {
            const auto parsed = std::from_chars(configured, configured + std::strlen(configured), workers);
            if (parsed.ec != std::errc{} || parsed.ptr != configured + std::strlen(configured) || workers < 1 || workers > 64)
                throw std::invalid_argument("LAMINA_CPU_THREADS must be 1..64");
        }
        impl_->cpu_experts = std::make_unique<CpuExperts>(workers);
    }
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
            throw std::invalid_argument("LAMINA_CUDA_CACHE_MB exceeds the available VRAM budget");
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
    check(cudaMemcpyAsync(impl_->input, x.data(), x.size() * sizeof(float),
                          cudaMemcpyHostToDevice, impl_->stream), "upload activation");
    strata::kernels::native_mmvq_f32(tensor.type, device_weight, impl_->input, impl_->output,
                                     n_in, n_out, impl_->stream);
    check(cudaGetLastError(), "launch projection");
    std::vector<float> result(static_cast<size_t>(n_out));
    check(cudaMemcpyAsync(result.data(), impl_->output, result.size() * sizeof(float),
                          cudaMemcpyDeviceToHost, impl_->stream), "download projection");
    check(cudaStreamSynchronize(impl_->stream), "finish projection");
    return result;
}

std::vector<float> CudaProjection::normalize_columns(const strata::TensorInfo& gamma, const uint8_t* data,
                                                     const std::vector<float>& x, int columns, float epsilon) {
    if (gamma.type != 0 || gamma.shape.size() != 1 || columns < 1 || columns > 2048 || x.size() != size_t(columns) * gamma.shape[0])
        throw std::invalid_argument("invalid prefill normalization");
    impl_->ensure(impl_->input, impl_->input_capacity, x.size());
    auto* weight = static_cast<const float*>(impl_->weight(gamma, data, -1, size_t(gamma.shape[0]) * sizeof(float)));
    check(cudaMemcpyAsync(impl_->input, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice, impl_->stream), "upload normalization columns");
    cuda::rms_norm_columns(impl_->input, weight, int(gamma.shape[0]), columns, epsilon, impl_->stream);
    check(cudaGetLastError(), "normalize prefill columns");
    std::vector<float> result(x.size());
    check(cudaMemcpyAsync(result.data(), impl_->input, result.size() * sizeof(float), cudaMemcpyDeviceToHost, impl_->stream), "download normalization columns");
    check(cudaStreamSynchronize(impl_->stream), "finish normalization columns");
    return result;
}

std::vector<float> CudaProjection::matvec_columns(const strata::TensorInfo& t, const uint8_t* data,
                                                  const std::vector<float>& x, int columns, int64_t expert) {
    if (columns < 1 || columns > 2048 || !data || t.shape.size() != (expert < 0 ? 2u : 3u) ||
        x.size() != t.shape[0] * size_t(columns) || (!supports(t.type) && t.type != 0))
        throw std::invalid_argument("invalid CUDA column projection: " + t.name);
    const int out = int(t.shape[1]);
    impl_->reserve(x.size(), size_t(columns) * out);
    check(cudaMemcpyAsync(impl_->input, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice,
                          impl_->stream), "upload prefill columns");
    project_columns_device(t, data, impl_->input, impl_->output, columns, expert);
    std::vector<float> result(size_t(columns) * out);
    check(cudaMemcpyAsync(result.data(), impl_->output, result.size() * sizeof(float),
                          cudaMemcpyDeviceToHost, impl_->stream), "download prefill columns");
    check(cudaStreamSynchronize(impl_->stream), "finish prefill columns");
    return result;
}

void CudaProjection::project_columns_device(const strata::TensorInfo& t, const uint8_t* data,
                                             const float* x_dev, float* y_dev, int columns, int64_t expert) {
    const int in = int(t.shape[0]), out = int(t.shape[1]);
#ifdef LAMINA_PREFILL_BLAS
    const char* configured = std::getenv("LAMINA_PREFILL_BLAS");
    // Small expert groups favor the native reduction; GEMM amortizes its
    // dequantization and launch cost once there are at least sixteen columns.
    const bool use_blas = columns >= 16 && (!configured || configured[0] != '0');
    if (use_blas) {
        if (!impl_->blas) {
            check_blas(cublasCreate(&impl_->blas), "create prefill handle");
            check_blas(cublasSetStream(impl_->blas, impl_->stream), "set prefill stream");
            check_blas(cublasSetMathMode(impl_->blas, CUBLAS_PEDANTIC_MATH), "select full FP32 math");
            check_blas(cublasSetAtomicsMode(impl_->blas, CUBLAS_ATOMICS_NOT_ALLOWED), "select deterministic accumulation");
            impl_->allocate(&impl_->blas_workspace, 8 * MIB);
            check_blas(cublasSetWorkspace(impl_->blas, impl_->blas_workspace, 8 * MIB), "set bounded prefill workspace");
        }
        if (t.type != 0) impl_->ensure(impl_->blas_weights, impl_->blas_weights_cap, size_t(in) * out);
    }
#endif
    void* w = t.type == 0 ? impl_->weight(t, data, -1, size_t(in) * out * sizeof(float))
                         : impl_->projection_weight(t, data, expert);
    bool projected = false;
#ifdef LAMINA_PREFILL_BLAS
    if (use_blas) {
        const float* matrix = static_cast<const float*>(w);
        if (t.type != 0) {
            strata::kernels::native_mmvq_dequant_f32(t.type, w, impl_->blas_weights, in, out, impl_->stream);
            check(cudaGetLastError(), "dequantize prefill matrix");
            matrix = impl_->blas_weights;
        }
        const float one = 1.0f, zero = 0.0f;
        // GGUF rows are columns of the transposed BLAS view. Explicit pedantic
        // compute prevents TF32/FP16 activation rounding and implicit conversion.
        check_blas(cublasGemmEx(impl_->blas, CUBLAS_OP_T, CUBLAS_OP_N, out, columns, in,
            &one, matrix, CUDA_R_32F, in, x_dev, CUDA_R_32F, in, &zero,
            y_dev, CUDA_R_32F, out, CUBLAS_COMPUTE_32F_PEDANTIC, CUBLAS_GEMM_DEFAULT), "project prefill columns");
        projected = true;
    }
#endif
    if (!projected && t.type == 0) {
        cuda::gemv_f32_columns(static_cast<const float*>(w), x_dev, y_dev, in, out, columns, impl_->stream);
    } else if (!projected) strata::kernels::native_mmvq_f32_columns(t.type, w, x_dev, y_dev,
                                                  in, out, columns, impl_->stream);
    check(cudaGetLastError(), "launch prefill columns");
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
    impl_->lease_active = true;
    impl_->leases.clear();
    size_t output_offset = 0;
    for (size_t i = 0; i < count; ++i) {
        weights[i] = impl_->projection_weight(*tensors[i], data[i], experts[i]);
        output_ptrs[i] = impl_->output + output_offset;
        output_offset += static_cast<size_t>(row_counts[i]);
    }
    check(cudaMemcpyAsync(impl_->input, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice, impl_->stream),
          "upload batched activation");
    strata::kernels::native_mmvq_f32_many(static_cast<int>(count), types.data(), weights.data(),
                                          output_ptrs.data(), row_counts.data(), impl_->input,
                                          n_in, impl_->stream);
    check(cudaGetLastError(), "launch batched projections");
    impl_->lease_active = false;
    impl_->leases.clear();
    check(cudaStreamSynchronize(impl_->stream), "finish batched projections");
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
        impl_->allocate(reinterpret_cast<void**>(&state.conv), static_cast<size_t>(kChannels) * 3 * sizeof(float));
        check(cudaMemset(state.conv, 0, static_cast<size_t>(kChannels) * 3 * sizeof(float)),
              "clear conv state");
    }
    if (!state.rec) {
        impl_->allocate(reinterpret_cast<void**>(&state.rec), static_cast<size_t>(kS) * kHeads * kS * sizeof(float));
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

std::vector<float> CudaProjection::delta_net_columns(int layer, const std::vector<float>& x,
                                                     int columns, const GdnWeights& w) {
    if (!supports_gdn(w)) throw std::invalid_argument("unsupported DeltaNet prefill");
    auto qkv = matvec_columns(*w.qkv, w.qkv_data, x, columns);
    auto z = matvec_columns(*w.gate, w.gate_data, x, columns);
    auto alpha = matvec_columns(*w.alpha, w.alpha_data, x, columns);
    auto beta = matvec_columns(*w.beta, w.beta_data, x, columns);
    if (layer < 0) throw std::invalid_argument("invalid DeltaNet prefill layer");
    if (size_t(layer) >= impl_->gdn_states.size()) impl_->gdn_states.resize(size_t(layer) + 1);
    auto& state = impl_->gdn_states[layer];
    if (!state.conv) {
        impl_->allocate(reinterpret_cast<void**>(&state.conv), 8192 * 3 * sizeof(float));
        check(cudaMemsetAsync(state.conv, 0, 8192 * 3 * sizeof(float), impl_->stream), "initialize prefill convolution");
    }
    if (!state.rec) {
        impl_->allocate(reinterpret_cast<void**>(&state.rec), 128 * 32 * 128 * sizeof(float));
        check(cudaMemsetAsync(state.rec, 0, 128 * 32 * 128 * sizeof(float), impl_->stream), "initialize prefill recurrence");
    }
    std::vector<float> result(size_t(columns) * 4096);
    impl_->ensure(impl_->b_g_qkv, impl_->b_g_qkv_cap, qkv.size());
    impl_->ensure(impl_->b_g_z, impl_->b_g_z_cap, z.size());
    impl_->ensure(impl_->b_g_alpha, impl_->b_g_alpha_cap, alpha.size());
    impl_->ensure(impl_->b_g_beta, impl_->b_g_beta_cap, beta.size());
    impl_->ensure(impl_->b_g_gate, impl_->b_g_gate_cap, alpha.size());
    impl_->ensure(impl_->b_g_conv, impl_->b_g_conv_cap, qkv.size());
    impl_->ensure(impl_->batch_gdn, impl_->batch_gdn_cap, result.size());
    impl_->ensure(impl_->b_g_y, impl_->b_g_y_cap, result.size());
    check(cudaMemcpyAsync(impl_->b_g_qkv, qkv.data(), qkv.size() * sizeof(float), cudaMemcpyHostToDevice, impl_->stream), "upload prefill qkv");
    check(cudaMemcpyAsync(impl_->b_g_z, z.data(), z.size() * sizeof(float), cudaMemcpyHostToDevice, impl_->stream), "upload prefill gates");
    check(cudaMemcpyAsync(impl_->b_g_alpha, alpha.data(), alpha.size() * sizeof(float), cudaMemcpyHostToDevice, impl_->stream), "upload prefill alpha");
    check(cudaMemcpyAsync(impl_->b_g_beta, beta.data(), beta.size() * sizeof(float), cudaMemcpyHostToDevice, impl_->stream), "upload prefill beta");
    const auto dense = [&](const strata::TensorInfo* t, const uint8_t* data, size_t count) {
        return static_cast<const float*>(impl_->weight(*t, data, -1, count * sizeof(float)));
    };
    const float* conv = dense(w.conv, w.conv_data, 8192 * 4);
    const float* dt = dense(w.dt, w.dt_data, 32);
    const float* a = dense(w.a, w.a_data, 32);
    const float* gamma = dense(w.norm, w.norm_data, 128);
    strata::kernels::native_gdn_preprocess_columns(state.conv, impl_->b_g_qkv, conv, impl_->b_g_conv,
        impl_->b_g_alpha, impl_->b_g_beta, dt, a, impl_->b_g_gate, columns, impl_->stream);
    strata::kernels::native_gdn_step_columns(state.rec, impl_->b_g_conv, impl_->b_g_gate,
        impl_->b_g_beta, impl_->batch_gdn, columns, impl_->stream);
    cuda::gdn_out_norm_silu(impl_->batch_gdn, impl_->b_g_z, gamma, impl_->b_g_y, columns * 32, 1e-6f, impl_->stream);
    check(cudaGetLastError(), "launch prefill recurrence");
    check(cudaMemcpyAsync(result.data(), impl_->b_g_y, result.size() * sizeof(float),
                          cudaMemcpyDeviceToHost, impl_->stream), "download recurrence columns");
    check(cudaStreamSynchronize(impl_->stream), "finish recurrence columns");
    return matvec_columns(*w.out, w.out_data, result, columns);
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
    if (experts.empty() || experts.size() > 32)
        throw std::invalid_argument("CUDA MoE supports 1..32 routed experts");
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
    if (!impl_->scales_host)
        check(cudaMallocHost(reinterpret_cast<void**>(&impl_->scales_host), 65 * sizeof(float)), "allocate expert scales staging");
    std::copy(scales.begin(), scales.end(), impl_->scales_host);
    // The device router doorbell (or the blocking public moe call) completed
    // the preceding expert block before this persistent staging is reused.
    check(cudaMemcpyAsync(impl_->scales_dev, impl_->scales_host, slots * sizeof(float),
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
    std::vector<std::pair<size_t, std::future<std::vector<float>>>> cpu_jobs;
    std::shared_ptr<std::vector<float>> cpu_input;
    impl_->lease_active = true;
    for (size_t s = 0; s < slots; ++s) {
        const MoeWeights& w = moe_weights(s);
        const int64_t expert = expert_id(s);
        const bool is_routed = s < experts.size();
        if (is_routed && impl_->cpu_misses && impl_->stat_evicted &&
            !(impl_->resident(*w.gate, int(expert)) && impl_->resident(*w.up, int(expert)) && impl_->resident(*w.down, int(expert)))) {
            if (!cpu_input) {
                cpu_input = std::make_shared<std::vector<float>>(n_in);
                check(cudaMemcpyAsync(cpu_input->data(), x_dev, size_t(n_in) * sizeof(float), cudaMemcpyDeviceToHost, impl_->stream), "publish CPU expert input");
                check(cudaStreamSynchronize(impl_->stream), "finish CPU input");
                if (!impl_->cpu_down_host)
                    check(cudaMallocHost(reinterpret_cast<void**>(&impl_->cpu_down_host), 9 * size_t(hidden) * sizeof(float)), "allocate CPU expert staging");
            }
            cpu_jobs.emplace_back(s, impl_->cpu_experts->submit(w, int(expert), cpu_input));
            ++impl_->stat_cpu_experts;
            check(cudaMemsetAsync(impl_->gate_dev + s * ff, 0, size_t(ff) * sizeof(float), impl_->stream), "clear CPU gate slot");
            check(cudaMemsetAsync(impl_->up_dev + s * ff, 0, size_t(ff) * sizeof(float), impl_->stream), "clear CPU up slot");
            continue;
        }
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
    const char* graphs = std::getenv("LAMINA_MOE_GRAPHS");
    const bool use_tables = cpu_jobs.empty() && (!graphs || graphs[0] != '0');
    if (use_tables) {
        const size_t bytes = 4 * sizeof(strata::kernels::NativeF32Grouped);
        if (!impl_->expert_table_host) {
            check(cudaMallocHost(reinterpret_cast<void**>(&impl_->expert_table_host), bytes), "allocate expert pointer staging");
            impl_->allocate(reinterpret_cast<void**>(&impl_->expert_table_gpu), bytes);
        }
        int index = 0;
        for (const auto* group : {&gate_up_routed, &gate_up_shared, &down_routed, &down_shared}) {
            auto& table = impl_->expert_table_host[index++];
            table = {};
            table.count = int(group->size());
            for (size_t i = 0; i < group->size(); ++i) {
                table.weights[i] = (*group)[i].weight; table.inputs[i] = (*group)[i].input;
                table.outputs[i] = (*group)[i].output; table.n_outs[i] = (*group)[i].rows;
            }
        }
        check(cudaMemcpyAsync(impl_->expert_table_gpu, impl_->expert_table_host, bytes,
                              cudaMemcpyHostToDevice, impl_->stream), "publish expert pointer tables");
    }
    const auto launch_table = [&](int index, int type, int width, const std::vector<GroupItem>& items) {
        int rows = 0;
        for (const auto& item : items) rows += item.rows;
        strata::kernels::native_mmvq_f32_grouped_table(type, impl_->expert_table_gpu + index,
                                                      int(items.size()), rows, width, impl_->stream);
    };
    const int down_n_in = static_cast<int>(routed.down->shape[0]);
    const auto gpu_experts = [&] {
        if (use_tables) {
            launch_table(0, routed.gate->type, n_in, gate_up_routed);
            launch_table(1, shared.gate->type, n_in, gate_up_shared);
        } else {
            launch_group(static_cast<int>(routed.gate->type), n_in, gate_up_routed);
            launch_group(static_cast<int>(shared.gate->type), n_in, gate_up_shared);
        }
        cuda::swiglu(impl_->gate_dev, impl_->up_dev, impl_->swiglu_dev,
                     static_cast<int>(slots) * ff, impl_->stream);
        if (use_tables) {
            launch_table(2, routed.down->type, down_n_in, down_routed);
            launch_table(3, shared.down->type, down_n_in, down_shared);
        } else {
            launch_group(static_cast<int>(routed.down->type), down_n_in, down_routed);
            launch_group(static_cast<int>(shared.down->type), down_n_in, down_shared);
        }
        cuda::moe_combine(impl_->down_dev, impl_->scales_dev, static_cast<int>(slots), hidden, out_dev, impl_->stream);
    };
    if (use_tables) {
        // Graphs reference stable device pointer tables, refreshed on the stream
        // before each replay. Cache identity covers tensor layouts and scratch;
        // selected expert weight addresses are intentionally dynamic.
        std::vector<uintptr_t> addresses{reinterpret_cast<uintptr_t>(routed.gate),
            reinterpret_cast<uintptr_t>(shared.gate), reinterpret_cast<uintptr_t>(x_dev),
            reinterpret_cast<uintptr_t>(out_dev), reinterpret_cast<uintptr_t>(impl_->gate_dev),
            reinterpret_cast<uintptr_t>(impl_->up_dev), reinterpret_cast<uintptr_t>(impl_->swiglu_dev),
            reinterpret_cast<uintptr_t>(impl_->down_dev), reinterpret_cast<uintptr_t>(impl_->scales_dev),
            reinterpret_cast<uintptr_t>(impl_->expert_table_gpu), slots};
        const std::string key(reinterpret_cast<const char*>(addresses.data()), addresses.size() * sizeof(uintptr_t));
        auto found = impl_->moe_graphs.find(key);
        if (found == impl_->moe_graphs.end()) {
            ++impl_->moe_graph_misses;
            if (impl_->moe_graphs.size() >= 128) {
                auto oldest = impl_->moe_graphs.begin();
                for (auto it = impl_->moe_graphs.begin(); it != impl_->moe_graphs.end(); ++it)
                    if (it->second.used < oldest->second.used) oldest = it;
                if (oldest->second.exec) check(cudaGraphExecDestroy(oldest->second.exec), "release expert graph");
                impl_->moe_graphs.erase(oldest);
            }
            impl_->moe_graphs.emplace(key, Impl::MoeGraph{nullptr, ++impl_->clock});
            gpu_experts();
        } else {
            auto& graph = found->second;
            graph.used = ++impl_->clock;
            if (!graph.exec) {
                check(cudaStreamSynchronize(impl_->stream), "flush before expert capture");
                check(cudaStreamBeginCapture(impl_->stream, cudaStreamCaptureModeThreadLocal), "begin expert capture");
                gpu_experts();
                cudaGraph_t captured = nullptr;
                check(cudaStreamEndCapture(impl_->stream, &captured), "end expert capture");
                const auto status = cudaGraphInstantiate(&graph.exec, captured, nullptr, nullptr, 0);
                cudaGraphDestroy(captured);
                check(status, "instantiate expert graph");
            }
            check(cudaGraphLaunch(graph.exec, impl_->stream), "launch expert graph");
            ++impl_->moe_graph_hits;
        }
        impl_->lease_active = false;
        impl_->leases.clear();
        check(cudaGetLastError(), "launch captured experts");
        return;
    }
    launch_group(static_cast<int>(routed.gate->type), n_in, gate_up_routed);
    launch_group(static_cast<int>(shared.gate->type), n_in, gate_up_shared);
    const auto t_after_gu = std::chrono::steady_clock::now();
    cuda::swiglu(impl_->gate_dev, impl_->up_dev, impl_->swiglu_dev,
                 static_cast<int>(slots) * ff, impl_->stream);
    launch_group(static_cast<int>(routed.down->type), down_n_in, down_routed);
    launch_group(static_cast<int>(shared.down->type), down_n_in, down_shared);
    impl_->lease_active = false;
    impl_->leases.clear();
    for (auto& [slot, job] : cpu_jobs) {
        const auto result = job.get();
        std::copy(result.begin(), result.end(), impl_->cpu_down_host + slot * hidden);
        check(cudaMemcpyAsync(impl_->down_dev + slot * hidden, impl_->cpu_down_host + slot * hidden,
                              size_t(hidden) * sizeof(float), cudaMemcpyHostToDevice, impl_->stream), "publish CPU expert result");
    }
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

std::vector<float> CudaProjection::moe_columns(const std::vector<float>& x, int columns,
                                               const strata::TensorInfo& router, const uint8_t* router_data,
                                               const strata::TensorInfo& shared_gate, const uint8_t* shared_gate_data,
                                               const MoeWeights& routed, const MoeWeights& shared) {
    if (!supports_moe(routed, shared) || columns < 1 || columns > 2048 || x.size() != size_t(columns) * 2048 ||
        router.type != 0 || router.shape != std::vector<uint64_t>{2048, 256} ||
        shared_gate.type != 0 || shared_gate.shape != std::vector<uint64_t>{2048})
        throw std::invalid_argument("unsupported MoE prefill block");
    const int hidden = 2048, ff = int(routed.gate->shape[1]);
    impl_->ensure(impl_->bm_x, impl_->bm_x_cap, x.size());
    impl_->ensure(impl_->bm_router, impl_->bm_router_cap, size_t(columns) * 256);
    impl_->ensure(impl_->bm_scales, impl_->bm_scales_cap, size_t(columns) * 9);
    impl_->ensure(impl_->bm_ids, impl_->bm_ids_cap, size_t(columns) * 8);
    impl_->ensure(impl_->bm_map, impl_->bm_map_cap, size_t(columns) * 9);
    impl_->ensure(impl_->bm_input, impl_->bm_input_cap, x.size());
    impl_->ensure(impl_->bm_gate, impl_->bm_gate_cap, size_t(columns) * ff);
    impl_->ensure(impl_->bm_up, impl_->bm_up_cap, size_t(columns) * ff);
    impl_->ensure(impl_->bm_swiglu, impl_->bm_swiglu_cap, size_t(columns) * ff);
    impl_->ensure(impl_->bm_down, impl_->bm_down_cap, x.size());
    impl_->ensure(impl_->bm_slots, impl_->bm_slots_cap, size_t(columns) * 9 * hidden);
    check(cudaMemcpyAsync(impl_->bm_x, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice, impl_->stream), "upload MoE prefill input");
    project_columns_device(router, router_data, impl_->bm_x, impl_->bm_router, columns, -1);
    auto* shared_weight = static_cast<const float*>(impl_->weight(shared_gate, shared_gate_data, -1, 2048 * sizeof(float)));
    cuda::dot_sigmoid_columns(shared_weight, impl_->bm_x, columns, impl_->bm_scales, impl_->stream);
    cuda::router_topk_columns(impl_->bm_router, columns, impl_->bm_ids, impl_->bm_scales, impl_->stream);
    std::vector<int> ids(size_t(columns) * 8);
    check(cudaMemcpyAsync(ids.data(), impl_->bm_ids, ids.size() * sizeof(int), cudaMemcpyDeviceToHost, impl_->stream), "download prefill expert ids");
    check(cudaStreamSynchronize(impl_->stream), "finish prefill routing");
    std::array<std::vector<int>, 256> groups;
    for (int c = 0; c < columns; ++c) for (int s = 0; s < 8; ++s) {
        const int expert = ids[size_t(c)*8+s];
        if (expert < 0 || expert >= 256) throw std::runtime_error("invalid prefill route");
        groups[expert].push_back(c*9+s);
    }
    std::vector<int> map; map.reserve(size_t(columns) * 9);
    for (const auto& group : groups) map.insert(map.end(), group.begin(), group.end());
    for (int c = 0; c < columns; ++c) map.push_back(c*9+8);
    check(cudaMemcpyAsync(impl_->bm_map, map.data(), map.size() * sizeof(int), cudaMemcpyHostToDevice, impl_->stream), "publish prefill expert groups");
    const auto apply = [&](const MoeWeights& w, int expert, int count, const float* input, const int* slots) {
        project_columns_device(*w.gate, w.gate_data, input, impl_->bm_gate, count, expert);
        project_columns_device(*w.up, w.up_data, input, impl_->bm_up, count, expert);
        cuda::swiglu(impl_->bm_gate, impl_->bm_up, impl_->bm_swiglu, count*ff, impl_->stream);
        project_columns_device(*w.down, w.down_data, impl_->bm_swiglu, impl_->bm_down, count, expert);
        cuda::scatter_expert_outputs(impl_->bm_down, slots, count, hidden, impl_->bm_slots, impl_->stream);
    };
    size_t offset = 0;
    for (int expert = 0; expert < 256; ++expert) {
        const int count = int(groups[expert].size());
        if (!count) continue;
        const int* slots = impl_->bm_map + offset;
        cuda::gather_expert_inputs(impl_->bm_x, slots, count, hidden, impl_->bm_input, impl_->stream);
        apply(routed, expert, count, impl_->bm_input, slots);
        offset += size_t(count);
    }
    apply(shared, -1, columns, impl_->bm_x, impl_->bm_map + offset);
    cuda::moe_combine_columns(impl_->bm_slots, impl_->bm_scales, columns, hidden, impl_->bm_x, impl_->stream);
    check(cudaGetLastError(), "launch grouped prefill experts");
    std::vector<float> result(x.size());
    check(cudaMemcpyAsync(result.data(), impl_->bm_x, result.size() * sizeof(float), cudaMemcpyDeviceToHost, impl_->stream), "download combined prefill experts");
    check(cudaStreamSynchronize(impl_->stream), "finish prefill experts");
    return result;
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

void CudaProjection::finish_prefill() {
    if (!impl_->batch_gdn && !impl_->bm_x && !impl_->batch_q) return;
    check(cudaStreamSynchronize(impl_->stream), "finish prefill workspace");
    const auto drop = [&](auto*& pointer, size_t& capacity) {
        impl_->release(pointer); pointer = nullptr; capacity = 0;
    };
    drop(impl_->input, impl_->input_capacity); drop(impl_->output, impl_->output_capacity);
    drop(impl_->batch_gdn, impl_->batch_gdn_cap);
    drop(impl_->b_g_qkv, impl_->b_g_qkv_cap); drop(impl_->b_g_z, impl_->b_g_z_cap);
    drop(impl_->b_g_alpha, impl_->b_g_alpha_cap); drop(impl_->b_g_beta, impl_->b_g_beta_cap);
    drop(impl_->b_g_gate, impl_->b_g_gate_cap); drop(impl_->b_g_conv, impl_->b_g_conv_cap);
    drop(impl_->b_g_y, impl_->b_g_y_cap);
    drop(impl_->batch_q, impl_->batch_q_cap); drop(impl_->batch_k, impl_->batch_k_cap);
    drop(impl_->batch_v, impl_->batch_v_cap); drop(impl_->batch_partial, impl_->batch_partial_cap);
    drop(impl_->batch_accum, impl_->batch_accum_cap); drop(impl_->batch_ctx, impl_->batch_ctx_cap);
    drop(impl_->bm_x, impl_->bm_x_cap); drop(impl_->bm_router, impl_->bm_router_cap);
    drop(impl_->bm_scales, impl_->bm_scales_cap); drop(impl_->bm_input, impl_->bm_input_cap);
    drop(impl_->bm_gate, impl_->bm_gate_cap); drop(impl_->bm_up, impl_->bm_up_cap);
    drop(impl_->bm_swiglu, impl_->bm_swiglu_cap); drop(impl_->bm_down, impl_->bm_down_cap);
    drop(impl_->bm_slots, impl_->bm_slots_cap); drop(impl_->bm_ids, impl_->bm_ids_cap);
    drop(impl_->bm_map, impl_->bm_map_cap);
#ifdef LAMINA_PREFILL_BLAS
    drop(impl_->blas_weights, impl_->blas_weights_cap);
#endif
}

void CudaProjection::reset() {
    check(cudaStreamSynchronize(impl_->stream), "wait before resetting conversation");
    for (auto& state : impl_->gdn_states) {
        if (state.conv) check(cudaMemsetAsync(state.conv, 0, 8192 * 3 * sizeof(float), impl_->stream), "reset convolution");
        if (state.rec) check(cudaMemsetAsync(state.rec, 0, 32 * 128 * 128 * sizeof(float), impl_->stream), "reset recurrence");
    }
    for (auto& state : impl_->attn_states) {
        state.host_keys.clear();
        state.host_values.clear();
        state.half_keys.clear(); state.half_values.clear();
    }
}

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
    unsigned polls = 0;
    while (*flag != static_cast<int>(seq)) {
        const auto status = cudaStreamQuery(impl_->stream);
        if (status != cudaSuccess && status != cudaErrorNotReady) check(status, "wait for router");
        if (!(++polls & 131071) && std::chrono::steady_clock::now() - route_start > std::chrono::seconds(30))
            throw std::runtime_error("CUDA router timed out");
    }
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
                                        float rope_base, const std::array<int, 3>& rope_positions) {
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
    cuda::attn_norm_mrope(impl_->a_q, kHeads, 2 * kHeadDim, kHeadDim, kRot,
                          static_cast<const float*>(q_norm), kEps, rope_base,
                          rope_positions[0], rope_positions[1], rope_positions[2], impl_->stream);
    cuda::attn_norm_mrope(impl_->a_k, kKvHeads, kHeadDim, kHeadDim, kRot,
                          static_cast<const float*>(k_norm), kEps, rope_base,
                          rope_positions[0], rope_positions[1], rope_positions[2], impl_->stream);
    if (static_cast<size_t>(layer) >= impl_->attn_states.size())
        impl_->attn_states.resize(static_cast<size_t>(layer) + 1);
    auto& state = impl_->attn_states[static_cast<size_t>(layer)];
    const int needed = position + 1;
    if (needed > impl_->max_context) throw std::out_of_range("CUDA attention context exceeded");
    const int tiles = (needed + 127) / 128;
    impl_->ensure(impl_->a_partial, impl_->a_partial_cap, static_cast<size_t>(kHeads) * tiles * (kHeadDim + 2));
    if (impl_->host_kv) {
        impl_->append_host(state, impl_->a_k, impl_->a_v, position, 1);
        for (int start = 0; start < needed; start += Impl::kKvTile) {
            const int count = std::min(Impl::kKvTile, needed - start);
            auto& slot = impl_->stage_kv(state, start, count);
            if (impl_->half_kv)
                cuda::attn_half_partials(impl_->a_q, slot.keys, slot.values, count, start / 128,
                                        tiles, impl_->a_partial, impl_->stream);
            else cuda::attn_partials(impl_->a_q, static_cast<float*>(slot.keys), static_cast<float*>(slot.values),
                count, start / 128, tiles, kHeads, kKvHeads, kHeadDim, 1.0f/16.0f, impl_->a_partial, impl_->stream);
            check(cudaEventRecord(slot.consumed, impl_->stream), "record KV consumer");
        }
    } else {
    if (state.capacity < needed) {
        int capacity = state.capacity ? state.capacity : std::min(2048, std::max(needed, 1));
        while (capacity < needed) capacity *= 2;
        if (capacity > impl_->max_context) capacity = impl_->max_context;
        if (capacity < needed) throw std::out_of_range("CUDA attention context exceeded");
        float* keys = nullptr;
        float* values = nullptr;
        impl_->allocate(reinterpret_cast<void**>(&keys), impl_->kv_bytes(capacity));
        impl_->allocate(reinterpret_cast<void**>(&values), impl_->kv_bytes(capacity));
        if (state.keys) {
            const size_t bytes = impl_->kv_bytes(state.capacity);
            check(cudaMemcpy(keys, state.keys, bytes, cudaMemcpyDeviceToDevice), "grow KV keys");
            check(cudaMemcpy(values, state.values, bytes, cudaMemcpyDeviceToDevice), "grow KV values");
            impl_->release(state.keys);
            impl_->release(state.values);
        }
        state.keys = keys;
        state.values = values;
        state.capacity = capacity;
    }
    cuda::kv_store(impl_->a_k, impl_->kv_offset(state.keys, position), kKv, impl_->half_kv, impl_->stream);
    cuda::kv_store(impl_->a_v, impl_->kv_offset(state.values, position), kKv, impl_->half_kv, impl_->stream);
    if (impl_->half_kv) cuda::attn_half_partials(impl_->a_q, state.keys, state.values, needed, 0, tiles,
                                               impl_->a_partial, impl_->stream);
    else cuda::attn_partials(impl_->a_q, state.keys, state.values, needed, 0, tiles,
                            kHeads, kKvHeads, kHeadDim, 1.0f / 16.0f, impl_->a_partial, impl_->stream);
    }
    cuda::attn_merge(impl_->a_q, impl_->a_partial, tiles, kHeads, kHeadDim, impl_->a_ctx, impl_->stream);
    const void* out_weight = impl_->projection_weight(*w.out, w.out_data, -1);
    const float* out_input = impl_->a_ctx;
    float* out_output = impl_->mix_dev;
    const int out_rows = kHidden;
    strata::kernels::native_mmvq_f32_grouped(static_cast<int>(w.out->type), &out_weight, &out_input,
                                             &out_output, &out_rows, 1, kHidden, kHeads * kHeadDim,
                                             impl_->stream);
    check(cudaGetLastError(), "launch attention");
}

std::vector<float> CudaProjection::attention_columns(int layer, const std::vector<float>& x,
                                                     int columns, const AttnWeights& w,
                                                     int position, int rope_position, float rope_base,
                                                     const std::vector<std::array<int, 3>>& positions) {
    if (!supports_attention(w) || columns < 1 || columns > 2048 || position < 0 ||
        position + columns > impl_->max_context || layer < 0 || (!positions.empty() && positions.size() != size_t(columns)))
        throw std::invalid_argument("invalid attention prefill");
    auto q = matvec_columns(*w.q, w.q_data, x, columns);
    auto k = matvec_columns(*w.k, w.k_data, x, columns);
    auto v = matvec_columns(*w.v, w.v_data, x, columns);
    impl_->ensure(impl_->batch_q, impl_->batch_q_cap, q.size());
    impl_->ensure(impl_->batch_k, impl_->batch_k_cap, k.size());
    impl_->ensure(impl_->batch_v, impl_->batch_v_cap, v.size());
    impl_->ensure(impl_->batch_ctx, impl_->batch_ctx_cap, size_t(columns) * 4096);
    // Fused attention has no per-key-tile partials. All query columns can
    // share one history sweep, bounded to a 32 MiB running accumulator.
    const int query_tile = columns;
    impl_->ensure(impl_->batch_accum, impl_->batch_accum_cap, size_t(query_tile) * 16 * 258);
    check(cudaMemcpyAsync(impl_->batch_q, q.data(), q.size() * sizeof(float), cudaMemcpyHostToDevice, impl_->stream), "upload query columns");
    check(cudaMemcpyAsync(impl_->batch_k, k.data(), k.size() * sizeof(float), cudaMemcpyHostToDevice, impl_->stream), "upload key columns");
    check(cudaMemcpyAsync(impl_->batch_v, v.data(), v.size() * sizeof(float), cudaMemcpyHostToDevice, impl_->stream), "upload value columns");
    auto* qnorm = static_cast<const float*>(impl_->weight(*w.q_norm, w.q_norm_data, -1, 256 * sizeof(float)));
    auto* knorm = static_cast<const float*>(impl_->weight(*w.k_norm, w.k_norm_data, -1, 256 * sizeof(float)));
    for (int c = 0; c < columns; ++c) {
        const int p = rope_position + c;
        const auto pos = positions.empty() ? std::array<int, 3>{p, p, p} : positions[c];
        cuda::attn_norm_mrope(impl_->batch_q + size_t(c) * 8192, 16, 512, 256, 64, qnorm, 1e-6f, rope_base, pos[0], pos[1], pos[2], impl_->stream);
        cuda::attn_norm_mrope(impl_->batch_k + size_t(c) * 512, 2, 256, 256, 64, knorm, 1e-6f, rope_base, pos[0], pos[1], pos[2], impl_->stream);
    }
    if (size_t(layer) >= impl_->attn_states.size()) impl_->attn_states.resize(size_t(layer) + 1);
    auto& state = impl_->attn_states[size_t(layer)];
    const int needed = position + columns;
    if (needed > impl_->max_context) throw std::out_of_range("CUDA attention context exceeded");
    if (impl_->host_kv) {
        impl_->append_host(state, impl_->batch_k, impl_->batch_v, position, columns);
    } else {
        if (state.capacity < needed) {
            int capacity = state.capacity ? state.capacity : 2048;
            while (capacity < needed) capacity *= 2;
            capacity = std::min(capacity, impl_->max_context);
            float *keys = nullptr, *values = nullptr;
            impl_->allocate(reinterpret_cast<void**>(&keys), impl_->kv_bytes(capacity));
            impl_->allocate(reinterpret_cast<void**>(&values), impl_->kv_bytes(capacity));
            if (position) {
                check(cudaMemcpyAsync(keys, state.keys, impl_->kv_bytes(position), cudaMemcpyDeviceToDevice, impl_->stream), "grow key columns");
                check(cudaMemcpyAsync(values, state.values, impl_->kv_bytes(position), cudaMemcpyDeviceToDevice, impl_->stream), "grow value columns");
            }
            check(cudaStreamSynchronize(impl_->stream), "finish KV growth");
            impl_->release(state.keys); impl_->release(state.values);
            state.keys = keys; state.values = values; state.capacity = capacity;
        }
        cuda::kv_store(impl_->batch_k, impl_->kv_offset(state.keys, position), columns * 512, impl_->half_kv, impl_->stream);
        cuda::kv_store(impl_->batch_v, impl_->kv_offset(state.values, position), columns * 512, impl_->half_kv, impl_->stream);
    }
    for (int query = 0; query < columns; query += query_tile) {
    const int query_count = std::min(query_tile, columns - query);
    const int prefix = position + query + query_count;
    for (int start = 0; start < prefix; start += Impl::kKvTile) {
        const int count = std::min(Impl::kKvTile, prefix - start);
        const void *keys = state.keys ? impl_->kv_offset(state.keys, start) : nullptr;
        const void *values = state.values ? impl_->kv_offset(state.values, start) : nullptr;
        Impl::KvSlot* slot = nullptr;
        if (impl_->host_kv) {
            slot = &impl_->stage_kv(state, start, count);
            keys = slot->keys; values = slot->values;
        }
        cuda::attn_fused_columns(impl_->batch_q + size_t(query) * 8192, keys, values, count,
            start, position + query, query_count, impl_->batch_accum, start == 0, impl_->half_kv, impl_->stream);
        if (slot) check(cudaEventRecord(slot->consumed, impl_->stream), "record column KV consumer");
    }
    cuda::attn_columns_finish(impl_->batch_q + size_t(query) * 8192, impl_->batch_accum, query_count,
                              impl_->batch_ctx + size_t(query) * 4096, impl_->stream);
    }
    check(cudaGetLastError(), "launch causal attention columns");
    std::vector<float> result(size_t(columns) * 4096);
    check(cudaMemcpyAsync(result.data(), impl_->batch_ctx, result.size() * sizeof(float), cudaMemcpyDeviceToHost, impl_->stream), "download attention columns");
    check(cudaStreamSynchronize(impl_->stream), "finish attention columns");
    return matvec_columns(*w.out, w.out_data, result, columns);
}

void* CudaProjection::cuda_stream() const { return impl_->stream; }

void CudaProjection::mark_begin() {
    if (!impl_->mark_start) check(cudaEventCreateWithFlags(&impl_->mark_start, cudaEventDefault), "create event");
    if (!impl_->mark_stop) check(cudaEventCreateWithFlags(&impl_->mark_stop, cudaEventDefault), "create event");
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
    stats.allocated_bytes = impl_->allocated_bytes;
    stats.memory_limit = impl_->memory_limit;
    stats.graph_hits = impl_->moe_graph_hits;
    stats.graph_misses = impl_->moe_graph_misses;
    stats.cpu_experts = impl_->stat_cpu_experts;
    return stats;
}

}  // namespace lamina::model
#else
namespace lamina::model {
struct CudaProjection::Impl {};
CudaProjection::CudaProjection(size_t, bool, bool) { throw std::runtime_error("Lamina was built without CUDA"); }
CudaProjection::~CudaProjection() = default;
bool CudaProjection::supports(uint32_t) const { return false; }
std::vector<float> CudaProjection::normalize_columns(const strata::TensorInfo&, const uint8_t*, const std::vector<float>&, int, float) {
    throw std::runtime_error("Lamina was built without CUDA");
}
std::vector<float> CudaProjection::matvec_columns(const strata::TensorInfo&, const uint8_t*, const std::vector<float>&, int, int64_t) {
    throw std::runtime_error("Lamina was built without CUDA");
}
std::vector<float> CudaProjection::delta_net_columns(int, const std::vector<float>&, int, const GdnWeights&) {
    throw std::runtime_error("Lamina was built without CUDA");
}
std::vector<float> CudaProjection::attention_columns(int, const std::vector<float>&, int, const AttnWeights&, int, int, float, const std::vector<std::array<int, 3>>&) {
    throw std::runtime_error("Lamina was built without CUDA");
}
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
std::vector<float> CudaProjection::moe_columns(const std::vector<float>&, int, const strata::TensorInfo&, const uint8_t*,
                                               const strata::TensorInfo&, const uint8_t*, const MoeWeights&, const MoeWeights&) {
    throw std::runtime_error("Lamina was built without CUDA");
}
std::vector<float> CudaProjection::moe(const std::vector<float>&, const MoeWeights&,
                                       const std::vector<int>&, const std::vector<float>&,
                                       const MoeWeights&, float) {
    throw std::runtime_error("Lamina was built without CUDA");
}
void CudaProjection::set_context(int) { throw std::runtime_error("Lamina was built without CUDA"); }
void CudaProjection::reset() { throw std::runtime_error("Lamina was built without CUDA"); }
void CudaProjection::finish_prefill() { throw std::runtime_error("Lamina was built without CUDA"); }
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
void CudaProjection::attention_into_mix(int, const AttnWeights&, int, float, const std::array<int, 3>&) {
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
