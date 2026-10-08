#include "lamina/model/cuda_projection.hpp"

#include <stdexcept>

#ifdef LAMINA_ENABLE_CUDA
#include "lamina/model/cuda_kernels.hpp"
#include "lamina/model/cpu_experts.hpp"
#include "lamina/model/weight_stager.hpp"
#include "strata/kernels/gdn.hpp"
#include "strata/kernels/native_gdn.hpp"
#include "strata/kernels/native_gdn_preprocess.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>
#ifdef LAMINA_PREFILL_BLAS
#include <cublas_v2.h>
#endif

#include <algorithm>
#include <numeric>
#include <charconv>
#include <chrono>
#include <cmath>
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
// Hot-path settings are read once: on Windows each getenv takes a CRT lock and
// scans the whole environment block.
bool env_is(const char* name, char value) {
    const char* v = std::getenv(name);
    return v && v[0] == value;
}
bool atomic_expert_cache() {
    static const bool on = env_is("LAMINA_ATOMIC_EXPERT_CACHE", '1');
    return on;
}
bool pageable_uploads() {
    static const bool on = env_is("LAMINA_PINNED_UPLOADS", '0');
    return on;
}

bool device_profile() {
    static const bool enabled = std::getenv("LAMINA_DEV_PROFILE") != nullptr;
    return enabled;
}
}

struct CudaProjection::Impl {
    struct ExpertBlock { void* base; size_t bytes; std::array<std::string, 3> keys; uint64_t hits = 1; };
    struct Entry {
        void* device = nullptr;
        size_t bytes = 0;
        uint64_t used = 0;
        bool pinned = false;  // decode dense/shared; only current layer during long prefill
        std::list<std::string>::iterator lru;
        std::shared_ptr<ExpertBlock> block;
        uint64_t hits = 1;
        bool admitted = false;  // lives in admit_lru (request-local pool), not lru
        bool kept = false;      // lives in keep_lru: a per-layer top prompt expert, not evicted by evict_oldest
    };
    std::unordered_map<std::string, Entry> weights;
    std::list<std::string> lru;
    std::unordered_map<const strata::TensorInfo*, MoeWeights> expert_layouts;
    bool frequency_cache = false;
    uint64_t decode_tokens = 0;
    std::unique_ptr<WeightStager> stager;
    std::unordered_map<const uint8_t*, WeightStager::Ticket> prepared;
    cudaStream_t weight_copy = nullptr;
    cudaEvent_t expert_epoch = nullptr, expert_ready = nullptr;
    bool expert_pipeline_running = false, pipeline_copy_fenced = false;
    const uint8_t* registered_weights = nullptr; size_t registered_bytes = 0;
    std::unordered_map<void*, cudaEvent_t> upload_ready;
    std::unordered_map<void*, cudaEvent_t> retired;
    std::unordered_set<void*> retired_in_pipeline;
    // Reuse evicted blocks instead of cudaMalloc/cudaFree per miss: on WDDM the
    // malloc serializes the device and dominated a token once the cache churned.
    std::unordered_map<size_t, std::vector<void*>> free_blocks;
    size_t cached_bytes = 0;
    size_t cache_limit = 0;
    size_t memory_limit = 0, allocated_bytes = 0;
    std::unordered_map<void*, size_t> allocations;
    bool host_kv = false, half_kv = false, lease_active = false, fast = false;
    uint8_t* expert_q8 = nullptr; size_t expert_q8_cap = 0;
    uint8_t* dense_q8 = nullptr; size_t dense_q8_cap = 0;
    std::unordered_set<void*> leases;
    std::unique_ptr<CpuExperts> cpu_experts;
    bool cpu_misses = false;
    // Mapped host staging for CPU experts. Kernels move these few kilobytes
    // through the device aliases, so they never queue behind weight DMA.
    float* cpu_down_host = nullptr; float* cpu_down_dev = nullptr; size_t cpu_down_cap = 0;
    // Expert input published by the decode router before the doorbell, so CPU
    // experts start without a separate copy and stream synchronization.
    float* cpu_input_host = nullptr; float* cpu_input_dev = nullptr; size_t cpu_input_cap = 0;
    // Hybrid prefill staging: the layer's expert inputs, CPU results, and the
    // slot of each result (int32 values in a float-sized mapped buffer).
    float* pf_in_host = nullptr; float* pf_in_dev = nullptr; size_t pf_in_cap = 0;
    float* pf_out_host = nullptr; float* pf_out_dev = nullptr; size_t pf_out_cap = 0;
    float* pf_map_host = nullptr; float* pf_map_dev = nullptr; size_t pf_map_cap = 0;
    bool cpu_input_published = false;
    cudaEvent_t cpu_down_copied = nullptr;  // CPU results consumed; staging may be rewritten
    int* argmax_host = nullptr; int* argmax_dev = nullptr;  // mapped greedy-token result
    uint64_t clock = 0;
    struct Timing { cudaEvent_t start, stop; int kind; };
    std::vector<Timing> timings;
    double copy_ms = 0, expert_ms = 0, staging_wait_ms = 0, router_wait_ms = 0;
    cudaEvent_t begin_time(cudaStream_t target, int kind) {
        if (!device_profile()) return nullptr;
        // Profiling must remain bounded even for a full 128K prefill. This
        // instrumentation fence is excluded from headline performance runs.
        if (timings.size() >= 1024) {
            check(cudaEventSynchronize(timings.back().stop), "bound timing event pool");
            collect_times();
        }
        Timing timing{}; timing.kind = kind;
        check(cudaEventCreate(&timing.start), "create timing start");
        check(cudaEventCreate(&timing.stop), "create timing stop");
        check(cudaEventRecord(timing.start, target), "record timing start");
        timings.push_back(timing);
        return timing.stop;
    }
    void end_time(cudaEvent_t stop, cudaStream_t target) {
        if (stop) check(cudaEventRecord(stop, target), "record timing stop");
    }
    void collect_times() {
        for (const auto& timing : timings) {
            float ms = 0;
            check(cudaEventSynchronize(timing.stop), "finish timing event");
            check(cudaEventElapsedTime(&ms, timing.start, timing.stop), "read completed timings");
            if (timing.kind == 0) copy_ms += ms; else expert_ms += ms;
            cudaEventDestroy(timing.start); cudaEventDestroy(timing.stop);
        }
        timings.clear();
    }
    // Next-layer expert prefetch. Entries inserted by a prefetch are visible in
    // `weights` while their copy may still be in flight, so every reader must
    // first order `stream` after prefetch_ready (drain_prefetch).
    cudaEvent_t prefetch_ready = nullptr;
    bool prefetch_pending = false, has_prefetch = false;
    std::vector<int> prefetch_pred, prefetch_expected;
    MoeWeights prefetch_target;
    float* pred_scales = nullptr; size_t pred_scales_cap = 0;
    uint64_t stat_pf_uploaded = 0, stat_pf_predicted = 0, stat_pf_useful = 0;
    // Diagnostic stage timeline; see CudaProjection::timeline_mark.
    bool timeline = [] { const char* v = std::getenv("LAMINA_TIMELINE"); return v && v[0] == '1'; }();
    std::vector<cudaEvent_t> tl_events;
    std::vector<int> tl_stages;
    size_t tl_used = 0;
    std::array<double, CudaProjection::kTlCount> tl_ms{};
    // Host-side milliseconds per section of moe_core (timeline diagnostics).
    std::array<double, 9> tl_host{};
    std::unordered_map<const strata::TensorInfo*, uint64_t> tl_cpu_by_layer;  // CPU experts per routed tensor
    std::vector<const strata::TensorInfo*> tl_layer_order;  // cpu start, setup to launch, cpu wait, admission, setup parts 4..8
    uint64_t tl_tokens = 0, tl_seen = 0;
    void tl_mark(int stage) {
        if (!timeline) return;
        if (tl_used == tl_events.size()) {
            tl_events.push_back(nullptr);
            check(cudaEventCreate(&tl_events.back()), "create timeline event");
        }
        check(cudaEventRecord(tl_events[tl_used++], stream), "record timeline mark");
        tl_stages.push_back(stage);
    }
    // Background admission for the CPU-miss policy: an expert computed on the
    // CPU is also copied into the cache on the copy stream. It stays invisible
    // to readers (absent from `weights`) until its copy event has completed, so
    // no kernel can read a partially copied weight and compute never waits.
    //
    // Admitted experts form a separate, request-local pool with its own LRU and
    // byte budget. They only evict each other, never the base cache, and RESET
    // empties the pool. Every request therefore starts from the same base
    // cache, so identical requests produce identical output even though fast
    // CPU and GPU expert arithmetic differ in their last bits.
    struct Admission { std::string key; void* device; size_t bytes; };
    std::vector<Admission> admissions;
    std::unordered_set<std::string> admitting;
    std::list<std::string> admit_lru;
    // Fast CPU-miss mode: during layer-major prefill each layer's most-routed
    // prompt experts (already uploaded by the prefill itself) are kept out of
    // the base LRU, so the base cache ends up balanced across layers instead of
    // holding only the last layers prefill touched. Rebuilt for every request.
    std::list<std::string> keep_lru;
    std::unordered_map<const strata::TensorInfo*, std::array<uint32_t, 256>> prefill_counts;
    std::unordered_map<const strata::TensorInfo*, std::vector<int>> kept_experts;
    int base_keep = [] {
        const char* v = std::getenv("LAMINA_BASE_KEEP");
        return v && *v ? std::max(0, std::atoi(v)) : 12;
    }();
    std::string expert_key(const strata::TensorInfo& t, int expert) const {
        return t.name + "#type=" + std::to_string(t.type) + "#expert=" + std::to_string(expert);
    }
    void set_kept(const MoeWeights& w, int expert, bool keep) {
        const std::array<std::pair<const strata::TensorInfo*, const uint8_t*>, 3> tensors{
            std::pair{w.gate, w.gate_data}, std::pair{w.up, w.up_data}, std::pair{w.down, w.down_data}};
        for (const auto& [t, data] : tensors) {
            // A chosen expert may already have been evicted under cache pressure,
            // which differs between a cold first request and later ones; upload
            // it so the kept set is exactly the prompt's top experts.
            if (keep && !weights.contains(expert_key(*t, expert))) projection_weight(*t, data, expert);
            const auto found = weights.find(expert_key(*t, expert));
            if (found == weights.end() || found->second.pinned || found->second.admitted || found->second.kept == keep) continue;
            if (keep) keep_lru.splice(keep_lru.end(), lru, found->second.lru);
            else lru.splice(lru.begin(), keep_lru, found->second.lru);  // released entries are evicted first
            found->second.kept = keep;
        }
    }
    void keep_prompt_experts(const MoeWeights& w) {
        const auto& counts = prefill_counts[w.gate];
        std::vector<int> order(256);
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return counts[size_t(a)] > counts[size_t(b)]; });
        std::vector<int> chosen;
        for (int i = 0; i < base_keep && i < 256 && counts[size_t(order[size_t(i)])] > 0; ++i) chosen.push_back(order[size_t(i)]);
        auto& previous = kept_experts[w.gate];
        for (const int expert : previous)
            if (std::find(chosen.begin(), chosen.end(), expert) == chosen.end()) set_kept(w, expert, false);
        for (const int expert : chosen) set_kept(w, expert, true);
        previous = std::move(chosen);
    }
    // After a prefill with kept experts, drop every other routed expert from the
    // base LRU. Which leftovers survive prefill depends on earlier requests, and
    // in fast mode GPU and CPU expert results differ in their last bits, so the
    // base must depend on the current prompt alone for repeatable output.
    void drop_unkept_routed() {
        if (keep_lru.empty()) return;
        for (auto it = lru.begin(); it != lru.end();) {
            const auto entry = weights.find(*it);
            if (entry == weights.end() || entry->first.find("#expert=-1") != std::string::npos) { ++it; continue; }
            device_release(entry->second.device, entry->second.bytes);
            cached_bytes -= entry->second.bytes; stat_evicted += entry->second.bytes;
            weights.erase(entry);
            it = lru.erase(it);
        }
    }
    void release_kept() {
        for (const auto& key : keep_lru) weights.at(key).kept = false;
        lru.splice(lru.begin(), keep_lru);
        prefill_counts.clear();
        kept_experts.clear();
    }
    size_t admitted_bytes = 0;  // promoted plus in-flight admissions
    uint64_t stat_admitted = 0;
    int admit_per_layer = [] {
        const char* v = std::getenv("LAMINA_ADMIT_PER_LAYER");
        return v && *v ? std::max(0, std::atoi(v)) : 1;
    }();
    size_t admit_limit = [] {
        const char* v = std::getenv("LAMINA_ADMIT_MB");
        return size_t(v && *v ? std::max(0, std::atoi(v)) : 2048) << 20;
    }();
    bool admission_enabled() const {
        return cpu_misses && admit_per_layer > 0 && registered_weights && !atomic_expert_cache();
    }
    // The base cache keeps the admission pool's share free.
    size_t base_limit() const {
        return admission_enabled() ? cache_limit - std::min(admit_limit, cache_limit / 2) : cache_limit;
    }
    bool base_room(size_t bytes) const {
        const size_t limit = base_limit(), used = std::min(cached_bytes - admitted_bytes, limit);
        return bytes <= limit - used;
    }
    // Room for a base-cache upload. Base entries are evicted first to keep the
    // admission share free; when only pinned or leased entries remain (a small
    // forced cache), the base borrows from the admission pool instead of failing.
    bool make_base_room(size_t bytes) {
        while (!base_room(bytes) && evict_oldest()) {}
        if (base_room(bytes)) return true;
        while (cached_bytes + bytes > cache_limit && evict_oldest_admitted()) {}
        return cached_bytes + bytes <= cache_limit;
    }
    bool evict_oldest_admitted() {
        if (admit_lru.empty()) return false;
        const auto entry = weights.find(admit_lru.front());
        if (lease_active && leases.contains(entry->second.device)) return false;
        device_release(entry->second.device, entry->second.bytes);
        cached_bytes -= entry->second.bytes; admitted_bytes -= entry->second.bytes; stat_evicted += entry->second.bytes;
        weights.erase(entry);
        admit_lru.pop_front();
        return true;
    }
    // Blocks evicted from the admission pool by admit() are reused only by later
    // admissions. One epoch event, recorded on the compute stream after such
    // evictions, fences all of their readers, and the copy stream waits on it
    // once; this replaces a retirement event and a wait per evicted tensor.
    std::unordered_map<size_t, std::vector<void*>> admit_free;
    cudaEvent_t admit_epoch = nullptr;
    bool admit_epoch_pending = false;
    bool evict_admitted_for_reuse() {
        if (admit_lru.empty()) return false;
        const auto entry = weights.find(admit_lru.front());
        if (lease_active && leases.contains(entry->second.device)) return false;
        admit_free[entry->second.bytes].push_back(entry->second.device);
        cached_bytes -= entry->second.bytes; admitted_bytes -= entry->second.bytes; stat_evicted += entry->second.bytes;
        weights.erase(entry);
        admit_lru.pop_front();
        admit_epoch_pending = true;
        return true;
    }
    // Called once per decode token. Waiting for the previous token's copies
    // keeps expert placement, and therefore fast-mode output, independent of
    // copy timing.
    // The copy stream is FIFO, so one synchronization covers every admission
    // queued during the previous token; no per-admission events are recorded.
    void promote_admissions() {
        if (!admissions.empty()) check(cudaStreamSynchronize(weight_copy), "admit expert weights");
        for (size_t i = 0; i < admissions.size();) {
            auto& a = admissions[i];
            admitting.erase(a.key);
            if (weights.contains(a.key)) {  // another path uploaded it meanwhile
                device_release(a.device, a.bytes);
                cached_bytes -= a.bytes; admitted_bytes -= a.bytes;
            } else {
                Entry entry{a.device, a.bytes, ++clock, false};
                entry.admitted = true;
                entry.lru = admit_lru.insert(admit_lru.end(), a.key);
                weights.emplace(a.key, std::move(entry));
            }
            admissions[i] = std::move(admissions.back());
            admissions.pop_back();
        }
    }
    // Drops the whole admission pool (RESET). The caller has synchronized the
    // compute stream, so no kernel can still read an admitted weight.
    void clear_admissions() {
        if (weight_copy) check(cudaStreamSynchronize(weight_copy), "finish admissions");
        for (auto& a : admissions) {
            device_release(a.device, a.bytes);
            cached_bytes -= a.bytes; admitted_bytes -= a.bytes;
        }
        admissions.clear();
        admitting.clear();
        while (evict_oldest_admitted()) {}
        // Both streams are idle here, so these blocks may rejoin the general pool.
        for (auto& [bytes, blocks] : admit_free)
            for (void* block : blocks) free_blocks[bytes].push_back(block);
        admit_free.clear();
        admit_epoch_pending = false;
    }
    bool admit(const strata::TensorInfo& t, const uint8_t* data, int expert) {
        const std::string key = t.name + "#type=" + std::to_string(t.type) + "#expert=" + std::to_string(expert);
        if (weights.contains(key) || admitting.contains(key)) return true;
        const size_t bytes = strata::kernels::native_mmvq_weight_bytes(t.type, int(t.shape[0]), int(t.shape[1]));
        const uint8_t* source = data + size_t(expert) * bytes;
        if (!admission_enabled() || source < registered_weights || size_t(source - registered_weights) > registered_bytes ||
            bytes > registered_bytes - size_t(source - registered_weights)) return false;
        const size_t limit = std::min(admit_limit, cache_limit / 2);
        while ((admitted_bytes + bytes > limit || cached_bytes + bytes > cache_limit) && evict_admitted_for_reuse()) {}
        if (admitted_bytes + bytes > limit || cached_bytes + bytes > cache_limit) return false;
        if (!weight_copy) check(cudaStreamCreateWithFlags(&weight_copy, cudaStreamNonBlocking), "create expert copy stream");
        void* device = nullptr;
        if (auto reuse = admit_free.find(bytes); reuse != admit_free.end() && !reuse->second.empty()) {
            if (admit_epoch_pending) {
                if (!admit_epoch) check(cudaEventCreateWithFlags(&admit_epoch, cudaEventDisableTiming), "create admission epoch");
                check(cudaEventRecord(admit_epoch, stream), "record admission epoch");
                check(cudaStreamWaitEvent(weight_copy, admit_epoch, 0), "wait admission epoch");
                admit_epoch_pending = false;
            }
            device = reuse->second.back();
            reuse->second.pop_back();
        } else {
            device = device_alloc(bytes);
            if (retired_in_pipeline.erase(device) && expert_epoch)
                check(cudaStreamWaitEvent(weight_copy, expert_epoch, 0), "wait pipeline retirement epoch");
            else if (auto found = retired.find(device); found != retired.end())
                check(cudaStreamWaitEvent(weight_copy, found->second, 0), "wait retired weight readers");
        }
        const auto timed = begin_time(weight_copy, 0);
        check(cudaMemcpyAsync(device, source, bytes, cudaMemcpyHostToDevice, weight_copy), "admit expert weight");
        end_time(timed, weight_copy);
        admissions.push_back({key, device, bytes});
        admitting.insert(key);
        cached_bytes += bytes; admitted_bytes += bytes; stat_uploaded += bytes; ++stat_admitted;
        return true;
    }
    // Fast CPU-miss decode computes the shared expert before the host has read
    // the routing, into its usual slot. These are the same kernels, inputs and
    // outputs the expert graph used for that slot, so results are unchanged.
    bool shared_early = false;
    void shared_expert_early(const MoeWeights& shared, const float* x, int slot) {
        const int n_in = int(shared.gate->shape[0]), ff = int(shared.gate->shape[1]), hidden = int(shared.down->shape[1]);
        const size_t slots = size_t(slot) + 1;
        ensure(gate_dev, gate_capacity, slots * ff);
        ensure(up_dev, up_capacity, slots * ff);
        ensure(swiglu_dev, swiglu_capacity, slots * ff);
        ensure(down_dev, down_capacity, slots * hidden);
        const size_t iq = strata::kernels::native_q8_1_bytes(n_in), dq = strata::kernels::native_q8_1_bytes(ff);
        ensure(expert_q8, expert_q8_cap, iq + slots * dq);
        strata::kernels::native_quantize_q8_1(x, expert_q8, n_in, 1, stream);
        strata::kernels::NativeF32Grouped gate_up{};
        gate_up.count = 2;
        gate_up.weights[0] = projection_weight(*shared.gate, shared.gate_data, -1);
        gate_up.weights[1] = projection_weight(*shared.up, shared.up_data, -1);
        gate_up.inputs[0] = gate_up.inputs[1] = reinterpret_cast<const float*>(expert_q8);
        gate_up.outputs[0] = gate_dev + size_t(slot) * ff;
        gate_up.outputs[1] = up_dev + size_t(slot) * ff;
        gate_up.n_outs[0] = gate_up.n_outs[1] = ff;
        strata::kernels::native_mmvq_q8_grouped(int(shared.gate->type), gate_up, 2 * ff, n_in, stream);
        cuda::swiglu(gate_dev + size_t(slot) * ff, up_dev + size_t(slot) * ff, swiglu_dev + size_t(slot) * ff, ff, stream);
        strata::kernels::native_quantize_q8_1(swiglu_dev + size_t(slot) * ff, expert_q8 + iq + size_t(slot) * dq, ff, 1, stream);
        strata::kernels::NativeF32Grouped down{};
        down.count = 1;
        down.weights[0] = projection_weight(*shared.down, shared.down_data, -1);
        down.inputs[0] = reinterpret_cast<const float*>(expert_q8 + iq + size_t(slot) * dq);
        down.outputs[0] = down_dev + size_t(slot) * hidden;
        down.n_outs[0] = hidden;
        strata::kernels::native_mmvq_q8_grouped(int(shared.down->type), down, hidden, ff, stream);
    }
    void mapped_floats(float*& host, float*& device, size_t& capacity, size_t elements) {
        if (elements <= capacity) return;
        if (host) check(cudaFreeHost(host), "release mapped staging");
        host = nullptr; device = nullptr; capacity = 0;
        check(cudaHostAlloc(reinterpret_cast<void**>(&host), elements * sizeof(float), cudaHostAllocMapped), "allocate mapped staging");
        check(cudaHostGetDevicePointer(reinterpret_cast<void**>(&device), host, 0), "map staging");
        capacity = elements;
    }
    void drain_prefetch() {
        if (!prefetch_pending) return;
        check(cudaStreamWaitEvent(stream, prefetch_ready, 0), "wait prefetched experts");
        prefetch_pending = false;
    }
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
    uint8_t *matrix_q = nullptr, *matrix_k = nullptr, *matrix_v = nullptr, *matrix_probability = nullptr;
    size_t matrix_q_cap = 0, matrix_k_cap = 0, matrix_v_cap = 0, matrix_probability_cap = 0;
    float *matrix_scores = nullptr, *matrix_output = nullptr, *matrix_metadata = nullptr;
    size_t matrix_scores_cap = 0, matrix_output_cap = 0, matrix_metadata_cap = 0;
    cublasHandle_t blas = nullptr;
    void* blas_workspace = nullptr;
    uint8_t *blas_half_weights = nullptr, *blas_half_input = nullptr;
    size_t blas_half_weights_cap = 0, blas_half_input_cap = 0;
    float* blas_weights = nullptr;
    size_t blas_weights_cap = 0;
#endif
    int prefill_columns = 0, long_columns = 0;
    float* long_hidden = nullptr; size_t long_hidden_cap = 0;
    int* long_positions = nullptr; size_t long_positions_cap = 0;
    int* p_positions = nullptr; size_t p_positions_cap = 0;
    float *p_hidden = nullptr, *p_norm = nullptr, *p_mix = nullptr;
    size_t p_hidden_cap = 0, p_norm_cap = 0, p_mix_cap = 0;
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
    std::vector<MixerGraph> mixer_graphs, pair_graphs;
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
        for (auto& timing : timings) { cudaEventDestroy(timing.start); cudaEventDestroy(timing.stop); }
#ifdef LAMINA_PREFILL_BLAS
        if (blas) cublasDestroy(blas);
        cudaFree(matrix_q); cudaFree(matrix_k); cudaFree(matrix_v); cudaFree(matrix_probability);
        cudaFree(matrix_scores); cudaFree(matrix_output); cudaFree(matrix_metadata);
        cudaFree(blas_workspace); cudaFree(blas_weights);
        cudaFree(blas_half_weights); cudaFree(blas_half_input);
#endif
        if (weight_copy) cudaStreamSynchronize(weight_copy);
        stager.reset();
        for (auto& [pointer, event] : retired) cudaEventDestroy(event);
        for (auto& [pointer, event] : upload_ready) cudaEventDestroy(event);
        if (registered_weights) cudaHostUnregister(const_cast<uint8_t*>(registered_weights));
        if (weight_copy) cudaStreamDestroy(weight_copy);
        if (expert_epoch) cudaEventDestroy(expert_epoch);
        if (expert_ready) cudaEventDestroy(expert_ready);
        if (prefetch_ready) cudaEventDestroy(prefetch_ready);
        for (auto event : tl_events) cudaEventDestroy(event);
        cudaFree(pred_scales);
        for (auto& [name, entry] : weights)
            if (!entry.block || entry.device == entry.block->base) cudaFree(entry.device);
        for (auto& [bytes, blocks] : free_blocks)
            for (void* block : blocks) cudaFree(block);
        for (auto& state : gdn_states) {
            cudaFree(state.conv);
            cudaFree(state.rec);
        }
        for (auto* graphs : {&mixer_graphs, &pair_graphs})
            for (auto& graph : *graphs)
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
        if (cpu_input_host) cudaFreeHost(cpu_input_host);
        for (float* host : {pf_in_host, pf_out_host, pf_map_host}) if (host) cudaFreeHost(host);
        if (argmax_host) cudaFreeHost(argmax_host);
        if (cpu_down_copied) cudaEventDestroy(cpu_down_copied);
        for (auto& a : admissions) cudaFree(a.device);
        for (auto& [bytes, blocks] : admit_free) for (void* block : blocks) cudaFree(block);
        if (admit_epoch) cudaEventDestroy(admit_epoch);
        cudaFree(r_logits);
        cudaFree(r_ids);
        if (h_ids_map) cudaFreeHost(h_ids_map);
        if (h_scales_map) cudaFreeHost(h_scales_map);
        if (h_flag_map) cudaFreeHost(h_flag_map);
        cudaFree(input);
        cudaFree(output);
        cudaFree(long_hidden); cudaFree(long_positions); cudaFree(p_positions); cudaFree(p_hidden); cudaFree(p_norm); cudaFree(p_mix);
        cudaFree(batch_gdn);
        cudaFree(b_g_qkv); cudaFree(b_g_z); cudaFree(b_g_alpha); cudaFree(b_g_beta);
        cudaFree(b_g_gate); cudaFree(b_g_conv); cudaFree(b_g_y);
        cudaFree(batch_q); cudaFree(batch_k); cudaFree(batch_v);
        cudaFree(batch_partial); cudaFree(batch_accum); cudaFree(batch_ctx);
        cudaFree(bm_x); cudaFree(bm_router); cudaFree(bm_scales); cudaFree(bm_input);
        cudaFree(bm_gate); cudaFree(bm_up); cudaFree(bm_swiglu); cudaFree(bm_down); cudaFree(bm_slots);
        cudaFree(bm_ids); cudaFree(bm_map);
        cudaFree(expert_q8); cudaFree(dense_q8);
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

    uint64_t stat_mallocs = 0, stat_trims = 0;
    void trim_free() {
        ++stat_trims;
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
            if (!evict_oldest()) throw std::runtime_error("VRAM working set exceeds available headroom (request=" + std::to_string(bytes / MIB) + " MiB, free=" + std::to_string(available / MIB) + " MiB); use --kv-cache host or --kv-type f16");
        }
        check(cudaMalloc(pointer, bytes), "allocate bounded device memory");
        ++stat_mallocs;
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

    void ensure(uint8_t*& pointer, size_t& capacity, size_t bytes) {
        if (bytes <= capacity) return;
        release(pointer);
        pointer = nullptr;
        allocate(reinterpret_cast<void**>(&pointer), bytes);
        capacity = bytes;
    }

#ifdef LAMINA_PREFILL_BLAS
    void ensure_blas() {
        if (blas) return;
        check_blas(cublasCreate(&blas), "create prefill handle");
        check_blas(cublasSetStream(blas, stream), "set prefill stream");
        check_blas(cublasSetMathMode(blas, fast ? CUBLAS_DEFAULT_MATH : CUBLAS_PEDANTIC_MATH), "select compute math");
        check_blas(cublasSetAtomicsMode(blas, CUBLAS_ATOMICS_NOT_ALLOWED), "select deterministic accumulation");
        allocate(&blas_workspace, 8 * MIB);
        check_blas(cublasSetWorkspace(blas, blas_workspace, 8 * MIB), "set bounded workspace");
    }
    void matrix_tile(const void* keys, const void* values, int count, int start, int position, int columns) {
        const size_t bytes = fast ? 2 : 4;
        const auto type = fast ? CUDA_R_16BF : CUDA_R_32F;
        const auto compute = fast ? CUBLAS_COMPUTE_32F : CUBLAS_COMPUTE_32F_PEDANTIC;
        cuda::attn_matrix_kv(keys, values, count, half_kv, matrix_k, matrix_v, fast, stream);
        for (int query = 0; query < columns; query += 128) {
            const int n = std::min(128, columns - query);
            if (start > position + query + n - 1) continue;
            const float scale = 1.f / 16.f, one = 1.f, zero = 0.f;
            for (int kv = 0; kv < 2; ++kv) {
                check_blas(cublasGemmStridedBatchedEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, count, n, 256,
                    &scale, matrix_k + size_t(kv) * count * 256 * bytes, type, 256, 0,
                    matrix_q + (size_t(kv) * 8 * columns + query) * 256 * bytes, type, 256, size_t(columns) * 256,
                    &zero, matrix_scores + size_t(kv) * 8 * count * n, CUDA_R_32F, count, size_t(count) * n,
                    8, compute, CUBLAS_GEMM_DEFAULT), "matrix attention scores");
            }
            cuda::attn_matrix_softmax(matrix_scores, count, n, start, position + query, matrix_metadata,
                matrix_probability, fast, stream);
            const void* probability = fast ? static_cast<void*>(matrix_probability) : static_cast<void*>(matrix_scores);
            for (int kv = 0; kv < 2; ++kv) {
                check_blas(cublasGemmStridedBatchedEx(blas, CUBLAS_OP_N, CUBLAS_OP_N, 256, n, count,
                    &one, matrix_v + size_t(kv) * count * 256 * bytes, type, 256, 0,
                    static_cast<const uint8_t*>(probability) + size_t(kv) * 8 * count * n * bytes, type, count, size_t(count) * n,
                    &zero, matrix_output + size_t(kv) * 8 * n * 256, CUDA_R_32F, 256, size_t(n) * 256,
                    8, compute, CUBLAS_GEMM_DEFAULT), "matrix attention values");
            }
            cuda::attn_matrix_merge(matrix_output, matrix_metadata, n, batch_accum + size_t(query) * 16 * 258, start == 0, stream);
        }
    }
#endif

    void reserve(size_t n_in, size_t n_out) {
        ensure(input, input_capacity, n_in);
        ensure(output, output_capacity, n_out);
    }

    void decode_group(int type, const void* const* weights, const float* const* inputs,
                      float* const* outputs, const int* rows, int count, int total_rows,
                      int width, void* launch_stream) {
        if (!fast) {
            strata::kernels::native_mmvq_f32_grouped(type, weights, inputs, outputs, rows, count, total_rows, width, launch_stream);
            return;
        }
        // Sized for two columns up front: captured DeltaNet graphs hold this
        // pointer, so the two-token step must never reallocate it.
        ensure(dense_q8, dense_q8_cap, 2 * strata::kernels::native_q8_1_bytes(4096));
        strata::kernels::native_quantize_q8_1(inputs[0], dense_q8, width, 1, launch_stream);
        strata::kernels::NativeF32Grouped args{};
        args.count = count;
        for (int i = 0; i < count; ++i) {
            if (inputs[i] != inputs[0]) throw std::logic_error("decode group must share input");
            args.weights[i] = weights[i]; args.outputs[i] = outputs[i]; args.n_outs[i] = rows[i];
            args.inputs[i] = reinterpret_cast<const float*>(dense_q8);
        }
        strata::kernels::native_mmvq_q8_grouped(type, args, total_rows, width, launch_stream);
    }

    // Two-token step scratch. Index names keep the pair functions readable.
    enum PairBuffer { kPHidden, kPNorm, kPMix, kPQkv, kPZ, kPAlpha, kPBeta, kPGate, kPConvRaw, kPConvSilu,
                      kPOutNorm, kPY, kPQ, kPK, kPV, kPCtx, kPLogits, kPHead, kPairBuffers };
    std::array<float*, kPairBuffers> pair_buf{};
    std::array<size_t, kPairBuffers> pair_cap{};
    float* pair(PairBuffer index, size_t elements) {
        ensure(pair_buf[index], pair_cap[index], elements);
        return pair_buf[index];
    }
    // DeltaNet states after the pair's first column, per layer.
    struct GdnSnapshot { float* conv = nullptr; float* rec = nullptr; bool taken = false; };
    std::vector<GdnSnapshot> gdn_snapshots;
    // decode_group for two columns: inputs[i] holds two contiguous width-wide
    // columns, outputs[i] two contiguous rows[i]-long columns. Each column is
    // bitwise equal to decode_group on that column.
    void decode_pair(int type, const void* const* weights, const float* input, float* const* outputs,
                     const int* rows, int count, int total_rows, int width) {
        if (!fast) throw std::logic_error("the two-token step requires fast mode");
        if (count > 32) throw std::invalid_argument("too many pair matrices");
        const size_t q_bytes = strata::kernels::native_q8_1_bytes(width);
        ensure(dense_q8, dense_q8_cap, 2 * strata::kernels::native_q8_1_bytes(4096));
        strata::kernels::native_quantize_q8_1(input, dense_q8, width, 2, stream);
        strata::kernels::NativeF32Pairs args{};
        args.count = count;
        for (int i = 0; i < count; ++i) {
            args.weights[i] = weights[i]; args.n_outs[i] = rows[i];
            for (int c = 0; c < 2; ++c) {
                args.inputs[c][i] = reinterpret_cast<const float*>(dense_q8 + c * q_bytes);
                args.outputs[c][i] = outputs[i] + size_t(c) * rows[i];
            }
        }
        strata::kernels::native_mmvq_q8_pairs(type, args, total_rows, width, stream);
    }
    // Grows a layer's device KV cache to hold `needed` positions.
    void grow_kv(AttnState& state, int needed) {
        if (state.capacity >= needed) return;
        int capacity = state.capacity ? state.capacity : std::min(2048, std::max(needed, 1));
        while (capacity < needed) capacity *= 2;
        if (capacity > max_context) capacity = max_context;
        if (capacity < needed) throw std::out_of_range("CUDA attention context exceeded");
        float* keys = nullptr;
        float* values = nullptr;
        allocate(reinterpret_cast<void**>(&keys), kv_bytes(capacity));
        allocate(reinterpret_cast<void**>(&values), kv_bytes(capacity));
        if (state.keys) {
            const size_t bytes = kv_bytes(state.capacity);
            check(cudaMemcpy(keys, state.keys, bytes, cudaMemcpyDeviceToDevice), "grow KV keys");
            check(cudaMemcpy(values, state.values, bytes, cudaMemcpyDeviceToDevice), "grow KV values");
            release(state.keys);
            release(state.values);
        }
        state.keys = keys;
        state.values = values;
        state.capacity = capacity;
    }

    void expert_layout(const MoeWeights& w) {
        for (const auto* t : {w.gate, w.up, w.down}) expert_layouts.emplace(t, w);
    }
    void* expert_weight(const strata::TensorInfo& tensor, int expert) {
        const auto& layout = expert_layouts.at(&tensor);
        const std::array<const strata::TensorInfo*, 3> tensors{layout.gate, layout.up, layout.down};
        const std::array<const uint8_t*, 3> data{layout.gate_data, layout.up_data, layout.down_data};
        std::array<size_t, 3> sizes{}, offsets{}; std::array<std::string, 3> keys{};
        size_t total = 0; int role = 0;
        for (int i = 0; i < 3; ++i) {
            sizes[i] = strata::kernels::native_mmvq_weight_bytes(tensors[i]->type, int(tensors[i]->shape[0]), int(tensors[i]->shape[1]));
            offsets[i] = total; total += sizes[i];
            keys[i] = tensors[i]->name + "#type=" + std::to_string(tensors[i]->type) + "#expert=" + std::to_string(expert);
            if (tensors[i] == &tensor) role = i;
        }
        if (auto entry = weights.find(keys[role]); entry != weights.end()) {
            if (!expert_pipeline_running) drain_prefetch();
            ++stat_hits; entry->second.used = ++clock; ++entry->second.block->hits;
            lru.splice(lru.end(), lru, entry->second.lru);
            if (lease_active) for (const auto& key : keys) leases.insert(weights.at(key).device);
            return entry->second.device;
        }
        ++stat_misses;
        if (!make_base_room(total)) throw std::runtime_error("expert block exceeds available cache");
        auto block = std::make_shared<ExpertBlock>(ExpertBlock{nullptr, total, keys});
        void* base = device_alloc(total); block->base = base;
        auto position = lru.end();
        try {
            for (int i = 0; i < 3; ++i) upload(static_cast<uint8_t*>(base) + offsets[i], data[i] + size_t(expert) * sizes[i], sizes[i]);
            position = lru.insert(lru.end(), keys[0]);
            for (int i = 0; i < 3; ++i) {
                void* pointer = static_cast<uint8_t*>(base) + offsets[i];
                weights.emplace(keys[i], Entry{pointer, sizes[i], ++clock, false, position, block});
                if (lease_active) leases.insert(pointer);
            }
            cached_bytes += total; stat_uploaded += total;
            return static_cast<uint8_t*>(base) + offsets[role];
        } catch (...) {
            for (const auto& key : keys) weights.erase(key);
            if (position != lru.end()) lru.erase(position);
            device_release(base, total); throw;
        }
    }

    void* projection_weight(const strata::TensorInfo& tensor, const uint8_t* data, int64_t expert) {
        const int n_in = static_cast<int>(tensor.shape.at(0));
        const int n_out = static_cast<int>(tensor.shape.at(1));
        const size_t bytes = strata::kernels::native_mmvq_weight_bytes(tensor.type, n_in, n_out);
        if (expert >= 0 && expert >= static_cast<int64_t>(tensor.shape.at(2)))
            throw std::out_of_range("CUDA projection expert index");
        if (expert >= 0 && atomic_expert_cache() && expert_layouts.contains(&tensor)) return expert_weight(tensor, int(expert));
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

    void device_release(void* block, size_t bytes) {
        auto& event = retired[block];
        if (!event) check(cudaEventCreateWithFlags(&event, cudaEventDisableTiming), "create weight retirement");
        if (expert_pipeline_running) retired_in_pipeline.insert(block);
        else {
            retired_in_pipeline.erase(block);
            check(cudaEventRecord(event, stream), "retire weight readers");
        }
        free_blocks[bytes].push_back(block);
    }

    void prepare(const strata::TensorInfo& t, const uint8_t* data, int64_t expert) {
        if (registered_weights || pageable_uploads()) return;
        if (expert < 0 || resident(t, int(expert))) return;
        const size_t bytes = strata::kernels::native_mmvq_weight_bytes(t.type, int(t.shape[0]), int(t.shape[1]));
        const uint8_t* source = data + size_t(expert) * bytes;
        if (!stager) stager = std::make_unique<WeightStager>();
        for (size_t offset = 0; offset < bytes; offset += WeightStager::bytes) {
            const uint8_t* slice = source + offset;
            if (!prepared.contains(slice)) prepared.emplace(slice, stager->prepare(slice, std::min(WeightStager::bytes, bytes - offset)));
        }
    }

    bool evict_oldest() {
        auto candidate = lru.end(); double utility = INFINITY; int examined = 0;
        for (auto it = lru.begin(); it != lru.end(); ++it) {
            auto& entry = weights.at(*it);
            bool leased = leases.contains(entry.device);
            if (entry.block) for (const auto& key : entry.block->keys) leased |= leases.contains(weights.at(key).device);
            if (leased) continue;
            if (!frequency_cache) { candidate = it; break; }
            const double value = entry.block ? double(entry.block->hits) / entry.block->bytes : double(entry.hits) / entry.bytes;
            if (value < utility) { utility = value; candidate = it; }
            if (++examined >= 64) break;
        }
        if (candidate == lru.end()) return false;
        auto entry = weights.find(*candidate);
        if (entry->second.block) {
            auto block = entry->second.block;
            device_release(block->base, block->bytes);
            cached_bytes -= block->bytes; stat_evicted += block->bytes;
            for (const auto& key : block->keys) weights.erase(key);
        } else {
            device_release(entry->second.device, entry->second.bytes);
            cached_bytes -= entry->second.bytes; stat_evicted += entry->second.bytes;
            weights.erase(entry);
        }
        lru.erase(candidate); return true;
    }

    void upload(void* device, const uint8_t* source, size_t bytes) {
        if (pageable_uploads()) {
            check(cudaMemcpyAsync(device, source, bytes, cudaMemcpyHostToDevice, stream), "upload pageable weight");
            return;
        }
        if (!weight_copy) check(cudaStreamCreateWithFlags(&weight_copy, cudaStreamNonBlocking), "create expert copy stream");
        if (expert_pipeline_running) {
            if (!pipeline_copy_fenced) {
                check(cudaStreamWaitEvent(weight_copy, expert_epoch, 0), "wait previous expert readers");
                pipeline_copy_fenced = true;
            }
        } else if (retired_in_pipeline.erase(device)) {
            // A fallback/prefill upload can reuse an earlier pipeline victim.
            // Its readers precede the latest epoch, rather than its individual
            // retirement event, which intentionally was not recorded.
            check(cudaStreamWaitEvent(weight_copy, expert_epoch, 0), "wait pipeline retirement epoch");
        } else if (auto found = retired.find(device); found != retired.end())
            check(cudaStreamWaitEvent(weight_copy, found->second, 0), "wait retired weight readers");
        if (registered_weights && source >= registered_weights && size_t(source - registered_weights) <= registered_bytes &&
            bytes <= registered_bytes - size_t(source - registered_weights)) {
            const auto timed = begin_time(weight_copy, 0);
            check(cudaMemcpyAsync(device, source, bytes, cudaMemcpyHostToDevice, weight_copy), "DMA registered weights");
            end_time(timed, weight_copy);
            if (!expert_pipeline_running) {
                auto& ready = upload_ready[device];
                if (!ready) check(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming), "create weight ready event");
                check(cudaEventRecord(ready, weight_copy), "record weight ready");
                check(cudaStreamWaitEvent(stream, ready, 0), "wait direct DMA");
            }
            return;
        }
        if (!stager) stager = std::make_unique<WeightStager>();
        for (size_t offset = 0; offset < bytes; offset += WeightStager::bytes) {
            const uint8_t* slice = source + offset;
            const size_t count = std::min(WeightStager::bytes, bytes - offset);
            if (!prepared.contains(slice)) prepared.emplace(slice, stager->prepare(slice, count));
        }
        cudaEvent_t last = nullptr;
        for (size_t offset = 0; offset < bytes; offset += WeightStager::bytes) {
            const uint8_t* slice = source + offset;
            auto job = prepared.extract(slice);
            const auto waited = std::chrono::steady_clock::now();
            job.mapped().ready.get();
            if (device_profile()) staging_wait_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - waited).count();
            const auto sequence = job.mapped().sequence;
            const unsigned slot = unsigned(sequence % WeightStager::slots);
            const size_t count = std::min(WeightStager::bytes, bytes - offset);
            const auto timed = begin_time(weight_copy, 0);
            check(cudaMemcpyAsync(static_cast<uint8_t*>(device) + offset, stager->host[slot], count,
                cudaMemcpyHostToDevice, weight_copy), "upload staged weight");
            end_time(timed, weight_copy);
            stager->issued(sequence, weight_copy);
            last = stager->done[slot];
        }
        if (!expert_pipeline_running) check(cudaStreamWaitEvent(stream, last, 0), "wait uploaded weight");
    }

    void* weight(const strata::TensorInfo& tensor, const uint8_t* data,
                 int64_t expert, size_t bytes) {
        const std::string key = tensor.name + "#type=" + std::to_string(tensor.type) +
                                "#expert=" + std::to_string(expert);
        if (auto found = weights.find(key); found != weights.end()) {
            if (expert >= 0 && !expert_pipeline_running) drain_prefetch();
            found->second.used = ++clock; ++found->second.hits;
            if (!found->second.pinned) {
                if (expert < 0) {
                    lru.erase(found->second.lru);
                    found->second.pinned = true;
                } else if (found->second.admitted) admit_lru.splice(admit_lru.end(), admit_lru, found->second.lru);
                else if (found->second.kept) keep_lru.splice(keep_lru.end(), keep_lru, found->second.lru);
                else lru.splice(lru.end(), lru, found->second.lru);
            }
            ++stat_hits;
            if (lease_active) leases.insert(found->second.device);
            return found->second.device;
        }
        ++stat_misses;
        stat_uploaded += bytes;
        if (!make_base_room(bytes))
            throw std::runtime_error("CUDA weight-cache limit is below the pinned/active working set");
        void* device = device_alloc(bytes);
        try {
            // The copy stream waits for retired readers; compute waits for
            // upload completion (one batch fence in the overlap pipeline).
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

CudaProjection::CudaProjection(size_t vram_limit_mb, bool host_kv, bool half_kv, bool fast) : impl_(std::make_unique<Impl>()) {
    impl_->fast = fast;
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
    // This 8 GiB WDDM card regressed with a larger cache: leave room for
    // allocation reuse and driver/BLAS state instead of chasing residency.
    if (total_bytes <= 8192 * MIB) impl_->cache_limit = std::min(impl_->cache_limit, 5000 * MIB);
    if (const char* policy = std::getenv("LAMINA_CACHE_POLICY")) {
        const std::string value(policy);
        if (value != "lru" && value != "lfu") throw std::invalid_argument("LAMINA_CACHE_POLICY must be lru or lfu");
        impl_->frequency_cache = value == "lfu";
    }
    impl_->host_kv = host_kv;
    impl_->half_kv = half_kv;
    // Fast computation defaults to CPU experts for cache misses: on the RTX 3050
    // test machine it measured about 50% faster than streaming every miss over
    // PCIe, with an unchanged quality gate. FP32 keeps streaming, the checked
    // reference path. LAMINA_EXPERT_POLICY overrides either default.
    impl_->cpu_misses = fast;
    if (const char* policy = std::getenv("LAMINA_EXPERT_POLICY"); policy && *policy) {
        const std::string value(policy);
        if (value != "stream" && value != "cpu-miss") throw std::invalid_argument("LAMINA_EXPERT_POLICY must be stream or cpu-miss");
        impl_->cpu_misses = value == "cpu-miss";
    }
    if (impl_->cpu_misses) {
        // The CPU experts are memory-bandwidth bound; more spinning workers only
        // slow the host thread that drives the GPU (measured: 4 of 16 threads best).
        unsigned workers = std::min(8u, std::max(1u, std::thread::hardware_concurrency() / 4));
        if (const char* configured = std::getenv("LAMINA_CPU_THREADS")) {
            const auto parsed = std::from_chars(configured, configured + std::strlen(configured), workers);
            if (parsed.ec != std::errc{} || parsed.ptr != configured + std::strlen(configured) || workers < 1 || workers > 64)
                throw std::invalid_argument("LAMINA_CPU_THREADS must be 1..64");
        }
        // Prefill batches are compute bound and use more threads. On a 16-thread
        // Ryzen 7 5700X, 12 instead of 4 cut the first token 0.87 -> 0.57 s
        // (8: 0.64 s, 16: 0.63 s) without changing decode speed.
        unsigned prefill_workers = std::max(workers, std::min(12u, std::max(1u, std::thread::hardware_concurrency() * 3 / 4)));
        if (const char* configured = std::getenv("LAMINA_PREFILL_CPU_THREADS"))
            prefill_workers = std::max(workers, unsigned(std::clamp(std::atoi(configured), 1, 64)));
        impl_->cpu_experts = std::make_unique<CpuExperts>(workers, impl_->fast, prefill_workers - workers);
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
            mib > impl_->memory_limit / MIB)
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
                        (impl_->kRouteMax + 2) * sizeof(float), cudaHostAllocMapped),
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

void CudaProjection::register_weight_ram(const uint8_t* source, size_t bytes) {
    const char* setting = std::getenv("LAMINA_HOST_REGISTER");
    if (!setting || setting[0] != '1' || impl_->registered_weights) return;
    const auto result = cudaHostRegister(const_cast<uint8_t*>(source), bytes, cudaHostRegisterPortable | cudaHostRegisterReadOnly);
    if (result == cudaSuccess) {
        impl_->registered_weights = source; impl_->registered_bytes = bytes;
        std::fprintf(stderr, "registered model RAM: %.1f MiB, direct expert DMA enabled\n", double(bytes) / MIB);
    } else {
        std::fprintf(stderr, "model RAM registration refused: %s; worker staging retained\n", cudaGetErrorString(result));
        cudaGetLastError();
    }
}


bool CudaProjection::supports(uint32_t type) const {
    return type == 8 || type == 12 || type == 13 || type == 14;
}

std::vector<float> CudaProjection::matvec(const strata::TensorInfo& tensor, const uint8_t* data,
                                          const std::vector<float>& x, int64_t expert) {
    const int n_out = project_output(tensor, data, x, expert);
    impl_->tl_mark(kTlHeadCopy);
    const auto download_start = std::chrono::steady_clock::now();
    std::vector<float> result(static_cast<size_t>(n_out));
    check(cudaMemcpyAsync(result.data(), impl_->output, result.size() * sizeof(float),
                          cudaMemcpyDeviceToHost, impl_->stream), "download projection");
    check(cudaStreamSynchronize(impl_->stream), "finish projection");
    if (impl_->timeline)
        impl_->tl_host[8] += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - download_start).count();
    return result;
}

int CudaProjection::matvec_argmax(const strata::TensorInfo& tensor, const uint8_t* data,
                                  const std::vector<float>& x, int rows) {
    const int n_out = project_output(tensor, data, x, -1, rows);
    impl_->tl_mark(kTlHeadCopy);
    if (!impl_->argmax_host) {
        check(cudaHostAlloc(reinterpret_cast<void**>(&impl_->argmax_host), 4 * sizeof(int), cudaHostAllocMapped), "allocate argmax result");
        check(cudaHostGetDevicePointer(reinterpret_cast<void**>(&impl_->argmax_dev), impl_->argmax_host, 0), "map argmax result");
    }
    cuda::argmax_f32(impl_->output, n_out, impl_->argmax_dev, impl_->stream);
    check(cudaGetLastError(), "launch argmax");
    check(cudaStreamSynchronize(impl_->stream), "finish argmax");
    return impl_->argmax_host[1] ? -1 : impl_->argmax_host[0];
}

int CudaProjection::project_output(const strata::TensorInfo& tensor, const uint8_t* data,
                                   const std::vector<float>& x, int64_t expert, int rows) {
    if (!supports(tensor.type) || tensor.shape.size() != (expert < 0 ? 2u : 3u) ||
        tensor.shape[0] != x.size())
        throw std::invalid_argument("unsupported CUDA projection: " + tensor.name);
    const int n_in = static_cast<int>(x.size());
    const int all_rows = static_cast<int>(tensor.shape[1]);
    const size_t bytes = strata::kernels::native_mmvq_weight_bytes(tensor.type, n_in, all_rows);
    // The whole matrix stays the cached unit; only the kernel stops early.
    const int n_out = rows > 0 ? std::min(rows, all_rows) : all_rows;
    if (expert >= 0 && expert >= static_cast<int64_t>(tensor.shape[2]))
        throw std::out_of_range("expert index");
    impl_->reserve(x.size(), static_cast<size_t>(n_out));
    const auto* slice = data + (expert < 0 ? 0 : static_cast<size_t>(expert) * bytes);
    void* device_weight = impl_->weight(tensor, slice, expert, bytes);
    check(cudaMemcpyAsync(impl_->input, x.data(), x.size() * sizeof(float),
                          cudaMemcpyHostToDevice, impl_->stream), "upload activation");
    impl_->tl_mark(kTlHeadKernel);
    if (impl_->fast) {
        impl_->ensure(impl_->dense_q8, impl_->dense_q8_cap, 2 * strata::kernels::native_q8_1_bytes(4096));
        strata::kernels::native_quantize_q8_1(impl_->input, impl_->dense_q8, n_in, 1, impl_->stream);
        strata::kernels::native_mmvq(tensor.type, device_weight, impl_->dense_q8, impl_->output, n_in, n_out, 1, impl_->stream);
    } else strata::kernels::native_mmvq_f32(tensor.type, device_weight, impl_->input, impl_->output,
                                     n_in, n_out, impl_->stream);
    check(cudaGetLastError(), "launch projection");
    return n_out;
}

void CudaProjection::prefill_upload(const std::vector<float>& x, int columns, const std::vector<std::array<int, 3>>& positions) {
    if (columns < 1 || columns > 2048 || x.size() != size_t(columns) * 2048 || positions.size() != size_t(columns))
        throw std::invalid_argument("invalid device prefill input");
    impl_->ensure(impl_->p_positions, impl_->p_positions_cap, size_t(columns)*3);
    check(cudaMemcpyAsync(impl_->p_positions, positions.data(), size_t(columns)*3*sizeof(int), cudaMemcpyHostToDevice, impl_->stream), "upload prefill mRoPE positions");
    impl_->ensure(impl_->p_hidden, impl_->p_hidden_cap, x.size());
    impl_->ensure(impl_->p_norm, impl_->p_norm_cap, x.size());
    impl_->ensure(impl_->p_mix, impl_->p_mix_cap, x.size());
    impl_->prefill_columns = columns;
    check(cudaMemcpyAsync(impl_->p_hidden, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice, impl_->stream), "upload prefill hidden");
}

void CudaProjection::prefill_layer(int layer) {
    // Cached decode graphs capture dense addresses. Complete their readers
    // before making previous layers eligible for eviction, then invalidate.
    check(cudaStreamSynchronize(impl_->stream), "retire previous prefill layer");
    for (auto* graphs : {&impl_->mixer_graphs, &impl_->pair_graphs})
        for (auto& graph : *graphs) {
            if (graph.exec) check(cudaGraphExecDestroy(graph.exec), "invalidate dense graph");
            graph = {};
        }
    const std::string current = "blk." + std::to_string(layer) + ".";
    for (auto& [key, entry] : impl_->weights) {
        if (key.rfind("blk.", 0) != 0 || key.find("#expert=-1") == std::string::npos) continue;
        const bool pinned = key.rfind(current, 0) == 0;
        if (entry.pinned && !pinned) {
            entry.lru = impl_->lru.insert(impl_->lru.end(), key);
            entry.pinned = false;
        } else if (!entry.pinned && pinned) {
            impl_->lru.erase(entry.lru);
            entry.pinned = true;
        }
    }
}

void CudaProjection::prefill_long_upload(const std::vector<float>& x, const std::vector<std::array<int, 3>>& positions) {
    if (positions.empty() || positions.size() > size_t(impl_->max_context) || x.size() != positions.size()*2048)
        throw std::invalid_argument("invalid long prefill input");
    impl_->ensure(impl_->long_hidden,impl_->long_hidden_cap,x.size());
    impl_->ensure(impl_->long_positions,impl_->long_positions_cap,positions.size()*3);
    impl_->long_columns=int(positions.size());
    check(cudaMemcpyAsync(impl_->long_hidden,x.data(),x.size()*sizeof(float),cudaMemcpyHostToDevice,impl_->stream),"upload long residual");
    check(cudaMemcpyAsync(impl_->long_positions,positions.data(),positions.size()*3*sizeof(int),cudaMemcpyHostToDevice,impl_->stream),"upload long positions");
}
void CudaProjection::prefill_long_tile(int start,int columns) {
    if(start<0 || columns<1 || columns>2048 || start>impl_->long_columns-columns)
        throw std::invalid_argument("invalid long prefill tile");
    const size_t elements=size_t(columns)*2048;
    impl_->ensure(impl_->p_hidden,impl_->p_hidden_cap,elements);
    impl_->ensure(impl_->p_norm,impl_->p_norm_cap,elements);
    impl_->ensure(impl_->p_mix,impl_->p_mix_cap,elements);
    impl_->ensure(impl_->p_positions,impl_->p_positions_cap,size_t(columns)*3);
    impl_->prefill_columns=columns;
    check(cudaMemcpyAsync(impl_->p_hidden,impl_->long_hidden+size_t(start)*2048,elements*sizeof(float),cudaMemcpyDeviceToDevice,impl_->stream),"select long residual tile");
    check(cudaMemcpyAsync(impl_->p_positions,impl_->long_positions+size_t(start)*3,size_t(columns)*3*sizeof(int),cudaMemcpyDeviceToDevice,impl_->stream),"select long position tile");
}
void CudaProjection::prefill_long_store(int start) {
    if(start<0 || !impl_->prefill_columns || start>impl_->long_columns-impl_->prefill_columns)
        throw std::invalid_argument("invalid long prefill store");
    check(cudaMemcpyAsync(impl_->long_hidden+size_t(start)*2048,impl_->p_hidden,size_t(impl_->prefill_columns)*2048*sizeof(float),cudaMemcpyDeviceToDevice,impl_->stream),"store long residual tile");
}
std::vector<float> CudaProjection::prefill_long_download() {
    if(!impl_->long_columns)throw std::logic_error("long prefill not active");
    std::vector<float> hidden(2048);
    check(cudaMemcpyAsync(hidden.data(),impl_->long_hidden+size_t(impl_->long_columns-1)*2048,hidden.size()*sizeof(float),cudaMemcpyDeviceToHost,impl_->stream),"download long residual");
    check(cudaStreamSynchronize(impl_->stream),"finish long prefill");
    if(device_profile())impl_->collect_times();
    return hidden;
}

void CudaProjection::prefill_rms(const strata::TensorInfo& gamma, const uint8_t* data, float epsilon) {
    if (!impl_->prefill_columns || gamma.type != 0 || gamma.shape != std::vector<uint64_t>{2048})
        throw std::invalid_argument("invalid device prefill norm");
    const size_t bytes = size_t(impl_->prefill_columns) * 2048 * sizeof(float);
    auto* w = static_cast<const float*>(impl_->weight(gamma, data, -1, 2048 * sizeof(float)));
    check(cudaMemcpyAsync(impl_->p_norm, impl_->p_hidden, bytes, cudaMemcpyDeviceToDevice, impl_->stream), "copy prefill residual");
    cuda::rms_norm_columns(impl_->p_norm, w, 2048, impl_->prefill_columns, epsilon, impl_->stream);
}

void CudaProjection::prefill_add_mix() {
    if (!impl_->prefill_columns) throw std::logic_error("device prefill not active");
    cuda::add_inplace(impl_->p_hidden, impl_->p_mix, impl_->prefill_columns * 2048, impl_->stream);
}

std::vector<float> CudaProjection::prefill_download() {
    if (!impl_->prefill_columns) throw std::logic_error("device prefill not active");
    std::vector<float> result(2048);
    check(cudaMemcpyAsync(result.data(), impl_->p_hidden + size_t(impl_->prefill_columns - 1) * 2048,
        result.size() * sizeof(float), cudaMemcpyDeviceToHost, impl_->stream), "download prefill last row");
    check(cudaStreamSynchronize(impl_->stream), "finish device prefill");
    if (device_profile()) impl_->collect_times();
    impl_->prefill_columns = 0;
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
    const bool use_blas = columns >= (impl_->fast && t.type != 0 ? 8 : 16) && (!configured || configured[0] != '0');
    if (use_blas) {
        impl_->ensure_blas();
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
        if (impl_->fast && t.type != 0) {
            impl_->ensure(impl_->blas_half_weights, impl_->blas_half_weights_cap, size_t(in) * out * 2);
            impl_->ensure(impl_->blas_half_input, impl_->blas_half_input_cap, size_t(in) * columns * 2);
            cuda::to_bf16(matrix, impl_->blas_half_weights, in * out, impl_->stream);
            cuda::to_bf16(x_dev, impl_->blas_half_input, in * columns, impl_->stream);
            check_blas(cublasGemmEx(impl_->blas, CUBLAS_OP_T, CUBLAS_OP_N, out, columns, in,
                &one, impl_->blas_half_weights, CUDA_R_16BF, in, impl_->blas_half_input, CUDA_R_16BF, in, &zero,
                y_dev, CUDA_R_32F, out, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT), "project BF16 prefill columns");
            return;
        }
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

    impl_->tl_mark(kTlGdnProject);
    // qkv and attn_gate share the activation; one grouped MMVQ dispatch.
    const void* qkv_gate_weights[2] = {impl_->projection_weight(*w.qkv, w.qkv_data, -1),
                                       impl_->projection_weight(*w.gate, w.gate_data, -1)};
    const float* qkv_gate_inputs[2] = {x_dev, x_dev};
    float* qkv_gate_outputs[2] = {impl_->g_qkv, impl_->g_z};
    const int qkv_gate_rows[2] = {kChannels, 4096};
    impl_->decode_group(static_cast<int>(w.qkv->type), qkv_gate_weights,
                                             qkv_gate_inputs, qkv_gate_outputs, qkv_gate_rows, 2,
                                             kChannels + 4096, kHidden, impl_->stream);
    impl_->tl_mark(kTlGdnSmall);
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
    impl_->tl_mark(kTlGdnStep);
    strata::kernels::native_gdn_step(state.rec, impl_->g_conv_silu, impl_->g_conv_silu + 2048,
                                     impl_->g_conv_silu + 4096, impl_->g_gate, impl_->g_beta,
                                     impl_->g_outnorm, shapes, impl_->stream);
    impl_->tl_mark(kTlGdnOut);
    // Closing RMS norm with gamma and SiLU(z), then the ssm_out projection.
    void* norm_weight = impl_->weight(*w.norm, w.norm_data, -1,
                                      static_cast<size_t>(kS) * sizeof(float));
    cuda::gdn_out_norm_silu(impl_->g_outnorm, impl_->g_z, static_cast<const float*>(norm_weight),
                            impl_->g_y, kHeads, kEps, impl_->stream);
    const void* ssm_out_weight = impl_->projection_weight(*w.out, w.out_data, -1);
    const float* ssm_out_input = impl_->g_y;
    float* ssm_out_output = out_dev;
    const int ssm_out_rows = kHidden;
    impl_->decode_group(static_cast<int>(w.out->type), &ssm_out_weight,
                                             &ssm_out_input, &ssm_out_output, &ssm_out_rows, 1,
                                             kHidden, 4096, impl_->stream);
    check(cudaGetLastError(), "launch DeltaNet");
}

std::vector<float> CudaProjection::delta_net_columns(int layer, const std::vector<float>& x,
                                                     int columns, const GdnWeights& w) {
    if (!supports_gdn(w)) throw std::invalid_argument("unsupported DeltaNet prefill");
    const bool device = x.empty() && impl_->prefill_columns == columns;
    if (!device && x.size() != size_t(columns) * 2048) throw std::invalid_argument("invalid GDN columns");
    if (!device) impl_->ensure(impl_->input, impl_->input_capacity, x.size());
    const float* input = device ? impl_->p_norm : impl_->input;
    if (!device) check(cudaMemcpyAsync(impl_->input, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice, impl_->stream), "upload GDN columns");
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
    impl_->ensure(impl_->b_g_qkv, impl_->b_g_qkv_cap, size_t(columns) * 8192);
    impl_->ensure(impl_->b_g_z, impl_->b_g_z_cap, size_t(columns) * 4096);
    impl_->ensure(impl_->b_g_alpha, impl_->b_g_alpha_cap, size_t(columns) * 32);
    impl_->ensure(impl_->b_g_beta, impl_->b_g_beta_cap, size_t(columns) * 32);
    impl_->ensure(impl_->b_g_gate, impl_->b_g_gate_cap, size_t(columns) * 32);
    impl_->ensure(impl_->b_g_conv, impl_->b_g_conv_cap, size_t(columns) * 8192);
    impl_->ensure(impl_->batch_gdn, impl_->batch_gdn_cap, size_t(columns) * 4096);
    impl_->ensure(impl_->b_g_y, impl_->b_g_y_cap, size_t(columns) * 4096);
    project_columns_device(*w.qkv, w.qkv_data, input, impl_->b_g_qkv, columns, -1);
    project_columns_device(*w.gate, w.gate_data, input, impl_->b_g_z, columns, -1);
    project_columns_device(*w.alpha, w.alpha_data, input, impl_->b_g_alpha, columns, -1);
    project_columns_device(*w.beta, w.beta_data, input, impl_->b_g_beta, columns, -1);
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
    if (!device) impl_->ensure(impl_->output, impl_->output_capacity, size_t(columns) * 2048);
    project_columns_device(*w.out, w.out_data, impl_->b_g_y, device ? impl_->p_mix : impl_->output, columns, -1);
    if (device) return {};
    std::vector<float> result(size_t(columns) * 2048);
    check(cudaMemcpyAsync(result.data(), impl_->output, result.size() * sizeof(float), cudaMemcpyDeviceToHost, impl_->stream), "download GDN columns");
    check(cudaStreamSynchronize(impl_->stream), "finish GDN columns");
    return result;
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

// Strata's hit/miss split. Resident experts run while DMA brings misses from
// RAM; two pointer-table graphs preserve original expert slots and FP32 sums.
void CudaProjection::moe_pipeline(const float* x, float* output, const MoeWeights& routed,
    const std::vector<int>& ids, const std::vector<float>& scales, const MoeWeights& shared, float shared_scale) {
    impl_->expert_layout(routed);
    static const bool persistent = env_is("LAMINA_Q8_PERSISTENT", '1');
    const bool compact=impl_->fast && persistent;
    const int hidden = 2048, ff = int(routed.gate->shape[1]), slots = int(ids.size()) + 1;
    impl_->ensure(impl_->gate_dev, impl_->gate_capacity, size_t(slots) * ff);
    impl_->ensure(impl_->up_dev, impl_->up_capacity, size_t(slots) * ff);
    impl_->ensure(impl_->swiglu_dev, impl_->swiglu_capacity, size_t(slots) * ff);
    impl_->ensure(impl_->down_dev, impl_->down_capacity, size_t(slots) * hidden);
    impl_->ensure(impl_->scales_dev, impl_->scales_capacity, slots);
    const size_t iq = strata::kernels::native_q8_1_bytes(hidden), dq = strata::kernels::native_q8_1_bytes(ff);
    if (impl_->fast) impl_->ensure(impl_->expert_q8, impl_->expert_q8_cap, iq + size_t(slots) * dq);
    if (!impl_->scales_host) check(cudaMallocHost(reinterpret_cast<void**>(&impl_->scales_host), 33 * sizeof(float)), "allocate scales");
    std::copy(scales.begin(), scales.end(), impl_->scales_host); impl_->scales_host[slots-1] = shared_scale;
    impl_->tl_mark(kTlMoeSetup);
    check(cudaMemcpyAsync(impl_->scales_dev, impl_->scales_host, size_t(slots) * sizeof(float), cudaMemcpyHostToDevice, impl_->stream), "publish expert scales");
    const size_t table_bytes = 8 * sizeof(strata::kernels::NativeF32Grouped);
    if (!impl_->expert_table_host) {
        check(cudaMallocHost(reinterpret_cast<void**>(&impl_->expert_table_host), table_bytes), "allocate pipeline tables");
        impl_->allocate(reinterpret_cast<void**>(&impl_->expert_table_gpu), table_bytes);
    }
    if (!impl_->expert_epoch) {
        check(cudaEventCreateWithFlags(&impl_->expert_epoch, cudaEventDisableTiming), "create expert epoch");
        check(cudaEventCreateWithFlags(&impl_->expert_ready, cudaEventDisableTiming), "create expert ready");
    }
    impl_->lease_active = true; impl_->leases.clear();
    std::vector<bool> resident(slots);
    for (int slot = 0; slot < slots; ++slot) {
        const auto& w = slot < slots-1 ? routed : shared;
        const int expert = slot < slots-1 ? ids[slot] : -1;
        resident[slot] = true;
        for (const auto* t : {w.gate, w.up, w.down}) {
            const auto key = t->name + "#type=" + std::to_string(t->type) + "#expert=" + std::to_string(expert);
            const auto entry = impl_->weights.find(key);
            resident[slot] = resident[slot] && entry != impl_->weights.end();
            if (entry != impl_->weights.end()) impl_->leases.insert(entry->second.device);
        }
    }
    const bool misses = std::find(resident.begin(), resident.end(), false) != resident.end();
    check(cudaEventRecord(impl_->expert_epoch, impl_->stream), "record previous readers");
    impl_->expert_pipeline_running = true; impl_->pipeline_copy_fenced = false;
    struct Guard { Impl* p; ~Guard() { p->expert_pipeline_running = false; p->lease_active = false; p->leases.clear(); } } guard{impl_.get()};
    const auto fill = [&](int phase) {
        for (int i = 0; i < 4; ++i) {
            auto& table = impl_->expert_table_host[phase*4+i]; table = {};
            const bool down = i >= 2, common = i % 2 != 0;
            const int count = common ? 1 : slots - 1;
            table.clear_missing = phase == 0;
            const int matrices=count*(down ? 1 : 2);
            table.count=compact ? 0 : matrices;
            for(int row=0;row<matrices;++row) {
                const int slot = common ? slots-1 : down ? row : row / 2;
                if(compact && resident[slot]!=(phase==0))continue;
                const int index=compact ? table.count++ : row;
                const auto& w = common ? shared : routed;
                const int expert = common ? -1 : ids[slot];
                const auto* tensor = down ? w.down : row % 2 ? w.up : w.gate;
                const auto* data = down ? w.down_data : row % 2 ? w.up_data : w.gate_data;
                table.n_outs[index] = down ? hidden : ff;
                table.outputs[index] = down ? impl_->down_dev + size_t(slot) * hidden : (row % 2 ? impl_->up_dev : impl_->gate_dev) + size_t(slot) * ff;
                table.inputs[index] = down ? (impl_->fast ? reinterpret_cast<const float*>(impl_->expert_q8 + iq + size_t(slot) * dq) : impl_->swiglu_dev + size_t(slot) * ff) :
                    (impl_->fast ? reinterpret_cast<const float*>(impl_->expert_q8) : x);
                if (resident[slot] == (phase == 0)) table.weights[index] = impl_->projection_weight(*tensor, data, expert);
            }
        }
        check(cudaMemcpyAsync(impl_->expert_table_gpu + phase * 4, impl_->expert_table_host + phase * 4,
            table_bytes / 2, cudaMemcpyHostToDevice, impl_->stream), "publish pipeline tables");
    };
    const auto execute = [&](int phase) {
        const auto launch = [&](int i, int type, int width) {
            const int count = i % 2 ? 1 : slots-1;
            const int matrices = i < 2 ? count * 2 : count;
            const int rows = count * (i < 2 ? ff * 2 : hidden);
            const auto* table = impl_->expert_table_gpu + phase * 4 + i;
            if (impl_->fast) strata::kernels::native_mmvq_q8_grouped_table(type, table, matrices, rows, width, impl_->stream);
            else strata::kernels::native_mmvq_f32_grouped_table(type, table, matrices, rows, width, impl_->stream);
        };
        const auto body = [&] {
            if(compact && !phase) {
                check(cudaMemsetAsync(impl_->gate_dev,0,size_t(slots)*ff*sizeof(float),impl_->stream),"clear expert gate slots");
                check(cudaMemsetAsync(impl_->up_dev,0,size_t(slots)*ff*sizeof(float),impl_->stream),"clear expert up slots");
                check(cudaMemsetAsync(impl_->down_dev,0,size_t(slots)*hidden*sizeof(float),impl_->stream),"clear expert down slots");
            }
            if (impl_->fast && (!compact || !phase)) strata::kernels::native_quantize_q8_1(x, impl_->expert_q8, hidden, 1, impl_->stream);
            launch(0, routed.gate->type, hidden); launch(1, shared.gate->type, hidden);
            cuda::swiglu(impl_->gate_dev, impl_->up_dev, impl_->swiglu_dev, slots * ff, impl_->stream);
            if (impl_->fast) for (int base = 0; base < slots; base += 8)
                strata::kernels::native_quantize_q8_1(impl_->swiglu_dev + size_t(base) * ff,
                    impl_->expert_q8 + iq + size_t(base) * dq, ff, std::min(8, slots-base), impl_->stream);
            launch(2, routed.down->type, ff); launch(3, shared.down->type, ff);
        };
        std::vector<uintptr_t> identity{reinterpret_cast<uintptr_t>(routed.gate), reinterpret_cast<uintptr_t>(shared.gate),
            reinterpret_cast<uintptr_t>(x), reinterpret_cast<uintptr_t>(impl_->gate_dev), reinterpret_cast<uintptr_t>(impl_->up_dev),
            reinterpret_cast<uintptr_t>(impl_->down_dev), reinterpret_cast<uintptr_t>(impl_->swiglu_dev),
            reinterpret_cast<uintptr_t>(impl_->expert_q8), reinterpret_cast<uintptr_t>(impl_->expert_table_gpu), uintptr_t(slots), uintptr_t(phase), uintptr_t(compact)};
        const std::string key(reinterpret_cast<const char*>(identity.data()), identity.size() * sizeof(uintptr_t));
        const auto timed = impl_->begin_time(impl_->stream, 1);
        auto it = impl_->moe_graphs.find(key);
        if (it == impl_->moe_graphs.end()) {
            if (impl_->moe_graphs.size() >= 128) {
                auto oldest = impl_->moe_graphs.begin();
                for (auto q = impl_->moe_graphs.begin(); q != impl_->moe_graphs.end(); ++q) if (q->second.used < oldest->second.used) oldest = q;
                if (oldest->second.exec) check(cudaGraphExecDestroy(oldest->second.exec), "release pipeline graph");
                impl_->moe_graphs.erase(oldest);
            }
            ++impl_->moe_graph_misses;
            impl_->moe_graphs.emplace(key, Impl::MoeGraph{nullptr, ++impl_->clock}); body();
        } else {
            auto& graph = it->second; graph.used = ++impl_->clock;
            if (!graph.exec) {
                check(cudaStreamSynchronize(impl_->stream), "flush pipeline capture");
                check(cudaStreamBeginCapture(impl_->stream, cudaStreamCaptureModeThreadLocal), "capture pipeline phase");
                body(); cudaGraph_t captured = nullptr;
                check(cudaStreamEndCapture(impl_->stream, &captured), "end pipeline phase");
                const auto result = cudaGraphInstantiate(&graph.exec, captured, nullptr, nullptr, 0);
                cudaGraphDestroy(captured); check(result, "instantiate pipeline phase");
            }
            check(cudaGraphLaunch(graph.exec, impl_->stream), "replay pipeline phase"); ++impl_->moe_graph_hits;
        }
        impl_->end_time(timed, impl_->stream);
    };
    // Experts uploaded by the previous layer's prefetch count as resident above,
    // so the resident phase must not start before their copies finish.
    impl_->tl_mark(kTlPrefetchWait);
    impl_->drain_prefetch();
    fill(0);
    impl_->tl_mark(kTlResident);
    execute(0);
    auto status = cudaStreamQuery(impl_->stream);
    if (status != cudaSuccess && status != cudaErrorNotReady) check(status, "submit resident experts");
    if (misses) {
        impl_->tl_mark(kTlMissWait);
        for (int i = 0; i < 4; ++i) {
            const bool common = i % 2 != 0, down = i >= 2;
            for (int slot = common ? slots-1 : 0; slot < (common ? slots : slots-1); ++slot) {
                if (resident[slot]) continue;
                const auto& w = common ? shared : routed; const int e = common ? -1 : ids[slot];
                if (down) impl_->prepare(*w.down, w.down_data, e);
                else { impl_->prepare(*w.gate, w.gate_data, e); impl_->prepare(*w.up, w.up_data, e); }
            }
        }
        fill(1);
        if (impl_->weight_copy) {
            check(cudaEventRecord(impl_->expert_ready, impl_->weight_copy), "record expert batch ready");
            check(cudaStreamWaitEvent(impl_->stream, impl_->expert_ready, 0), "wait missing experts");
        }
        impl_->tl_mark(kTlMissExperts);
        execute(1);
    }
    impl_->tl_mark(kTlPrefetchIssue);
    if (impl_->has_prefetch && impl_->weight_copy) {
        // Queue the predicted next-layer experts behind this layer's own misses.
        // Leases are still active, so nothing this layer reads can be evicted.
        const auto& next = impl_->prefetch_target;
        impl_->expert_layout(next);
        const auto hits_before = impl_->stat_hits, misses_before = impl_->stat_misses;
        int issued = 0;
        for (const int e : impl_->prefetch_pred) {
            if (e < 0 || e >= int(next.gate->shape.at(2))) continue;
            const bool present = impl_->resident(*next.gate, e) && impl_->resident(*next.up, e) && impl_->resident(*next.down, e);
            impl_->projection_weight(*next.gate, next.gate_data, e);
            impl_->projection_weight(*next.up, next.up_data, e);
            impl_->projection_weight(*next.down, next.down_data, e);
            if (!present) ++issued;
        }
        impl_->stat_hits = hits_before; impl_->stat_misses = misses_before;  // prefetch must not skew the hit rate
        if (issued) {
            if (!impl_->prefetch_ready)
                check(cudaEventCreateWithFlags(&impl_->prefetch_ready, cudaEventDisableTiming), "create prefetch event");
            check(cudaEventRecord(impl_->prefetch_ready, impl_->weight_copy), "record prefetch ready");
            impl_->prefetch_pending = true;
            impl_->stat_pf_uploaded += issued;
        }
    }
    impl_->tl_mark(kTlCombine);
    cuda::moe_combine(impl_->down_dev, impl_->scales_dev, slots, hidden, output, impl_->stream);
    check(cudaGetLastError(), "launch expert pipeline");
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
    static const bool no_pipeline = env_is("LAMINA_EXPERT_PIPELINE", '0'), no_graphs = env_is("LAMINA_MOE_GRAPHS", '0');
    if (!impl_->cpu_misses && experts.size() <= 8 && !no_pipeline && !no_graphs) {
        moe_pipeline(x_dev, out_dev, routed, experts, weights, shared, shared_weight);
        return;
    }
    impl_->expert_layout(routed);
    const int n_in = static_cast<int>(routed.gate->shape[0]);
    const int ff = static_cast<int>(routed.gate->shape[1]);
    const int hidden = static_cast<int>(routed.down->shape[1]);
    const size_t slots = experts.size() + 1;  // routed experts plus the shared expert

    using HostClock = std::chrono::steady_clock;
    const auto host_ms = [](HostClock::time_point a, HostClock::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };
    const auto h_begin = HostClock::now();
    // Non-resident routed experts run on the CPU pool. Start them before any GPU
    // setup so the CPU and the GPU's resident experts overlap as much as possible.
    std::vector<bool> on_cpu(slots, false);
    std::vector<size_t> cpu_slots;
    if (impl_->cpu_misses && impl_->stat_evicted) {
        for (size_t s = 0; s < experts.size(); ++s)
            if (!(impl_->resident(*routed.gate, experts[s]) && impl_->resident(*routed.up, experts[s]) &&
                  impl_->resident(*routed.down, experts[s]))) {
                on_cpu[s] = true; cpu_slots.push_back(s);
            }
    }
    const bool published = impl_->cpu_input_published && x_dev == impl_->norm_dev;
    impl_->cpu_input_published = false;
    // moe_into_mix already computed the shared expert into its slot.
    const bool shared_early = impl_->shared_early && x_dev == impl_->norm_dev;
    impl_->shared_early = false;
    struct CpuBatchGuard {
        CpuExperts* pool = nullptr;
        ~CpuBatchGuard() { if (pool) try { pool->wait(); } catch (...) {} }
    } cpu_batch;
    if (!cpu_slots.empty()) {
        impl_->mapped_floats(impl_->cpu_down_host, impl_->cpu_down_dev, impl_->cpu_down_cap, slots * size_t(hidden));
        if (impl_->cpu_down_copied) check(cudaEventSynchronize(impl_->cpu_down_copied), "reuse CPU expert staging");
        if (!published) {
            impl_->mapped_floats(impl_->cpu_input_host, impl_->cpu_input_dev, impl_->cpu_input_cap, size_t(n_in));
            check(cudaMemcpyAsync(impl_->cpu_input_host, x_dev, size_t(n_in) * sizeof(float), cudaMemcpyDeviceToHost, impl_->stream), "publish CPU expert input");
            check(cudaStreamSynchronize(impl_->stream), "finish CPU input");
        }
        std::vector<int> ids;
        std::vector<float*> outputs;
        for (const size_t s : cpu_slots) { ids.push_back(experts[s]); outputs.push_back(impl_->cpu_down_host + s * hidden); }
        impl_->cpu_experts->start(routed, ids, impl_->cpu_input_host, outputs);
        cpu_batch.pool = impl_->cpu_experts.get();
        impl_->stat_cpu_experts += cpu_slots.size();
    }
    if (impl_->timeline) {
        auto [it, inserted] = impl_->tl_cpu_by_layer.try_emplace(routed.gate, 0);
        if (inserted) impl_->tl_layer_order.push_back(routed.gate);
        it->second += cpu_slots.size();
    }
    const auto h_started = HostClock::now();

    const size_t input_q8_bytes = strata::kernels::native_q8_1_bytes(n_in);
    const size_t down_q8_bytes = strata::kernels::native_q8_1_bytes(ff);
    if (impl_->fast) impl_->ensure(impl_->expert_q8, impl_->expert_q8_cap, input_q8_bytes + slots * down_q8_bytes);

    impl_->ensure(impl_->accum_dev, impl_->accum_capacity, static_cast<size_t>(hidden));
    impl_->ensure(impl_->gate_dev, impl_->gate_capacity, slots * ff);
    impl_->ensure(impl_->up_dev, impl_->up_capacity, slots * ff);
    impl_->ensure(impl_->swiglu_dev, impl_->swiglu_capacity, slots * ff);
    impl_->ensure(impl_->down_dev, impl_->down_capacity, slots * hidden);
    impl_->ensure(impl_->scales_dev, impl_->scales_capacity, slots);

    const auto t_scales = std::chrono::steady_clock::now();
    impl_->tl_mark(kTlMoeSetup);
    std::vector<float> scales(experts.size());
    for (size_t i = 0; i < experts.size(); ++i) scales[i] = weights[i];
    scales.push_back(shared_weight);
    if (!impl_->scales_host)
        check(cudaMallocHost(reinterpret_cast<void**>(&impl_->scales_host), 65 * sizeof(float)), "allocate expert scales staging");
    std::copy(scales.begin(), scales.end(), impl_->scales_host);
    // With the graph path, the scales travel with the pointer tables as kernel
    // arguments (one launch instead of two small copies, cheaper under WDDM).
    const bool publish_args = !no_graphs && 2 * experts.size() <= size_t(cuda::kPublishItems) &&
                              slots <= size_t(cuda::kPublishScales);
    // The device router doorbell (or the blocking public moe call) completed
    // the preceding expert block before this persistent staging is reused.
    if (!publish_args)
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
        if (impl_->fast) {
            strata::kernels::NativeF32Grouped args{};
            args.count = int(n);
            for (size_t i = 0; i < n; ++i) {
                args.weights[i] = weights[i]; args.outputs[i] = outputs[i]; args.n_outs[i] = rows[i];
                const bool down = batch_n_in == ff;
                const size_t slot = down ? size_t(inputs[i] - impl_->swiglu_dev) / ff : 0;
                args.inputs[i] = reinterpret_cast<const float*>(impl_->expert_q8 + (down ? input_q8_bytes + slot * down_q8_bytes : 0));
            }
            strata::kernels::native_mmvq_q8_grouped(type, args, total_rows, batch_n_in, impl_->stream);
        } else strata::kernels::native_mmvq_f32_grouped(type, weights.data(), inputs.data(),
                                                 outputs.data(), rows.data(), static_cast<int>(n),
                                                 total_rows, batch_n_in, impl_->stream);
    };
    std::vector<GroupItem> gate_up_routed, gate_up_shared, down_routed, down_shared;
    const auto h_leases = HostClock::now();
    impl_->lease_active = true;
    for (size_t slot = 0; slot < slots; ++slot) {
        const auto& w = moe_weights(slot);
        const auto expert = expert_id(slot);
        for (auto pair : {std::pair{w.gate, w.gate_data}, std::pair{w.up, w.up_data}, std::pair{w.down, w.down_data}}) {
            const auto key = pair.first->name + "#type=" + std::to_string(pair.first->type) + "#expert=" + std::to_string(expert);
            if (auto found = impl_->weights.find(key); found != impl_->weights.end()) impl_->leases.insert(found->second.device);
            else if (!impl_->cpu_misses) impl_->prepare(*pair.first, pair.second, expert);
        }
    }
    for (size_t s = 0; s < slots; ++s) {
        const MoeWeights& w = moe_weights(s);
        const int64_t expert = expert_id(s);
        const bool is_routed = s < experts.size();
        // A CPU slot's gate/up/SwiGLU scratch is never read: no down projection is
        // launched for it and its down row is overwritten by the CPU result.
        if (on_cpu[s]) continue;
        if (shared_early && !is_routed) continue;  // its gate/up/down outputs already exist
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
    const auto h_slots_done = HostClock::now();
    const bool use_tables = !no_graphs;
    // With CPU slots the combine must follow their results, so it stays outside the graph.
    const bool combine_in_graph = cpu_slots.empty();
    if (use_tables) {
        const size_t bytes = 4 * sizeof(strata::kernels::NativeF32Grouped);
        if (!impl_->expert_table_host) {
            check(cudaMallocHost(reinterpret_cast<void**>(&impl_->expert_table_host), bytes), "allocate expert pointer staging");
            impl_->allocate(reinterpret_cast<void**>(&impl_->expert_table_gpu), bytes);
        }
        // The same entries either way: each group's weight, input (the Q8 view in
        // fast mode), output and row count.
        const auto entry_input = [&](int index, const GroupItem& item) -> const float* {
            if (!impl_->fast) return item.input;
            const bool down = index >= 2;
            const size_t slot = down ? size_t(item.input - impl_->swiglu_dev) / ff : 0;
            return reinterpret_cast<const float*>(impl_->expert_q8 + (down ? input_q8_bytes + slot * down_q8_bytes : 0));
        };
        const std::array<const std::vector<GroupItem>*, 4> groups{&gate_up_routed, &gate_up_shared, &down_routed, &down_shared};
        if (publish_args) {
            cuda::ExpertPublish args{};
            for (int index = 0; index < 4; ++index) {
                const auto& group = *groups[size_t(index)];
                args.counts[index] = int(group.size());
                for (size_t i = 0; i < group.size(); ++i) {
                    args.weights[index][i] = group[i].weight;
                    args.inputs[index][i] = entry_input(index, group[i]);
                    args.outputs[index][i] = group[i].output;
                    args.n_outs[index][i] = group[i].rows;
                }
            }
            for (size_t i = 0; i < slots; ++i) args.scales[i] = scales[i];
            args.scale_count = int(slots);
            cuda::publish_expert_tables(args, impl_->expert_table_gpu, impl_->scales_dev, impl_->stream);
        } else {
            for (int index = 0; index < 4; ++index) {
                const auto& group = *groups[size_t(index)];
                auto& table = impl_->expert_table_host[index];
                table = {};
                table.count = int(group.size());
                for (size_t i = 0; i < group.size(); ++i) {
                    table.weights[i] = group[i].weight; table.inputs[i] = entry_input(index, group[i]);
                    table.outputs[i] = group[i].output; table.n_outs[i] = group[i].rows;
                }
            }
            check(cudaMemcpyAsync(impl_->expert_table_gpu, impl_->expert_table_host, bytes,
                                  cudaMemcpyHostToDevice, impl_->stream), "publish expert pointer tables");
        }
    }
    const auto h_tables = HostClock::now();
    const auto launch_table = [&](int index, int type, int width, const std::vector<GroupItem>& items, cudaStream_t on) {
        if (items.empty()) return;  // every routed expert of this layer runs on the CPU
        int rows = 0;
        for (const auto& item : items) rows += item.rows;
        if (impl_->fast) strata::kernels::native_mmvq_q8_grouped_table(type, impl_->expert_table_gpu + index,
                                                      int(items.size()), rows, width, on);
        else strata::kernels::native_mmvq_f32_grouped_table(type, impl_->expert_table_gpu + index,
                                                      int(items.size()), rows, width, on);
    };
    // Running the shared expert concurrently on a second stream measured about
    // 1% slower: the routed launches already fill every SM.
    const auto launch_pair = [&](int routed_index, int routed_type, int shared_index, int shared_type, int width,
                                 const std::vector<GroupItem>& routed_items, const std::vector<GroupItem>& shared_items) {
        launch_table(routed_index, routed_type, width, routed_items, impl_->stream);
        launch_table(shared_index, shared_type, width, shared_items, impl_->stream);
    };
    const int down_n_in = static_cast<int>(routed.down->shape[0]);
    const auto gpu_experts = [&] {
        if (impl_->fast) strata::kernels::native_quantize_q8_1(x_dev, impl_->expert_q8, n_in, 1, impl_->stream);
        if (use_tables) {
            launch_pair(0, routed.gate->type, 1, shared.gate->type, n_in, gate_up_routed, gate_up_shared);
        } else {
            launch_group(static_cast<int>(routed.gate->type), n_in, gate_up_routed);
            launch_group(static_cast<int>(shared.gate->type), n_in, gate_up_shared);
        }
        cuda::swiglu(impl_->gate_dev, impl_->up_dev, impl_->swiglu_dev,
                     static_cast<int>(slots) * ff, impl_->stream);
        if (impl_->fast) for (size_t base = 0; base < slots; base += 8)
            strata::kernels::native_quantize_q8_1(impl_->swiglu_dev + base * ff, impl_->expert_q8 + input_q8_bytes + base * down_q8_bytes, ff, int(std::min(size_t(8), slots - base)), impl_->stream);
        if (use_tables) {
            launch_pair(2, routed.down->type, 3, shared.down->type, down_n_in, down_routed, down_shared);
        } else {
            launch_group(static_cast<int>(routed.down->type), down_n_in, down_routed);
            launch_group(static_cast<int>(shared.down->type), down_n_in, down_shared);
        }
        if (combine_in_graph)
            cuda::moe_combine(impl_->down_dev, impl_->scales_dev, static_cast<int>(slots), hidden, out_dev, impl_->stream);
    };
    impl_->tl_mark(kTlResident);
    if (impl_->timeline) {
        impl_->tl_host[0] += host_ms(h_begin, h_started);
        impl_->tl_host[1] += host_ms(h_started, HostClock::now());
        impl_->tl_host[4] += host_ms(h_started, h_leases);       // ensures, scales upload
        impl_->tl_host[5] += host_ms(h_leases, h_slots_done);    // lease and slot loops
        impl_->tl_host[6] += host_ms(h_slots_done, h_tables);    // table fill and upload
    }
    const auto expert_timing = impl_->begin_time(impl_->stream, 1);
    if (use_tables) {
        // Graphs reference stable device pointer tables, refreshed on the stream
        // before each replay. Cache identity covers tensor layouts and scratch;
        // selected expert weight addresses are intentionally dynamic.
        std::vector<uintptr_t> addresses{reinterpret_cast<uintptr_t>(impl_->expert_q8), reinterpret_cast<uintptr_t>(routed.gate),
            reinterpret_cast<uintptr_t>(shared.gate), reinterpret_cast<uintptr_t>(x_dev),
            reinterpret_cast<uintptr_t>(out_dev), reinterpret_cast<uintptr_t>(impl_->gate_dev),
            reinterpret_cast<uintptr_t>(impl_->up_dev), reinterpret_cast<uintptr_t>(impl_->swiglu_dev),
            reinterpret_cast<uintptr_t>(impl_->down_dev), reinterpret_cast<uintptr_t>(impl_->scales_dev),
            reinterpret_cast<uintptr_t>(impl_->expert_table_gpu), slots, gate_up_routed.size(), uintptr_t(combine_in_graph),
            uintptr_t(shared_early)};
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
        impl_->end_time(expert_timing, impl_->stream);
        if (combine_in_graph) {
            impl_->lease_active = false;
            impl_->leases.clear();
            check(cudaGetLastError(), "launch captured experts");
            return;
        }
    } else {
        // Ungraphed path (LAMINA_MOE_GRAPHS=0); the timeline marks between its
        // launches give per-kernel expert times for diagnosis.
        if (impl_->fast) strata::kernels::native_quantize_q8_1(x_dev, impl_->expert_q8, n_in, 1, impl_->stream);
        impl_->tl_mark(kTlExGateUp);
        launch_group(static_cast<int>(routed.gate->type), n_in, gate_up_routed);
        impl_->tl_mark(kTlExSharedGateUp);
        launch_group(static_cast<int>(shared.gate->type), n_in, gate_up_shared);
        impl_->tl_mark(kTlExSwiglu);
        cuda::swiglu(impl_->gate_dev, impl_->up_dev, impl_->swiglu_dev,
                     static_cast<int>(slots) * ff, impl_->stream);
        if (impl_->fast) for (size_t base = 0; base < slots; base += 8)
            strata::kernels::native_quantize_q8_1(impl_->swiglu_dev + base * ff, impl_->expert_q8 + input_q8_bytes + base * down_q8_bytes,
                ff, int(std::min(size_t(8), slots - base)), impl_->stream);
        impl_->tl_mark(kTlExDown);
        launch_group(static_cast<int>(routed.down->type), down_n_in, down_routed);
        impl_->tl_mark(kTlExSharedDown);
        launch_group(static_cast<int>(shared.down->type), down_n_in, down_shared);
        impl_->end_time(expert_timing, impl_->stream);
    }
    const auto t_after_gu = std::chrono::steady_clock::now();
    impl_->lease_active = false;
    impl_->leases.clear();
    impl_->tl_mark(kTlMissWait);  // the stream idles here if the CPU experts finish after the GPU ones
    if (!cpu_slots.empty()) {
        cpu_batch.pool = nullptr;
        const auto h_wait = HostClock::now();
        impl_->cpu_experts->wait();  // the host thread helps with the remaining CPU rows
        if (impl_->timeline) impl_->tl_host[2] += host_ms(h_wait, HostClock::now());
    }
    const auto t_after_down = std::chrono::steady_clock::now();
    impl_->tl_mark(kTlCombine);
    if (!cpu_slots.empty()) {
        // The combine reads the CPU slots straight from mapped memory, which
        // replaces one copy launch per CPU expert.
        unsigned long long mask = 0;
        for (const size_t slot : cpu_slots) mask |= 1ull << slot;
        cuda::moe_combine_mixed(impl_->down_dev, impl_->cpu_down_dev, mask, impl_->scales_dev, static_cast<int>(slots),
                                hidden, 1, out_dev, impl_->stream);
        if (!impl_->cpu_down_copied)
            check(cudaEventCreateWithFlags(&impl_->cpu_down_copied, cudaEventDisableTiming), "create CPU staging event");
        check(cudaEventRecord(impl_->cpu_down_copied, impl_->stream), "record CPU staging reuse");
    } else cuda::moe_combine(impl_->down_dev, impl_->scales_dev, static_cast<int>(slots), hidden, out_dev,
                             impl_->stream);
    check(cudaGetLastError(), "launch MoE");
    // Copy this layer's CPU experts into the cache in the background, highest
    // router weight first, so following tokens run them on the GPU. Evicted
    // blocks are fenced by their retirement events. (Deferring this host work
    // into the next layer's doorbell wait measured slower: 31.0 -> 29.0 tok/s.)
    const auto h_admit = HostClock::now();
    // (Admitting only experts already used two or three times on the CPU, or
    // more than one per layer, measured no better.)
    for (size_t i = 0; i < cpu_slots.size() && int(i) < impl_->admit_per_layer; ++i) {
        const int expert = experts[cpu_slots[i]];
        if (!(impl_->admit(*routed.gate, routed.gate_data, expert) && impl_->admit(*routed.up, routed.up_data, expert) &&
              impl_->admit(*routed.down, routed.down_data, expert))) break;
    }
    if (impl_->timeline) impl_->tl_host[3] += host_ms(h_admit, HostClock::now());
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
    const bool device = x.empty() && impl_->prefill_columns == columns;
    const size_t elements = size_t(columns) * 2048;
    if (!supports_moe(routed, shared) || columns < 1 || columns > 2048 || (!device && x.size() != size_t(columns) * 2048) ||
        router.type != 0 || router.shape != std::vector<uint64_t>{2048, 256} ||
        shared_gate.type != 0 || shared_gate.shape != std::vector<uint64_t>{2048})
        throw std::invalid_argument("unsupported MoE prefill block");
    impl_->expert_layout(routed);
    const int hidden = 2048, ff = int(routed.gate->shape[1]);
    impl_->ensure(impl_->bm_x, impl_->bm_x_cap, elements);
    impl_->ensure(impl_->bm_router, impl_->bm_router_cap, size_t(columns) * 256);
    impl_->ensure(impl_->bm_scales, impl_->bm_scales_cap, size_t(columns) * 9);
    impl_->ensure(impl_->bm_ids, impl_->bm_ids_cap, size_t(columns) * 8);
    impl_->ensure(impl_->bm_map, impl_->bm_map_cap, size_t(columns) * 9);
    impl_->ensure(impl_->bm_input, impl_->bm_input_cap, elements);
    impl_->ensure(impl_->bm_gate, impl_->bm_gate_cap, size_t(columns) * ff);
    impl_->ensure(impl_->bm_up, impl_->bm_up_cap, size_t(columns) * ff);
    impl_->ensure(impl_->bm_swiglu, impl_->bm_swiglu_cap, size_t(columns) * ff);
    impl_->ensure(impl_->bm_down, impl_->bm_down_cap, elements);
    impl_->ensure(impl_->bm_slots, impl_->bm_slots_cap, size_t(columns) * 9 * hidden);
    check(cudaMemcpyAsync(impl_->bm_x, device ? impl_->p_norm : x.data(), elements * sizeof(float), device ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice, impl_->stream), "upload MoE prefill input");
    project_columns_device(router, router_data, impl_->bm_x, impl_->bm_router, columns, -1);
    auto* shared_weight = static_cast<const float*>(impl_->weight(shared_gate, shared_gate_data, -1, 2048 * sizeof(float)));
    cuda::dot_sigmoid_columns(shared_weight, impl_->bm_x, columns, impl_->bm_scales, impl_->stream);
    cuda::router_topk_columns(impl_->bm_router, columns, impl_->bm_ids, impl_->bm_scales, impl_->stream);
    // Hybrid prefill (fast CPU-miss mode): an expert routed by
    // at most LAMINA_PREFILL_CPU_TOKENS prompt tokens runs on the CPU pool
    // instead of being uploaded; the CPU reads an expert from RAM several times
    // faster than PCIe delivers it, and prefill uploads are mostly discarded
    // afterwards. The split depends on token counts only, never on residency,
    // so output does not depend on earlier requests.
    static const int cpu_tokens = [] {
        const char* v = std::getenv("LAMINA_PREFILL_CPU_TOKENS");
        return v && *v ? std::clamp(std::atoi(v), 0, CpuExperts::kMaxTokens) : 12;  // best of 2..16 on the RTX 3050
    }();
    // CPU staging stays bounded: at most 256 experts x cpu_tokens results per layer.
    const bool hybrid = impl_->fast && impl_->cpu_misses && impl_->cpu_experts && cpu_tokens > 0;
    if (hybrid) {
        impl_->mapped_floats(impl_->pf_in_host, impl_->pf_in_dev, impl_->pf_in_cap, elements);
        check(cudaMemcpyAsync(impl_->pf_in_host, impl_->bm_x, elements * sizeof(float), cudaMemcpyDeviceToHost, impl_->stream),
              "publish prefill expert inputs");
    }
    std::vector<int> ids(size_t(columns) * 8);
    check(cudaMemcpyAsync(ids.data(), impl_->bm_ids, ids.size() * sizeof(int), cudaMemcpyDeviceToHost, impl_->stream), "download prefill expert ids");
    check(cudaStreamSynchronize(impl_->stream), "finish prefill routing");
    std::array<std::vector<int>, 256> groups;
    for (int c = 0; c < columns; ++c) for (int s = 0; s < 8; ++s) {
        const int expert = ids[size_t(c)*8+s];
        if (expert < 0 || expert >= 256) throw std::runtime_error("invalid prefill route");
        groups[expert].push_back(c*9+s);
    }
    const bool keep_prompt = impl_->fast && impl_->cpu_misses && impl_->base_keep > 0 && !atomic_expert_cache();
    if (keep_prompt) {
        auto& counts = impl_->prefill_counts.try_emplace(routed.gate).first->second;
        for (int e = 0; e < 256; ++e) counts[size_t(e)] += uint32_t(groups[size_t(e)].size());
    }
    std::vector<int> map; map.reserve(size_t(columns) * 9);
    for (const auto& group : groups) map.insert(map.end(), group.begin(), group.end());
    for (int c = 0; c < columns; ++c) map.push_back(c*9+8);
    check(cudaMemcpyAsync(impl_->bm_map, map.data(), map.size() * sizeof(int), cudaMemcpyHostToDevice, impl_->stream), "publish prefill expert groups");
    std::array<bool, 256> on_cpu{};
    std::vector<CpuExperts::Job> cpu_jobs;
    size_t cpu_results = 0;
    if (hybrid) {
        for (int e = 0; e < 256; ++e) {
            const size_t count = groups[size_t(e)].size();
            if (count && count <= size_t(cpu_tokens)) cpu_results += count;
        }
    }
    if (cpu_results) {
        impl_->mapped_floats(impl_->pf_out_host, impl_->pf_out_dev, impl_->pf_out_cap, cpu_results * hidden);
        impl_->mapped_floats(impl_->pf_map_host, impl_->pf_map_dev, impl_->pf_map_cap, cpu_results);
        auto* result_slots = reinterpret_cast<int*>(impl_->pf_map_host);
        size_t next = 0;
        for (int e = 0; e < 256; ++e) {
            const auto& group = groups[size_t(e)];
            if (group.empty() || group.size() > size_t(cpu_tokens)) continue;
            on_cpu[size_t(e)] = true;
            CpuExperts::Job job;
            job.expert = e;
            for (const int slot : group) {
                const int k = job.tokens++;
                job.inputs[size_t(k)] = impl_->pf_in_host + size_t(slot / 9) * hidden;
                job.outputs[size_t(k)] = impl_->pf_out_host + next * hidden;
                result_slots[next++] = slot;
            }
            cpu_jobs.push_back(job);
        }
        impl_->cpu_experts->start(routed, cpu_jobs, /*wide=*/true);
        impl_->stat_cpu_experts += cpu_jobs.size();
    }
    struct CpuBatchGuard {
        CpuExperts* pool = nullptr;
        ~CpuBatchGuard() { if (pool) try { pool->wait(); } catch (...) {} }
    } cpu_batch{cpu_jobs.empty() ? nullptr : impl_->cpu_experts.get()};
    const auto apply = [&](const MoeWeights& w, int expert, int count, const float* input, const int* slots) {
        impl_->lease_active = true;
        for (const auto* t : {w.gate, w.up, w.down}) {
            const auto key = t->name + "#type=" + std::to_string(t->type) + "#expert=" + std::to_string(expert);
            if (auto entry = impl_->weights.find(key); entry != impl_->weights.end()) impl_->leases.insert(entry->second.device);
        }
        impl_->prepare(*w.gate, w.gate_data, expert);
        impl_->prepare(*w.up, w.up_data, expert);
        impl_->prepare(*w.down, w.down_data, expert);
        project_columns_device(*w.gate, w.gate_data, input, impl_->bm_gate, count, expert);
        project_columns_device(*w.up, w.up_data, input, impl_->bm_up, count, expert);
        cuda::swiglu(impl_->bm_gate, impl_->bm_up, impl_->bm_swiglu, count*ff, impl_->stream);
        project_columns_device(*w.down, w.down_data, impl_->bm_swiglu, impl_->bm_down, count, expert);
        cuda::scatter_expert_outputs(impl_->bm_down, slots, count, hidden, impl_->bm_slots, impl_->stream);
        impl_->lease_active = false; impl_->leases.clear();
    };
    size_t offset = 0;
    for (int expert = 0; expert < 256; ++expert) {
        const int count = int(groups[expert].size());
        if (!count) continue;
        const int* slots = impl_->bm_map + offset;
        offset += size_t(count);
        if (on_cpu[size_t(expert)]) continue;
        cuda::gather_expert_inputs(impl_->bm_x, slots, count, hidden, impl_->bm_input, impl_->stream);
        apply(routed, expert, count, impl_->bm_input, slots);
    }
    apply(shared, -1, columns, impl_->bm_x, impl_->bm_map + offset);
    if (!cpu_jobs.empty()) {
        cpu_batch.pool = nullptr;
        impl_->cpu_experts->wait();
        cuda::scatter_expert_outputs(impl_->pf_out_dev, reinterpret_cast<const int*>(impl_->pf_map_dev), int(cpu_results),
                                     hidden, impl_->bm_slots, impl_->stream);
    }
    if (keep_prompt) impl_->keep_prompt_experts(routed);
    cuda::moe_combine_columns(impl_->bm_slots, impl_->bm_scales, columns, hidden, impl_->bm_x, impl_->stream);
    check(cudaGetLastError(), "launch grouped prefill experts");
    if (device) {
        check(cudaMemcpyAsync(impl_->p_mix, impl_->bm_x, elements * sizeof(float), cudaMemcpyDeviceToDevice, impl_->stream), "retain prefill experts");
        return {};
    }
    std::vector<float> result(elements);
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
    impl_->expert_layout(routed);
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
    impl_->drop_unkept_routed();
    const auto drop = [&](auto*& pointer, size_t& capacity) {
        impl_->release(pointer); pointer = nullptr; capacity = 0;
    };
    drop(impl_->input, impl_->input_capacity); drop(impl_->output, impl_->output_capacity);
    drop(impl_->long_hidden,impl_->long_hidden_cap); drop(impl_->long_positions,impl_->long_positions_cap); impl_->long_columns=0;
    drop(impl_->p_positions, impl_->p_positions_cap); drop(impl_->p_hidden, impl_->p_hidden_cap); drop(impl_->p_norm, impl_->p_norm_cap); drop(impl_->p_mix, impl_->p_mix_cap);
    impl_->prefill_columns = 0;
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
    drop(impl_->matrix_q, impl_->matrix_q_cap); drop(impl_->matrix_k, impl_->matrix_k_cap); drop(impl_->matrix_v, impl_->matrix_v_cap);
    drop(impl_->matrix_probability, impl_->matrix_probability_cap); drop(impl_->matrix_scores, impl_->matrix_scores_cap);
    drop(impl_->matrix_output, impl_->matrix_output_cap); drop(impl_->matrix_metadata, impl_->matrix_metadata_cap);
    drop(impl_->blas_weights, impl_->blas_weights_cap);
    drop(impl_->blas_half_weights, impl_->blas_half_weights_cap); drop(impl_->blas_half_input, impl_->blas_half_input_cap);
#endif
}

void CudaProjection::reset() {
    check(cudaStreamSynchronize(impl_->stream), "wait before resetting conversation");
    impl_->clear_admissions();  // every request starts from the same base cache
    impl_->release_kept();     // the next prefill chooses its own kept experts
    impl_->prefill_columns = 0;
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
    const auto promote_start = std::chrono::steady_clock::now();
    if (impl_->cpu_misses) impl_->promote_admissions();
    if (impl_->timeline)
        impl_->tl_host[7] += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - promote_start).count();
    if (!(++impl_->decode_tokens % 256)) for (auto& [name, entry] : impl_->weights) {
        entry.hits = std::max(uint64_t(1), entry.hits / 2);
        if (entry.block && entry.device == entry.block->base) entry.block->hits = std::max(uint64_t(1), entry.block->hits / 2);
    }

    impl_->ensure(impl_->hidden_dev, impl_->hidden_cap, x.size());
    check(cudaMemcpyAsync(impl_->hidden_dev, x.data(), x.size() * sizeof(float),
                          cudaMemcpyHostToDevice, impl_->stream), "upload hidden");
}

std::vector<float> CudaProjection::hidden_download() {
    std::vector<float> result(static_cast<size_t>(impl_->hidden_cap));
    check(cudaMemcpyAsync(result.data(), impl_->hidden_dev, result.size() * sizeof(float),
                          cudaMemcpyDeviceToHost, impl_->stream), "download hidden");
    check(cudaStreamSynchronize(impl_->stream), "finish hidden");
    if (device_profile()) impl_->collect_times();
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

void CudaProjection::hidden_project(const strata::TensorInfo& tensor, const uint8_t* data, const std::vector<float>& x) {
    if (!supports(tensor.type) || tensor.shape.size() != 2 || tensor.shape[0] != x.size() || tensor.shape[1] != 2048)
        throw std::invalid_argument("unsupported hidden projection: " + tensor.name);
    const int n_in = int(x.size()), n_out = 2048;
    impl_->ensure(impl_->hidden_dev, impl_->hidden_cap, size_t(n_out));
    const void* weight = impl_->projection_weight(tensor, data, -1);
    float* input = impl_->pair(Impl::kPHead, x.size());
    check(cudaMemcpyAsync(input, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice, impl_->stream), "upload projection input");
    float* output = impl_->hidden_dev;
    impl_->decode_group(int(tensor.type), &weight, &input, &output, &n_out, 1, n_out, n_in, impl_->stream);
    check(cudaGetLastError(), "launch hidden projection");
}

bool CudaProjection::supports_pair() const { return impl_->fast && !impl_->host_kv; }

void CudaProjection::pair_upload(const std::vector<float>& x) {
    if (x.size() != 2 * 2048) throw std::invalid_argument("pair hidden width");
    if (!supports_pair()) throw std::logic_error("the two-token step requires fast mode and device KV");
    if (impl_->cpu_misses) impl_->promote_admissions();
    float* hidden = impl_->pair(Impl::kPHidden, x.size());
    check(cudaMemcpyAsync(hidden, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice, impl_->stream),
          "upload pair hidden");
    for (auto& snapshot : impl_->gdn_snapshots) snapshot.taken = false;
}

std::vector<float> CudaProjection::pair_download() {
    std::vector<float> result(2 * 2048);
    check(cudaMemcpyAsync(result.data(), impl_->pair_buf[Impl::kPHidden], result.size() * sizeof(float),
                          cudaMemcpyDeviceToHost, impl_->stream), "download pair hidden");
    check(cudaStreamSynchronize(impl_->stream), "finish pair step");
    return result;
}

void CudaProjection::pair_rms(const strata::TensorInfo& gamma, const uint8_t* data, float epsilon) {
    float* norm = impl_->pair(Impl::kPNorm, 2 * 2048);
    check(cudaMemcpyAsync(norm, impl_->pair_buf[Impl::kPHidden], 2 * 2048 * sizeof(float),
                          cudaMemcpyDeviceToDevice, impl_->stream), "copy pair hidden");
    const auto* weight = static_cast<const float*>(impl_->weight(gamma, data, -1, size_t(gamma.shape[0]) * sizeof(float)));
    for (int c = 0; c < 2; ++c) cuda::rms_norm(norm + c * 2048, weight, 2048, epsilon, impl_->stream);
}

void CudaProjection::pair_add_mix() {
    cuda::add_inplace(impl_->pair_buf[Impl::kPHidden], impl_->pair_buf[Impl::kPMix], 2 * 2048, impl_->stream);
}

void CudaProjection::pair_delta(int layer, const GdnWeights& w) {
    constexpr int kHidden = 2048, kChannels = 8192, kHeads = 32, kS = 128;
    constexpr float kEps = 1e-6f;
    if (!supports_gdn(w) || layer < 0 || size_t(layer) >= impl_->gdn_states.size() ||
        !impl_->gdn_states[size_t(layer)].rec)
        throw std::invalid_argument("pair DeltaNet needs a decoded layer");
    auto& state = impl_->gdn_states[size_t(layer)];
    if (size_t(layer) >= impl_->gdn_snapshots.size()) impl_->gdn_snapshots.resize(size_t(layer) + 1);
    auto& snapshot = impl_->gdn_snapshots[size_t(layer)];
    const size_t conv_bytes = size_t(kChannels) * 3 * sizeof(float), rec_bytes = size_t(kS) * kHeads * kS * sizeof(float);
    if (!snapshot.conv) impl_->allocate(reinterpret_cast<void**>(&snapshot.conv), conv_bytes);
    if (!snapshot.rec) impl_->allocate(reinterpret_cast<void**>(&snapshot.rec), rec_bytes);
    const float* x = impl_->pair_buf[Impl::kPNorm];
    float* qkv = impl_->pair(Impl::kPQkv, 2 * kChannels);
    float* z = impl_->pair(Impl::kPZ, 2 * 4096);
    float* alpha = impl_->pair(Impl::kPAlpha, 2 * kHeads);
    float* beta = impl_->pair(Impl::kPBeta, 2 * kHeads);
    float* gate = impl_->pair(Impl::kPGate, 2 * kHeads);
    float* raw = impl_->pair(Impl::kPConvRaw, 2 * kChannels);
    float* silu = impl_->pair(Impl::kPConvSilu, 2 * kChannels);
    float* outnorm = impl_->pair(Impl::kPOutNorm, 2 * 4096);
    float* y = impl_->pair(Impl::kPY, 2 * 4096);
    float* mix = impl_->pair(Impl::kPMix, 2 * kHidden);

    const void* weights[2] = {impl_->projection_weight(*w.qkv, w.qkv_data, -1),
                              impl_->projection_weight(*w.gate, w.gate_data, -1)};
    float* outputs[2] = {qkv, z};
    const int rows[2] = {kChannels, 4096};
    impl_->decode_pair(int(w.qkv->type), weights, x, outputs, rows, 2, kChannels + 4096, kHidden);
    const auto dense = [&](const strata::TensorInfo* t, const uint8_t* data, size_t count) {
        return static_cast<const float*>(impl_->weight(*t, data, -1, count * sizeof(float)));
    };
    const float* alpha_w = dense(w.alpha, w.alpha_data, size_t(kHidden) * kHeads);
    const float* beta_w = dense(w.beta, w.beta_data, size_t(kHidden) * kHeads);
    const float* conv_w = dense(w.conv, w.conv_data, size_t(4) * kChannels);
    const float* a_w = dense(w.a, w.a_data, kHeads);
    const float* dt_w = dense(w.dt, w.dt_data, kHeads);
    const float* gamma = dense(w.norm, w.norm_data, kS);
    const strata::kernels::GdnShapes shapes{kS, 16, kHeads};
    // The recurrence is sequential: column 0 updates the states in place, they
    // are snapshotted, then column 1 continues from them.
    for (int c = 0; c < 2; ++c) {
        float* sc = silu + size_t(c) * kChannels;
        cuda::gemv_f32(alpha_w, x + c * kHidden, alpha + c * kHeads, kHidden, kHeads, impl_->stream);
        cuda::gemv_f32(beta_w, x + c * kHidden, beta + c * kHeads, kHidden, kHeads, impl_->stream);
        strata::kernels::native_gdn_conv_silu(state.conv, qkv + size_t(c) * kChannels, conv_w,
                                              raw + size_t(c) * kChannels, sc, kChannels, 4, impl_->stream);
        strata::kernels::native_gdn_l2_norm(sc, 16, kS, kEps, impl_->stream);
        strata::kernels::native_gdn_l2_norm(sc + 2048, 16, kS, kEps, impl_->stream);
        strata::kernels::native_gdn_beta_gate(beta + c * kHeads, kHeads, impl_->stream);
        strata::kernels::native_gdn_gate(alpha + c * kHeads, dt_w, a_w, gate + c * kHeads, kHeads, impl_->stream);
        strata::kernels::native_gdn_step(state.rec, sc, sc + 2048, sc + 4096, gate + c * kHeads, beta + c * kHeads,
                                         outnorm + c * 4096, shapes, impl_->stream);
        if (c == 0) {
            check(cudaMemcpyAsync(snapshot.conv, state.conv, conv_bytes, cudaMemcpyDeviceToDevice, impl_->stream), "snapshot conv");
            check(cudaMemcpyAsync(snapshot.rec, state.rec, rec_bytes, cudaMemcpyDeviceToDevice, impl_->stream), "snapshot recurrence");
            snapshot.taken = true;
        }
    }
    // Per-head norm: the two columns are 64 consecutive heads.
    cuda::gdn_out_norm_silu(outnorm, z, gamma, y, 2 * kHeads, kEps, impl_->stream);
    const void* out_weight = impl_->projection_weight(*w.out, w.out_data, -1);
    const int out_rows = kHidden;
    impl_->decode_pair(int(w.out->type), &out_weight, y, &mix, &out_rows, 1, kHidden, 4096);
    check(cudaGetLastError(), "launch pair DeltaNet");
}

void CudaProjection::pair_delta_layer(int layer, const GdnWeights& w, const strata::TensorInfo& input_norm,
                                      const uint8_t* input_norm_data, const strata::TensorInfo& post_norm,
                                      const uint8_t* post_norm_data, float epsilon) {
    if (layer < 0) throw std::invalid_argument("CUDA DeltaNet layer index");
    if (size_t(layer) >= impl_->pair_graphs.size()) impl_->pair_graphs.resize(size_t(layer) + 1);
    auto& graph = impl_->pair_graphs[size_t(layer)];
    const auto record = [&] {
        pair_rms(input_norm, input_norm_data, epsilon);
        pair_delta(layer, w);
        pair_add_mix();
        pair_rms(post_norm, post_norm_data, epsilon);
    };
    if (graph.exec) {
        check(cudaGraphLaunch(graph.exec, impl_->stream), "launch pair DeltaNet graph");
        impl_->gdn_snapshots[size_t(layer)].taken = true;  // the replay refreshed the snapshot
        return;
    }
    // The first pass allocates every buffer and caches every weight, so the
    // captured second pass performs no allocations.
    if (!graph.warmed) { graph.warmed = true; record(); return; }
    check(cudaStreamSynchronize(impl_->stream), "flush before pair capture");
    check(cudaStreamBeginCapture(impl_->stream, cudaStreamCaptureModeThreadLocal), "begin pair capture");
    record();
    cudaGraph_t captured = nullptr;
    check(cudaStreamEndCapture(impl_->stream, &captured), "end pair capture");
    const auto status = cudaGraphInstantiate(&graph.exec, captured, nullptr, nullptr, 0);
    cudaGraphDestroy(captured);
    check(status, "instantiate pair graph");
    check(cudaGraphLaunch(graph.exec, impl_->stream), "launch pair DeltaNet graph");
}

void CudaProjection::pair_attention(int layer, const AttnWeights& w, int position, float rope_base,
                                    const std::array<std::array<int, 3>, 2>& rope_positions) {
    constexpr int kHidden = 2048, kHeads = 16, kKvHeads = 2, kHeadDim = 256, kRot = 64;
    constexpr int kQ = kHeads * 2 * kHeadDim, kKv = kKvHeads * kHeadDim;
    constexpr float kEps = 1e-6f;
    if (!supports_attention(w) || layer < 0 || position < 0) throw std::invalid_argument("pair attention layer");
    if (position + 2 > impl_->max_context) throw std::out_of_range("CUDA attention context exceeded");
    const float* x = impl_->pair_buf[Impl::kPNorm];
    float* q = impl_->pair(Impl::kPQ, 2 * kQ);
    float* k = impl_->pair(Impl::kPK, 2 * kKv);
    float* v = impl_->pair(Impl::kPV, 2 * kKv);
    float* ctx = impl_->pair(Impl::kPCtx, 2 * kHeads * kHeadDim);
    float* mix = impl_->pair(Impl::kPMix, 2 * kHidden);
    const void* weights[3] = {impl_->projection_weight(*w.q, w.q_data, -1), impl_->projection_weight(*w.k, w.k_data, -1),
                              impl_->projection_weight(*w.v, w.v_data, -1)};
    float* outputs[3] = {q, k, v};
    const int rows[3] = {kQ, kKv, kKv};
    impl_->decode_pair(int(w.q->type), weights, x, outputs, rows, 3, kQ + 2 * kKv, kHidden);
    const auto* q_norm = static_cast<const float*>(impl_->weight(*w.q_norm, w.q_norm_data, -1, kHeadDim * sizeof(float)));
    const auto* k_norm = static_cast<const float*>(impl_->weight(*w.k_norm, w.k_norm_data, -1, kHeadDim * sizeof(float)));
    if (size_t(layer) >= impl_->attn_states.size()) impl_->attn_states.resize(size_t(layer) + 1);
    auto& state = impl_->attn_states[size_t(layer)];
    impl_->grow_kv(state, position + 2);
    for (int c = 0; c < 2; ++c) {
        const auto& r = rope_positions[size_t(c)];
        cuda::attn_norm_mrope(q + c * kQ, kHeads, 2 * kHeadDim, kHeadDim, kRot, q_norm, kEps, rope_base,
                              r[0], r[1], r[2], impl_->stream);
        cuda::attn_norm_mrope(k + c * kKv, kKvHeads, kHeadDim, kHeadDim, kRot, k_norm, kEps, rope_base,
                              r[0], r[1], r[2], impl_->stream);
        cuda::kv_store(k + c * kKv, impl_->kv_offset(state.keys, position + c), kKv, impl_->half_kv, impl_->stream);
        cuda::kv_store(v + c * kKv, impl_->kv_offset(state.values, position + c), kKv, impl_->half_kv, impl_->stream);
    }
    // Causal: column c attends to positions 0..position+c. Tiles are reused in
    // order, so column 1's partials overwrite column 0's after its merge.
    for (int c = 0; c < 2; ++c) {
        const int needed = position + 1 + c, tiles = (needed + 127) / 128;
        impl_->ensure(impl_->a_partial, impl_->a_partial_cap, size_t(kHeads) * tiles * (kHeadDim + 2));
        if (impl_->half_kv) cuda::attn_half_partials(q + c * kQ, state.keys, state.values, needed, 0, tiles,
                                                    impl_->a_partial, impl_->stream);
        else cuda::attn_partials(q + c * kQ, state.keys, state.values, needed, 0, tiles, kHeads, kKvHeads, kHeadDim,
                                 1.0f / 16.0f, impl_->a_partial, impl_->stream);
        cuda::attn_merge(q + c * kQ, impl_->a_partial, tiles, kHeads, kHeadDim, ctx + c * kHeads * kHeadDim, impl_->stream);
    }
    const void* out_weight = impl_->projection_weight(*w.out, w.out_data, -1);
    const int out_rows = kHidden;
    impl_->decode_pair(int(w.out->type), &out_weight, ctx, &mix, &out_rows, 1, kHidden, kHeads * kHeadDim);
    check(cudaGetLastError(), "launch pair attention");
}

void CudaProjection::pair_moe(const strata::TensorInfo& router, const uint8_t* router_data,
                              const strata::TensorInfo& shared_gate, const uint8_t* shared_gate_data,
                              const MoeWeights& routed, const MoeWeights& shared) {
    constexpr int kTopK = 8, kSlots = kTopK + 1;  // per column: routed experts, then the shared expert
    if (!supports_moe(routed, shared)) throw std::invalid_argument("unsupported pair MoE block");
    impl_->expert_layout(routed);
    const int n_in = int(routed.gate->shape[0]), ff = int(routed.gate->shape[1]), hidden = int(routed.down->shape[1]);
    const int experts_count = int(router.shape[1]);
    const float* x = impl_->pair_buf[Impl::kPNorm];
    float* mix = impl_->pair(Impl::kPMix, 2 * size_t(hidden));
    float* logits = impl_->pair(Impl::kPLogits, 2 * size_t(experts_count));
    const auto* router_w = static_cast<const float*>(impl_->weight(router, router_data, -1, size_t(n_in) * experts_count * sizeof(float)));
    const auto* shared_w = static_cast<const float*>(impl_->weight(shared_gate, shared_gate_data, -1, size_t(n_in) * sizeof(float)));
    // Both routings, then one doorbell. Mapped layout: ids [c*8+k], scales
    // [c*8+k], shared gates [16+c].
    const bool cpu_route = impl_->cpu_misses && impl_->stat_evicted;
    if (cpu_route) impl_->mapped_floats(impl_->cpu_input_host, impl_->cpu_input_dev, impl_->cpu_input_cap, 2 * size_t(n_in));
    const unsigned seq = ++impl_->route_seq;
    cuda::gemv_f32_batch(router_w, x, logits, n_in, experts_count, 2, impl_->stream);
    cuda::router_finish_pair(logits, experts_count, kTopK, impl_->d_ids_map, impl_->d_scales_map, shared_w, x, n_in,
                             impl_->d_scales_map + 2 * kTopK, cpu_route ? impl_->cpu_input_dev : nullptr,
                             impl_->d_flag_map, int(seq), impl_->stream);
    // LAMINA_PAIR_PROFILE=1: host milliseconds per call for the routing wait,
    // the setup and launches, and the CPU-expert wait, every 2560 calls.
    static const bool profile = env_is("LAMINA_PAIR_PROFILE", '1');
    static double prof_ms[3] = {};
    static uint64_t prof_calls = 0, prof_union = 0, prof_cpu = 0;
    const auto route_start = std::chrono::steady_clock::now();
    volatile const int* flag = impl_->h_flag_map;
    unsigned polls = 0;
    while (*flag != int(seq)) {
        if (!(polls & 1023)) {
            const auto status = cudaStreamQuery(impl_->stream);
            if (status != cudaSuccess && status != cudaErrorNotReady) check(status, "wait for pair router");
        }
        if (!(++polls & 131071) && std::chrono::steady_clock::now() - route_start > std::chrono::seconds(30))
            throw std::runtime_error("CUDA pair router timed out");
    }
    const auto routed_at = std::chrono::steady_clock::now();
    int ids[2][kTopK];
    float scales[2 * kSlots];
    std::memcpy(ids, impl_->h_ids_map, sizeof(ids));
    for (int c = 0; c < 2; ++c) {
        for (int k = 0; k < kTopK; ++k) scales[c * kSlots + k] = impl_->h_scales_map[c * kTopK + k];
        scales[c * kSlots + kTopK] = impl_->h_scales_map[2 * kTopK + c];
    }

    // The union of both routings. slot[u][c] is expert u's slot in column c, or -1.
    struct Unique { int expert; int slot[2]; bool cpu; };
    std::vector<Unique> unique;
    for (int c = 0; c < 2; ++c)
        for (int k = 0; k < kTopK; ++k) {
            auto found = std::find_if(unique.begin(), unique.end(), [&](const Unique& u) { return u.expert == ids[c][k]; });
            if (found == unique.end()) { unique.push_back({ids[c][k], {-1, -1}, false}); found = unique.end() - 1; }
            found->slot[c] = c * kSlots + k;
        }
    // Non-resident experts run on the CPU pool (both tokens per row read).
    std::vector<int> cpu_ids;
    std::vector<std::array<float*, CpuExperts::kMaxTokens>> cpu_outputs;
    if (cpu_route) {
        impl_->mapped_floats(impl_->cpu_down_host, impl_->cpu_down_dev, impl_->cpu_down_cap, 2 * kSlots * size_t(hidden));
        if (impl_->cpu_down_copied) check(cudaEventSynchronize(impl_->cpu_down_copied), "reuse CPU expert staging");
        for (auto& u : unique) {
            u.cpu = !(impl_->resident(*routed.gate, u.expert) && impl_->resident(*routed.up, u.expert) &&
                      impl_->resident(*routed.down, u.expert));
            if (!u.cpu) continue;
            cpu_ids.push_back(u.expert);
            std::array<float*, CpuExperts::kMaxTokens> out{};
            for (int c = 0; c < 2; ++c) if (u.slot[c] >= 0) out[size_t(c)] = impl_->cpu_down_host + size_t(u.slot[c]) * hidden;
            cpu_outputs.push_back(out);
        }
    }
    struct CpuBatchGuard {
        CpuExperts* pool = nullptr;
        ~CpuBatchGuard() { if (pool) try { pool->wait(); } catch (...) {} }
    } cpu_batch;
    if (!cpu_ids.empty()) {
        impl_->cpu_experts->start(routed, cpu_ids, {impl_->cpu_input_host, impl_->cpu_input_host + n_in}, 2, cpu_outputs);
        cpu_batch.pool = impl_->cpu_experts.get();
        impl_->stat_cpu_experts += cpu_ids.size();
    }

    const size_t iq = strata::kernels::native_q8_1_bytes(n_in), dq = strata::kernels::native_q8_1_bytes(ff);
    impl_->ensure(impl_->expert_q8, impl_->expert_q8_cap, 2 * iq + 2 * kSlots * dq);
    impl_->ensure(impl_->gate_dev, impl_->gate_capacity, 2 * kSlots * size_t(ff));
    impl_->ensure(impl_->up_dev, impl_->up_capacity, 2 * kSlots * size_t(ff));
    impl_->ensure(impl_->swiglu_dev, impl_->swiglu_capacity, 2 * kSlots * size_t(ff));
    impl_->ensure(impl_->down_dev, impl_->down_capacity, 2 * kSlots * size_t(hidden));
    impl_->ensure(impl_->scales_dev, impl_->scales_capacity, 2 * kSlots);
    if (!impl_->scales_host)
        check(cudaMallocHost(reinterpret_cast<void**>(&impl_->scales_host), 65 * sizeof(float)), "allocate expert scales staging");
    std::copy(scales, scales + 2 * kSlots, impl_->scales_host);
    check(cudaMemcpyAsync(impl_->scales_dev, impl_->scales_host, 2 * kSlots * sizeof(float), cudaMemcpyHostToDevice,
                          impl_->stream), "upload pair MoE scales");

    // Weight pointers, leased so that uploads for later experts cannot evict them.
    impl_->lease_active = true;
    struct GpuExpert { const void* gate; const void* up; const void* down; int slot[2]; };
    std::vector<GpuExpert> gpu;
    for (const auto& u : unique)
        if (!u.cpu) gpu.push_back({impl_->projection_weight(*routed.gate, routed.gate_data, u.expert),
                                   impl_->projection_weight(*routed.up, routed.up_data, u.expert),
                                   impl_->projection_weight(*routed.down, routed.down_data, u.expert), {u.slot[0], u.slot[1]}});
    const GpuExpert shared_expert{impl_->projection_weight(*shared.gate, shared.gate_data, -1),
                                  impl_->projection_weight(*shared.up, shared.up_data, -1),
                                  impl_->projection_weight(*shared.down, shared.down_data, -1), {kTopK, kSlots + kTopK}};
    const auto q8 = [&](size_t offset) { return reinterpret_cast<const float*>(impl_->expert_q8 + offset); };
    const auto x_q8 = [&](int slot) { return q8(size_t(slot / kSlots) * iq); };          // the slot's column input
    const auto swiglu_q8 = [&](int slot) { return q8(2 * iq + size_t(slot) * dq); };
    // Experts routed by both tokens use the two-column kernel; the rest run as
    // ordinary grouped matrices, each on its own column's input.
    const auto launch = [&](int type, int width, bool down, const std::vector<const GpuExpert*>& experts) {
        strata::kernels::NativeF32Pairs both{};
        strata::kernels::NativeF32Grouped single{};
        int both_rows = 0, single_rows = 0;
        for (const auto* e : experts) {
            const int rows = down ? hidden : ff;
            const void* matrices[2] = {down ? e->down : e->gate, e->up};
            float* bases[2] = {down ? impl_->down_dev : impl_->gate_dev, impl_->up_dev};
            for (int j = 0; j < (down ? 1 : 2); ++j) {
                const void* weight = matrices[j];
                float* base = bases[j];
                if (e->slot[0] >= 0 && e->slot[1] >= 0) {
                    const int i = both.count++;
                    both.weights[i] = weight; both.n_outs[i] = rows; both_rows += rows;
                    for (int c = 0; c < 2; ++c) {
                        both.inputs[c][i] = down ? swiglu_q8(e->slot[c]) : x_q8(e->slot[c]);
                        both.outputs[c][i] = base + size_t(e->slot[c]) * rows;
                    }
                } else {
                    const int slot = e->slot[0] >= 0 ? e->slot[0] : e->slot[1];
                    const int i = single.count++;
                    single.weights[i] = weight; single.n_outs[i] = rows; single_rows += rows;
                    single.inputs[i] = down ? swiglu_q8(slot) : x_q8(slot);
                    single.outputs[i] = base + size_t(slot) * rows;
                }
            }
        }
        if (both.count) strata::kernels::native_mmvq_q8_pairs(type, both, both_rows, width, impl_->stream);
        if (single.count) strata::kernels::native_mmvq_q8_grouped(type, single, single_rows, width, impl_->stream);
    };
    std::vector<const GpuExpert*> routed_gpu;
    for (const auto& e : gpu) routed_gpu.push_back(&e);
    strata::kernels::native_quantize_q8_1(x, impl_->expert_q8, n_in, 2, impl_->stream);
    launch(int(routed.gate->type), n_in, false, routed_gpu);
    launch(int(shared.gate->type), n_in, false, {&shared_expert});
    cuda::swiglu(impl_->gate_dev, impl_->up_dev, impl_->swiglu_dev, 2 * kSlots * ff, impl_->stream);
    for (int base = 0; base < 2 * kSlots; base += 8)
        strata::kernels::native_quantize_q8_1(impl_->swiglu_dev + size_t(base) * ff, impl_->expert_q8 + 2 * iq + size_t(base) * dq,
                                              ff, std::min(8, 2 * kSlots - base), impl_->stream);
    launch(int(routed.down->type), int(routed.down->shape[0]), true, routed_gpu);
    launch(int(shared.down->type), int(shared.down->shape[0]), true, {&shared_expert});
    impl_->lease_active = false;
    impl_->leases.clear();
    const auto launched = std::chrono::steady_clock::now();
    if (profile) {
        prof_ms[0] += std::chrono::duration<double, std::milli>(routed_at - route_start).count();
        prof_ms[1] += std::chrono::duration<double, std::milli>(launched - routed_at).count();
        prof_union += unique.size(); prof_cpu += cpu_ids.size();
    }
    if (!cpu_ids.empty()) {
        cpu_batch.pool = nullptr;
        impl_->cpu_experts->wait();
        if (profile) prof_ms[2] += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - launched).count();
    }
    unsigned long long cpu_mask = 0;
    for (const auto& u : unique)
        if (u.cpu) for (int c = 0; c < 2; ++c) if (u.slot[c] >= 0) cpu_mask |= 1ull << u.slot[c];
    cuda::moe_combine_mixed(impl_->down_dev, cpu_mask ? impl_->cpu_down_dev : impl_->down_dev, cpu_mask, impl_->scales_dev,
                            kSlots, hidden, 2, mix, impl_->stream);
    if (cpu_mask) {
        if (!impl_->cpu_down_copied)
            check(cudaEventCreateWithFlags(&impl_->cpu_down_copied, cudaEventDisableTiming), "create CPU staging event");
        check(cudaEventRecord(impl_->cpu_down_copied, impl_->stream), "record CPU staging reuse");
    }
    check(cudaGetLastError(), "launch pair MoE");
    // Admit CPU experts into the cache as the single-token path does, first
    // column's highest router weight first.
    int admitted = 0;
    for (const auto& u : unique) {
        if (!u.cpu) continue;
        if (admitted++ >= impl_->admit_per_layer) break;
        if (!(impl_->admit(*routed.gate, routed.gate_data, u.expert) && impl_->admit(*routed.up, routed.up_data, u.expert) &&
              impl_->admit(*routed.down, routed.down_data, u.expert))) break;
    }
    if (profile && ++prof_calls % 2560 == 0)
        std::fprintf(stderr, "pair moe calls=%llu route_wait_ms=%.3f setup_ms=%.3f cpu_wait_ms=%.3f union=%.2f cpu=%.2f\n",
                     static_cast<unsigned long long>(prof_calls), prof_ms[0] / double(prof_calls), prof_ms[1] / double(prof_calls),
                     prof_ms[2] / double(prof_calls), double(prof_union) / double(prof_calls), double(prof_cpu) / double(prof_calls));
}

void CudaProjection::pair_commit(bool keep_second) {
    if (keep_second) return;
    for (size_t layer = 0; layer < impl_->gdn_snapshots.size(); ++layer) {
        auto& snapshot = impl_->gdn_snapshots[layer];
        if (!snapshot.taken) continue;
        auto& state = impl_->gdn_states[layer];
        check(cudaMemcpyAsync(state.conv, snapshot.conv, size_t(8192) * 3 * sizeof(float), cudaMemcpyDeviceToDevice,
                              impl_->stream), "restore conv");
        check(cudaMemcpyAsync(state.rec, snapshot.rec, size_t(128) * 32 * 128 * sizeof(float), cudaMemcpyDeviceToDevice,
                              impl_->stream), "restore recurrence");
        snapshot.taken = false;
    }
}

std::array<int, 2> CudaProjection::matvec_argmax2(const strata::TensorInfo& tensor, const uint8_t* data,
                                                  const std::vector<float>& x) {
    if (!impl_->fast || !supports(tensor.type) || tensor.shape.size() != 2 || 2 * tensor.shape[0] != x.size())
        throw std::invalid_argument("unsupported two-column LM head: " + tensor.name);
    const int n_in = int(tensor.shape[0]), n_out = int(tensor.shape[1]);
    const size_t bytes = strata::kernels::native_mmvq_weight_bytes(tensor.type, n_in, n_out);
    void* weight = impl_->weight(tensor, data, -1, bytes);
    float* input = impl_->pair(Impl::kPHead, x.size());
    impl_->reserve(size_t(n_in), 2 * size_t(n_out));
    check(cudaMemcpyAsync(input, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice, impl_->stream), "upload pair head input");
    impl_->ensure(impl_->dense_q8, impl_->dense_q8_cap, 2 * strata::kernels::native_q8_1_bytes(4096));
    strata::kernels::native_quantize_q8_1(input, impl_->dense_q8, n_in, 2, impl_->stream);
    // Generic MMVQ with two columns: bitwise per column in the default exact layout.
    strata::kernels::native_mmvq(tensor.type, weight, impl_->dense_q8, impl_->output, n_in, n_out, 2, impl_->stream);
    if (!impl_->argmax_host) {
        check(cudaHostAlloc(reinterpret_cast<void**>(&impl_->argmax_host), 4 * sizeof(int), cudaHostAllocMapped), "allocate argmax result");
        check(cudaHostGetDevicePointer(reinterpret_cast<void**>(&impl_->argmax_dev), impl_->argmax_host, 0), "map argmax result");
    }
    cuda::argmax_f32(impl_->output, n_out, impl_->argmax_dev, impl_->stream);
    cuda::argmax_f32(impl_->output + n_out, n_out, impl_->argmax_dev + 2, impl_->stream);
    check(cudaGetLastError(), "launch pair argmax");
    check(cudaStreamSynchronize(impl_->stream), "finish pair argmax");
    return {impl_->argmax_host[1] ? -1 : impl_->argmax_host[0], impl_->argmax_host[3] ? -1 : impl_->argmax_host[2]};
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
    // The timeline records events inside the layer, which capture would forbid.
    if (!graph.warmed || impl_->timeline) {
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
                                  const MoeWeights& shared, const MoePrefetch* next) {
    if (!supports_moe(routed, shared)) throw std::invalid_argument("unsupported CUDA MoE block");
    impl_->expert_layout(routed);
    const int n_in = static_cast<int>(routed.gate->shape[0]);
    const int experts_count = static_cast<int>(router.shape[1]);
    const int top_k = 8;
    impl_->ensure(impl_->r_logits, impl_->r_logits_cap, static_cast<size_t>(experts_count));
    void* router_weight = impl_->weight(router, router_data, -1,
                                        static_cast<size_t>(n_in) * experts_count * sizeof(float));
    void* shared_gate_weight = impl_->weight(shared_gate, shared_gate_data, -1,
                                             static_cast<size_t>(n_in) * sizeof(float));
    const auto route_start = std::chrono::steady_clock::now();
    impl_->tl_mark(kTlRouter);
    // Router -> top-k -> doorbell, all writing straight into mapped pinned memory.
    cuda::gemv_f32(static_cast<const float*>(router_weight), impl_->norm_dev, impl_->r_logits,
                   n_in, experts_count, impl_->stream);
    // Predict the next layer's experts by applying its router to this layer's
    // normalized input. The hidden state changes by one layer's residual, so
    // the prediction is approximate; its ids ride in the second half of the
    // mapped id buffer and are published by the same doorbell.
    // Default on only with registered model RAM, where an upload is one async
    // DMA call. Through the staging pool each upload blocks the host on a worker
    // copy, which measured 20% slower; LAMINA_PREFETCH=1/0 overrides.
    static const char prefetch_mode = [] { const char* v = std::getenv("LAMINA_PREFETCH"); return v ? v[0] : char(0); }();
    const bool allowed = prefetch_mode ? prefetch_mode == '1' : impl_->registered_weights != nullptr;
    const bool predict = allowed && next && next->router && !impl_->cpu_misses;
    const bool publish = impl_->cpu_misses && impl_->stat_evicted;
    if (publish) {
        impl_->mapped_floats(impl_->cpu_input_host, impl_->cpu_input_dev, impl_->cpu_input_cap, static_cast<size_t>(n_in));
        impl_->cpu_input_published = true;
    }
    const unsigned seq = ++impl_->route_seq;
    if (!predict) {
        // Shared gate, top-k, CPU-input publication and doorbell in one launch;
        // on WDDM each separate launch cost the waiting GPU several microseconds.
        cuda::router_finish(impl_->r_logits, experts_count, top_k, impl_->d_ids_map, impl_->d_scales_map,
                            static_cast<const float*>(shared_gate_weight), impl_->norm_dev, n_in,
                            impl_->d_scales_map + top_k, publish ? impl_->cpu_input_dev : nullptr,
                            impl_->d_flag_map, static_cast<int>(seq), impl_->stream);
    } else {
        cuda::dot_sigmoid(static_cast<const float*>(shared_gate_weight), impl_->norm_dev, n_in,
                          impl_->d_scales_map + top_k, impl_->stream);
        cuda::router_topk(impl_->r_logits, experts_count, top_k, impl_->d_ids_map, impl_->d_scales_map,
                          impl_->stream);
        void* next_router = impl_->weight(*next->router, next->router_data, -1,
                                          static_cast<size_t>(n_in) * experts_count * sizeof(float));
        impl_->ensure(impl_->pred_scales, impl_->pred_scales_cap, static_cast<size_t>(top_k));
        cuda::gemv_f32(static_cast<const float*>(next_router), impl_->norm_dev, impl_->r_logits,
                       n_in, experts_count, impl_->stream);
        cuda::router_topk(impl_->r_logits, experts_count, top_k, impl_->d_ids_map + top_k,
                          impl_->pred_scales, impl_->stream);
        if (publish) cuda::copy_f32(impl_->norm_dev, impl_->cpu_input_dev, n_in, impl_->stream);
        cuda::doorbell_signal(impl_->d_flag_map, static_cast<int>(seq), impl_->stream);
    }
    // The shared expert does not depend on routing. Queued behind the doorbell,
    // it keeps the GPU busy while the host reads the routing and prepares the
    // routed experts, instead of running afterwards on the critical path.
    static const bool no_early = env_is("LAMINA_SHARED_EARLY", '0');
    impl_->shared_early = impl_->fast && impl_->cpu_misses && !no_early;
    if (impl_->shared_early) impl_->shared_expert_early(shared, impl_->norm_dev, top_k);
    impl_->tl_mark(kTlHostGap);  // the stream idles from here until the host submits the experts
    volatile const int* flag = impl_->h_flag_map;
    unsigned polls = 0;
    while (*flag != static_cast<int>(seq)) {
        if (!(polls & 1023)) {
            const auto status = cudaStreamQuery(impl_->stream);
            if (status != cudaSuccess && status != cudaErrorNotReady) check(status, "wait for router");
        }
        if (!(++polls & 131071) && std::chrono::steady_clock::now() - route_start > std::chrono::seconds(30))
            throw std::runtime_error("CUDA router timed out");
    }
    const auto route_end = std::chrono::steady_clock::now();
    if (device_profile()) impl_->router_wait_ms += std::chrono::duration<double, std::milli>(route_end - route_start).count();
    std::vector<int> ids(static_cast<size_t>(top_k));
    std::vector<float> weights(static_cast<size_t>(top_k));
    float shared_weight = 0.0f;
    std::memcpy(ids.data(), impl_->h_ids_map, ids.size() * sizeof(int));
    std::memcpy(weights.data(), impl_->h_scales_map, weights.size() * sizeof(float));
    std::memcpy(&shared_weight, impl_->h_scales_map + top_k, sizeof(float));
    // Score the previous layer's guess against what this router really chose.
    for (const int guess : impl_->prefetch_expected) {
        impl_->stat_pf_predicted++;
        if (std::find(ids.begin(), ids.end(), guess) != ids.end()) impl_->stat_pf_useful++;
    }
    impl_->prefetch_pred.clear();
    impl_->has_prefetch = false;
    if (predict) {
        impl_->prefetch_pred.assign(impl_->h_ids_map + top_k, impl_->h_ids_map + 2 * top_k);
        impl_->prefetch_target = next->routed;
        impl_->has_prefetch = true;
    }
    impl_->prefetch_expected = impl_->prefetch_pred;
    impl_->ensure(impl_->mix_dev, impl_->mix_cap, static_cast<size_t>(n_in));
    moe_core(impl_->norm_dev, impl_->mix_dev, routed, ids, weights, shared, shared_weight);
    impl_->has_prefetch = false;
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
    impl_->tl_mark(kTlAttnProject);
    impl_->decode_group(static_cast<int>(w.q->type), proj_weights, proj_inputs,
                                             proj_outputs, proj_rows, 3, kQ + 2 * kKv, kHidden,
                                             impl_->stream);
    void* q_norm = impl_->weight(*w.q_norm, w.q_norm_data, -1,
                                 static_cast<size_t>(kHeadDim) * sizeof(float));
    void* k_norm = impl_->weight(*w.k_norm, w.k_norm_data, -1,
                                 static_cast<size_t>(kHeadDim) * sizeof(float));
    impl_->tl_mark(kTlAttnRope);
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
    impl_->tl_mark(kTlAttnCore);
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
    impl_->grow_kv(state, needed);
    cuda::kv_store(impl_->a_k, impl_->kv_offset(state.keys, position), kKv, impl_->half_kv, impl_->stream);
    cuda::kv_store(impl_->a_v, impl_->kv_offset(state.values, position), kKv, impl_->half_kv, impl_->stream);
    if (impl_->half_kv) cuda::attn_half_partials(impl_->a_q, state.keys, state.values, needed, 0, tiles,
                                               impl_->a_partial, impl_->stream);
    else cuda::attn_partials(impl_->a_q, state.keys, state.values, needed, 0, tiles,
                            kHeads, kKvHeads, kHeadDim, 1.0f / 16.0f, impl_->a_partial, impl_->stream);
    }
    cuda::attn_merge(impl_->a_q, impl_->a_partial, tiles, kHeads, kHeadDim, impl_->a_ctx, impl_->stream);
    impl_->tl_mark(kTlAttnOut);
    const void* out_weight = impl_->projection_weight(*w.out, w.out_data, -1);
    const float* out_input = impl_->a_ctx;
    float* out_output = impl_->mix_dev;
    const int out_rows = kHidden;
    impl_->decode_group(static_cast<int>(w.out->type), &out_weight, &out_input,
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
    const bool device = x.empty() && impl_->prefill_columns == columns;
    if (!device && x.size() != size_t(columns) * 2048) throw std::invalid_argument("invalid attention columns");
    if (!device) impl_->ensure(impl_->input, impl_->input_capacity, x.size());
    const float* input = device ? impl_->p_norm : impl_->input;
    if (!device) check(cudaMemcpyAsync(impl_->input, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice, impl_->stream), "upload attention columns");
    impl_->ensure(impl_->batch_q, impl_->batch_q_cap, size_t(columns) * 8192);
    impl_->ensure(impl_->batch_k, impl_->batch_k_cap, size_t(columns) * 512);
    impl_->ensure(impl_->batch_v, impl_->batch_v_cap, size_t(columns) * 512);
    project_columns_device(*w.q, w.q_data, input, impl_->batch_q, columns, -1);
    project_columns_device(*w.k, w.k_data, input, impl_->batch_k, columns, -1);
    project_columns_device(*w.v, w.v_data, input, impl_->batch_v, columns, -1);
    impl_->ensure(impl_->batch_ctx, impl_->batch_ctx_cap, size_t(columns) * 4096);
    // Fused attention has no per-key-tile partials. All query columns can
    // share one history sweep, bounded to a 32 MiB running accumulator.
    const int query_tile = columns;
    impl_->ensure(impl_->batch_accum, impl_->batch_accum_cap, size_t(query_tile) * 16 * 258);
    bool matrix = false;
#ifdef LAMINA_PREFILL_BLAS
    const char* matrix_setting = std::getenv("LAMINA_MATRIX_ATTN");
    matrix = columns >= 16 && (!matrix_setting || matrix_setting[0] != '0');
    if (matrix) {
        impl_->ensure_blas();
        const size_t bytes = impl_->fast ? 2 : 4;
        impl_->ensure(impl_->matrix_q, impl_->matrix_q_cap, size_t(columns) * 16 * 256 * bytes);
        impl_->ensure(impl_->matrix_k, impl_->matrix_k_cap, size_t(Impl::kKvTile) * 2 * 256 * bytes);
        impl_->ensure(impl_->matrix_v, impl_->matrix_v_cap, size_t(Impl::kKvTile) * 2 * 256 * bytes);
        impl_->ensure(impl_->matrix_scores, impl_->matrix_scores_cap, size_t(128) * 16 * Impl::kKvTile);
        impl_->ensure(impl_->matrix_output, impl_->matrix_output_cap, size_t(128) * 16 * 256);
        impl_->ensure(impl_->matrix_metadata, impl_->matrix_metadata_cap, size_t(128) * 16 * 2);
        if (impl_->fast) impl_->ensure(impl_->matrix_probability, impl_->matrix_probability_cap, size_t(128) * 16 * Impl::kKvTile * 2);
    }
#endif
    auto* qnorm = static_cast<const float*>(impl_->weight(*w.q_norm, w.q_norm_data, -1, 256 * sizeof(float)));
    auto* knorm = static_cast<const float*>(impl_->weight(*w.k_norm, w.k_norm_data, -1, 256 * sizeof(float)));
    if (device) {
        cuda::attn_norm_mrope_columns(impl_->batch_q,16,512,columns,qnorm,1e-6f,rope_base,impl_->p_positions,impl_->stream);
        cuda::attn_norm_mrope_columns(impl_->batch_k,2,256,columns,knorm,1e-6f,rope_base,impl_->p_positions,impl_->stream);
    } else for (int c = 0; c < columns; ++c) {
        const int p = rope_position + c;
        const auto pos = positions.empty() ? std::array<int, 3>{p, p, p} : positions[c];
        cuda::attn_norm_mrope(impl_->batch_q + size_t(c) * 8192, 16, 512, 256, 64, qnorm, 1e-6f, rope_base, pos[0], pos[1], pos[2], impl_->stream);
        cuda::attn_norm_mrope(impl_->batch_k + size_t(c) * 512, 2, 256, 256, 64, knorm, 1e-6f, rope_base, pos[0], pos[1], pos[2], impl_->stream);
    }
#ifdef LAMINA_PREFILL_BLAS
    if (matrix) cuda::attn_matrix_queries(impl_->batch_q, columns, impl_->matrix_q, impl_->fast, impl_->stream);
#endif
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
#ifdef LAMINA_PREFILL_BLAS
        if (matrix) impl_->matrix_tile(keys, values, count, start, position, columns);
        else
#endif
        cuda::attn_fused_columns(impl_->batch_q + size_t(query) * 8192, keys, values, count,
            start, position + query, query_count, impl_->batch_accum, start == 0, impl_->half_kv, impl_->stream);
        if (slot) check(cudaEventRecord(slot->consumed, impl_->stream), "record column KV consumer");
    }
    cuda::attn_columns_finish(impl_->batch_q + size_t(query) * 8192, impl_->batch_accum, query_count,
                              impl_->batch_ctx + size_t(query) * 4096, impl_->stream);
    }
    check(cudaGetLastError(), "launch causal attention columns");
    if (!device) impl_->ensure(impl_->output, impl_->output_capacity, size_t(columns) * 2048);
    project_columns_device(*w.out, w.out_data, impl_->batch_ctx, device ? impl_->p_mix : impl_->output, columns, -1);
    if (device) return {};
    std::vector<float> result(size_t(columns) * 2048);
    check(cudaMemcpyAsync(result.data(), impl_->output, result.size() * sizeof(float), cudaMemcpyDeviceToHost, impl_->stream), "download attention columns");
    check(cudaStreamSynchronize(impl_->stream), "finish attention columns");
    return result;
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

void CudaProjection::timeline_mark(TimelineStage stage) { impl_->tl_mark(stage); }

void CudaProjection::timeline_token() {
    auto& p = *impl_;
    if (!p.timeline) return;
    if (p.tl_used >= 2) {
        p.tl_mark(kTlCount);  // closing event for the last stage
        check(cudaEventSynchronize(p.tl_events[p.tl_used - 1]), "finish timeline token");
        // The first tokens include graph capture and cache fill.
        if (++p.tl_seen > 8) {
            for (size_t i = 0; i + 1 < p.tl_used; ++i) {
                float ms = 0;
                check(cudaEventElapsedTime(&ms, p.tl_events[i], p.tl_events[i + 1]), "timeline elapsed");
                p.tl_ms[size_t(p.tl_stages[i])] += ms;
            }
            if (++p.tl_tokens % 32 == 0) {
                static const char* names[kTlCount] = {"dense", "router", "host_gap", "moe_setup", "prefetch_wait",
                    "resident", "miss_wait", "miss_experts", "prefetch_issue", "combine", "tail", "attention",
                    "gdn_project", "gdn_small", "gdn_step", "gdn_out", "head_kernel", "head_copy_and_host",
                    "attn_project", "attn_rope_kv", "attn_core", "attn_out",
                    "ex_gate_up", "ex_shared_gate_up", "ex_swiglu_quant", "ex_down", "ex_shared_down"};
                double total = 0;
                for (double v : p.tl_ms) total += v;
                std::fprintf(stderr, "timeline tokens=%llu ms_per_token=%.3f", static_cast<unsigned long long>(p.tl_tokens),
                             total / double(p.tl_tokens));
                for (int i = 0; i < kTlCount; ++i) std::fprintf(stderr, " %s=%.3f", names[i], p.tl_ms[size_t(i)] / double(p.tl_tokens));
                std::fprintf(stderr, " host_cpu_start=%.3f host_setup=%.3f host_cpu_wait=%.3f host_admit=%.3f"
                             " host_scales=%.3f host_slots=%.3f host_tables=%.3f host_promote=%.3f host_head_download=%.3f\n",
                             p.tl_host[0] / double(p.tl_seen), p.tl_host[1] / double(p.tl_seen),
                             p.tl_host[2] / double(p.tl_seen), p.tl_host[3] / double(p.tl_seen),
                             p.tl_host[4] / double(p.tl_seen), p.tl_host[5] / double(p.tl_seen),
                             p.tl_host[6] / double(p.tl_seen), p.tl_host[7] / double(p.tl_seen),
                             p.tl_host[8] / double(p.tl_seen));
                if (!p.tl_layer_order.empty()) {
                    std::fprintf(stderr, "cpu_experts_by_layer");
                    for (const auto* t : p.tl_layer_order) std::fprintf(stderr, " %.2f", double(p.tl_cpu_by_layer[t]) / double(p.tl_seen));
                    std::fprintf(stderr, "\n");
                }
                if (p.cpu_experts) {
                    const auto c = p.cpu_experts->stats();
                    if (c.batches)
                        std::fprintf(stderr, "cpu_batches per_token=%.2f batch_us=%.1f first_claim_us=%.1f experts_per_batch=%.2f\n",
                                     double(c.batches) / double(p.tl_seen), 1000.0 * c.batch_ms / double(c.batches),
                                     1000.0 * c.first_claim_ms / double(c.batches), double(c.experts) / double(c.batches));
                }
            }
        }
    }
    p.tl_used = 0;
    p.tl_stages.clear();
}

CudaProjection::Stats CudaProjection::stats() const {
    Stats stats;
    if (device_profile()) {
        impl_->collect_times();
        std::fprintf(stderr, "timing cumulative dma_ms=%.3f expert_gpu_ms=%.3f staging_wait_ms=%.3f router_wait_ms=%.3f\n",
            impl_->copy_ms, impl_->expert_ms, impl_->staging_wait_ms, impl_->router_wait_ms);
    }
    stats.hits = impl_->stat_hits;
    stats.misses = impl_->stat_misses;
    stats.uploaded_bytes = impl_->stat_uploaded;
    stats.evicted_bytes = impl_->stat_evicted;
    stats.resident_bytes = impl_->cached_bytes;
    stats.cache_limit = impl_->cache_limit;
    stats.allocated_bytes = impl_->allocated_bytes;
    stats.memory_limit = impl_->memory_limit;
    stats.device_mallocs = impl_->stat_mallocs;
    stats.free_trims = impl_->stat_trims;
    stats.graph_hits = impl_->moe_graph_hits;
    stats.graph_misses = impl_->moe_graph_misses;
    stats.cpu_experts = impl_->stat_cpu_experts;
    stats.admitted = impl_->stat_admitted;
    stats.prefetch_uploaded = impl_->stat_pf_uploaded;
    stats.prefetch_predicted = impl_->stat_pf_predicted;
    stats.prefetch_useful = impl_->stat_pf_useful;
    return stats;
}

}  // namespace lamina::model
#else
namespace lamina::model {
struct CudaProjection::Impl {};
CudaProjection::CudaProjection(size_t, bool, bool, bool) { throw std::runtime_error("Lamina was built without CUDA"); }
CudaProjection::~CudaProjection() = default;
bool CudaProjection::supports(uint32_t) const { return false; }
std::vector<float> CudaProjection::normalize_columns(const strata::TensorInfo&, const uint8_t*, const std::vector<float>&, int, float) {
    throw std::runtime_error("Lamina was built without CUDA");
}
void CudaProjection::register_weight_ram(const uint8_t*, size_t) {}
void CudaProjection::prefill_layer(int) { throw std::runtime_error("CUDA unavailable"); }
void CudaProjection::prefill_long_upload(const std::vector<float>&, const std::vector<std::array<int, 3>>&) { throw std::runtime_error("CUDA unavailable"); }
void CudaProjection::prefill_long_tile(int,int) { throw std::runtime_error("CUDA unavailable"); }
void CudaProjection::prefill_long_store(int) { throw std::runtime_error("CUDA unavailable"); }
std::vector<float> CudaProjection::prefill_long_download() { throw std::runtime_error("CUDA unavailable"); }
void CudaProjection::prefill_upload(const std::vector<float>&, int, const std::vector<std::array<int, 3>>&) { throw std::runtime_error("CUDA unavailable"); }
void CudaProjection::prefill_rms(const strata::TensorInfo&, const uint8_t*, float) { throw std::runtime_error("CUDA unavailable"); }
void CudaProjection::prefill_add_mix() { throw std::runtime_error("CUDA unavailable"); }
std::vector<float> CudaProjection::prefill_download() { throw std::runtime_error("CUDA unavailable"); }
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
int CudaProjection::matvec_argmax(const strata::TensorInfo&, const uint8_t*, const std::vector<float>&, int) {
    throw std::runtime_error("Lamina was built without CUDA");
}
int CudaProjection::project_output(const strata::TensorInfo&, const uint8_t*, const std::vector<float>&, int64_t, int) {
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
void CudaProjection::hidden_project(const strata::TensorInfo&, const uint8_t*, const std::vector<float>&) {
    throw std::runtime_error("Lamina was built without CUDA");
}
bool CudaProjection::supports_pair() const { return false; }
void CudaProjection::pair_upload(const std::vector<float>&) { throw std::runtime_error("Lamina was built without CUDA"); }
std::vector<float> CudaProjection::pair_download() { throw std::runtime_error("Lamina was built without CUDA"); }
void CudaProjection::pair_rms(const strata::TensorInfo&, const uint8_t*, float) { throw std::runtime_error("Lamina was built without CUDA"); }
void CudaProjection::pair_add_mix() { throw std::runtime_error("Lamina was built without CUDA"); }
void CudaProjection::pair_delta(int, const GdnWeights&) { throw std::runtime_error("Lamina was built without CUDA"); }
void CudaProjection::pair_delta_layer(int, const GdnWeights&, const strata::TensorInfo&, const uint8_t*,
                                      const strata::TensorInfo&, const uint8_t*, float) {
    throw std::runtime_error("Lamina was built without CUDA");
}
void CudaProjection::pair_attention(int, const AttnWeights&, int, float, const std::array<std::array<int, 3>, 2>&) {
    throw std::runtime_error("Lamina was built without CUDA");
}
void CudaProjection::pair_moe(const strata::TensorInfo&, const uint8_t*, const strata::TensorInfo&, const uint8_t*,
                              const MoeWeights&, const MoeWeights&) {
    throw std::runtime_error("Lamina was built without CUDA");
}
void CudaProjection::pair_commit(bool) { throw std::runtime_error("Lamina was built without CUDA"); }
std::array<int, 2> CudaProjection::matvec_argmax2(const strata::TensorInfo&, const uint8_t*, const std::vector<float>&) {
    throw std::runtime_error("Lamina was built without CUDA");
}
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
                                  const MoeWeights&, const MoePrefetch*) {
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
void CudaProjection::timeline_mark(TimelineStage) {}
void CudaProjection::timeline_token() {}
}  // namespace lamina::model
#endif
