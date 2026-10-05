#include "lamina/model/cuda_projection.hpp"

#include <stdexcept>

#ifdef LAMINA_ENABLE_CUDA
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <charconv>
#include <cstdlib>
#include <string>
#include <unordered_map>

namespace lamina::model {
namespace {
void check(cudaError_t status, const char* action) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("CUDA ") + action + ": " + cudaGetErrorString(status));
}
constexpr size_t MIB = 1024ULL * 1024;
}

struct CudaProjection::Impl {
    struct Entry {
        void* device = nullptr;
        size_t bytes = 0;
        uint64_t used = 0;
    };
    std::unordered_map<std::string, Entry> weights;
    size_t cached_bytes = 0;
    size_t cache_limit = 0;
    uint64_t clock = 0;
    cudaStream_t stream = nullptr;
    float* input = nullptr;
    float* output = nullptr;
    void* q8 = nullptr;
    size_t input_capacity = 0, output_capacity = 0, q8_capacity = 0;

    ~Impl() {
        for (auto& [name, entry] : weights) cudaFree(entry.device);
        cudaFree(input);
        cudaFree(output);
        cudaFree(q8);
        if (stream) cudaStreamDestroy(stream);
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
        const size_t needed = strata::kernels::native_q8_1_bytes(static_cast<int>(n_in));
        if (needed > q8_capacity) {
            check(cudaFree(q8), "free Q8 scratch");
            q8 = nullptr;
            check(cudaMalloc(&q8, needed), "allocate Q8 scratch");
            q8_capacity = needed;
        }
    }

    bool evict_oldest() {
        if (weights.empty()) return false;
        const auto oldest = std::min_element(weights.begin(), weights.end(),
            [](const auto& a, const auto& b) { return a.second.used < b.second.used; });
        check(cudaFree(oldest->second.device), "evict weight");
        cached_bytes -= oldest->second.bytes;
        weights.erase(oldest);
        return true;
    }

    void* weight(const strata::TensorInfo& tensor, const uint8_t* data,
                 int64_t expert, size_t bytes) {
        const std::string key = tensor.name + "#" + std::to_string(expert);
        if (auto found = weights.find(key); found != weights.end()) {
            found->second.used = ++clock;
            return found->second.device;
        }
        while (bytes > cache_limit - std::min(cached_bytes, cache_limit) && evict_oldest()) {}
        void* device = nullptr;
        auto status = cudaMalloc(&device, bytes);
        while (status == cudaErrorMemoryAllocation && evict_oldest())
            status = cudaMalloc(&device, bytes);
        check(status, "allocate weight");
        try {
            check(cudaMemcpy(device, data, bytes, cudaMemcpyHostToDevice), "upload weight");
            weights.emplace(key, Entry{device, bytes, ++clock});
            cached_bytes += bytes;
            return device;
        } catch (...) {
            cudaFree(device);
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
    strata::kernels::native_quantize_q8_1(impl_->input, impl_->q8, n_in, 1, impl_->stream);
    strata::kernels::native_mmvq(tensor.type, device_weight, impl_->q8, impl_->output,
                                 n_in, n_out, 1, impl_->stream);
    check(cudaGetLastError(), "launch projection");
    std::vector<float> result(static_cast<size_t>(n_out));
    check(cudaMemcpyAsync(result.data(), impl_->output, result.size() * sizeof(float),
                          cudaMemcpyDeviceToHost, impl_->stream), "download projection");
    check(cudaStreamSynchronize(impl_->stream), "finish projection");
    return result;
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
}  // namespace lamina::model
#endif
