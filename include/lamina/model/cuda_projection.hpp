#pragma once

#include "strata/artifact/gguf_reader.hpp"

#include <cstdint>
#include <memory>
#include <vector>

namespace lamina::model {

// Optional hybrid backend: GGUF matrix-vector projections run on CUDA while
// Qwen3.6 recurrence, attention state, and routing stay in the scalar graph.
class CudaProjection {
public:
    CudaProjection();
    ~CudaProjection();
    CudaProjection(const CudaProjection&) = delete;
    CudaProjection& operator=(const CudaProjection&) = delete;

    bool supports(uint32_t ggml_type) const;
    std::vector<float> matvec(const strata::TensorInfo& tensor, const uint8_t* data,
                              const std::vector<float>& x, int64_t expert);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace lamina::model
