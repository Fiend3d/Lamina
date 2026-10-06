#include "lamina/model/cuda_projection.hpp"
#include "strata/artifact/dequant.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void decode_row(const strata::TensorInfo& tensor, const uint8_t* source, float* output) {
    const int64_t width = static_cast<int64_t>(tensor.shape.at(0));
    int block_elements = 0;
    int block_bytes = 0;
    if (!strata::block_geometry(tensor.type, block_elements, block_bytes) || width % block_elements)
        throw std::runtime_error("unsupported row encoding: " + tensor.name);
    for (int64_t i = 0; i < width; i += block_elements, source += block_bytes) {
        switch (tensor.type) {
        case 8: strata::dequantize_q8_0(source, output + i); break;
        case 12: strata::dequantize_q4_K(source, output + i); break;
        case 13: strata::dequantize_q5_K(source, output + i); break;
        case 14: strata::dequantize_q6_K(source, output + i); break;
        default: throw std::runtime_error("unexpected tensor type: " + tensor.name);
        }
    }
}

bool check_projection(lamina::model::CudaProjection& cuda, const strata::GgufFile& file,
                      const std::string& name, int64_t expert) {
    const auto* tensor = file.find(name);
    if (!tensor) throw std::runtime_error("missing tensor: " + name);
    const int64_t n_in = static_cast<int64_t>(tensor->shape.at(0));
    const int64_t n_out = static_cast<int64_t>(tensor->shape.at(1));
    if (tensor->shape.size() != (expert < 0 ? 2u : 3u))
        throw std::runtime_error("unexpected tensor rank: " + name);

    std::vector<float> input(static_cast<size_t>(n_in));
    for (int64_t i = 0; i < n_in; ++i)
        input[static_cast<size_t>(i)] = 0.75f * std::sin(static_cast<float>(i) * 0.013f) +
                                       0.25f * std::cos(static_cast<float>(i) * 0.071f);

    const uint64_t experts = expert < 0 ? 1 : tensor->shape.at(2);
    const uint64_t row_bytes = strata::tensor_payload_bytes(*tensor) /
                               (static_cast<uint64_t>(n_out) * experts);
    const auto* weights = file.tensor_data(*tensor) +
                          (expert < 0 ? 0 : static_cast<uint64_t>(expert) * n_out * row_bytes);
    const auto actual = cuda.matvec(*tensor, file.tensor_data(*tensor), input, expert);
    double repeat_max = 0.0;
    for (int attempt = 0; attempt < 2; ++attempt) {
        const auto repeated = cuda.matvec(*tensor, file.tensor_data(*tensor), input, expert);
        for (size_t i = 0; i < actual.size(); ++i)
            repeat_max = std::max(repeat_max, std::abs(double(actual[i]) - repeated[i]));
    }

    std::vector<float> row(static_cast<size_t>(n_in));
    double error_squared = 0.0;
    double reference_squared = 0.0;
    double max_error = 0.0;
    double max_reference = 0.0;
    size_t zero_rows = 0;
    size_t reference_nonzero_rows = 0;
    std::vector<double> reference_values;
    reference_values.reserve(static_cast<size_t>(n_out));
    for (int64_t r = 0; r < n_out; ++r) {
        decode_row(*tensor, weights + static_cast<uint64_t>(r) * row_bytes, row.data());
        double reference = 0.0;
        for (int64_t i = 0; i < n_in; ++i)
            reference += static_cast<double>(row[static_cast<size_t>(i)]) * input[static_cast<size_t>(i)];
        const double got = actual[static_cast<size_t>(r)];
        const double error = got - reference;
        reference_values.push_back(reference);
        error_squared += error * error;
        reference_squared += reference * reference;
        max_error = std::max(max_error, std::abs(error));
        max_reference = std::max(max_reference, std::abs(reference));
        zero_rows += got == 0.0;
        reference_nonzero_rows += std::abs(reference) > 1e-6;
    }
    const double relative_l2 = std::sqrt(error_squared / std::max(reference_squared, 1e-30));
    std::printf("%s type=%s shape=%lldx%lld expert=%lld relative_l2=%.6g max_abs=%.6g repeat_max=%.6g "
                "zero_rows=%zu/%lld reference_nonzero=%zu first(cuda/ref)=%.6g/%.6g\n",
                name.c_str(), tensor->type_name(), static_cast<long long>(n_in),
                static_cast<long long>(n_out), static_cast<long long>(expert), relative_l2,
                max_error, repeat_max, zero_rows, static_cast<long long>(n_out), reference_nonzero_rows,
                actual.empty() ? 0.0f : actual[0], reference_values.empty() ? 0.0 : reference_values[0]);
    const bool passed = relative_l2 <= 0.01 && max_error <= 0.02 + 0.01 * max_reference;
    if (!passed)
        std::fprintf(stderr, "  PARITY FAILURE: %s\n", name.c_str());
    return passed;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: lamina-cuda-projection-check <model.gguf>\n");
        return 2;
    }
    try {
        strata::GgufFile file(argv[1]);
        bool passed = true;
        for (const auto& [name, expert] : std::vector<std::pair<std::string, int64_t>>{
                 {"blk.0.attn_gate.weight", -1}, {"blk.0.ffn_gate_exps.weight", 0},
                 {"blk.0.ffn_up_exps.weight", 112}, {"blk.0.ffn_down_exps.weight", 112},
                 {"blk.34.ffn_down_exps.weight", 0}}) {
            lamina::model::CudaProjection cuda;
            passed = check_projection(cuda, file, name, expert) && passed;
        }
        if (!passed) return 1;
        std::puts("CUDA projection checks passed");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "CUDA projection check failed: %s\n", error.what());
        return 1;
    }
}
