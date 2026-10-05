#pragma once

#include "strata/artifact/gguf_reader.hpp"

#include <map>
#include <string>
#include <vector>

namespace lamina::model {

// The published UD-Q4_K_M artifact's tensor geometry. This is independent of
// the inherited Qwen3.8 layer/weight layout in strata::core.
inline std::map<std::string, std::vector<uint64_t>> qwen36_shapes() {
    using Shape = std::vector<uint64_t>;
    std::map<std::string, Shape> result{
        {"token_embd.weight", {2048, 248320}},
        {"output_norm.weight", {2048}},
        {"output.weight", {2048, 248320}},
    };
    const std::map<std::string, Shape> common{
        {"attn_norm.weight", {2048}},
        {"post_attention_norm.weight", {2048}},
        {"ffn_gate_inp.weight", {2048, 256}},
        {"ffn_gate_inp_shexp.weight", {2048}},
        {"ffn_gate_shexp.weight", {2048, 512}},
        {"ffn_up_shexp.weight", {2048, 512}},
        {"ffn_down_shexp.weight", {512, 2048}},
        {"ffn_gate_exps.weight", {2048, 512, 256}},
        {"ffn_up_exps.weight", {2048, 512, 256}},
        {"ffn_down_exps.weight", {512, 2048, 256}},
    };
    const std::map<std::string, Shape> linear{
        {"attn_qkv.weight", {2048, 8192}},
        {"attn_gate.weight", {2048, 4096}},
        {"ssm_out.weight", {4096, 2048}},
        {"ssm_conv1d.weight", {4, 8192}},
        {"ssm_alpha.weight", {2048, 32}},
        {"ssm_beta.weight", {2048, 32}},
        {"ssm_a", {32}},
        {"ssm_dt.bias", {32}},
        {"ssm_norm.weight", {128}},
    };
    const std::map<std::string, Shape> attention{
        {"attn_q.weight", {2048, 8192}},
        {"attn_k.weight", {2048, 512}},
        {"attn_v.weight", {2048, 512}},
        {"attn_output.weight", {4096, 2048}},
        {"attn_q_norm.weight", {256}},
        {"attn_k_norm.weight", {256}},
    };
    for (int layer = 0; layer < 40; ++layer) {
        const std::string prefix = "blk." + std::to_string(layer) + ".";
        for (const auto& [name, shape] : common) result.emplace(prefix + name, shape);
        for (const auto& [name, shape] : (layer % 4 == 3 ? attention : linear))
            result.emplace(prefix + name, shape);
    }
    return result;
}

inline std::string check_qwen36_tensors(const strata::GgufFile& file) {
    const auto shapes = qwen36_shapes();
    if (file.tensors().size() != shapes.size())
        return "GGUF has " + std::to_string(file.tensors().size()) + " tensors, expected " +
               std::to_string(shapes.size());
    for (const auto& tensor : file.tensors()) {
        const auto found = shapes.find(tensor.name);
        if (found == shapes.end()) return "unexpected tensor " + tensor.name;
        if (tensor.shape != found->second) return "wrong shape for " + tensor.name;
        int block_elements = 0, block_bytes = 0;
        if (!strata::block_geometry(tensor.type, block_elements, block_bytes) ||
            tensor.elements() % static_cast<uint64_t>(block_elements))
            return "unsupported tensor encoding for " + tensor.name;
        if (tensor.type != 0 && tensor.type != 8 && tensor.type != 12 &&
            tensor.type != 13 && tensor.type != 14)
            return "tensor encoding is not implemented by Lamina for " + tensor.name;
    }
    return {};
}

}  // namespace lamina::model
