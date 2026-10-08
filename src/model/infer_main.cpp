#include "lamina/model/inference.hpp"
#include "lamina/model/sampling.hpp"
#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
int integer(const std::string& value) {
    int result = 0;
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
    if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() || result < 0)
        throw std::invalid_argument("invalid integer: " + value);
    return result;
}
void image(lamina::model::Inference& model, const std::string& path, int context) {
    std::ifstream input(std::filesystem::path(std::u8string(path.begin(), path.end())), std::ios::binary);
    int32_t header[5]{};
    input.read(reinterpret_cast<char*>(header), sizeof(header));
    if (!input || header[0] != 0x31455653 || header[1] < 1 || header[2] < 1 || header[3] < 1 ||
        int64_t(header[2]) * header[3] != header[1] || header[4] != 2048 ||
        header[1] > context - model.position()) throw std::invalid_argument("invalid image embedding file or context overflow");
    const auto payload = input.tellg(); input.seekg(0, std::ios::end);
    if (input.tellg() != payload + std::streamoff(header[1]) * 2048 * std::streamoff(sizeof(float)))
        throw std::invalid_argument("image embedding payload size mismatch");
    input.seekg(payload);
    const int base = model.rope_position();
    for (int start = 0; start < header[1]; start += 1024) {
        const int count = std::min(1024, header[1] - start);
        std::vector<float> embeddings(size_t(count) * 2048);
        std::vector<std::array<int, 3>> positions(count);
        input.read(reinterpret_cast<char*>(embeddings.data()), embeddings.size() * sizeof(float));
        if (!input) throw std::runtime_error("short image embedding read");
        for (int c = 0; c < count; ++c) {
            const int i = start + c;
            positions[c] = {base, base + i / header[2], base + i % header[2]};
        }
        model.prefill_embeddings(embeddings, positions);
    }
}
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: lamina-infer MODEL [--cuda] [--max-context N] [--kv-cache auto|device|host] [--kv-type f32|f16] [--compute-mode f32|fast] [--vram-limit-mb N] token...|--interactive|--prefix layers token...\n"); return 2;
    }
    try {
        bool cuda = false; int context = 32768, first = 2; size_t vram_mb = 0; std::string kv_cache = "auto", kv_type = "f32", compute_mode = "f32";
        for (; first < argc; ++first) {
            const std::string arg = argv[first];
            if (arg == "--cuda") cuda = true;
            else if (arg == "--max-context" || arg == "--kv-cache" || arg == "--kv-type" || arg == "--compute-mode" || arg == "--vram-limit-mb") {
                if (++first >= argc) throw std::invalid_argument("missing value for " + arg);
                if (arg == "--max-context") context = integer(argv[first]);
                else if (arg == "--kv-cache") kv_cache = argv[first];
                else if (arg == "--compute-mode") compute_mode = argv[first];
                else if (arg == "--kv-type") kv_type = argv[first]; else vram_mb = integer(argv[first]);
            } else break;
        }
        if (first >= argc) throw std::invalid_argument("missing token or mode");
        if (std::string(argv[first]) == "--prefix" || std::string(argv[first]) == "--batch-prefix") {
            const bool batched = std::string(argv[first]) == "--batch-prefix";
            if (argc < first + 3) throw std::invalid_argument("prefix requires layers and tokens");
            const int layers = integer(argv[++first]);
            lamina::model::Inference model(argv[1], context, layers, cuda, kv_cache, vram_mb, kv_type, compute_mode); std::vector<float> hidden;
            std::vector<int> tokens;
            while (++first < argc) tokens.push_back(integer(argv[first]));
            if (batched) hidden = model.prefill_hidden(tokens);
            else for (int token : tokens) hidden = model.step_hidden(token);
            for (float value : hidden) std::printf("%.9g\n", value); return 0;
        }
        lamina::model::Inference model(argv[1], context, 40, cuda, kv_cache, vram_mb, kv_type, compute_mode); lamina::model::Sampler sampler;
        // Greedy sampling selects the token on the GPU and skips downloading the
        // logits; any other setting, or a non-finite logit, takes the full path.
        const auto next_token = [&](std::vector<float> hidden) {
            if (sampler.greedy())
                if (const int token = model.greedy(hidden); token >= 0) return token;
            return sampler.sample(model.logits(std::move(hidden)));
        };
        if (argc == first + 1 && std::string(argv[first]) == "--interactive") {
            std::string line;
            // LAMINA_TIMELINE=1: time spent waiting for the client's next command.
            const char* timeline = std::getenv("LAMINA_TIMELINE");
            const bool time_input = timeline && timeline[0] == '1';
            double input_ms = 0; uint64_t commands = 0;
            for (;;) {
                const auto waited = std::chrono::steady_clock::now();
                if (!std::getline(std::cin, line)) break;
                if (time_input) {
                    input_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - waited).count();
                    if (++commands % 64 == 0)
                        std::fprintf(stderr, "client_wait commands=%llu ms_per_command=%.3f\n",
                                     static_cast<unsigned long long>(commands), input_ms / double(commands));
                }
                if (line == "QUIT") break;
                if (line == "RESET") { model.reset(); std::puts("."); }
                else if (line.rfind("SAMPLE ", 0) == 0) {
                    std::istringstream args(line.substr(7)); double temperature, p; int k; uint64_t seed; std::string extra;
                    if (!(args >> temperature >> p >> k >> seed) || args >> extra) throw std::invalid_argument("invalid SAMPLE command");
                    sampler.configure(temperature, p, k, seed); std::puts(".");
                } else if (line.rfind("IMAGE ", 0) == 0) { image(model, line.substr(6), context); std::puts("."); }
                else if (line.rfind("PROMPT ",0)==0) {
                    std::istringstream args(line.substr(7));std::string word;
                    if(!(args>>word))throw std::invalid_argument("PROMPT requires chunk and tokens");
                    const int chunk=integer(word);std::vector<int> tokens;
                    while(args>>word)tokens.push_back(integer(word));
                    auto hidden=model.prefill_long(tokens,chunk);
                    std::printf("%d\n",next_token(std::move(hidden)));
                }
                else if (line.rfind("PREFILL ", 0) == 0 || line.rfind("BATCH ", 0) == 0) {
                    const bool logits = line.rfind("PREFILL ", 0) == 0; std::istringstream args(line.substr(logits ? 8 : 6));
                    std::vector<int> tokens; std::string word; while (args >> word) tokens.push_back(integer(word));
                    if (tokens.empty() || tokens.size() > size_t(context - model.position())) throw std::invalid_argument("empty or oversized prefill");
                    auto hidden = model.prefill_hidden(tokens);
                    if (logits) std::printf("%d\n", next_token(std::move(hidden)));
                    else std::puts(".");
                } else {
                    const bool hidden_only = !line.empty() && line[0] == '+'; const int token = integer(line.substr(hidden_only ? 1 : 0));
                    if (hidden_only) { model.step_hidden(token); std::puts("."); } else std::printf("%d\n", next_token(model.step_hidden(token)));
                }
                std::fflush(stdout);
            }
            return 0;
        }
        for (; first < argc; ++first) {
            const int token = integer(argv[first]); auto logits = model.step(token); const int next = sampler.sample(logits);
            std::printf("position=%d token=%d next=%d logit=%.6f\n", model.position(), token, next, logits[next]); std::fflush(stdout);
        }
        return 0;
    } catch (const std::exception& error) { std::fprintf(stderr, "lamina-infer: %s\n", error.what()); return 1; }
}
