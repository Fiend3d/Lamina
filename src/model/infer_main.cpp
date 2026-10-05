#include "lamina/model/inference.hpp"

#include <algorithm>
#include <charconv>
#include <cstdio>
#include <exception>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: lamina-infer <model.gguf> <token-id> [token-id ...]\n"
                             "       lamina-infer <model.gguf> [--cuda] --interactive\n"
                             "       lamina-infer <model.gguf> --prefix <layers> <token-id> [token-id ...]\n"
                             "--cuda uses the hybrid GPU projection path when built with CUDA.\n");
        return 2;
    }
    try {
        const bool cuda = std::string(argv[2]) == "--cuda";
        const int first = cuda ? 3 : 2;
        if (argc <= first) throw std::invalid_argument("missing token or mode");
        if (argc >= first + 3 && std::string(argv[first]) == "--prefix") {
            int layers = 0;
            const std::string layer_arg = argv[first + 1];
            const char* layer_end = layer_arg.data() + layer_arg.size();
            if (std::from_chars(layer_arg.data(), layer_end, layers).ec != std::errc{})
                throw std::invalid_argument("invalid layer count");
            lamina::model::Inference model(argv[1], 32768, layers, cuda);
            std::vector<float> hidden;
            for (int i = first + 2; i < argc; ++i) {
                int token = -1;
                const char* end = argv[i] + std::char_traits<char>::length(argv[i]);
                if (std::from_chars(argv[i], end, token).ec != std::errc{} || token < 0)
                    throw std::invalid_argument("invalid token id");
                hidden = model.step_hidden(token);
            }
            for (float value : hidden) std::printf("%.9g\n", value);
            return 0;
        }
        lamina::model::Inference model(argv[1], 32768, 40, cuda);
        if (argc == first + 1 && std::string(argv[first]) == "--interactive") {
            // A leading '+' consumes a prompt token without computing logits;
            // the final prompt token and generated tokens have no prefix.
            std::string line;
            while (std::getline(std::cin, line)) {
                const bool hidden_only = !line.empty() && line[0] == '+';
                const char* begin = line.data() + (hidden_only ? 1 : 0);
                int token = -1;
                const char* end = line.data() + line.size();
                if (std::from_chars(begin, end, token).ec != std::errc{} || token < 0)
                    throw std::invalid_argument("invalid token id");
                if (hidden_only) {
                    model.step_hidden(token);
                    std::puts(".");
                    std::fflush(stdout);
                    continue;
                }
                auto logits = model.step(token);
                const int next = static_cast<int>(std::max_element(logits.begin(), logits.end()) - logits.begin());
                std::printf("%d\n", next);
                std::fflush(stdout);
            }
            return 0;
        }
        for (int i = first; i < argc; ++i) {
            int token = -1;
            const char* end = argv[i] + std::char_traits<char>::length(argv[i]);
            if (std::from_chars(argv[i], end, token).ec != std::errc{} || token < 0)
                throw std::invalid_argument("invalid token id");
            auto logits = model.step(token);
            const int next = static_cast<int>(std::max_element(logits.begin(), logits.end()) - logits.begin());
            std::printf("position=%d token=%d next=%d logit=%.6f\n", model.position(), token,
                        next, logits[next]);
            std::fflush(stdout);
        }
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "lamina-infer: %s\n", error.what());
        return 1;
    }
}
