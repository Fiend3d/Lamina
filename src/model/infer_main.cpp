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
                             "       lamina-infer <model.gguf> --interactive\n"
                             "Runs the scalar reference path and prints the highest-logit token.\n");
        return 2;
    }
    try {
        lamina::model::Inference model(argv[1]);
        if (argc == 3 && std::string(argv[2]) == "--interactive") {
            std::string line;
            while (std::getline(std::cin, line)) {
                int token = -1;
                const char* end = line.data() + line.size();
                if (std::from_chars(line.data(), end, token).ec != std::errc{} || token < 0)
                    throw std::invalid_argument("invalid token id");
                auto logits = model.step(token);
                const int next = static_cast<int>(std::max_element(logits.begin(), logits.end()) - logits.begin());
                std::printf("%d\n", next);
                std::fflush(stdout);
            }
            return 0;
        }
        for (int i = 2; i < argc; ++i) {
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
