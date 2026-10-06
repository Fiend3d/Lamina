#include "lamina/model/sampling.hpp"
#include <cstdio>
#include <limits>
#include <stdexcept>

int main() {
    try {
        lamina::model::Sampler a, b;
        const std::vector<float> logits{-100.0f, 2.0f, 1.5f, 2.0f};
        if (a.sample(logits) != 1) throw std::runtime_error("greedy tie ordering");
        a.configure(1, 1, 2, 42); b.configure(1, 1, 2, 42);
        for (int i = 0; i < 100; ++i) {
            const int token = a.sample(logits);
            if (token != b.sample(logits) || (token != 1 && token != 3)) throw std::runtime_error("seed or top-k mismatch");
        }
        a.configure(0.7, 0.01, 0, 19);
        for (int i = 0; i < 20; ++i) if (a.sample(logits) != 1) throw std::runtime_error("top-p cutoff");
        bool rejected = false;
        try { a.configure(1, 0, 20, 0); } catch (const std::invalid_argument&) { rejected = true; }
        if (!rejected) throw std::runtime_error("invalid options accepted");
        rejected = false;
        try { a.sample({std::numeric_limits<float>::quiet_NaN()}); } catch (const std::runtime_error&) { rejected = true; }
        if (!rejected) throw std::runtime_error("non-finite logits accepted");
        std::puts("sampling checks passed"); return 0;
    } catch (const std::exception& error) { std::fprintf(stderr, "%s\n", error.what()); return 1; }
}
