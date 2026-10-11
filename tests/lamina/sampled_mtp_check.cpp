#include "lamina/model/inference.hpp"
#include "lamina/model/sampling.hpp"
#include <cstdio>
#include <stdexcept>
#include <algorithm>
#include <cmath>

int main(int argc, char** argv) {
    if (argc != 3) return 2;
    try {
        lamina::model::Inference model(argv[1], 4096, 40, true, "device", 0, "f16", "fast");
        model.load_mtp(argv[2]);
        model.prefill_hidden({42,43,44,45,46,47,48,49});
        model.cache_prefix();
        auto hidden = model.step_hidden(50);
        int token = model.greedy(hidden);
        auto expected = model.generate_greedy(token, hidden, 24);
        model.restore_prefix(); hidden = model.step_hidden(50);
        lamina::model::Sampler sampler; sampler.configure(.6, .95, 1, 42);
        token = sampler.sample(model.logits(hidden));
        const auto actual = model.generate_sampled(token, hidden, 24, sampler);
        if (actual != expected) throw std::runtime_error("top-k=1 sampled MTP differs from greedy");
        const int first_position = model.position();
        model.restore_prefix(); hidden = model.step_hidden(50);
        sampler.configure(.6, .95, 20, 2718);
        token = sampler.sample(model.logits(hidden));
        const int initial_sample = token;
        const auto first = model.generate_sampled(token, hidden, 24, sampler);
        const auto speculative_hidden = hidden;
        if (model.position() != first_position) throw std::runtime_error("sampled MTP position accounting differs");
        model.restore_prefix(); hidden = model.step_hidden(50);
        sampler.configure(.6, .95, 20, 2718);
        token = sampler.sample(model.logits(hidden));
        const auto repeated = model.generate_sampled(token, hidden, 24, sampler);
        if (first != repeated) throw std::runtime_error("sampled MTP seed/cache replay differs");
        model.restore_prefix(); hidden = model.step_hidden(50);
        hidden = model.step_hidden(initial_sample);
        for (size_t i = 0; i + 1 < first.size(); ++i) hidden = model.step_hidden(first[i]);
        double error = 0;
        for (size_t i = 0; i < hidden.size(); ++i) error = std::max(error, std::abs(double(hidden[i]) - speculative_hidden[i]));
        std::printf("sampled/serial same-token hidden max_abs_diff=%.9g\n", error);
        if (error >= 1e-4) throw std::runtime_error("sampled MTP rollback/continuation state differs from serial");
        // Ending with a single token must use ordinary sampling without a pair.
        const int before = model.position();
        model.generate_sampled(repeated.back(), hidden, 1, sampler);
        if (model.position() != before + 1) throw std::runtime_error("single-token sampled tail advanced twice");
        std::printf("sampled MTP: top-k=1 greedy parity, seeded replay and positions passed; steps=%llu accepted=%llu\n",
            static_cast<unsigned long long>(model.spec_stats().steps),
            static_cast<unsigned long long>(model.spec_stats().accepted));
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
