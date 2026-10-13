#include "lamina/model/sampling.hpp"
#include <cmath>
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
        // Verify the rejection correction against target distributions, not
        // just against this implementation's control flow.
        for (const auto& proposal : std::vector<std::vector<double>>{{.9,.05,.05}, {0,1,0}, {.2,.3,.5}}) {
            const std::vector<double> target{.2,.3,.5};
            a.configure(.6, .95, 20, 2718);
            std::vector<int> counts(3);
            for (int i = 0; i < 100000; ++i) {
                const int draft = a.sample_distribution(proposal);
                const auto [token, accepted] = a.verify_draft(target, proposal, draft);
                if (accepted && token != draft) throw std::runtime_error("accepted draft changed");
                ++counts[token];
            }
            for (size_t i = 0; i < target.size(); ++i)
                if (std::abs(double(counts[i]) / 100000 - target[i]) > .008)
                    throw std::runtime_error("speculative distribution biased");
        }
        a.configure(.7, .01, 0, 19);
        const auto filtered = a.distribution(logits);
        if (filtered != std::vector<double>({0,1,0,0})) throw std::runtime_error("proposal top-p filter differs");
        if (a.verify_draft({1,0}, {0,1}, 1) != std::pair<int,bool>{0,false})
            throw std::runtime_error("disjoint support correction failed");
        // Presence penalty: generated tokens lose a fixed amount once, and
        // configure() starts a new history.
        const std::vector<float> scores{0.0f, 2.0f, 1.5f};
        a.configure(0, 1, 20, 0, 1.0);
        if (a.sample(scores) != 1) throw std::runtime_error("penalty applied without history");
        a.accept(1); a.accept(1);
        if (a.sample(scores) != 2) throw std::runtime_error("greedy presence penalty");
        if (a.device_greedy() || !a.penalized()) throw std::runtime_error("penalized greedy must take the logits path");
        a.configure(0, 1, 20, 0, 1.0);
        if (a.sample(scores) != 1) throw std::runtime_error("configure kept the penalty history");
        a.configure(0, 1, 20, 0, -1.0); a.accept(2);
        if (a.sample(scores) != 2) throw std::runtime_error("negative presence penalty");
        a.configure(1, 1, 0, 7, 0.5); a.accept(2);
        const auto penalized = a.distribution({0.0f, 1.0f, 1.0f});
        const double z = std::exp(0.0) + std::exp(1.0) + std::exp(0.5);
        const std::vector<double> exact{std::exp(0.0) / z, std::exp(1.0) / z, std::exp(0.5) / z};
        for (size_t i = 0; i < exact.size(); ++i)
            if (std::abs(penalized[i] - exact[i]) > 1e-12) throw std::runtime_error("penalized distribution differs");
        std::vector<int> draws(3);
        for (int i = 0; i < 100000; ++i) ++draws[size_t(a.sample({0.0f, 1.0f, 1.0f}))];
        for (size_t i = 0; i < exact.size(); ++i)
            if (std::abs(double(draws[i]) / 100000 - exact[i]) > .008) throw std::runtime_error("penalized sampling biased");
        for (const double invalid : {2.5, -2.5, std::numeric_limits<double>::quiet_NaN()}) {
            rejected = false;
            try { a.configure(1, 1, 20, 0, invalid); } catch (const std::invalid_argument&) { rejected = true; }
            if (!rejected) throw std::runtime_error("invalid presence penalty accepted");
        }
        std::puts("sampling checks passed"); return 0;
    } catch (const std::exception& error) { std::fprintf(stderr, "%s\n", error.what()); return 1; }
}
