#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>
namespace lamina::model {
class Sampler {
    double temperature_ = 0, top_p_ = 1;
    int top_k_ = 20;
    std::mt19937_64 random_{0};
public:
    void configure(double temperature, double top_p, int top_k, uint64_t seed) {
        if (!std::isfinite(temperature) || temperature < 0 || temperature > 10 ||
            !std::isfinite(top_p) || top_p <= 0 || top_p > 1 || top_k < 0 || top_k > 248320)
            throw std::invalid_argument("invalid temperature, top_p or top_k");
        temperature_ = temperature; top_p_ = top_p; top_k_ = top_k; random_.seed(seed);
    }
    int sample(const std::vector<float>& logits) {
        if (logits.empty() || !std::all_of(logits.begin(), logits.end(), [](float v) { return std::isfinite(v); }))
            throw std::runtime_error("non-finite or empty logits");
        if (temperature_ == 0) return int(std::max_element(logits.begin(), logits.end()) - logits.begin());
        std::vector<int> ids(logits.size()); std::iota(ids.begin(), ids.end(), 0);
        const size_t keep = top_k_ ? std::min(size_t(top_k_), ids.size()) : ids.size();
        auto greater = [&](int a, int b) { return logits[a] == logits[b] ? a < b : logits[a] > logits[b]; };
        std::partial_sort(ids.begin(), ids.begin() + keep, ids.end(), greater); ids.resize(keep);
        std::vector<double> probabilities(keep); double sum = 0;
        for (size_t i = 0; i < keep; ++i) sum += probabilities[i] = std::exp((double(logits[ids[i]]) - logits[ids[0]]) / temperature_);
        double cumulative = 0; size_t nucleus = 0;
        do { cumulative += probabilities[nucleus++] / sum; } while (nucleus < keep && cumulative < top_p_);
        probabilities.resize(nucleus);
        return ids[std::discrete_distribution<size_t>(probabilities.begin(), probabilities.end())(random_)];
    }
};
}
