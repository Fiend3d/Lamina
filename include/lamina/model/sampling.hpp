#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>
#include <utility>
namespace lamina::model {
class Sampler {
    double temperature_ = 0, top_p_ = 1, presence_ = 0;
    int top_k_ = 20;
    std::mt19937_64 random_{0};
    // Presence penalty (OpenAI/vLLM semantics): every token generated since the
    // last configure() has `presence_` subtracted from its logit, once,
    // however often it occurred. Prompt tokens are not counted.
    std::vector<uint8_t> seen_;
    std::vector<int> seen_ids_;
    std::vector<float> penalized_;
    const std::vector<float>& penalize(const std::vector<float>& logits) {
        if (presence_ == 0 || seen_ids_.empty()) return logits;
        penalized_ = logits;
        for (const int id : seen_ids_)
            if (size_t(id) < penalized_.size()) penalized_[size_t(id)] -= float(presence_);
        return penalized_;
    }
public:
    void configure(double temperature, double top_p, int top_k, uint64_t seed, double presence_penalty = 0) {
        if (!std::isfinite(temperature) || temperature < 0 || temperature > 10 ||
            !std::isfinite(top_p) || top_p <= 0 || top_p > 1 || top_k < 0 || top_k > 248320 ||
            !std::isfinite(presence_penalty) || presence_penalty < -2 || presence_penalty > 2)
            throw std::invalid_argument("invalid temperature, top_p, top_k or presence_penalty");
        temperature_ = temperature; top_p_ = top_p; top_k_ = top_k; random_.seed(seed);
        presence_ = presence_penalty;
        for (const int id : seen_ids_) seen_[size_t(id)] = 0;
        seen_ids_.clear();
    }
    bool greedy() const { return temperature_ == 0; }
    bool penalized() const { return presence_ != 0; }
    // Greedy selection on the GPU cannot apply the penalty.
    bool device_greedy() const { return greedy() && !penalized(); }
    // Records an emitted token; later draws in this request are penalized for it.
    void accept(int token) {
        if (token < 0) return;
        if (seen_.size() <= size_t(token)) seen_.resize(size_t(token) + 1, 0);
        if (!seen_[size_t(token)]) { seen_[size_t(token)] = 1; seen_ids_.push_back(token); }
    }
    // Dense normalized target/proposal probabilities, after the same penalty,
    // temperature, top-k and top-p filters as ordinary sampling.
    std::vector<double> distribution(const std::vector<float>& raw) {
        if (raw.empty() || !std::all_of(raw.begin(), raw.end(), [](float v) { return std::isfinite(v); }))
            throw std::runtime_error("non-finite or empty logits");
        const auto& logits = penalize(raw);
        std::vector<double> result(logits.size(), 0);
        if (greedy()) {
            result[size_t(std::max_element(logits.begin(), logits.end()) - logits.begin())] = 1;
            return result;
        }
        std::vector<int> ids(logits.size()); std::iota(ids.begin(), ids.end(), 0);
        const size_t keep = top_k_ ? std::min(size_t(top_k_), ids.size()) : ids.size();
        auto greater = [&](int a, int b) { return logits[a] == logits[b] ? a < b : logits[a] > logits[b]; };
        std::partial_sort(ids.begin(), ids.begin() + keep, ids.end(), greater); ids.resize(keep);
        std::vector<double> weights(keep); double sum = 0;
        for (size_t i = 0; i < keep; ++i) sum += weights[i] = std::exp((double(logits[ids[i]]) - logits[ids[0]]) / temperature_);
        double cumulative = 0; size_t nucleus = 0;
        do { cumulative += weights[nucleus++] / sum; } while (nucleus < keep && cumulative < top_p_);
        const double retained = std::accumulate(weights.begin(), weights.begin() + nucleus, 0.0);
        for (size_t i = 0; i < nucleus; ++i) result[ids[i]] = weights[i] / retained;
        return result;
    }
    int sample_distribution(const std::vector<double>& probabilities) {
        if (probabilities.empty() || !std::all_of(probabilities.begin(), probabilities.end(),
                [](double p) { return std::isfinite(p) && p >= 0; }) ||
            std::accumulate(probabilities.begin(), probabilities.end(), 0.0) <= 0)
            throw std::runtime_error("invalid sampling probabilities");
        return int(std::discrete_distribution<size_t>(probabilities.begin(), probabilities.end())(random_));
    }
    // Exact rejection correction: accepted mass is min(p,q); rejected mass is
    // max(0,p-q). Their sum is p, including disjoint filtered supports.
    std::pair<int, bool> verify_draft(const std::vector<double>& target,
                                     const std::vector<double>& proposal, int draft) {
        if (target.size() != proposal.size() || draft < 0 || size_t(draft) >= target.size() ||
            !std::isfinite(target[draft]) || !std::isfinite(proposal[draft]) ||
            target[draft] < 0 || proposal[draft] <= 0)
            throw std::runtime_error("invalid speculative proposal");
        if (target[draft] >= proposal[draft] ||
            std::uniform_real_distribution<double>(0, 1)(random_) < target[draft] / proposal[draft])
            return {draft, true};
        std::vector<double> residual(target.size());
        for (size_t i = 0; i < residual.size(); ++i) residual[i] = std::max(0.0, target[i] - proposal[i]);
        return {sample_distribution(residual), false};
    }
    int sample(const std::vector<float>& raw) {
        if (raw.empty() || !std::all_of(raw.begin(), raw.end(), [](float v) { return std::isfinite(v); }))
            throw std::runtime_error("non-finite or empty logits");
        const auto& logits = penalize(raw);
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
