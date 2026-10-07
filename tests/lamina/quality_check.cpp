#include "lamina/model/inference.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <vector>

// Serial baseline/fast processes avoid double model residency. Logits belong
// outside the source tree; token i is scored using the distribution at i-1.
int main(int argc, char** argv) {
    if (argc != 5) return 2;
    try {
        const bool baseline = std::string(argv[3]) == "write";
        if (!baseline && std::string(argv[3]) != "compare") throw std::runtime_error("expected write or compare");
        std::ifstream input(argv[2]); std::vector<int> tokens; int token;
        while (input >> token) tokens.push_back(token);
        constexpr size_t warmup = 256;
        if (tokens.size() < warmup + 1025) throw std::runtime_error("need 256 warmup and at least 1024 scored tokens");
        std::fstream rows(argv[4], std::ios::binary | (baseline ? std::ios::out | std::ios::trunc : std::ios::in));
        if (!rows) throw std::runtime_error("cannot open baseline logits");
        lamina::model::Inference engine(argv[1], int(tokens.size()+1), 40, true, "device", 0, "f32", baseline ? "f32" : "fast");
        auto hidden = engine.prefill_hidden(std::vector<int>(tokens.begin(), tokens.begin()+warmup));
        double kl = 0, base_nll = 0, fast_nll = 0; size_t count = 0, agree = 0;
        for (size_t i = warmup; i < tokens.size(); ++i) {
            auto actual = engine.logits(std::move(hidden));
            if (tokens[i] < 0 || size_t(tokens[i]) >= actual.size()) throw std::runtime_error("invalid target token");
            const auto bytes = std::streamsize(actual.size()*sizeof(float));
            if (baseline) { rows.write(reinterpret_cast<const char*>(actual.data()), bytes); if (!rows) throw std::runtime_error("baseline write failed"); }
            else {
                std::vector<float> expected(actual.size());
                rows.read(reinterpret_cast<char*>(expected.data()), bytes);
                if (!rows) throw std::runtime_error("baseline logits truncated");
                const double bm = *std::max_element(expected.begin(), expected.end());
                const double fm = *std::max_element(actual.begin(), actual.end());
                double bs = 0, fs = 0;
                for (size_t j=0; j<actual.size(); ++j) { bs += std::exp(double(expected[j])-bm); fs += std::exp(double(actual[j])-fm); }
                const double bz = bm+std::log(bs), fz = fm+std::log(fs);
                for (size_t j=0; j<actual.size(); ++j) kl += std::exp(double(expected[j])-bz)*(double(expected[j])-bz-double(actual[j])+fz);
                base_nll += bz-expected[tokens[i]]; fast_nll += fz-actual[tokens[i]];
                agree += std::max_element(expected.begin(),expected.end())-expected.begin() == std::max_element(actual.begin(),actual.end())-actual.begin();
            }
            ++count;
            if (count % 128 == 0) { std::printf("scored=%zu\n",count); std::fflush(stdout); }
            if (i+1 < tokens.size()) hidden = engine.step_hidden(tokens[i]);
        }
        if (baseline) { rows.flush(); if (!rows) throw std::runtime_error("baseline flush failed"); }
        if (!baseline) {
            const double mean_kl=kl/count, ratio=std::exp((fast_nll-base_nll)/count);
            std::printf("tokens=%zu mean_kl_nats=%.9g perplexity_ratio=%.9g argmax_agreement=%.6f baseline_nll=%.9g fast_nll=%.9g\n",count,mean_kl,ratio,double(agree)/count,base_nll/count,fast_nll/count);
            if (!std::isfinite(mean_kl) || !std::isfinite(ratio) || mean_kl > 0.1 || ratio > 1.05) throw std::runtime_error("fast quality gate failed");
        }
        return 0;
    } catch(const std::exception& e) { std::fprintf(stderr,"%s\n",e.what()); return 1; }
}
