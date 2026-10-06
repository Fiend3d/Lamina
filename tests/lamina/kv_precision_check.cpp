// FP16 is a lossy storage option. Keep FP32's strict numerical gate separate;
// here check storage location, reset, decode continuity, and report FP32 drift.
#include "lamina/model/inference.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

void parity(const std::vector<float>& a, const std::vector<float>& b, const char* label) {
    if (a.size() != b.size()) throw std::runtime_error("KV result shape");
    double error = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) throw std::runtime_error("nonfinite KV result");
        error = std::max(error, std::abs(double(a[i]) - b[i]));
    }
    std::printf("%s max_abs_diff=%.9g\n", label, error);
    if (error >= 1e-5) throw std::runtime_error("KV storage location or reset mismatch");
}

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    try {
        std::vector<int> tokens(64);
        for (int i = 0; i < 64; ++i) tokens[i] = 42 + i * 17 % 113;
        std::vector<float> reference, host, host_next, host_image, host_after_image;
        std::vector<float> image(8 * 2048);
        std::vector<std::array<int, 3>> positions(8);
        for (size_t i = 0; i < image.size(); ++i) image[i] = .04f * std::sin(float(i) * .0007f);
        for (int i = 0; i < 8; ++i) positions[i] = {64, 64 + i / 4, 64 + i % 4};
        {
            lamina::model::Inference model(argv[1], 131072, 40, true, "device");
            reference = model.prefill_hidden(tokens);
        }
        {
            lamina::model::Inference model(argv[1], 131072, 40, true, "host", 0, "f16");
            host = model.prefill_hidden(tokens);
            host_next = model.step_hidden(299);
            model.reset();
            parity(host, model.prefill_hidden(tokens), "FP16 host reset");
            host_image = model.prefill_embeddings(image, positions);
            host_after_image = model.step_hidden(299);
        }
        {
            lamina::model::Inference model(argv[1], 131072, 40, true, "device", 0, "f16");
            parity(host, model.prefill_hidden(tokens), "FP16 host/device prefill");
            parity(host_next, model.step_hidden(299), "FP16 host/device decode");
            model.reset();
            parity(host, model.prefill_hidden(tokens), "FP16 device reset");
            parity(host_image, model.prefill_embeddings(image, positions), "FP16 image mRoPE");
            parity(host_after_image, model.step_hidden(299), "FP16 text after image");
        }
        double error = 0, squared = 0, norm = 0;
        for (size_t i = 0; i < host.size(); ++i) {
            const double delta = double(host[i]) - reference[i];
            error = std::max(error, std::abs(delta)); squared += delta * delta;
            norm += double(reference[i]) * reference[i];
        }
        std::printf("FP16 versus FP32 hidden drift max_abs=%.9g relative_l2=%.9g (lossy, not FP32 parity)\n",
            error, std::sqrt(squared/norm));
        // Varied tokens across three history tiles exercise both DMA slots
        // and reuse slot zero while the previous consumer is queued. Compare
        // identical layer math in host/device storage, avoiding a lossy-mode
        // comparison to FP32 or a sequential/GEMM rounding comparison.
        std::vector<int> long_tokens(4097);
        for (int i = 0; i < 4097; ++i) long_tokens[i] = 42 + i * 17 % 113;
        for (const std::string type : {"f32", "f16"}) {
            std::vector<float> expected, expected_next;
            for (const std::string location : {"device", "host"}) {
                lamina::model::Inference model(argv[1], 131072, 4, true, location, 0, type);
                std::vector<float> actual;
                for (int begin = 0; begin < 4097; begin += 2048)
                    actual = model.prefill_hidden(std::vector<int>(long_tokens.begin() + begin,
                        long_tokens.begin() + std::min(begin + 2048, 4097)));
                auto next = model.step_hidden(299);
                if (location == "device") { expected = actual; expected_next = next; }
                else {
                    std::printf("three-tile cache type=%s\n", type.c_str());
                    parity(expected, actual, "host/device varied-token prefill");
                    parity(expected_next, next, "host/device varied-token decode");
                }
            }
        }
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
