#include "lamina/model/inference.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>

int main(int argc, char** argv) {
    if (argc != 3) return 2; // model and a temporary snapshot path outside the source tree
    try {
        for (const char* kv : {"f16", "f32"}) for (const char* mode : {"f32", "fast"}) {
            std::vector<float> expected;
            int position = 0, rope = 0;
            {
                lamina::model::Inference model(argv[1], 4096, 40, true, "device", 0, kv, mode);
                model.prefill_hidden({42,43,44,45,46,47,48,49});
                // Synthetic image positions have a different mRoPE extent from token count.
                std::vector<float> embeddings(4 * 2048);
                for (size_t i = 0; i < embeddings.size(); ++i) embeddings[i] = float(std::sin(double(i) * .01) * .1);
                model.prefill_embeddings(embeddings, {{8,8,8},{8,8,9},{8,9,8},{8,9,9}});
                model.cache_prefix(); position = model.position(); rope = model.rope_position();
                expected = model.prefill_hidden({50,51,52,53});
                for (int i = 0; i < 5; ++i) model.step_hidden(60 + i);
                model.export_prefix(argv[2]);
            } // old CUDA allocations and model mapping really disappear
            {
                lamina::model::Inference restored(argv[1], 4096, 40, true, "device", 0, kv, mode);
                restored.import_prefix(argv[2]);
                if (restored.position() != position || restored.rope_position() != rope)
                    throw std::runtime_error("snapshot position mismatch");
                const auto actual = restored.prefill_hidden({50,51,52,53});
                double error = 0;
                for (size_t i = 0; i < actual.size(); ++i) error = std::max(error, std::abs(double(actual[i]) - expected[i]));
                std::printf("%s %s reload hidden max_abs_diff=%.9g\n", kv, mode, error);
                if (error >= 1e-5) throw std::runtime_error("snapshot hidden state differs");
                restored.restore_prefix();
                if (restored.position() != position) throw std::runtime_error("import did not rebuild checkpoint");
            }
            {
                lamina::model::Inference wrong(argv[1], 2048, 40, true, "device", 0, kv, mode);
                bool rejected = false;
                try { wrong.import_prefix(argv[2]); } catch (const std::exception&) { rejected = true; }
                if (!rejected) throw std::runtime_error("snapshot settings mismatch accepted");
            }
            std::filesystem::resize_file(std::filesystem::u8path(argv[2]), 12);
            {
                lamina::model::Inference truncated(argv[1], 4096, 40, true, "device", 0, kv, mode);
                bool rejected = false;
                try { truncated.import_prefix(argv[2]); } catch (const std::exception&) { rejected = true; }
                if (!rejected) throw std::runtime_error("truncated snapshot accepted");
            }
        }
        std::filesystem::remove(std::filesystem::u8path(argv[2]));
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
