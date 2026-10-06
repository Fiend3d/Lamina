#include "lamina/model/inference.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

void compare(const std::vector<float>& a, const std::vector<float>& b, const char* label) {
    if (a.size() != b.size()) throw std::runtime_error("hidden size differs");
    double error = 0;
    for (size_t i = 0; i < a.size(); ++i) error = std::max(error, std::abs(double(a[i]) - b[i]));
    std::printf("%s max_abs_diff=%.9g\n", label, error);
    if (error >= 1e-5) throw std::runtime_error("prefill state parity failed");
}

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    try {
        for (const std::string cache : {"host", "device"}) {
            for (int layers : {4, 40}) {
                const int count = layers == 4 ? 1025 : 64;
                const int chunk = layers == 4 ? 1024 : 32;
                std::vector<int> tokens(count);
                for (int i = 0; i < count; ++i) tokens[i] = 42 + (i * 17 % 113);
                std::vector<float> expected, expected_next, expected_reset, expected_image, expected_image_next;
                std::vector<float> image(8 * 2048);
                std::vector<std::array<int, 3>> image_positions(8);
                for (size_t i = 0; i < image.size(); ++i) image[i] = 0.04f * std::sin(float(i) * 0.0007f);
                for (int i = 0; i < 8; ++i) image_positions[i] = {2, 2 + i / 4, 2 + i % 4};
                {
                    lamina::model::Inference sequential(argv[1], 131072, layers, true, cache);
                    for (int token : tokens) expected = sequential.step_hidden(token);
                    expected_next = sequential.step_hidden(299);
                    sequential.reset(); sequential.step_hidden(42);
                    expected_reset = sequential.step_hidden(43);
                    for (int i = 0; i < 8; ++i)
                        expected_image = sequential.step_hidden_embedding(
                            std::vector<float>(image.begin() + i * 2048, image.begin() + (i + 1) * 2048), image_positions[i]);
                    expected_image_next = sequential.step_hidden(299);
                }
                lamina::model::Inference batched(argv[1], 131072, layers, true, cache);
                std::vector<float> actual;
                for (int start = 0; start < count; start += chunk) {
                    const int end = std::min(start + chunk, count);
                    actual = batched.prefill_hidden(std::vector<int>(tokens.begin() + start, tokens.begin() + end));
                }
                std::printf("layers=%d cache=%s tokens=%d chunk=%d\n", layers, cache.c_str(), count, chunk);
                compare(expected, actual, "prefill");
                compare(expected_next, batched.step_hidden(299), "decode after prefill");
                batched.reset();
                compare(expected_reset, batched.prefill_hidden({42, 43}), "reset after prefill");
                if (batched.position() != 2 || batched.rope_position() != 2)
                    throw std::runtime_error("prefill position mismatch");
                compare(expected_image, batched.prefill_embeddings(image, image_positions), "image mRoPE prefill");
                if (batched.position() != 10 || batched.rope_position() != 6)
                    throw std::runtime_error("image mRoPE position mismatch");
                compare(expected_image_next, batched.step_hidden(299), "text after image");
            }
        }
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
