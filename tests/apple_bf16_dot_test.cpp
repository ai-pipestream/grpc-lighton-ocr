#include "ggml.h"
#include <cmath>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

void ggml_vec_dot_bf16(int, float*, size_t, ggml_bf16_t*, size_t, ggml_bf16_t*, size_t, int);

int main() {
    std::mt19937 random(1042);
    std::uniform_real_distribution<float> values(-16.0f, 16.0f);
    size_t cases = 0;
    for (int n : {1, 2, 7, 8, 9, 15, 16, 17, 63, 64, 65, 768, 1024, 4097}) {
        for (int offset : {0, 1, 3}) {
            std::vector<ggml_bf16_t> x(n + offset), y(n + offset);
            for (int repeat = 0; repeat < 20; ++repeat) {
                double expected = 0, magnitude = 0;
                for (int i = 0; i < n; ++i) {
                    x[i + offset] = ggml_fp32_to_bf16(values(random));
                    y[i + offset] = ggml_fp32_to_bf16(values(random));
                    float product = ggml_bf16_to_fp32(x[i + offset]) * ggml_bf16_to_fp32(y[i + offset]);
                    expected += product;
                    magnitude += std::abs(product);
                }
                float actual = 0;
                ggml_vec_dot_bf16(n, &actual, 0, x.data() + offset, 0, y.data() + offset, 0, 1);
                if (!std::isfinite(actual) || std::abs(actual - expected) > 1e-7 * magnitude + 1e-6) {
                    std::cerr << "BF16 dot mismatch n=" << n << " offset=" << offset << '\n';
                    return 1;
                }
                ++cases;
            }
        }
    }
    std::cout << "BF16 dot reference PASS cases=" << cases << " (tails and unaligned inputs)\n";
}
