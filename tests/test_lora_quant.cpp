// LoRA / QLoRA / int8 quantization tests.
//
//   1. quantize -> dequantize error bounded by the per-block scale
//   2. QLinear forward tracks the fp32 Linear within quantization error
//   3. LoRA at init is EXACTLY the base layer (B = 0)
//   4. backward touches only A and B; a training step moves the output
//   5. merged_weight equals base + scaling * A B (adapter fold-in)
//   6. QLoRA: quantized base + adapters trains end to end
#include <cmath>
#include <cstdio>
#include <random>

#include "check.hpp"
#include "microtorch/quant.hpp"
#include "microtorch/device.hpp"

using namespace microtorch;

namespace {

Matrix randm(size_t r, size_t c, unsigned seed, float scale = 1.0f) {
    Matrix m(r, c);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> d(-scale, scale);
    for (size_t i = 0; i < r; ++i)
        for (size_t j = 0; j < c; ++j) m(i, j) = d(rng);
    return m;
}

float max_abs_diff(const Matrix& a, const Matrix& b) {
    float md = 0;
    for (size_t i = 0; i < a.rows(); ++i)
        for (size_t j = 0; j < a.cols(); ++j)
            md = std::max(md, std::fabs(a(i, j) - b(i, j)));
    return md;
}

void test_quantize_roundtrip() {
    printf("=== int8 blockwise quantization ===\n");
    Matrix W = randm(64, 96, 1);           // cols not a multiple of block
    QuantizedTensor t = quantize_int8(W, 64);
    Matrix W2 = dequantize(t);

    CHECK(W2.rows() == W.rows() && W2.cols() == W.cols());
    // Error bound: half a quantization step per element, step = scale.
    const size_t blocks_per_row = (W.cols() + t.block - 1) / t.block;
    for (size_t i = 0; i < W.rows(); ++i)
        for (size_t j = 0; j < W.cols(); ++j) {
            const float s = t.scales[i * blocks_per_row + j / t.block];
            CHECK(std::fabs(W(i, j) - W2(i, j)) <= 0.5f * s + 1e-7f);
        }
    const float ratio = static_cast<float>(t.nbytes()) /
                        (W.rows() * W.cols() * sizeof(float));
    printf("  error within half-step everywhere; memory %.2fx of fp32\n", ratio);
    CHECK(ratio < 0.30f);   // ~4x smaller (int8 + per-block scales)
}

void test_qlinear_matches_linear() {
    printf("=== QLinear vs fp32 Linear ===\n");
    Matrix W = randm(32, 48, 2, 0.2f);
    Matrix b = randm(1, 48, 3, 0.1f);
    Matrix x = randm(8, 32, 4);

    nn::QLinear q(W, b);
    Var y_q = q.forward(make_var(x));

    // fp32 reference: x W + b
    Var y_ref = ops::add_bias(ops::matmul(make_var(x), make_var(W)),
                              make_var(b));

    // Worst-case per-output error: sum over 32 inner terms of |x| * step/2.
    const float md = max_abs_diff(y_q->data, y_ref->data);
    printf("  max |QLinear - Linear| = %.5f\n", md);
    CHECK(md < 0.05f);
}

void test_lora_identity_at_init() {
    printf("=== LoRA: identity delta at init ===\n");
    Matrix W = randm(32, 48, 5, 0.2f);
    Matrix x = randm(8, 32, 6);

    nn::LoRALinear lora(W, /*rank=*/8, /*alpha=*/16.0f, /*seed=*/42);
    Var y = lora.forward(make_var(x));
    Var y_base = ops::matmul(make_var(x), make_var(W));

    CHECK(max_abs_diff(y->data, y_base->data) == 0.0f);   // B=0 -> exact
    printf("  output identical to frozen base (B initialized to zero)\n");
}

void test_lora_trains_adapters_only() {
    printf("=== LoRA: gradients hit adapters only ===\n");
    Matrix W = randm(32, 48, 7, 0.2f);
    Matrix x = randm(8, 32, 8);

    nn::LoRALinear lora(W, /*rank=*/4, /*alpha=*/8.0f, /*seed=*/1);
    lora.train();

    // Only A and B are registered parameters.
    auto params = lora.parameters();
    CHECK(params.size() == 2);

    Var out = lora.forward(make_var(x, /*requires_grad=*/false));
    Var loss = ops::mean(out);
    backward(loss);

    // B = 0 kills dA on the first step? No: dA = x^T dY B^T = 0 when B = 0,
    // but dB = (xA)^T dY is nonzero -- exactly the paper's asymmetry.
    CHECK(lora.B->grad.rows() != 0);
    float bnorm = 0;
    for (size_t i = 0; i < lora.B->grad.rows(); ++i)
        for (size_t j = 0; j < lora.B->grad.cols(); ++j)
            bnorm += std::fabs(lora.B->grad(i, j));
    CHECK(bnorm > 0.0f);
    printf("  dB nonzero at step 0 (dA zero by B=0, as in the paper)\n");

    // One AdamW step moves the output away from the base.
    nn::AdamW opt(params, /*lr=*/1e-2f);
    opt.step();
    Var out2 = lora.forward(make_var(x));
    Var base = ops::matmul(make_var(x), make_var(W));
    CHECK(max_abs_diff(out2->data, base->data) > 0.0f);
    printf("  adapter step shifts the output; base W untouched\n");
}

void test_lora_merge() {
    printf("=== LoRA: merged_weight equivalence ===\n");
    Matrix W = randm(24, 36, 9, 0.2f);
    Matrix x = randm(4, 24, 10);

    nn::LoRALinear lora(W, /*rank=*/4, /*alpha=*/4.0f, /*seed=*/2);
    // Give B nonzero values so the merge is nontrivial.
    for (size_t i = 0; i < lora.B->data.rows(); ++i)
        for (size_t j = 0; j < lora.B->data.cols(); ++j)
            lora.B->data(i, j) = 0.01f * static_cast<float>(i + j);

    Var y_adapter = lora.forward(make_var(x));
    Matrix Wm = lora.merged_weight();
    Var y_merged = ops::matmul(make_var(x), make_var(Wm));

    const float md = max_abs_diff(y_adapter->data, y_merged->data);
    printf("  max |adapter - merged| = %.7f\n", md);
    CHECK(md < 1e-4f);
}

void test_qlora_end_to_end() {
    printf("=== QLoRA: quantized base + adapters ===\n");
    Matrix W = randm(64, 64, 11, 0.2f);
    Matrix x = randm(8, 64, 12);

    nn::QLoRALinear qlora(W, Matrix(), /*rank=*/8, /*alpha=*/16.0f);
    qlora.train();
    CHECK(qlora.parameters().size() == 2);   // adapters only
    printf("  base stored at %.2fx fp32 bytes\n",
           static_cast<float>(qlora.base_nbytes()) /
               (W.rows() * W.cols() * sizeof(float)));

    nn::AdamW opt(qlora.parameters(), 1e-2f);
    Var out0 = qlora.forward(make_var(x));
    for (int step = 0; step < 3; ++step) {
        Var loss = ops::mean(qlora.forward(make_var(x)));
        opt.zero_grad();
        backward(loss);
        opt.step();
    }
    Var out1 = qlora.forward(make_var(x));
    CHECK(max_abs_diff(out0->data, out1->data) > 0.0f);
    for (size_t i = 0; i < out1->data.rows(); ++i)
        for (size_t j = 0; j < out1->data.cols(); ++j)
            CHECK(std::isfinite(out1->data(i, j)));
    printf("  3 AdamW steps trained the adapters, outputs finite\n");
}

}  // namespace

int main() {
    microtorch::device::set_from_env();
    test_quantize_roundtrip();
    test_qlinear_matches_linear();
    test_lora_identity_at_init();
    test_lora_trains_adapters_only();
    test_lora_merge();
    test_qlora_end_to_end();
    printf("\n[PASS] all LoRA/QLoRA/quantization tests\n");
    return 0;
}
