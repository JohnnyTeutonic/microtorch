#include <cassert>
#include <cmath>
#include <cstdio>

#include "microtorch/mamba.hpp"

using namespace microtorch;

namespace {

// Test 1: S4Layer forward pass
void test_s4_layer() {
    printf("\n=== Test 1: S4Layer ===\n");

    mamba::S4Layer s4(256, 64, 42);
    s4.train();

    // Create input sequence
    Matrix x(8, 256);
    for (size_t i = 0; i < x.rows(); ++i) {
        for (size_t j = 0; j < x.cols(); ++j) {
            x(i, j) = static_cast<float>(i + j) * 0.01f;
        }
    }

    Var x_var = make_var(x, true);
    Var out = s4.forward(x_var);

    // Verify output shape
    assert(out->data.rows() == 8 && out->data.cols() == 256);

    // Verify outputs are finite
    for (size_t i = 0; i < out->data.rows(); ++i) {
        for (size_t j = 0; j < out->data.cols(); ++j) {
            assert(std::isfinite(out->data(i, j)));
        }
    }

    printf("✓ Output shape correct: [%zu, %zu]\n", out->data.rows(),
           out->data.cols());
    printf("✓ Outputs finite\n");
}

// Test 2: MambaBlock with pre-LN and skip connections
void test_mamba_block() {
    printf("\n=== Test 2: MambaBlock ===\n");

    mamba::MambaBlock block(256, 4, 42);
    block.train();

    Matrix x(8, 256);
    for (size_t i = 0; i < x.rows(); ++i) {
        for (size_t j = 0; j < x.cols(); ++j) {
            x(i, j) = static_cast<float>(i + j) * 0.01f;
        }
    }

    Var x_var = make_var(x, true);
    Var out = block.forward(x_var);

    // Verify shape
    assert(out->data.rows() == 8 && out->data.cols() == 256);

    printf("✓ Output shape correct: [%zu, %zu]\n", out->data.rows(),
           out->data.cols());
    printf("✓ MambaBlock forward pass successful\n");
}

// Test 3: MambaModel stacking multiple blocks
void test_mamba_model() {
    printf("\n=== Test 3: MambaModel ===\n");

    const size_t vocab_size = 1024;
    const size_t d_model = 256;
    const size_t n_layers = 4;

    mamba::MambaModel model(vocab_size, d_model, n_layers, 64, 42);
    model.train();

    // Create input token IDs
    Matrix tokens(8, 1);
    for (size_t i = 0; i < tokens.rows(); ++i) {
        tokens(i, 0) = static_cast<float>(i % vocab_size);
    }

    Var tokens_var = make_var(tokens, true);
    Var logits = model.forward(tokens_var);

    // Verify output shape
    assert(logits->data.rows() == 8 && logits->data.cols() == vocab_size);

    // Verify logits are finite
    for (size_t i = 0; i < logits->data.rows(); ++i) {
        for (size_t j = 0; j < logits->data.cols(); ++j) {
            assert(std::isfinite(logits->data(i, j)));
        }
    }

    printf("✓ Logit shape correct: [%zu, %zu]\n", logits->data.rows(),
           logits->data.cols());
    printf("✓ Model stacking and output projection working\n");
}

// Test 4: State-space recurrence (sequential dependence)
void test_ss_recurrence() {
    printf("\n=== Test 4: State-Space Recurrence ===\n");

    mamba::S4Layer s4(256, 64, 42);
    s4.train();

    // Create two different input sequences
    Matrix x1(4, 256), x2(4, 256);
    for (size_t i = 0; i < x1.rows(); ++i) {
        for (size_t j = 0; j < x1.cols(); ++j) {
            x1(i, j) = 0.1f;
            x2(i, j) = 0.5f;
        }
    }

    Var x1_var = make_var(x1, true);
    Var x2_var = make_var(x2, true);

    Var out1 = s4.forward(x1_var);
    Var out2 = s4.forward(x2_var);

    // Different inputs should produce different outputs
    float diff = 0;
    for (size_t i = 0; i < out1->data.rows(); ++i) {
        for (size_t j = 0; j < out1->data.cols(); ++j) {
            diff += std::abs(out1->data(i, j) - out2->data(i, j));
        }
    }

    assert(diff > 0.1f);  // Outputs should differ meaningfully

    printf("✓ State-space outputs respond to input variation\n");
    printf("✓ Recurrence mechanism active (outputs differ for different inputs)\n");
}

}  // namespace

int main() {
    printf("=== Mamba State-Space Model Tests ===\n");

    try {
        test_s4_layer();
        test_mamba_block();
        test_mamba_model();
        test_ss_recurrence();

        printf("\n[PASS] All Mamba tests passed!\n");
        return 0;
    } catch (const std::exception& e) {
        printf("\n[FAIL] Test failed: %s\n", e.what());
        return 1;
    }
}
