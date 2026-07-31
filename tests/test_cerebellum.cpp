#include <cmath>
#include <cstdio>
#include <random>
#include "check.hpp"

#include "microtorch/cerebellum.hpp"
#include "microtorch/nn.hpp"

using namespace microtorch;

namespace {

// Test 1: RoutinePredictor learns and produces reasonable output
void test_routine_predictor() {
    printf("\n=== Test 1: RoutinePredictor ===\n");

    cerebellum::RoutinePredictor predictor(256, 42);
    predictor.train();

    // Create input
    Matrix x(8, 256);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    for (size_t i = 0; i < x.rows(); ++i) {
        for (size_t j = 0; j < x.cols(); ++j) {
            x(i, j) = dist(rng);
        }
    }

    Var x_var = make_var(x, true);
    Var predicted = predictor.forward(x_var);

    // Verify output shape
    CHECK(predicted->data.rows() == 8 && predicted->data.cols() == 256);

    // Verify outputs are finite
    for (size_t i = 0; i < predicted->data.rows(); ++i) {
        for (size_t j = 0; j < predicted->data.cols(); ++j) {
            float val = predicted->data(i, j);
            CHECK(std::isfinite(val));
        }
    }

    printf("✓ Output shape correct: [%zu, %zu]\n", predicted->data.rows(), predicted->data.cols());
    printf("✓ Outputs finite\n");
}

// Test 2: SelectiveGate wraps a layer and produces gated output
void test_selective_gate() {
    printf("\n=== Test 2: SelectiveGate ===\n");

    auto attn = std::make_shared<nn::KimiLinearAttention>(256, 4, 42);
    auto attn_fn = [attn](const Var& x) { return attn->forward(x); };
    cerebellum::SelectiveGate gate(attn_fn, 256, 42);
    gate.train();

    // Create input
    Matrix x(8, 256);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    for (size_t i = 0; i < x.rows(); ++i) {
        for (size_t j = 0; j < x.cols(); ++j) {
            x(i, j) = dist(rng);
        }
    }

    Var x_var = make_var(x, true);
    Var gated_out = gate.forward(x_var);

    // Verify output shape
    CHECK(gated_out->data.rows() == 8 && gated_out->data.cols() == 256);

    // Verify gate logits are in reasonable range
    const Matrix& gates = gate.last_gate_logits();
    for (size_t t = 0; t < gates.rows(); ++t) {
        float gate_prob = gates(t, 0);
        CHECK(gate_prob >= 0.0f && gate_prob <= 1.0f);
    }

    printf("✓ Output shape correct: [%zu, %zu]\n", gated_out->data.rows(), gated_out->data.cols());
    printf("✓ Gate probabilities in [0, 1]\n");
    printf("✓ Gate logits (surprise signal): ");
    for (size_t t = 0; t < std::min(size_t(3), gates.rows()); ++t) {
        printf("%.4f ", gates(t, 0));
    }
    printf("...\n");
}

// Test 3: GatedBlock integrates LN + gated attention + gated MLP
void test_gated_block() {
    printf("\n=== Test 3: GatedBlock ===\n");

    cerebellum::GatedBlock block(256, 4, 42);
    block.train();

    // Create input
    Matrix x(8, 256);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    for (size_t i = 0; i < x.rows(); ++i) {
        for (size_t j = 0; j < x.cols(); ++j) {
            x(i, j) = dist(rng);
        }
    }

    Var x_var = make_var(x, true);
    Var out = block.forward(x_var);

    // Verify output shape
    CHECK(out->data.rows() == 8 && out->data.cols() == 256);

    // Verify outputs are finite
    for (size_t i = 0; i < out->data.rows(); ++i) {
        for (size_t j = 0; j < out->data.cols(); ++j) {
            float val = out->data(i, j);
            CHECK(std::isfinite(val));
        }
    }

    printf("✓ Output shape correct: [%zu, %zu]\n", out->data.rows(), out->data.cols());
    printf("✓ Block forward pass successful\n");
    printf("✓ Pre-LN + gated attention + gated MLP integrated\n");
}

// Test 4: Gating mechanism actually gates (high residual -> high gate prob)
void test_gating_mechanism() {
    printf("\n=== Test 4: Gating Mechanism ===\n");

    // Create a simple identity function for testing
    auto identity_fn = [](const Var& x) { return x; };

    cerebellum::SelectiveGate gate(identity_fn, 256, 42);
    gate.train();

    // Test 1: Routine input (close to zero) should get LOW gate probability
    Matrix routine_x(4, 256);
    for (size_t i = 0; i < routine_x.rows(); ++i) {
        for (size_t j = 0; j < routine_x.cols(); ++j) {
            routine_x(i, j) = 1e-4f;  // Very small (routine)
        }
    }

    Var routine_var = make_var(routine_x, true);
    gate.forward(routine_var);
    // Copy, not reference: the next forward() overwrites the cached gates.
    Matrix routine_gates = gate.last_gate_logits();

    // Test 2: Surprising input (large values) should get HIGH gate probability
    Matrix surprising_x(4, 256);
    for (size_t i = 0; i < surprising_x.rows(); ++i) {
        for (size_t j = 0; j < surprising_x.cols(); ++j) {
            surprising_x(i, j) = 0.5f;  // Large (surprising)
        }
    }

    Var surprising_var = make_var(surprising_x, true);
    gate.forward(surprising_var);
    const Matrix& surprising_gates = gate.last_gate_logits();

    printf("Routine input (1e-4): gate prob = %.4f (should be low)\n", routine_gates(0, 0));
    printf("Surprising input (0.5): gate prob = %.4f (should be high)\n", surprising_gates(0, 0));

    // Verify mechanism works
    CHECK(routine_gates(0, 0) < 0.5f);     // Routine should have low gate
    CHECK(surprising_gates(0, 0) > 0.5f);  // Surprising should have high gate
    CHECK(surprising_gates(0, 0) > routine_gates(0, 0));

    printf("✓ Gating mechanism works: high residual → high gate prob\n");
}

}  // namespace

int main() {
    printf("=== Cerebellum-Inspired Selective Computation Tests ===\n");

    try {
        test_routine_predictor();
        test_selective_gate();
        test_gated_block();
        test_gating_mechanism();

        printf("\n[PASS] All cerebellum tests passed!\n");
        return 0;
    } catch (const std::exception& e) {
        printf("\n[FAIL] Test failed: %s\n", e.what());
        return 1;
    }
}
