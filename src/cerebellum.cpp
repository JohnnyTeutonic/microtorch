#include "microtorch/cerebellum.hpp"

namespace microtorch {
namespace cerebellum {

// RoutinePredictor: Small MLP that learns routine token patterns
RoutinePredictor::RoutinePredictor(size_t d_model, unsigned seed) {
    // Lightweight: d -> d/4 -> d
    size_t hidden = std::max(size_t(1), d_model / 4);
    fc1 = mod<nn::Linear>("fc1", d_model, hidden, true, seed + 31);
    fc2 = mod<nn::Linear>("fc2", hidden, d_model, true, seed + 32);
}

Var RoutinePredictor::forward(const Var& x) const {
    // Small feedforward: input -> gelu -> output
    Var h = ops::gelu(fc1->forward(x));
    return fc2->forward(h);
}

// SelectiveGate: Wraps computation with residual-based gating
SelectiveGate::SelectiveGate(ForwardFn layer_fn, size_t d_model, unsigned seed)
    : layer_fn_(layer_fn), predictor_(std::make_shared<RoutinePredictor>(
                                 d_model, seed)) {}

Var SelectiveGate::forward(const Var& x) const {
    // 1. Predict routine features
    Var predicted = predictor_->forward(x);

    // 2. Compute residual = actual - predicted (surprise signal)
    Var residual = ops::sub(x, predicted);

    // 3. Compute gate probability per token
    // gate_prob[t] = sigmoid(||residual[t]||)
    const size_t T = x->data.rows(), d = x->data.cols();
    Matrix residual_norm(T, 1);
    for (size_t t = 0; t < T; ++t) {
        float norm = 0;
        for (size_t j = 0; j < d; ++j) {
            float val = residual->data(t, j);
            norm += val * val;
        }
        residual_norm(t, 0) = std::sqrt(norm);  // L2 norm of residual
    }

    // Apply sigmoid to gate probabilities
    Var gate_logits = make_var(residual_norm);
    Matrix gate_probs(T, 1);
    for (size_t t = 0; t < T; ++t) {
        float logit = gate_logits->data(t, 0);
        gate_probs(t, 0) = 1.0f / (1.0f + std::exp(-logit));
    }

    // 4. Run expensive layer via function
    Var layer_out = layer_fn_(x);

    // 5. Apply gating: output = gate_prob * layer_out + (1 - gate_prob) * x
    // Expand gate probabilities to full matrix for element-wise multiply
    Matrix gated_output(T, d);
    Matrix gate_complement(T, 1);
    for (size_t t = 0; t < T; ++t) {
        float gate = gate_probs(t, 0);
        gate_complement(t, 0) = 1.0f - gate;
        for (size_t j = 0; j < d; ++j) {
            gated_output(t, j) = gate * layer_out->data(t, j) +
                                gate_complement(t, 0) * x->data(t, j);
        }
    }

    last_gate_ = gate_probs;
    last_residual_ = residual->data;

    return make_var(gated_output);
}

// GatedBlock: Pre-LN block with selective gating on attention and MLP
GatedBlock::GatedBlock(size_t d, size_t n_heads, unsigned seed) {
    ln_1 = mod<nn::LayerNorm>("ln_1", d);
    ln_2 = mod<nn::LayerNorm>("ln_2", d);

    // Create attention and MLP
    auto attn = std::make_shared<nn::KimiLinearAttention>(d, n_heads, seed + 43);
    auto mlp = std::make_shared<nn::MLP>(d, 4 * d, seed + 44);

    // Wrap with gating using lambda functions
    auto attn_fn = [attn](const Var& x) { return attn->forward(x); };
    auto mlp_fn = [mlp](const Var& x) { return mlp->forward(x); };

    attn_gate = mod<SelectiveGate>("attn_gate", attn_fn, d, seed + 45);
    mlp_gate = mod<SelectiveGate>("mlp_gate", mlp_fn, d, seed + 46);
}

Var GatedBlock::forward(const Var& x) const {
    // Pre-LN transformer block with selective gating
    // Residual 1: gated attention
    Var ln1_out = ln_1->forward(x);
    Var attn_out = attn_gate->forward(ln1_out);
    Var x_with_attn = ops::add(x, attn_out);

    // Residual 2: gated MLP
    Var ln2_out = ln_2->forward(x_with_attn);
    Var mlp_out = mlp_gate->forward(ln2_out);
    return ops::add(x_with_attn, mlp_out);
}

}  // namespace cerebellum
}  // namespace microtorch
