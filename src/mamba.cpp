#include "microtorch/mamba.hpp"

#include <cmath>

namespace microtorch {
namespace mamba {

// S4Layer: State-space model layer
S4Layer::S4Layer(size_t d_model, size_t d_state, unsigned seed)
    : d_model_(d_model), d_state_(d_state) {
    // Input/output projections
    proj_in = mod<nn::Linear>("proj_in", d_model, d_state, true, seed + 1);
    proj_out = mod<nn::Linear>("proj_out", d_state, d_model, true, seed + 2);

    // State-space matrices (learnable parameters)
    // A: state transition matrix [d_state, d_state]
    // Initialize A with random eigenvalues (stable dynamics)
    Matrix A_init(d_state, d_state);
    std::mt19937 rng(seed + 3);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    for (size_t i = 0; i < d_state; ++i) {
        for (size_t j = 0; j < d_state; ++j) {
            A_init(i, j) = dist(rng);
        }
    }
    A = reg("A", A_init);

    // B: input coupling matrix [d_state, 1]
    Matrix B_init(d_state, 1);
    for (size_t i = 0; i < d_state; ++i) {
        B_init(i, 0) = dist(rng);
    }
    B = reg("B", B_init);

    // C: output coupling matrix [1, d_state]
    Matrix C_init(1, d_state);
    for (size_t i = 0; i < d_state; ++i) {
        C_init(0, i) = dist(rng);
    }
    C = reg("C", C_init);

    // D: feedthrough [1, 1]. Initialized to 1 (identity skip), the standard
    // S4 init -- a small random D crushes the input path to near-zero before
    // the state path has trained, killing all signal through the layer.
    D = reg("D", Matrix(1, 1, 1.0f));

    // Output gating: d_model -> d_model
    gate_proj = mod<nn::Linear>("gate", d_model, d_model, true, seed + 4);
}

Var S4Layer::forward(const Var& x) const {
    const size_t T = x->data.rows(), d = x->data.cols();

    // Project input to state dimension
    Var u = proj_in->forward(x);  // [T, d_state]

    // State-space forward pass (sequential scan; parallel scan is the
    // roadmap item). Per-channel formulation -- an earlier version averaged
    // u over channels into one scalar per step, which collapsed the output
    // to ~0 for every input (2026-07-30 Debug-suite audit):
    //   state = A @ state + B[i] * u[t, i]     (A mixes channels)
    //   y[t, i] = C[i] * state[i] + D * u[t, i]  (per-channel readout)
    Matrix y_out(T, d_state_);
    Vector state(d_state_, 0.0f);

    for (size_t t = 0; t < T; ++t) {
        Vector new_state(d_state_, 0.0f);
        for (size_t i = 0; i < d_state_; ++i) {
            float val = 0;
            for (size_t j = 0; j < d_state_; ++j) {
                val += A->data(i, j) * state[j];
            }
            val += B->data(i, 0) * u->data(t, i);
            new_state[i] = val;
        }
        state = new_state;

        for (size_t i = 0; i < d_state_; ++i) {
            y_out(t, i) = C->data(0, i) * state[i] +
                          D->data(0, 0) * u->data(t, i);
        }
    }

    Var y_var = make_var(y_out);

    // Project back to model dimension
    Var y_projected = proj_out->forward(y_var);

    // Apply output gating: y_gated = y * sigmoid(gate(x))
    Var gate = ops::gelu(gate_proj->forward(x));
    return ops::mul(y_projected, gate);
}

// MambaBlock: Pre-LN + S4 + skip connection
MambaBlock::MambaBlock(size_t d_model, size_t d_state, unsigned seed) {
    norm = mod<nn::LayerNorm>("norm", d_model);
    s4 = mod<S4Layer>("s4", d_model, d_state, seed + 11);
    out_gate = mod<nn::Linear>("out_gate", d_model, d_model, true, seed + 12);
}

Var MambaBlock::forward(const Var& x) const {
    // Pre-LN: normalize, apply S4, add skip connection
    Var normalized = norm->forward(x);
    Var s4_out = s4->forward(normalized);
    Var gated = ops::mul(s4_out, ops::gelu(out_gate->forward(x)));
    return ops::add(x, gated);
}

// MambaModel: Stack of Mamba blocks with embedding and output projection
MambaModel::MambaModel(size_t vocab_size, size_t d_model, size_t n_layers,
                       size_t d_state, unsigned seed) {
    embed = mod<nn::Embedding>("embed", vocab_size, d_model);
    norm_final = mod<nn::LayerNorm>("norm_final", d_model);
    lm_head = mod<nn::Linear>("lm_head", d_model, vocab_size, true, seed + 21);

    for (size_t i = 0; i < n_layers; ++i) {
        blocks.push_back(
            mod<MambaBlock>("block_" + std::to_string(i), d_model, d_state,
                           seed + 30 + i));
    }
}

Var MambaModel::forward(const Var& x) const {
    // x is [T, 1] with token IDs (float, will be cast to int)
    // Extract token IDs from matrix
    std::vector<int> token_ids;
    for (size_t i = 0; i < x->data.rows(); ++i) {
        token_ids.push_back(static_cast<int>(x->data(i, 0)));
    }

    // Embed tokens
    Var h = embed->forward(token_ids);

    // Stack of Mamba blocks
    for (const auto& block : blocks) {
        h = block->forward(h);
    }

    // Final normalization and output projection
    Var h_norm = norm_final->forward(h);
    return lm_head->forward(h_norm);
}

}  // namespace mamba
}  // namespace microtorch
