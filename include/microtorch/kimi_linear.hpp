#pragma once
// Phase 3a: Kimi Linear - Linear-time attention mechanism
// O(n) complexity vs O(n²) for standard attention
// Maintains expressiveness through efficient linear parameterization
//
// Reference: arXiv:2510.26692 "Kimi Linear: Linear-Time Attention"
//
// Core idea: Replace quadratic attention matrix with linear projection space
// that preserves model expressiveness while breaking the n² scaling bottleneck.

#include "microtorch/primitives.hpp"
#include <cmath>
#include <memory>

namespace microtorch::kimi {

// Linear-time attention forward + backward
// Input shapes: q, k, v all [batch*heads, seq, head_dim]
// Output: [batch*heads, seq, head_dim], same shape as v
//
// Mechanism:
// 1. Feature map: transform queries and keys through feature function φ
// 2. Numerator: Σ φ(k_i) * v_i (cumulative sum over sequence)
// 3. Denominator: Σ φ(k_i) (normalization)
// 4. Output: φ(q) * (numerator / denominator)
//
// This avoids materializing the [seq, seq] attention matrix entirely.
class KimiLinearAttention {
public:
    // Initialize with head dimension
    explicit KimiLinearAttention(size_t head_dim)
        : head_dim_(head_dim) {}

    // Forward: linear-time attention
    // q, k, v: [batch*heads, seq, head_dim]
    // causal: if true, use causal mask (tokens can only attend to past)
    // Returns: attention output [batch*heads, seq, head_dim]
    Matrix forward(const Matrix& q, const Matrix& k, const Matrix& v,
                   bool causal = true);

    // Backward: compute gradients w.r.t. q, k, v
    // grad_out: gradient from upstream [batch*heads, seq, head_dim]
    // q, k, v, attention_out: forward pass activations
    // Returns: (grad_q, grad_k, grad_v)
    std::tuple<Matrix, Matrix, Matrix> backward(
        const Matrix& grad_out, const Matrix& q, const Matrix& k,
        const Matrix& v, const Matrix& attention_out) const;

private:
    size_t head_dim_;

    // Feature map: elu(x) + 1 (always positive, smooth gradients)
    // This is the key difference from standard attention
    // Maintains expressiveness while enabling linear complexity
    static Matrix feature_map(const Matrix& x);
    static Matrix feature_map_grad(const Matrix& x);

    // Cumulative sum: [seq, head_dim] -> [seq, head_dim]
    // cumsum[t, :] = sum(values[0:t+1, :], axis=0)
    static Matrix cumsum(const Matrix& x);

    // Causal mask: cumsum only up to current position
    // (prevents tokens from attending to future)
    static Matrix cumsum_causal(const Matrix& x);

    // Elementwise division: safer than direct /
    // Handles near-zero denominators
    static Matrix safe_divide(const Matrix& numerator,
                              const Matrix& denominator,
                              float eps = 1e-8f);
};

}  // namespace microtorch::kimi
