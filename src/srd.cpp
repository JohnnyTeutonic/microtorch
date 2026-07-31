#include "microtorch/srd.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

namespace microtorch {
namespace nn {

SurpriseRoutedAttention::SurpriseRoutedAttention(size_t d, size_t n_heads, unsigned seed)
    : H(n_heads), dk(d / n_heads) {
    if (d % n_heads != 0) {
        throw std::runtime_error("srd: d must divide by n_heads");
    }
    // Shared projections: identical wiring to CausalSelfAttention /
    // KimiLinearAttention so SRD is comparable parameter-for-parameter.
    c_attn = mod<Linear>("c_attn", d, 3 * d, true, seed + 11);
    c_proj = mod<Linear>("c_proj", d, d, true, seed + 13);
    predictor = mod<cerebellum::RoutinePredictor>("predictor", d, seed + 17);
}

Var SurpriseRoutedAttention::forward(const Var& x) const {
    const size_t T = x->data.rows(), d = H * dk;

    // ---- the router: surprise -> gate in (0, 1), fully on the tape ----
    // Falsifier mode decouples the gate from the query by permuting rows
    // (no-grad copy: the gate becomes an unaligned signal of the same
    // distribution; both attention paths still train normally).
    Var pred_in = x;
    if (shuffle_predictor) {
        std::mt19937 rng(static_cast<unsigned>(1234 + shuffle_ctr_++));
        std::vector<size_t> perm(T);
        for (size_t i = 0; i < T; ++i) perm[i] = i;
        std::shuffle(perm.begin(), perm.end(), rng);
        Matrix pm(T, d);
        for (size_t i = 0; i < T; ++i)
            for (size_t j = 0; j < d; ++j) pm(i, j) = x->data(perm[i], j);
        pred_in = make_var(std::move(pm));
    }
    Var predicted = predictor->forward(pred_in);
    Var residual = ops::sub(pred_in, predicted);
    Var g = ops::sigmoid(
        ops::add_scalar(ops::scale(ops::rms_row(residual), GATE_SCALE), GATE_BIAS));  // [T,1]
    last_gate_ = g;
    // 1 - g for the linear path's share.
    Var g_inv = ops::add_scalar(ops::scale(g, -1.0f), 1.0f);

    // ---- shared qkv ----
    Var qkv = c_attn->forward(x);  // [T, 3d]

    // Additive causal mask (no-grad constant), as in CausalSelfAttention.
    Matrix maskm(T, T);
    for (size_t i = 0; i < T; ++i)
        for (size_t j = i + 1; j < T; ++j) maskm(i, j) = -1e9f;
    Var mask = make_var(std::move(maskm));

    std::vector<Var> heads;
    heads.reserve(H);
    for (size_t h = 0; h < H; ++h) {
        Var q = ops::slice_cols(qkv, h * dk, (h + 1) * dk);
        Var k = ops::slice_cols(qkv, d + h * dk, d + (h + 1) * dk);
        Var v = ops::slice_cols(qkv, 2 * d + h * dk, 2 * d + (h + 1) * dk);

        // Exact path: masked softmax attention.
        Var s =
            ops::scale(ops::matmul(q, ops::transpose(k)), 1.0f / std::sqrt(static_cast<float>(dk)));
        Var exact = ops::matmul(ops::softmax_row(ops::add(s, mask)), v);

        // Cheap path: Kimi linear attention on the SAME q, k, v.
        Var linear = ops::kimi_attention(q, k, v, /*causal=*/true);

        // Per-query blend: surprising queries lean exact, routine lean
        // linear. mul_col keeps the blend differentiable in both paths
        // AND in g (so the predictor trains from the task loss).
        heads.push_back(ops::add(ops::mul_col(exact, g), ops::mul_col(linear, g_inv)));
    }
    return c_proj->forward(ops::concat_cols(heads));
}

}  // namespace nn
}  // namespace microtorch
