#include "microtorch/llama.hpp"

#include <cmath>
#include <stdexcept>

namespace microtorch {
namespace nn {

LlamaBlock::LlamaBlock(const LlamaConfig& cfg, unsigned seed)
    : H(cfg.n_heads), dk(cfg.d / cfg.n_heads),
      rope_theta_(cfg.rope_theta), rms_eps_(cfg.rms_eps) {
    if (cfg.d % cfg.n_heads != 0)
        throw std::runtime_error("llama: d must divide by n_heads");
    // HF names throughout; Linear registers its matrix as "weight", so the
    // collected dotted paths match LlamaForCausalLM exactly.
    ln1_w = reg("input_layernorm.weight", Matrix(1, cfg.d, 1.0f));
    ln2_w = reg("post_attention_layernorm.weight", Matrix(1, cfg.d, 1.0f));
    q_proj = mod<Linear>("self_attn.q_proj", cfg.d, cfg.d, false, seed + 1);
    k_proj = mod<Linear>("self_attn.k_proj", cfg.d, cfg.d, false, seed + 2);
    v_proj = mod<Linear>("self_attn.v_proj", cfg.d, cfg.d, false, seed + 3);
    o_proj = mod<Linear>("self_attn.o_proj", cfg.d, cfg.d, false, seed + 4);
    gate_proj = mod<Linear>("mlp.gate_proj", cfg.d, cfg.d_ff, false, seed + 5);
    up_proj = mod<Linear>("mlp.up_proj", cfg.d, cfg.d_ff, false, seed + 6);
    down_proj = mod<Linear>("mlp.down_proj", cfg.d_ff, cfg.d, false, seed + 7);
}

Var LlamaBlock::forward(const Var& x, const std::vector<int>& pos) const {
    const size_t T = x->data.rows(), d = H * dk;

    // ---- attention sublayer (pre-RMSNorm) ----
    Var h = ops::rmsnorm(x, ln1_w);
    Var q = q_proj->forward(h);
    Var k = k_proj->forward(h);
    Var v = v_proj->forward(h);
    // ops::apply_rope expects the fused [T, 3d] qkv layout; build it, rotate
    // q/k head-dim subspaces, slice back. concat/slice are tape ops, so
    // gradients flow.
    Var qkv = ops::apply_rope(ops::concat_cols({q, k, v}), pos, rope_theta_, dk);
    q = ops::slice_cols(qkv, 0, d);
    k = ops::slice_cols(qkv, d, 2 * d);
    v = ops::slice_cols(qkv, 2 * d, 3 * d);

    Matrix maskm(T, T);
    for (size_t i = 0; i < T; ++i)
        for (size_t j = i + 1; j < T; ++j) maskm(i, j) = -1e9f;
    Var mask = make_var(std::move(maskm));

    std::vector<Var> heads;
    heads.reserve(H);
    for (size_t hh = 0; hh < H; ++hh) {
        Var qh = ops::slice_cols(q, hh * dk, (hh + 1) * dk);
        Var kh = ops::slice_cols(k, hh * dk, (hh + 1) * dk);
        Var vh = ops::slice_cols(v, hh * dk, (hh + 1) * dk);
        Var s = ops::scale(ops::matmul(qh, ops::transpose(kh)),
                           1.0f / std::sqrt(static_cast<float>(dk)));
        heads.push_back(ops::matmul(ops::softmax_row(ops::add(s, mask)), vh));
    }
    Var attn = o_proj->forward(ops::concat_cols(heads));
    Var x1 = ops::add(x, attn);

    // ---- SwiGLU FFN sublayer ----
    Var h2 = ops::rmsnorm(x1, ln2_w);
    Var ffn = down_proj->forward(
        ops::mul(ops::silu(gate_proj->forward(h2)), up_proj->forward(h2)));
    return ops::add(x1, ffn);
}

Llama::Llama(const LlamaConfig& cfg_, unsigned seed) : cfg(cfg_) {
    embed_tokens = mod<Embedding>("embed_tokens", cfg.vocab, cfg.d, seed + 11);
    for (size_t i = 0; i < cfg.n_layers; ++i)
        blocks.push_back(mod<LlamaBlock>("layers." + std::to_string(i), cfg,
                                         seed + 100 * (unsigned)(i + 1)));
    norm_w = reg("norm.weight", Matrix(1, cfg.d, 1.0f));
    if (!cfg.tie_embeddings)
        lm_head = mod<Linear>("lm_head", cfg.d, cfg.vocab, false, seed + 12);
}

Var Llama::forward(const std::vector<int>& ids) const {
    std::vector<int> pos(ids.size());
    for (size_t i = 0; i < pos.size(); ++i) pos[i] = static_cast<int>(i);
    Var h = embed_tokens->forward(ids);
    for (const auto& b : blocks) h = b->forward(h, pos);
    h = ops::rmsnorm(h, norm_w);
    // Tied head: logits = h @ E^T (gradients flow into the embedding both
    // through the gather and through the head matmul).
    if (lm_head) return lm_head->forward(h);
    return ops::matmul(h, ops::transpose(embed_tokens->weight));
}

}  // namespace nn
}  // namespace microtorch
