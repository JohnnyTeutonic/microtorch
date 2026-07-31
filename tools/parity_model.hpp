#pragma once
// Shared 4-lane parity model for the SRD experiments (srd_parity.cpp,
// srd_needle.cpp): 2 pre-LN blocks, learned positional embeddings,
// attention flavor pluggable. Kept identical across experiments so results
// compare across runs.
#include <memory>
#include <string>
#include <vector>

#include "microtorch/nn.hpp"
#include "microtorch/srd.hpp"

namespace parity {

using namespace microtorch;

enum class AttnKind { EXACT, KIMI, SRD };

class ParityLM : public nn::Module {
public:
    ParityLM(AttnKind kind, size_t vocab, size_t d, size_t n_heads, size_t n_ctx, unsigned seed)
        : kind_(kind) {
        wte = mod<nn::Embedding>("wte", vocab, d, seed + 1);
        wpe = mod<nn::Embedding>("wpe", n_ctx, d, seed + 2);
        for (int b = 0; b < 2; ++b) {
            const unsigned s = seed + 10 * (b + 1);
            ln1.push_back(mod<nn::LayerNorm>("ln1_" + std::to_string(b), d));
            ln2.push_back(mod<nn::LayerNorm>("ln2_" + std::to_string(b), d));
            mlp.push_back(mod<nn::MLP>("mlp_" + std::to_string(b), d, 4 * d, s + 3));
            switch (kind) {
                case AttnKind::EXACT:
                    exact.push_back(
                        mod<nn::CausalSelfAttention>("attn_" + std::to_string(b), d, n_heads, s));
                    break;
                case AttnKind::KIMI:
                    kimi.push_back(
                        mod<nn::KimiLinearAttention>("attn_" + std::to_string(b), d, n_heads, s));
                    break;
                case AttnKind::SRD:
                    srd.push_back(mod<nn::SurpriseRoutedAttention>("attn_" + std::to_string(b), d,
                                                                   n_heads, s));
                    break;
            }
        }
        ln_f = mod<nn::LayerNorm>("ln_f", d);
        head = mod<nn::Linear>("head", d, vocab, false, seed + 99);
    }

    // seq_len 0 = one sequence; > 0 = stacked mini-batch. All three
    // attention kinds are block-aware (exact via the fused mask, kimi
    // via per-block prefix-sum reset, srd through both paths).
    Var forward(const std::vector<int>& ids, size_t seq_len = 0) const {
        const size_t sl = seq_len == 0 ? ids.size() : seq_len;
        std::vector<int> pos(ids.size());
        for (size_t i = 0; i < pos.size(); ++i) pos[i] = static_cast<int>(i % sl);
        Var h = ops::add(wte->forward(ids), wpe->forward(pos));
        for (int b = 0; b < 2; ++b) {
            Var a;
            Var n1 = ln1[b]->forward(h);
            switch (kind_) {
                case AttnKind::EXACT:
                    a = exact[b]->forward(n1, seq_len);
                    break;
                case AttnKind::KIMI:
                    a = kimi[b]->forward(n1, seq_len);
                    break;
                case AttnKind::SRD:
                    a = srd[b]->forward(n1, seq_len);
                    break;
            }
            h = ops::add(h, a);
            h = ops::add(h, mlp[b]->forward(ln2[b]->forward(h)));
        }
        return head->forward(ln_f->forward(h));
    }

    Var mean_gate() const {
        return ops::scale(ops::add(ops::mean(srd[0]->gate()), ops::mean(srd[1]->gate())), 0.5f);
    }
    void set_falsifier(bool on) {
        for (auto& s : srd) s->shuffle_predictor = on;
    }

    AttnKind kind_;
    std::shared_ptr<nn::Embedding> wte, wpe;
    std::vector<std::shared_ptr<nn::LayerNorm>> ln1, ln2;
    std::vector<std::shared_ptr<nn::MLP>> mlp;
    std::vector<std::shared_ptr<nn::CausalSelfAttention>> exact;
    std::vector<std::shared_ptr<nn::KimiLinearAttention>> kimi;
    std::vector<std::shared_ptr<nn::SurpriseRoutedAttention>> srd;
    std::shared_ptr<nn::LayerNorm> ln_f;
    std::shared_ptr<nn::Linear> head;
};

// The AttnRes LM (TECH_TRANSFER item 1 as a trainable preset): the same
// embedding/head shell as ParityLM, but the residual STREAM is replaced
// by nn::AttnResStack — attention over depth. Each transformer sublayer
// (attn-with-ln, mlp-with-ln) is one AttnRes layer with its own
// pseudo-query, so a 2-block model contributes 4 depth-attention
// sources plus the embedding. Block form (S=2) banks one representation
// per transformer block, K3-style.
class AttnResLM : public nn::Module {
public:
    AttnResLM(size_t vocab, size_t d, size_t n_heads, size_t n_ctx, unsigned seed,
              size_t n_blocks = 2, size_t attnres_block_size = 2) {
        wte = mod<nn::Embedding>("wte", vocab, d, seed + 1);
        wpe = mod<nn::Embedding>("wpe", n_ctx, d, seed + 2);
        std::vector<std::shared_ptr<nn::Module>> owned;
        std::vector<std::function<Var(const Var&)>> fns;
        for (size_t b = 0; b < n_blocks; ++b) {
            const unsigned s = seed + 10 * static_cast<unsigned>(b + 1);
            auto lna = std::make_shared<nn::LayerNorm>(d);
            auto attn = std::make_shared<nn::CausalSelfAttention>(d, n_heads, s);
            auto lnm = std::make_shared<nn::LayerNorm>(d);
            auto mlp = std::make_shared<nn::MLP>(d, 4 * d, s + 3);
            // Two composite sublayers per block; each is one AttnRes
            // source. seq_len flows through the mutable member so the
            // fixed closures stay batch-aware.
            struct AttnSub : nn::Module {
                AttnSub(std::shared_ptr<nn::LayerNorm> l,
                        std::shared_ptr<nn::CausalSelfAttention> a)
                    : ln(l), attn(a) {
                    adopt("ln", l);
                    adopt("attn", a);
                }
                std::shared_ptr<nn::LayerNorm> ln;
                std::shared_ptr<nn::CausalSelfAttention> attn;
            };
            struct MlpSub : nn::Module {
                MlpSub(std::shared_ptr<nn::LayerNorm> l, std::shared_ptr<nn::MLP> m)
                    : ln(l), mlp(m) {
                    adopt("ln", l);
                    adopt("mlp", m);
                }
                std::shared_ptr<nn::LayerNorm> ln;
                std::shared_ptr<nn::MLP> mlp;
            };
            auto asub = std::make_shared<AttnSub>(lna, attn);
            auto msub = std::make_shared<MlpSub>(lnm, mlp);
            owned.push_back(asub);
            owned.push_back(msub);
            fns.push_back([asub, this](const Var& x) {
                return asub->attn->forward(asub->ln->forward(x), cur_seq_len_);
            });
            fns.push_back(
                [msub](const Var& x) { return msub->mlp->forward(msub->ln->forward(x)); });
        }
        stack = mod<nn::AttnResStack>("stack", owned, fns, d, attnres_block_size);
        ln_f = mod<nn::LayerNorm>("ln_f", d);
        head = mod<nn::Linear>("head", d, vocab, false, seed + 99);
    }

    Var forward(const std::vector<int>& ids, size_t seq_len = 0) const {
        cur_seq_len_ = seq_len;
        const size_t sl = seq_len == 0 ? ids.size() : seq_len;
        std::vector<int> pos(ids.size());
        for (size_t i = 0; i < pos.size(); ++i) pos[i] = static_cast<int>(i % sl);
        Var h = ops::add(wte->forward(ids), wpe->forward(pos));
        return head->forward(ln_f->forward(stack->forward(h)));
    }

    std::shared_ptr<nn::Embedding> wte, wpe;
    std::shared_ptr<nn::AttnResStack> stack;
    std::shared_ptr<nn::LayerNorm> ln_f;
    std::shared_ptr<nn::Linear> head;

private:
    mutable size_t cur_seq_len_ = 0;
};

}  // namespace parity
