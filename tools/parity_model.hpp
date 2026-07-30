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
    ParityLM(AttnKind kind, size_t vocab, size_t d, size_t n_heads,
             size_t n_ctx, unsigned seed)
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
                    exact.push_back(mod<nn::CausalSelfAttention>(
                        "attn_" + std::to_string(b), d, n_heads, s));
                    break;
                case AttnKind::KIMI:
                    kimi.push_back(mod<nn::KimiLinearAttention>(
                        "attn_" + std::to_string(b), d, n_heads, s));
                    break;
                case AttnKind::SRD:
                    srd.push_back(mod<nn::SurpriseRoutedAttention>(
                        "attn_" + std::to_string(b), d, n_heads, s));
                    break;
            }
        }
        ln_f = mod<nn::LayerNorm>("ln_f", d);
        head = mod<nn::Linear>("head", d, vocab, false, seed + 99);
    }

    Var forward(const std::vector<int>& ids) const {
        std::vector<int> pos(ids.size());
        for (size_t i = 0; i < pos.size(); ++i) pos[i] = static_cast<int>(i);
        Var h = ops::add(wte->forward(ids), wpe->forward(pos));
        for (int b = 0; b < 2; ++b) {
            Var a;
            Var n1 = ln1[b]->forward(h);
            switch (kind_) {
                case AttnKind::EXACT: a = exact[b]->forward(n1); break;
                case AttnKind::KIMI: a = kimi[b]->forward(n1); break;
                case AttnKind::SRD: a = srd[b]->forward(n1); break;
            }
            h = ops::add(h, a);
            h = ops::add(h, mlp[b]->forward(ln2[b]->forward(h)));
        }
        return head->forward(ln_f->forward(h));
    }

    Var mean_gate() const {
        return ops::scale(
            ops::add(ops::mean(srd[0]->gate()), ops::mean(srd[1]->gate())),
            0.5f);
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

}  // namespace parity
