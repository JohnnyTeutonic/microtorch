// Phase-1b gate. The layers are compositions of gradchecked ops, so what
// needs testing here is what composition can still get wrong: wiring
// (finite differences through a FULL transformer block), state_dict
// round-tripping, strictness of loading, and that the optimizers actually
// optimize (overfit a tiny batch).
#include <cmath>
#include <cstdio>
#include <functional>
#include <random>

#include "microtorch/device.hpp"
#include "microtorch/nn.hpp"

using microtorch::make_var;
using microtorch::Var;
namespace ops = microtorch::ops;
namespace nn = microtorch::nn;

namespace {

int g_failures = 0;

void check(bool ok, const char* label, double measured) {
    std::printf("  [%s] %-52s %.3e\n", ok ? "ok" : "FAIL", label, measured);
    if (!ok) ++g_failures;
}

Matrix randn(size_t r, size_t c, unsigned seed, float s = 1.0f) {
    std::mt19937 gen(seed);
    std::normal_distribution<float> d(0.0f, s);
    Matrix m(r, c);
    for (size_t i = 0; i < r; ++i)
        for (size_t j = 0; j < c; ++j) m(i, j) = d(gen);
    return m;
}

double fd_vs_analytic(const std::function<float()>& forward, Var leaf, const Matrix& analytic,
                      float h = 1e-2f) {
    double worst = 0.0;
    for (size_t i = 0; i < leaf->data.rows(); ++i) {
        for (size_t j = 0; j < leaf->data.cols(); ++j) {
            const float keep = leaf->data(i, j);
            microtorch::NoGrad ng;
            leaf->data(i, j) = keep + h;
            const float up = forward();
            leaf->data(i, j) = keep - h;
            const float dn = forward();
            leaf->data(i, j) = keep;
            const double fd = (static_cast<double>(up) - dn) / (2.0 * h);
            const double a = analytic(i, j);
            const double err = std::abs(a - fd) / (1.0 + std::max(std::abs(a), std::abs(fd)));
            worst = std::max(worst, err);
        }
    }
    return worst;
}

}  // namespace

int main() {
    microtorch::device::set_from_env();
    std::printf("microtorch phase-1b nn gate\n\n");
    const double TOL = 5e-3;

    // ---- finite differences through one FULL pre-LN block ----
    // A tiny GPT-2-shaped model: every layer type participates (embedding,
    // layernorm, fused-qkv attention with mask, gelu MLP, weight-tied
    // logits, cross-entropy), so this single check exercises the entire 1b
    // surface end to end.
    nn::GPT2Config cfg;
    cfg.vocab = 11;
    cfg.n_ctx = 8;
    cfg.d = 12;
    cfg.n_layers = 1;
    cfg.n_heads = 3;
    nn::GPT2 model(cfg, /*seed=*/7);
    std::vector<int> ids{3, 1, 4, 1, 5}, tgt{1, 4, 1, 5, 9};

    auto forward = [&]() -> float {
        return ops::cross_entropy(model.forward(ids), tgt)->data(0, 0);
    };
    microtorch::backward(ops::cross_entropy(model.forward(ids), tgt));

    for (auto& [name, p] : model.named_parameters()) {
        if (p->grad.rows() == 0) {
            // wpe rows beyond T=5 legitimately get no grad; everything else must
            bool ok = name == "wpe.weight";
            check(ok, ("param got a grad: " + name).c_str(), 0.0);
            continue;
        }
        double err = fd_vs_analytic(forward, p, p->grad);
        check(err < TOL, ("FD through full model: " + name).c_str(), err);
    }

    // ---- state_dict round trip: save, wreck, load, bit-identical ----
    {
        auto sd = model.state_dict();
        float before = forward();
        for (auto& p : model.parameters()) p->data.fill(0.123f);  // wreck
        model.load_state_dict(sd);
        float after = forward();
        check(before == after, "state_dict round trip: bit-identical loss",
              std::abs(before - after));
    }

    // ---- strict loading fails loudly on missing and unexpected keys ----
    {
        auto sd = model.state_dict();
        sd.erase("ln_f.weight");
        bool threw = false;
        try {
            model.load_state_dict(sd);
        } catch (const std::exception&) {
            threw = true;
        }
        check(threw, "strict load: missing key throws", 0.0);
        sd = model.state_dict();
        sd.emplace("h.9.imaginary.weight", Matrix(1, 1));
        threw = false;
        try {
            model.load_state_dict(sd);
        } catch (const std::exception&) {
            threw = true;
        }
        check(threw, "strict load: unexpected key throws", 0.0);
    }

    // ---- optimizers: overfit one batch; loss must collapse ----
    for (int which = 0; which < 2; ++which) {
        nn::GPT2 m2(cfg, /*seed=*/9);
        std::vector<Var> ps = m2.parameters();
        nn::SGD sgd(ps, 0.5f, 0.9f);
        nn::AdamW adamw(ps, 3e-2f);
        float first = 0.0f, last = 0.0f;
        for (int it = 0; it < 60; ++it) {
            Var loss = ops::cross_entropy(m2.forward(ids), tgt);
            if (it == 0) first = loss->data(0, 0);
            last = loss->data(0, 0);
            if (which == 0)
                sgd.zero_grad();
            else
                adamw.zero_grad();
            microtorch::backward(loss);
            if (which == 0)
                sgd.step();
            else
                adamw.step();
        }
        check(last < 0.15f * first,
              which == 0 ? "SGD+momentum overfits one batch" : "AdamW overfits one batch",
              static_cast<double>(last / first));
    }

    // ---- causality: a future token must not move an earlier logit ----
    {
        nn::GPT2 m3(cfg, /*seed=*/13);
        microtorch::NoGrad ng;
        Var l1 = m3.forward({3, 1, 4, 1, 5});
        Var l2 = m3.forward({3, 1, 4, 2, 9});  // change tokens 3..4 only
        double drift = 0.0;
        for (size_t j = 0; j < cfg.vocab; ++j)
            for (size_t i = 0; i < 3; ++i)
                drift =
                    std::max(drift, static_cast<double>(std::abs(l1->data(i, j) - l2->data(i, j))));
        check(drift == 0.0, "causal mask: past logits untouched by future edit", drift);
    }

    if (g_failures == 0) {
        std::printf("\nALL NN CHECKS PASSED\n");
        return 0;
    }
    std::printf("\n%d FAILURE(S)\n", g_failures);
    return 1;
}
