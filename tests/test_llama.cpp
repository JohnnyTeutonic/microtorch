// Llama-family model tests:
//   1. HF-name contract: state_dict keys are exactly LlamaForCausalLM
//      paths (the GGUF-export and HF-load stories both depend on it)
//   2. forward shape + finiteness (tied head)
//   3. module-level FD gradcheck through RMSNorm/RoPE/SwiGLU/attention
//   4. short training: loss falls
#include <cmath>
#include <cstdio>
#include <random>

#include "check.hpp"
#include "microtorch/device.hpp"
#include "microtorch/llama.hpp"

using namespace microtorch;

int main() {
    device::set_from_env();
    nn::LlamaConfig cfg;
    cfg.vocab = 96; cfg.d = 32; cfg.n_layers = 2; cfg.n_heads = 4;
    cfg.d_ff = 64; cfg.n_ctx = 16;
    nn::Llama model(cfg, 42);
    model.train();

    printf("=== HF-name contract ===\n");
    auto sd = model.state_dict();
    const char* expect[] = {
        "embed_tokens.weight",
        "layers.0.input_layernorm.weight",
        "layers.0.self_attn.q_proj.weight",
        "layers.0.self_attn.o_proj.weight",
        "layers.0.mlp.gate_proj.weight",
        "layers.1.post_attention_layernorm.weight",
        "layers.1.mlp.down_proj.weight",
        "norm.weight",
    };
    for (const char* k : expect) {
        CHECK(sd.count(k) == 1);
    }
    // Tied config must NOT expose lm_head.
    CHECK(sd.count("lm_head.weight") == 0);
    printf("  %zu params, all sampled HF paths present, head tied\n", sd.size());

    printf("=== forward ===\n");
    std::vector<int> ids = {5, 17, 3, 42, 42, 9, 61, 2};
    Var logits = model.forward(ids);
    CHECK(logits->data.rows() == ids.size());
    CHECK(logits->data.cols() == cfg.vocab);
    for (size_t i = 0; i < logits->data.rows(); ++i)
        for (size_t j = 0; j < logits->data.cols(); ++j)
            CHECK(std::isfinite(logits->data(i, j)));
    printf("  [%zu, %zu] logits finite\n", logits->data.rows(),
           logits->data.cols());

    printf("=== module-level FD gradcheck ===\n");
    // d mean(logits) / d embed_tokens.weight at sampled entries, central
    // differences. Rows must be ones the forward actually touches (gather
    // rows for used ids); the tied head also touches every row, so any
    // row works -- probe a mix.
    Var E = model.embed_tokens->weight;
    backward(ops::mean(model.forward(ids)));
    std::mt19937 rng(7);
    const float eps = 1e-3f;
    float worst = 0;
    for (int p = 0; p < 8; ++p) {
        const size_t i = rng() % cfg.vocab, j = rng() % cfg.d;
        float fp, fm;
        {
            NoGrad ng;
            const float keep = E->data(i, j);
            E->data(i, j) = keep + eps;
            fp = ops::mean(model.forward(ids))->data(0, 0);
            E->data(i, j) = keep - eps;
            fm = ops::mean(model.forward(ids))->data(0, 0);
            E->data(i, j) = keep;
        }
        const float fd = (fp - fm) / (2 * eps), an = E->grad(i, j);
        worst = std::max(worst, std::fabs(fd - an) /
                                    std::max({std::fabs(fd), std::fabs(an),
                                              1e-4f}));
    }
    printf("  d mean(logits)/d embed rel err %.2e\n", worst);
    CHECK(worst < 1e-2f);

    printf("=== short training ===\n");
    nn::AdamW opt(model.parameters(), 3e-3f);
    std::mt19937 drng(123);
    float first = 0, last = 0;
    for (int step = 0; step < 25; ++step) {
        std::vector<int> x(12), y(12);
        // Learnable synthetic pattern: y[t] = (x[t] + 1) mod vocab.
        for (size_t t = 0; t < x.size(); ++t) {
            x[t] = drng() % cfg.vocab;
            y[t] = (x[t] + 1) % static_cast<int>(cfg.vocab);
        }
        Var loss = ops::cross_entropy(model.forward(x), y);
        opt.zero_grad();
        backward(loss);
        ops::clip_grad_norm(model.parameters(), 1.0f);
        opt.step();
        if (step == 0) first = loss->data(0, 0);
        last = loss->data(0, 0);
    }
    printf("  loss %.4f -> %.4f over 25 steps\n", first, last);
    CHECK(last < first - 0.3f);

    printf("\n[PASS] all Llama tests\n");
    return 0;
}
