// Activation-checkpointing receipts (STUDIO_PLAN section 9, gap 2).
//
// The claim: with checkpoint_blocks on, forward stores only block
// boundaries and recomputes block interiors on backward — same loss,
// same gradients, materially fewer live tape nodes between forward and
// backward. The tests assert all three:
//   1. loss identical with/without checkpointing (same seed, same data)
//   2. every parameter gradient identical (recompute runs the same ops
//      in the same order, so the tolerance is float-equality-tight)
//   3. live_variables() right after forward is a fraction of the
//      uncheckpointed count — the memory receipt
//   4. all of the above composed with mini-batching (seq_len path)
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "microtorch/llama.hpp"
#include "microtorch/nn.hpp"
#include "microtorch/ops.hpp"

#include "check.hpp"

using namespace microtorch;

namespace {

std::vector<int> random_ids(size_t n, unsigned seed, int vocab) {
    std::mt19937 rng(seed);
    std::vector<int> ids(n);
    for (auto& t : ids) t = static_cast<int>(rng() % vocab);
    return ids;
}

double max_abs_diff(const Matrix& a, const Matrix& b) {
    double worst = 0.0;
    for (size_t i = 0; i < a.rows(); ++i)
        for (size_t j = 0; j < a.cols(); ++j)
            worst = std::max(worst, std::fabs(static_cast<double>(a(i, j)) - b(i, j)));
    return worst;
}

// One fwd+bwd; returns loss and a copy of every parameter grad, plus the
// live-Variable count captured between forward and backward.
template <typename Model>
float run_once(Model& model, const std::vector<int>& x, const std::vector<int>& y, size_t seq_len,
               std::vector<Matrix>& grads_out, size_t& live_after_fwd) {
    auto params = model.parameters();
    zero_grad(params);
    const size_t live0 = live_variables();
    Var logits = model.forward(x, seq_len);
    Var loss = ops::cross_entropy(logits, y);
    live_after_fwd = live_variables() - live0;
    backward(loss);
    grads_out.clear();
    for (const auto& p : params) grads_out.push_back(p->grad);
    return loss->data(0, 0);
}

template <typename Model>
void run_family(const char* family, Model& plain, Model& ckpt, int vocab, size_t T,
                size_t seq_len) {
    auto ids = random_ids((seq_len ? 2 * T : T) + 1, 77, vocab);
    std::vector<int> x(ids.begin(), ids.end() - 1), y(ids.begin() + 1, ids.end());

    std::vector<Matrix> g_plain, g_ckpt;
    size_t live_plain = 0, live_ckpt = 0;
    ckpt.checkpoint_blocks = true;
    const float l_plain = run_once(plain, x, y, seq_len, g_plain, live_plain);
    const float l_ckpt = run_once(ckpt, x, y, seq_len, g_ckpt, live_ckpt);

    const double dl = std::fabs(static_cast<double>(l_plain) - l_ckpt);
    std::printf("  [%s%s] loss diff                 %.3e\n", family, seq_len ? "+batch" : "", dl);
    CHECK(dl < 1e-6);

    auto pp = plain.parameters();
    double worst = 0.0;
    for (size_t k = 0; k < g_plain.size(); ++k) {
        if (g_plain[k].rows() == 0 && g_ckpt[k].rows() == 0) continue;
        CHECK(g_plain[k].rows() != 0 && g_ckpt[k].rows() != 0);  // same coverage
        worst = std::max(worst, max_abs_diff(g_plain[k], g_ckpt[k]));
    }
    std::printf("  [%s%s] grad max diff             %.3e\n", family, seq_len ? "+batch" : "",
                worst);
    CHECK(worst < 1e-6);

    std::printf("  [%s%s] live vars after fwd       %zu -> %zu  (%.0f%%)\n", family,
                seq_len ? "+batch" : "", live_plain, live_ckpt,
                100.0 * static_cast<double>(live_ckpt) / static_cast<double>(live_plain));
    CHECK(live_ckpt * 2 < live_plain);  // at least halved
}

}  // namespace

int main() {
    const int vocab = 97;
    const size_t T = 16;

    {
        nn::GPT2Config cfg;
        cfg.vocab = vocab;
        cfg.n_ctx = 64;
        cfg.d = 32;
        cfg.n_layers = 4;
        cfg.n_heads = 4;
        nn::GPT2 plain(cfg, /*seed=*/9), ckpt(cfg, /*seed=*/9);
        run_family("gpt2", plain, ckpt, vocab, T, /*seq_len=*/0);
        nn::GPT2 plain_b(cfg, /*seed=*/9), ckpt_b(cfg, /*seed=*/9);
        run_family("gpt2", plain_b, ckpt_b, vocab, T, /*seq_len=*/T);
    }
    {
        nn::LlamaConfig cfg;
        cfg.vocab = vocab;
        cfg.d = 32;
        cfg.n_layers = 4;
        cfg.n_heads = 4;
        cfg.d_ff = 64;
        cfg.n_ctx = 64;
        nn::Llama plain(cfg, /*seed=*/9), ckpt(cfg, /*seed=*/9);
        run_family("llama", plain, ckpt, vocab, T, /*seq_len=*/0);
        nn::Llama plain_b(cfg, /*seed=*/9), ckpt_b(cfg, /*seed=*/9);
        run_family("llama", plain_b, ckpt_b, vocab, T, /*seq_len=*/T);
    }

    std::printf("[PASS] all checkpointing tests\n");
    return 0;
}
