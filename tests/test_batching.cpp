// Mini-batching receipts (STUDIO_PLAN section 9, gap 1).
//
// The claim: forwarding B sequences stacked as [B*T, d] rows with
// seq_len = T is EXACTLY the per-sequence computation — positions restart
// per block, the block-diagonal mask isolates the sequences, and every
// other op in the stack is row-wise. So the tests assert equality, not
// plausibility:
//   1. stacked logits == concat of single-sequence logits (GPT-2 + Llama)
//   2. stacked CE loss == mean of the individual CE losses
//   3. stacked backward grads == mean of the individual grads
//   4. the mask actually isolates: perturbing sequence 2 leaves sequence
//      1's logits bit-identical
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

std::vector<int> cat(const std::vector<int>& a, const std::vector<int>& b) {
    std::vector<int> out = a;
    out.insert(out.end(), b.begin(), b.end());
    return out;
}

double max_abs_diff(const Matrix& a, const Matrix& b, size_t row_off_a = 0) {
    double worst = 0.0;
    for (size_t i = 0; i < b.rows(); ++i)
        for (size_t j = 0; j < b.cols(); ++j)
            worst = std::max(worst, std::fabs(static_cast<double>(a(row_off_a + i, j)) - b(i, j)));
    return worst;
}

template <typename Model>
void run_family(const char* family, Model& model, int vocab, size_t T) {
    auto ids1 = random_ids(T + 1, 101, vocab);
    auto ids2 = random_ids(T + 1, 202, vocab);
    std::vector<int> x1(ids1.begin(), ids1.end() - 1), y1(ids1.begin() + 1, ids1.end());
    std::vector<int> x2(ids2.begin(), ids2.end() - 1), y2(ids2.begin() + 1, ids2.end());
    const auto xb = cat(x1, x2);
    const auto yb = cat(y1, y2);

    // 1. logits parity ------------------------------------------------------
    {
        NoGrad ng;
        Var l1 = model.forward(x1);
        Var l2 = model.forward(x2);
        Var lb = model.forward(xb, T);
        const double d1 = max_abs_diff(lb->data, l1->data, 0);
        const double d2 = max_abs_diff(lb->data, l2->data, T);
        std::printf("  [%s] stacked-vs-seq1 logits max diff  %.3e\n", family, d1);
        std::printf("  [%s] stacked-vs-seq2 logits max diff  %.3e\n", family, d2);
        CHECK(d1 < 1e-4);
        CHECK(d2 < 1e-4);
    }

    // 2 + 3. loss and gradient parity --------------------------------------
    // Per-sequence: two graphs, each backward scaled by 1/2, so the summed
    // grads equal the mean-over-batch gradient. Stacked: one graph, CE
    // already averages over 2T rows. Both must agree.
    auto params = model.parameters();
    float loss_sep = 0.0f;
    std::vector<Matrix> grads_sep;
    {
        zero_grad(params);
        Var la = model.forward(x1);
        Var ca = ops::cross_entropy(la, y1);
        loss_sep += 0.5f * ca->data(0, 0);
        backward(ops::scale(ca, 0.5f));
        Var lb = model.forward(x2);
        Var cb = ops::cross_entropy(lb, y2);
        loss_sep += 0.5f * cb->data(0, 0);
        backward(ops::scale(cb, 0.5f));
        for (const auto& p : params) grads_sep.push_back(p->grad);
    }
    {
        zero_grad(params);
        Var loss = ops::cross_entropy(model.forward(xb, T), yb);
        const double dl = std::fabs(static_cast<double>(loss->data(0, 0)) - loss_sep);
        std::printf("  [%s] stacked-vs-mean CE loss diff     %.3e\n", family, dl);
        CHECK(dl < 1e-5);
        backward(loss);
        double worst = 0.0;
        for (size_t k = 0; k < params.size(); ++k) {
            if (params[k]->grad.rows() == 0 || grads_sep[k].rows() == 0) continue;
            worst = std::max(worst, max_abs_diff(params[k]->grad, grads_sep[k]));
        }
        std::printf("  [%s] stacked-vs-mean grad max diff    %.3e\n", family, worst);
        CHECK(worst < 1e-4);
    }

    // 4. isolation: perturb sequence 2; sequence 1's logits must not move.
    {
        NoGrad ng;
        Var base = model.forward(xb, T);
        auto xb2 = xb;
        for (size_t i = T; i < 2 * T; ++i) xb2[i] = (xb2[i] + 7) % vocab;
        Var pert = model.forward(xb2, T);
        Matrix first(T, base->data.cols());
        for (size_t i = 0; i < T; ++i)
            for (size_t j = 0; j < first.cols(); ++j) first(i, j) = base->data(i, j);
        const double d = max_abs_diff(pert->data, first);
        std::printf("  [%s] batch isolation max leak         %.3e\n", family, d);
        CHECK(d == 0.0);
    }
}

}  // namespace

int main() {
    const int vocab = 97;
    const size_t T = 12;

    nn::GPT2Config gcfg;
    gcfg.vocab = vocab;
    gcfg.n_ctx = 64;
    gcfg.d = 32;
    gcfg.n_layers = 2;
    gcfg.n_heads = 4;
    nn::GPT2 gpt(gcfg, /*seed=*/5);
    run_family("gpt2", gpt, vocab, T);

    nn::LlamaConfig lcfg;
    lcfg.vocab = vocab;
    lcfg.d = 32;
    lcfg.n_layers = 2;
    lcfg.n_heads = 4;
    lcfg.d_ff = 64;
    lcfg.n_ctx = 64;
    nn::Llama llama(lcfg, /*seed=*/6);
    run_family("llama", llama, vocab, T);

    std::printf("[PASS] all batching tests\n");
    return 0;
}
