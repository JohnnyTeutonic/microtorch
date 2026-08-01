// FlexLM receipts — the paper-faithful decoder family (norm/activation/
// position/d_ff/depth all constructor-real, so a fetched arXiv config
// trains as WRITTEN, not approximated by the nearest preset):
//   1. THE EQUIVALENCE PIN: FlexLM(layernorm, gelu, learned, L=2,
//      d_ff=4d) reproduces ParityLM(EXACT) logits BIT-FOR-BIT at the
//      same seed — flex is a strict generalization, not a rewrite
//   2. sinusoidal table matches the Vaswani formula at spot positions,
//      and contributes zero parameters (learned - sinusoidal == n_ctx*d)
//   3. depth is real: L=6 builds 6 distinct "layers.N." param groups
//   4. rmsnorm + swiglu + sinusoidal trains: loss falls (composition of
//      FD-checked ops, exercised end-to-end)
//   5. relu path: grads reach the first block's fc through the ReLU
//   6. batch pin: stacked [2*T] forward at seq_len=T equals the two
//      single-sequence forwards row-for-row (mask isolation + positions
//      restart, sinusoidal included)
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "../tools/parity_model.hpp"
#include "check.hpp"
#include "microtorch/ops.hpp"

using namespace microtorch;

namespace {
std::vector<int> ids_mod(size_t n, size_t vocab, unsigned seed) {
    std::mt19937 g(seed);
    std::vector<int> v(n);
    for (auto& x : v) x = static_cast<int>(g() % vocab);
    return v;
}
}  // namespace

int main() {
    const size_t V = 31, d = 32, H = 4, T = 16;

    // 1. equivalence pin ------------------------------------------------------
    {
        parity::ParityLM ref(parity::AttnKind::EXACT, V, d, H, T, 7);
        parity::FlexConfig fc;
        fc.vocab = V;
        fc.d = d;
        fc.n_heads = H;
        fc.n_ctx = T;
        fc.n_layers = 2;
        fc.d_ff = 4 * d;
        parity::FlexLM flex(fc, 7);
        CHECK(flex.parameter_count() == ref.parameter_count());
        auto ids = ids_mod(T, V, 3);
        Var a = ref.forward(ids), b = flex.forward(ids);
        for (size_t i = 0; i < a->data.rows(); ++i)
            for (size_t j = 0; j < a->data.cols(); ++j)
                CHECK(a->data(i, j) == b->data(i, j));  // bitwise, not approx
        std::printf("1. equivalence pin: FlexLM(defaults) == ParityLM(EXACT) bitwise\n");
    }

    // 2. sinusoidal table -----------------------------------------------------
    {
        parity::FlexConfig fc;
        fc.vocab = V;
        fc.d = d;
        fc.n_heads = H;
        fc.n_ctx = T;
        fc.d_ff = 4 * d;
        parity::FlexLM learned(fc, 7);
        fc.pos = "sinusoidal";
        parity::FlexLM sinu(fc, 7);
        CHECK(learned.parameter_count() - sinu.parameter_count() == T * d);
        // Vaswani: PE(p, 2i) = sin(p / 10000^(2i/d)), PE(p, 2i+1) = cos(same)
        for (size_t p : {size_t(0), size_t(5), T - 1})
            for (size_t i : {size_t(0), size_t(7), d - 1}) {
                const double angle = p / std::pow(10000.0, double(2 * (i / 2)) / double(d));
                const float want =
                    static_cast<float>(i % 2 == 0 ? std::sin(angle) : std::cos(angle));
                CHECK(std::fabs(sinu.sin_table(p, i) - want) < 1e-6f);
            }
        std::printf("2. sinusoidal: formula matches, zero parameters added\n");
    }

    // 3. depth is real --------------------------------------------------------
    {
        parity::FlexConfig fc;
        fc.vocab = V;
        fc.d = d;
        fc.n_heads = H;
        fc.n_ctx = T;
        fc.n_layers = 6;
        fc.d_ff = 4 * d;
        parity::FlexLM deep(fc, 7);
        int block_params = 0;
        for (const auto& [name, p] : deep.named_parameters())
            if (name.rfind("layers.5.", 0) == 0) ++block_params;
        CHECK(block_params > 0);
        parity::FlexConfig f2 = fc;
        f2.n_layers = 2;
        parity::FlexLM two(f2, 7);
        CHECK(deep.parameter_count() > two.parameter_count());
        std::printf("3. depth: layers.5 exists, params grow with L\n");
    }

    // 4. rmsnorm + swiglu + sinusoidal trains ---------------------------------
    {
        parity::FlexConfig fc;
        fc.vocab = V;
        fc.d = d;
        fc.n_heads = H;
        fc.n_ctx = T;
        fc.d_ff = 3 * d;
        fc.norm = "rmsnorm";
        fc.act = "swiglu";
        fc.pos = "sinusoidal";
        parity::FlexLM m(fc, 11);
        nn::AdamW opt(m.parameters(), 3e-3f);
        auto ids = ids_mod(T + 1, V, 5);
        std::vector<int> x(ids.begin(), ids.end() - 1), y(ids.begin() + 1, ids.end());
        float first = 0, last = 0;
        for (int step = 0; step < 30; ++step) {
            Var loss = ops::cross_entropy(m.forward(x), y);
            if (step == 0) first = loss->data(0, 0);
            last = loss->data(0, 0);
            opt.zero_grad();
            backward(loss);
            opt.step();
        }
        CHECK(last < first * 0.8f);
        std::printf("4. rmsnorm+swiglu+sinusoidal trains: %.3f -> %.3f\n", first, last);
    }

    // 5. relu path grads ------------------------------------------------------
    {
        parity::FlexConfig fc;
        fc.vocab = V;
        fc.d = d;
        fc.n_heads = H;
        fc.n_ctx = T;
        fc.d_ff = 4 * d;
        fc.act = "relu";
        parity::FlexLM m(fc, 13);
        auto ids = ids_mod(T, V, 9);
        std::vector<int> y(ids.rbegin(), ids.rend());
        Var loss = ops::cross_entropy(m.forward(ids), y);
        backward(loss);
        double g = 0;
        for (const auto& [name, p] : m.named_parameters())
            if (name.find("layers.0.mlp.c_fc") != std::string::npos && p->grad.rows())
                for (size_t i = 0; i < p->grad.rows(); ++i)
                    for (size_t j = 0; j < p->grad.cols(); ++j) g += std::fabs(p->grad(i, j));
        CHECK(g > 0);
        std::printf("5. relu path: grad reaches layers.0.mlp.c_fc (|g|=%.3g)\n", g);
    }

    // 6. batch pin ------------------------------------------------------------
    {
        parity::FlexConfig fc;
        fc.vocab = V;
        fc.d = d;
        fc.n_heads = H;
        fc.n_ctx = T;
        fc.d_ff = 4 * d;
        fc.norm = "rmsnorm";
        fc.pos = "sinusoidal";
        parity::FlexLM m(fc, 17);
        auto s1 = ids_mod(T, V, 21), s2 = ids_mod(T, V, 22);
        std::vector<int> both = s1;
        both.insert(both.end(), s2.begin(), s2.end());
        Var a1 = m.forward(s1), a2 = m.forward(s2), ab = m.forward(both, T);
        float md = 0;
        for (size_t i = 0; i < T; ++i)
            for (size_t j = 0; j < V; ++j) {
                md = std::max(md, std::fabs(ab->data(i, j) - a1->data(i, j)));
                md = std::max(md, std::fabs(ab->data(T + i, j) - a2->data(i, j)));
            }
        CHECK(md < 2e-5f);
        std::printf("6. batch pin: stacked == singles (max diff %.2e)\n", md);
    }

    std::printf("FLEX-OK 6/6\n");
    return 0;
}
