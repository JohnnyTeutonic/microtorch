// Sliding-window + sink attention receipts (sparse phase S1 baseline):
//   1. THE EQUIVALENCE PIN: window >= T, sinks=0 must equal
//      ops::fused_attention(causal) BITWISE — the sparse path cannot
//      silently diverge from the exact one
//   2. finite differences through q, k, v at a windowed+sinked config
//   3. locality: perturbing v OUTSIDE (window ∪ sinks) of query i leaves
//      row i untouched; inside changes it
//   4. the sink works: with window too small to reach position 0, the
//      sink keeps token 0 visible
//   5. batch pin: stacked [2*T] at seq_len=T equals the two singles
//      row-for-row (blocks isolated; windows do not cross blocks)
//   6. module + lane: ParityLM(SWA) trains (loss falls)
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "../tools/parity_model.hpp"
#include "check.hpp"
#include "microtorch/ops.hpp"

using namespace microtorch;

namespace {
Matrix randn(size_t r, size_t c, unsigned seed) {
    std::mt19937 g(seed);
    std::normal_distribution<float> d(0.0f, 1.0f);
    Matrix m(r, c);
    for (size_t i = 0; i < r; ++i)
        for (size_t j = 0; j < c; ++j) m(i, j) = d(g);
    return m;
}

double loss_of(const Var& q, const Var& k, const Var& v, float sc, size_t w, size_t s) {
    Var y = ops::swa_attention(q, k, v, sc, w, s);
    double acc = 0;
    for (size_t i = 0; i < y->data.rows(); ++i)
        for (size_t j = 0; j < y->data.cols(); ++j) acc += y->data(i, j) * y->data(i, j);
    return 0.5 * acc;
}
}  // namespace

int main() {
    const size_t T = 12, dk = 4;
    const float sc = 0.5f;

    // 1. equivalence pin ------------------------------------------------------
    {
        Var q = make_var(randn(T, dk, 1), true);
        Var k = make_var(randn(T, dk, 2), true);
        Var v = make_var(randn(T, dk, 3), true);
        Var a = ops::fused_attention(q, k, v, sc, 0, /*causal=*/true);
        Var b = ops::swa_attention(q, k, v, sc, /*window=*/T, /*sinks=*/0);
        for (size_t i = 0; i < T; ++i)
            for (size_t j = 0; j < dk; ++j) CHECK(a->data(i, j) == b->data(i, j));  // bitwise
        std::printf("1. equivalence pin: swa(window>=T, sinks=0) == fused causal, bitwise\n");
    }

    // 2. finite differences ---------------------------------------------------
    {
        const size_t w = 4, s = 1;
        Var q = make_var(randn(T, dk, 4), true);
        Var k = make_var(randn(T, dk, 5), true);
        Var v = make_var(randn(T, dk, 6), true);
        Var y = ops::swa_attention(q, k, v, sc, w, s);
        // loss = 0.5*sum(y^2) via mean * N; dL/dy = y
        const float N = static_cast<float>(T * dk);
        Var loss = ops::scale(ops::mean(ops::mul(y, y)), 0.5f * N);
        backward(loss);
        const float eps = 1e-3f;
        double max_rel = 0;
        for (Var p : {q, k, v}) {
            for (size_t i = 0; i < T; i += 5) {
                for (size_t j = 0; j < dk; j += 3) {
                    const float keep = p->data(i, j);
                    p->data(i, j) = keep + eps;
                    const double lp = loss_of(q, k, v, sc, w, s);
                    p->data(i, j) = keep - eps;
                    const double lm = loss_of(q, k, v, sc, w, s);
                    p->data(i, j) = keep;
                    const double fd = (lp - lm) / (2 * eps);
                    const double an = p->grad(i, j);
                    const double rel = std::fabs(fd - an) / (std::fabs(fd) + 1e-6);
                    max_rel = std::max(max_rel, rel);
                }
            }
        }
        CHECK(max_rel < 2e-2);
        std::printf("2. FD gradcheck q/k/v (window=4, sinks=1): max rel err %.2e\n", max_rel);
    }

    // 3. locality -------------------------------------------------------------
    {
        const size_t w = 3, s = 0, i = 9;
        Var q = make_var(randn(T, dk, 7));
        Var k = make_var(randn(T, dk, 8));
        Var v = make_var(randn(T, dk, 9));
        Var y0 = ops::swa_attention(q, k, v, sc, w, s);
        v->data(2, 0) += 10.0f;  // j=2 far outside [7,9] for i=9
        Var y1 = ops::swa_attention(q, k, v, sc, w, s);
        CHECK(y0->data(i, 0) == y1->data(i, 0));
        v->data(8, 0) += 10.0f;  // inside the window
        Var y2 = ops::swa_attention(q, k, v, sc, w, s);
        CHECK(y0->data(i, 0) != y2->data(i, 0));
        std::printf("3. locality: out-of-window v untouched row %zu; in-window changed it\n", i);
    }

    // 4. sinks ---------------------------------------------------------------
    {
        const size_t w = 2, i = 9;
        Var q = make_var(randn(T, dk, 10));
        Var k = make_var(randn(T, dk, 11));
        Var v = make_var(randn(T, dk, 12));
        Var y_nosink = ops::swa_attention(q, k, v, sc, w, /*sinks=*/0);
        Var y_sink = ops::swa_attention(q, k, v, sc, w, /*sinks=*/1);
        v->data(0, 0) += 10.0f;  // token 0: outside window of i=9
        Var y_nosink2 = ops::swa_attention(q, k, v, sc, w, 0);
        Var y_sink2 = ops::swa_attention(q, k, v, sc, w, 1);
        CHECK(y_nosink->data(i, 0) == y_nosink2->data(i, 0));  // invisible without sink
        CHECK(y_sink->data(i, 0) != y_sink2->data(i, 0));      // visible with sink
        std::printf("4. sink: token 0 stays visible beyond the window\n");
    }

    // 5. batch pin ------------------------------------------------------------
    {
        const size_t w = 4, s = 1;
        Matrix q2(2 * T, dk), k2(2 * T, dk), v2(2 * T, dk);
        Matrix qa = randn(T, dk, 13), ka = randn(T, dk, 14), va = randn(T, dk, 15);
        Matrix qb = randn(T, dk, 16), kb = randn(T, dk, 17), vb = randn(T, dk, 18);
        for (size_t i = 0; i < T; ++i)
            for (size_t j = 0; j < dk; ++j) {
                q2(i, j) = qa(i, j);
                q2(T + i, j) = qb(i, j);
                k2(i, j) = ka(i, j);
                k2(T + i, j) = kb(i, j);
                v2(i, j) = va(i, j);
                v2(T + i, j) = vb(i, j);
            }
        Var ys = ops::swa_attention(make_var(std::move(q2)), make_var(std::move(k2)),
                                    make_var(std::move(v2)), sc, w, s, /*seq_len=*/T);
        Var y1 = ops::swa_attention(make_var(std::move(qa)), make_var(std::move(ka)),
                                    make_var(std::move(va)), sc, w, s);
        Var y2 = ops::swa_attention(make_var(std::move(qb)), make_var(std::move(kb)),
                                    make_var(std::move(vb)), sc, w, s);
        float md = 0;
        for (size_t i = 0; i < T; ++i)
            for (size_t j = 0; j < dk; ++j) {
                md = std::max(md, std::fabs(ys->data(i, j) - y1->data(i, j)));
                md = std::max(md, std::fabs(ys->data(T + i, j) - y2->data(i, j)));
            }
        // Not bitwise: the stacked [2T,2T] GEMM accumulates in a different
        // order than two [T,T] GEMMs (AVX2 blocking) — fp-epsilon only.
        CHECK(md < 1e-5f);
        std::printf("5. batch pin: stacked == singles to fp epsilon (max diff %.1e)\n", md);
    }

    // 6. lane trains ----------------------------------------------------------
    {
        const size_t V = 31, d = 32, H = 4, TT = 16;
        parity::ParityLM m(parity::AttnKind::SWA, V, d, H, TT, 7, /*window=*/6, /*sinks=*/1);
        nn::AdamW opt(m.parameters(), 3e-3f);
        std::mt19937 g(5);
        std::vector<int> ids(TT + 1);
        for (auto& x : ids) x = static_cast<int>(g() % V);
        std::vector<int> x(ids.begin(), ids.end() - 1), yv(ids.begin() + 1, ids.end());
        float first = 0, last = 0;
        for (int step = 0; step < 30; ++step) {
            Var loss = ops::cross_entropy(m.forward(x), yv);
            if (step == 0) first = loss->data(0, 0);
            last = loss->data(0, 0);
            opt.zero_grad();
            backward(loss);
            opt.step();
        }
        CHECK(last < first * 0.8f);
        std::printf("6. ParityLM(SWA) trains: %.3f -> %.3f\n", first, last);
    }

    std::printf("SWA-OK 6/6\n");
    return 0;
}
