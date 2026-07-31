// In-place elementwise receipts (performance-triage gap 4):
//   1. FD gradcheck through the blessed pattern (in-place after matmul)
//      for relu_ / sigmoid_ / scale_, on both leaves
//   2. parity: the in-place chain equals the out-of-place chain in loss
//      AND in every gradient, bit-for-bit-close
//   3. the memory receipt: the in-place chain creates FEWER live tape
//      nodes (the activation that would have been stored never exists)
#include <cmath>
#include <cstdio>
#include <functional>
#include <random>
#include <vector>

#include "microtorch/ops.hpp"

#include "check.hpp"

using namespace microtorch;
namespace ops = microtorch::ops;

namespace {

Matrix randn(size_t r, size_t c, unsigned seed, float scale = 1.0f) {
    std::mt19937 gen(seed);
    std::normal_distribution<float> d(0.0f, scale);
    Matrix m(r, c);
    for (size_t i = 0; i < r; ++i)
        for (size_t j = 0; j < c; ++j) m(i, j) = d(gen);
    return m;
}

double fd_vs_analytic(const std::function<float()>& forward, Var leaf, const Matrix& analytic,
                      float h = 1e-2f) {
    double worst = 0.0;
    for (size_t i = 0; i < leaf->data.rows(); ++i)
        for (size_t j = 0; j < leaf->data.cols(); ++j) {
            const float keep = leaf->data(i, j);
            NoGrad ng;
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
    return worst;
}

double max_abs_diff(const Matrix& a, const Matrix& b) {
    double worst = 0.0;
    for (size_t i = 0; i < a.rows(); ++i)
        for (size_t j = 0; j < a.cols(); ++j)
            worst = std::max(worst, std::fabs(static_cast<double>(a(i, j)) - b(i, j)));
    return worst;
}

}  // namespace

int main() {
    const double TOL = 5e-3;
    Var A = make_var(randn(5, 7, 1), true);
    Var W = make_var(randn(7, 6, 2, 0.6f), true);
    Var M = make_var(randn(5, 6, 3));  // no-grad mask so grads are non-uniform

    struct Case {
        const char* name;
        std::function<Var(const Var&)> in_place, out_of_place;
    };
    const Case cases[] = {
        {"relu", [](const Var& x) { return ops::relu_(x); },
         [](const Var& x) { return ops::relu(x); }},
        {"sigmoid", [](const Var& x) { return ops::sigmoid_(x); },
         [](const Var& x) { return ops::sigmoid(x); }},
        {"scale", [](const Var& x) { return ops::scale_(x, 1.7f); },
         [](const Var& x) { return ops::scale(x, 1.7f); }},
    };

    for (const auto& c : cases) {
        // 1. FD through the blessed pattern: f_(matmul(A, W)).
        auto f = [&] { return ops::mean(ops::mul(c.in_place(ops::matmul(A, W)), M))->data(0, 0); };
        zero_grad({A, W});
        backward(ops::mean(ops::mul(c.in_place(ops::matmul(A, W)), M)));
        const double ea = fd_vs_analytic(f, A, A->grad);
        const double ew = fd_vs_analytic(f, W, W->grad);
        std::printf("  [inplace] %s_: FD dA %.3e  dW %.3e\n", c.name, ea, ew);
        CHECK(ea < TOL);
        CHECK(ew < TOL);

        // 2. parity with the out-of-place sibling: loss + grads agree.
        zero_grad({A, W});
        Var l1 = ops::mean(ops::mul(c.in_place(ops::matmul(A, W)), M));
        backward(l1);
        Matrix gA = A->grad, gW = W->grad;
        const float v1 = l1->data(0, 0);
        zero_grad({A, W});
        Var l2 = ops::mean(ops::mul(c.out_of_place(ops::matmul(A, W)), M));
        backward(l2);
        const double dl = std::fabs(static_cast<double>(v1) - l2->data(0, 0));
        const double dg = std::max(max_abs_diff(gA, A->grad), max_abs_diff(gW, W->grad));
        std::printf("  [inplace] %s_: parity loss %.3e grads %.3e\n", c.name, dl, dg);
        CHECK(dl < 1e-6);
        CHECK(dg < 1e-6);
    }

    // 3. the memory receipt: nodes that never exist. A 4-op chain with
    // out-of-place activations allocates a node per activation; in-place
    // fuses each into its producer.
    {
        const size_t before_out = live_variables();
        Var h = ops::relu(ops::matmul(A, W));
        Var y1 = ops::sigmoid(h);
        const size_t out_nodes = live_variables() - before_out;
        (void)y1;

        const size_t before_in = live_variables();
        Var h2 = ops::relu_(ops::matmul(A, W));
        Var y2 = ops::sigmoid_(h2);
        const size_t in_nodes = live_variables() - before_in;
        (void)y2;
        std::printf("  [inplace] chain tape nodes: %zu out-of-place -> %zu in-place\n", out_nodes,
                    in_nodes);
        CHECK(in_nodes < out_nodes);
        CHECK(in_nodes == 1);  // exactly the matmul node; activations never exist
    }

    std::printf("[PASS] all in-place tests\n");
    return 0;
}
