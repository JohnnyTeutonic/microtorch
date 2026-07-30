// Training-utility tests: dropout (forward/backward mask agreement,
// eval-mode identity), clip_grad_norm, LR schedulers, and the
// save_safetensors -> load_safetensors round trip.
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "microtorch/nn.hpp"
#include "microtorch/safetensors.hpp"

using namespace microtorch;

namespace {

void test_dropout() {
    printf("=== dropout ===\n");
    const float p = 0.5f;
    Matrix x(64, 64);
    for (size_t i = 0; i < x.rows(); ++i)
        for (size_t j = 0; j < x.cols(); ++j) x(i, j) = 1.0f;

    Var xv = make_var(x, true);
    Var y = ops::dropout(xv, p, /*seed=*/7);

    // Forward: kept entries are exactly 1/(1-p), dropped are 0, and the
    // kept fraction is near 1-p.
    size_t kept = 0;
    const float inv_keep = 1.0f / (1.0f - p);
    for (size_t i = 0; i < y->data.rows(); ++i)
        for (size_t j = 0; j < y->data.cols(); ++j) {
            const float v = y->data(i, j);
            assert(v == 0.0f || std::fabs(v - inv_keep) < 1e-6f);
            if (v != 0.0f) ++kept;
        }
    const float frac = static_cast<float>(kept) / (64.0f * 64.0f);
    assert(std::fabs(frac - (1.0f - p)) < 0.05f);
    printf("  kept fraction %.3f (target %.3f)\n", frac, 1.0f - p);

    // Backward: the replayed mask must match the forward mask exactly --
    // dx is nonzero exactly where y was nonzero.
    Var loss = ops::mean(y);
    backward(loss);
    for (size_t i = 0; i < y->data.rows(); ++i)
        for (size_t j = 0; j < y->data.cols(); ++j) {
            const bool fwd_kept = y->data(i, j) != 0.0f;
            const bool bwd_kept = xv->grad(i, j) != 0.0f;
            assert(fwd_kept == bwd_kept);
        }
    printf("  backward mask identical to forward mask\n");

    // Module in eval mode is the identity (same Var, no tape node).
    nn::Dropout drop(p);
    drop.eval();
    Var z = drop.forward(xv);
    assert(z.get() == xv.get());
    printf("  eval mode is identity\n");
}

void test_clip_grad_norm() {
    printf("=== clip_grad_norm ===\n");
    // Two params, grads [3,0...] and [4,0...]: global norm 5.
    Var a = make_var(Matrix(1, 4), true);
    Var b = make_var(Matrix(1, 4), true);
    Matrix ga(1, 4), gb(1, 4);
    ga.fill(0); gb.fill(0);
    ga(0, 0) = 3.0f; gb(0, 0) = 4.0f;
    a->accumulate(ga); b->accumulate(gb);

    const float pre = ops::clip_grad_norm({a, b}, 1.0f);
    assert(std::fabs(pre - 5.0f) < 1e-5f);
    const float post = std::sqrt(a->grad(0, 0) * a->grad(0, 0) +
                                 b->grad(0, 0) * b->grad(0, 0));
    assert(std::fabs(post - 1.0f) < 1e-5f);
    printf("  norm 5.0 clipped to %.5f\n", post);
}

void test_schedulers() {
    printf("=== LR schedulers ===\n");
    Var w = make_var(Matrix(2, 2), true);
    nn::AdamW opt({w}, /*lr=*/1.0f);

    nn::CosineWarmupLR<nn::AdamW> sched(opt, /*warmup=*/10, /*total=*/110);
    for (int i = 0; i < 5; ++i) sched.step();
    assert(std::fabs(opt.lr - 0.5f) < 1e-5f);         // mid-warmup: t/warmup
    for (int i = 0; i < 5; ++i) sched.step();
    assert(std::fabs(opt.lr - 1.0f) < 1e-5f);         // warmup done
    for (int i = 0; i < 50; ++i) sched.step();        // halfway through decay
    assert(std::fabs(opt.lr - 0.5f) < 1e-3f);         // cos(pi/2) midpoint
    for (int i = 0; i < 50; ++i) sched.step();
    assert(opt.lr < 1e-3f);                           // fully decayed
    printf("  cosine-warmup checkpoints correct\n");

    nn::AdamW opt2({w}, /*lr=*/1.0f);
    nn::StepLR<nn::AdamW> step_sched(opt2, /*step_size=*/10, /*gamma=*/0.1f);
    for (int i = 0; i < 10; ++i) step_sched.step();
    assert(std::fabs(opt2.lr - 0.1f) < 1e-6f);
    for (int i = 0; i < 10; ++i) step_sched.step();
    assert(std::fabs(opt2.lr - 0.01f) < 1e-7f);
    printf("  StepLR decays correctly\n");
}

void test_safetensors_roundtrip() {
    printf("=== safetensors round trip ===\n");
    std::map<std::string, Matrix> dict;
    Matrix a(3, 5), b(1, 7);
    for (size_t i = 0; i < a.rows(); ++i)
        for (size_t j = 0; j < a.cols(); ++j)
            a(i, j) = 0.31f * static_cast<float>(i) - 1.7f * j;
    for (size_t j = 0; j < b.cols(); ++j) b(0, j) = std::sqrt(1.0f + j);
    dict.emplace("layer.weight", a);
    dict.emplace("layer.bias", b);

    const char* path = "roundtrip_test.safetensors";
    save_safetensors(path, dict);
    auto loaded = load_safetensors(path);
    std::remove(path);

    assert(loaded.size() == 2);
    for (const auto& [name, orig] : dict) {
        const Matrix& got = loaded.at(name);
        assert(got.rows() == orig.rows() && got.cols() == orig.cols());
        for (size_t i = 0; i < got.rows(); ++i)
            for (size_t j = 0; j < got.cols(); ++j)
                assert(got(i, j) == orig(i, j));   // bit-exact
    }
    printf("  2 tensors round-tripped bit-exact\n");
}

}  // namespace

int main() {
    test_dropout();
    test_clip_grad_norm();
    test_schedulers();
    test_safetensors_roundtrip();
    printf("\n[PASS] all training-utility tests\n");
    return 0;
}
