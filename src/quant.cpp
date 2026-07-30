#include "microtorch/quant.hpp"

#include "microtorch/device.hpp"

#include <cmath>
#include <random>
#include <stdexcept>

namespace microtorch {

QuantizedTensor quantize_int8(const Matrix& m, size_t block) {
    if (block == 0) throw std::runtime_error("quantize_int8: block must be > 0");
    QuantizedTensor t;
    t.rows = m.rows();
    t.cols = m.cols();
    t.block = block;
    t.q.resize(t.rows * t.cols);
    const size_t blocks_per_row = (t.cols + block - 1) / block;
    t.scales.resize(t.rows * blocks_per_row);

    for (size_t i = 0; i < t.rows; ++i) {
        for (size_t bidx = 0; bidx < blocks_per_row; ++bidx) {
            const size_t j0 = bidx * block;
            const size_t j1 = std::min(j0 + block, t.cols);
            float absmax = 0.0f;
            for (size_t j = j0; j < j1; ++j)
                absmax = std::max(absmax, std::fabs(m(i, j)));
            const float s = absmax > 0.0f ? absmax / 127.0f : 1.0f;
            t.scales[i * blocks_per_row + bidx] = s;
            for (size_t j = j0; j < j1; ++j) {
                const float v = m(i, j) / s;
                t.q[i * t.cols + j] = static_cast<int8_t>(
                    std::lround(std::max(-127.0f, std::min(127.0f, v))));
            }
        }
    }
    return t;
}

Matrix dequantize(const QuantizedTensor& t) {
    Matrix m(t.rows, t.cols);
    const size_t blocks_per_row = (t.cols + t.block - 1) / t.block;
    for (size_t i = 0; i < t.rows; ++i)
        for (size_t j = 0; j < t.cols; ++j)
            m(i, j) = static_cast<float>(t.q[i * t.cols + j]) *
                      t.scales[i * blocks_per_row + j / t.block];
    return m;
}

namespace nn {

namespace {
// LoRA A init: N(0, 1/r) as in the paper (B starts at zero).
Matrix lora_a_init(size_t in, size_t r, unsigned seed) {
    Matrix a(in, r);
    std::mt19937 rng(seed + 7331);
    std::normal_distribution<float> dist(0.0f, 1.0f / static_cast<float>(r));
    for (size_t i = 0; i < in; ++i)
        for (size_t j = 0; j < r; ++j) a(i, j) = dist(rng);
    return a;
}
}  // namespace

LoRALinear::LoRALinear(Matrix W, Matrix b, size_t rank, float alpha,
                       unsigned seed)
    : W_(std::move(W)), b_(std::move(b)), r_(rank),
      scaling_(alpha / static_cast<float>(rank)) {
    if (rank == 0) throw std::runtime_error("LoRALinear: rank must be > 0");
    A = reg("lora_A", lora_a_init(W_.rows(), r_, seed));
    B = reg("lora_B", Matrix(r_, W_.cols()));   // zeros: delta starts at 0
}

LoRALinear::LoRALinear(Matrix W, size_t rank, float alpha, unsigned seed)
    : LoRALinear(std::move(W), Matrix(), rank, alpha, seed) {}

Var LoRALinear::forward(const Var& x) const {
    // Frozen base path: plain data, no tape node for W.
    Var base = ops::matmul(x, make_var(W_));
    if (b_.rows() != 0) base = ops::add_bias(base, make_var(b_));
    // Adapter path: gradients flow into A and B only.
    Var delta = ops::matmul(ops::matmul(x, A), B);
    return ops::add(base, ops::scale(delta, scaling_));
}

Matrix LoRALinear::merged_weight() const {
    Matrix merged = W_;
    // W += scaling * A B
    Matrix AB = device::matmul(A->data, B->data);
    for (size_t i = 0; i < merged.rows(); ++i)
        for (size_t j = 0; j < merged.cols(); ++j)
            merged(i, j) += scaling_ * AB(i, j);
    return merged;
}

QLinear::QLinear(const Matrix& W, Matrix b, size_t block)
    : Wq_(quantize_int8(W, block)), Wdq_(dequantize(Wq_)), b_(std::move(b)) {}

QLinear::QLinear(const Matrix& W, size_t block)
    : QLinear(W, Matrix(), block) {}

Var QLinear::forward(const Var& x) const {
    Var y = ops::matmul(x, make_var(Wdq_));
    if (b_.rows() != 0) y = ops::add_bias(y, make_var(b_));
    return y;
}

QLoRALinear::QLoRALinear(const Matrix& W, Matrix b, size_t rank, float alpha,
                         size_t block, unsigned seed)
    : Wq_(quantize_int8(W, block)), Wdq_(dequantize(Wq_)), b_(std::move(b)),
      r_(rank), scaling_(alpha / static_cast<float>(rank)) {
    if (rank == 0) throw std::runtime_error("QLoRALinear: rank must be > 0");
    A = reg("lora_A", lora_a_init(W.rows(), r_, seed));
    B = reg("lora_B", Matrix(r_, W.cols()));
}

Var QLoRALinear::forward(const Var& x) const {
    Var base = ops::matmul(x, make_var(Wdq_));
    if (b_.rows() != 0) base = ops::add_bias(base, make_var(b_));
    Var delta = ops::matmul(ops::matmul(x, A), B);
    return ops::add(base, ops::scale(delta, scaling_));
}

}  // namespace nn
}  // namespace microtorch
