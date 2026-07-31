#pragma once
// Parameter-efficient fine-tuning + quantization:
//
//   LoRALinear   y = x W_frozen (+ b) + (alpha/r) * (x A) B
//                Only A [in, r] and B [r, out] train; B starts at zero so
//                the adapter is exactly the identity delta at init (the
//                LoRA paper's setup). merge() folds the delta into W for
//                zero-overhead inference.
//
//   QuantizedTensor / quantize_int8 / dequantize
//                Symmetric per-block absmax int8: q = round(x / s),
//                s = absmax(block) / 127. Blockwise along each row.
//
//   QLinear      Linear whose weight lives as int8 blocks; forward
//                dequantizes into a cached fp32 copy (4x memory saved on
//                the stored parameter, exact QLoRA-style compute).
//
//   QLoRALinear  QLinear base (frozen, quantized) + LoRA adapters -- the
//                QLoRA recipe with int8 in place of NF4 (int4/NF4 is the
//                documented follow-up).
//
// All trainable paths are tape ops, so gradients flow only into A and B by
// construction: the base weight is a plain Matrix, never a tape Variable.
#include <cstdint>
#include <vector>

#include "microtorch/nn.hpp"

namespace microtorch {

struct QuantizedTensor {
    std::vector<int8_t> q;      // rows*cols values, row-major
    std::vector<float> scales;  // one per block
    size_t rows = 0, cols = 0;
    size_t block = 64;  // block length along a row

    size_t nbytes() const { return q.size() + scales.size() * sizeof(float); }
};

QuantizedTensor quantize_int8(const Matrix& m, size_t block = 64);
Matrix dequantize(const QuantizedTensor& t);

namespace nn {

class LoRALinear : public Module {
public:
    // Wraps a FROZEN base weight W [in, out] (+ optional bias b [1, out])
    // with rank-r adapters. Only A and B are registered parameters, so
    // parameters()/state_dict() see the adapters alone -- an adapter
    // checkpoint, the LoRA convention.
    LoRALinear(Matrix W, Matrix b, size_t rank, float alpha, unsigned seed = 0);
    LoRALinear(Matrix W, size_t rank, float alpha, unsigned seed = 0);

    Var forward(const Var& x) const;

    // Fold the adapter into the base: W += (alpha/r) A B. Returns the
    // merged weight; the adapter keeps training from its current state.
    Matrix merged_weight() const;

    size_t rank() const { return r_; }
    Var A, B;  // trainable

protected:
    Matrix W_, b_;  // frozen base (b_ empty if no bias)
    size_t r_;
    float scaling_;  // alpha / r
};

class QLinear : public Module {
public:
    // Stores W as int8 blocks; forward uses a cached dequantized copy.
    QLinear(const Matrix& W, Matrix b, size_t block = 64);
    explicit QLinear(const Matrix& W, size_t block = 64);

    Var forward(const Var& x) const;
    const QuantizedTensor& weight_q() const { return Wq_; }
    size_t nbytes() const { return Wq_.nbytes(); }

private:
    QuantizedTensor Wq_;
    Matrix Wdq_;  // cached dequantized weight
    Matrix b_;    // empty if no bias
};

class QLoRALinear : public Module {
public:
    // Quantized frozen base + LoRA adapters: the QLoRA recipe.
    QLoRALinear(const Matrix& W, Matrix b, size_t rank, float alpha, size_t block = 64,
                unsigned seed = 0);

    Var forward(const Var& x) const;
    size_t base_nbytes() const { return Wq_.nbytes(); }
    Var A, B;  // trainable

private:
    QuantizedTensor Wq_;
    Matrix Wdq_, b_;
    size_t r_;
    float scaling_;
};

}  // namespace nn
}  // namespace microtorch
