#include "microtorch/ops.hpp"
#include "microtorch/device.hpp"
#include "microtorch/kimi_linear.hpp"

#include <cmath>
#include <random>
#include <stdexcept>

namespace microtorch {
namespace ops {

namespace {

// Attach a tape node to `out` unless grad is globally off or no input
// requires it. The closure convention throughout: read out->grad, guard
// each parent on requires_grad, accumulate.
Var record(Matrix result, std::vector<Var> parents,
           std::function<void(Variable*)> bw) {
    bool needs = false;
    if (grad_enabled()) {
        for (const auto& p : parents) needs = needs || p->requires_grad;
    }
    Var out = make_var(std::move(result), needs);
    if (needs) {
        out->parents = std::move(parents);
        Variable* self = out.get();
        out->backward_fn = [self, bw]() { bw(self); };
    }
    return out;
}

}  // namespace

Var matmul(const Var& a, const Var& b) {
    if (a->data.cols() != b->data.rows()) {
        throw std::runtime_error("matmul: inner dimensions disagree");
    }
    Matrix c = device::matmul(a->data, b->data);
    return record(std::move(c), {a, b}, [](Variable* self) {
        const Var& a = self->parents[0];
        const Var& b = self->parents[1];
        if (a->requires_grad) {
            a->accumulate(device::matmul(self->grad, b->data.transpose()));
        }
        if (b->requires_grad) {
            b->accumulate(device::matmul(a->data.transpose(), self->grad));
        }
    });
}

Var add(const Var& a, const Var& b) {
    return record(a->data + b->data, {a, b}, [](Variable* self) {
        if (self->parents[0]->requires_grad) self->parents[0]->accumulate(self->grad);
        if (self->parents[1]->requires_grad) self->parents[1]->accumulate(self->grad);
    });
}

Var sub(const Var& a, const Var& b) {
    return record(a->data - b->data, {a, b}, [](Variable* self) {
        if (self->parents[0]->requires_grad) self->parents[0]->accumulate(self->grad);
        if (self->parents[1]->requires_grad) {
            self->parents[1]->accumulate(self->grad * -1.0f);
        }
    });
}

Var mul(const Var& a, const Var& b) {
    return record(a->data.hadamard(b->data), {a, b}, [](Variable* self) {
        const Var& a = self->parents[0];
        const Var& b = self->parents[1];
        if (a->requires_grad) a->accumulate(self->grad.hadamard(b->data));
        if (b->requires_grad) b->accumulate(self->grad.hadamard(a->data));
    });
}

Var add_bias(const Var& x, const Var& b) {
    if (b->data.rows() != 1 || b->data.cols() != x->data.cols()) {
        throw std::runtime_error("add_bias: bias must be [1, cols(x)]");
    }
    Matrix out = x->data;
    for (size_t i = 0; i < out.rows(); ++i)
        for (size_t j = 0; j < out.cols(); ++j) out(i, j) += b->data(0, j);
    return record(std::move(out), {x, b}, [](Variable* self) {
        const Var& x = self->parents[0];
        const Var& b = self->parents[1];
        if (x->requires_grad) x->accumulate(self->grad);
        if (b->requires_grad) {
            Matrix db(1, b->data.cols());          // column-sum, the same
            for (size_t i = 0; i < self->grad.rows(); ++i)      // contract as
                for (size_t j = 0; j < self->grad.cols(); ++j)  // compute_bias_
                    db(0, j) += self->grad(i, j);  // gradients_kernel
            b->accumulate(db);
        }
    });
}

Var gelu(const Var& x) {
    Matrix out = x->data;
    out.apply_gelu();
    return record(std::move(out), {x}, [](Variable* self) {
        const Var& x = self->parents[0];
        if (!x->requires_grad) return;
        // Correct tanh-GELU derivative, matching the (verified) CUDA kernel
        // and NOT Matrix::apply_gelu_derivative -- see primitives.hpp.
        //   u  = sqrt(2/pi) (x + 0.044715 x^3)
        //   d  = cdf(u) + x * 0.5 sech^2(u) * u'
        constexpr float k = 0.7978845608028654f;
        Matrix dx = self->grad;
        for (size_t i = 0; i < dx.rows(); ++i) {
            for (size_t j = 0; j < dx.cols(); ++j) {
                float v = x->data(i, j);
                float u = k * (v + 0.044715f * v * v * v);
                float t = std::tanh(u);
                float cdf = 0.5f * (1.0f + t);
                float pdf = 0.5f * (1.0f - t * t) * k * (1.0f + 0.134145f * v * v);
                dx(i, j) *= cdf + v * pdf;
            }
        }
        x->accumulate(dx);
    });
}

Var softmax_row(const Var& x) {
    Matrix out = x->data;
    out.apply_softmax();
    return record(std::move(out), {x}, [](Variable* self) {
        const Var& x = self->parents[0];
        if (!x->requires_grad) return;
        // dX = S .* (dY - rowsum(dY .* S)) -- the same formula
        // attention_ops.cu's batched_softmax_backward_kernel uses.
        const Matrix& S = self->data;
        const Matrix& dY = self->grad;
        Matrix dX(S.rows(), S.cols());
        for (size_t i = 0; i < S.rows(); ++i) {
            float dot = 0.0f;
            for (size_t j = 0; j < S.cols(); ++j) dot += dY(i, j) * S(i, j);
            for (size_t j = 0; j < S.cols(); ++j) {
                dX(i, j) = S(i, j) * (dY(i, j) - dot);
            }
        }
        x->accumulate(dX);
    });
}

Var mean(const Var& x) {
    Matrix out(1, 1);
    float sum = 0.0f;
    for (size_t i = 0; i < x->data.rows(); ++i)
        for (size_t j = 0; j < x->data.cols(); ++j) sum += x->data(i, j);
    const float n = static_cast<float>(x->data.rows() * x->data.cols());
    out(0, 0) = sum / n;
    return record(std::move(out), {x}, [n](Variable* self) {
        const Var& x = self->parents[0];
        if (!x->requires_grad) return;
        Matrix dx(x->data.rows(), x->data.cols(), self->grad(0, 0) / n);
        x->accumulate(dx);
    });
}

Var scale(const Var& x, float s) {
    return record(x->data * s, {x}, [s](Variable* self) {
        if (self->parents[0]->requires_grad) {
            self->parents[0]->accumulate(self->grad * s);
        }
    });
}

Var transpose(const Var& x) {
    return record(x->data.transpose(), {x}, [](Variable* self) {
        if (self->parents[0]->requires_grad) {
            self->parents[0]->accumulate(self->grad.transpose());
        }
    });
}

Var slice_cols(const Var& x, size_t j0, size_t j1) {
    if (j1 <= j0 || j1 > x->data.cols()) {
        throw std::runtime_error("slice_cols: bad range");
    }
    Matrix out(x->data.rows(), j1 - j0);
    for (size_t i = 0; i < out.rows(); ++i)
        for (size_t j = 0; j < out.cols(); ++j) out(i, j) = x->data(i, j0 + j);
    return record(std::move(out), {x}, [j0](Variable* self) {
        const Var& x = self->parents[0];
        if (!x->requires_grad) return;
        Matrix dx(x->data.rows(), x->data.cols());
        for (size_t i = 0; i < self->grad.rows(); ++i)
            for (size_t j = 0; j < self->grad.cols(); ++j)
                dx(i, j0 + j) = self->grad(i, j);
        x->accumulate(dx);
    });
}

Var concat_cols(const std::vector<Var>& xs) {
    if (xs.empty()) throw std::runtime_error("concat_cols: empty input");
    size_t rows = xs[0]->data.rows(), cols = 0;
    for (const auto& x : xs) {
        if (x->data.rows() != rows) {
            throw std::runtime_error("concat_cols: row mismatch");
        }
        cols += x->data.cols();
    }
    Matrix out(rows, cols);
    size_t off = 0;
    for (const auto& x : xs) {
        for (size_t i = 0; i < rows; ++i)
            for (size_t j = 0; j < x->data.cols(); ++j)
                out(i, off + j) = x->data(i, j);
        off += x->data.cols();
    }
    return record(std::move(out), xs, [](Variable* self) {
        size_t off = 0;
        for (const auto& p : self->parents) {
            const size_t w = p->data.cols();
            if (p->requires_grad) {
                Matrix dp(p->data.rows(), w);
                for (size_t i = 0; i < dp.rows(); ++i)
                    for (size_t j = 0; j < w; ++j) dp(i, j) = self->grad(i, off + j);
                p->accumulate(dp);
            }
            off += w;
        }
    });
}

Var layernorm(const Var& x, const Var& gamma, const Var& beta, float eps) {
    const size_t R = x->data.rows(), C = x->data.cols();
    if (gamma->data.rows() != 1 || gamma->data.cols() != C ||
        beta->data.rows() != 1 || beta->data.cols() != C) {
        throw std::runtime_error("layernorm: gamma/beta must be [1, cols(x)]");
    }
    // Cache xhat and 1/std for the backward -- same normalisation the
    // canonical layer_norm_stats_kernel/layer_norm_kernel pair computes.
    auto xhat = std::make_shared<Matrix>(R, C);
    auto rstd = std::make_shared<std::vector<float>>(R);
    Matrix out(R, C);
    for (size_t i = 0; i < R; ++i) {
        float mu = 0.0f;
        for (size_t j = 0; j < C; ++j) mu += x->data(i, j);
        mu /= static_cast<float>(C);
        float var = 0.0f;
        for (size_t j = 0; j < C; ++j) {
            const float d = x->data(i, j) - mu;
            var += d * d;
        }
        var /= static_cast<float>(C);
        const float rs = 1.0f / std::sqrt(var + eps);
        (*rstd)[i] = rs;
        for (size_t j = 0; j < C; ++j) {
            const float xh = (x->data(i, j) - mu) * rs;
            (*xhat)(i, j) = xh;
            out(i, j) = gamma->data(0, j) * xh + beta->data(0, j);
        }
    }
    return record(std::move(out), {x, gamma, beta},
                  [xhat, rstd](Variable* self) {
        const Var& x = self->parents[0];
        const Var& g = self->parents[1];
        const Var& b = self->parents[2];
        const Matrix& dY = self->grad;
        const size_t R = dY.rows(), C = dY.cols();
        if (g->requires_grad || b->requires_grad) {
            Matrix dg(1, C), db(1, C);
            for (size_t i = 0; i < R; ++i)
                for (size_t j = 0; j < C; ++j) {
                    dg(0, j) += dY(i, j) * (*xhat)(i, j);
                    db(0, j) += dY(i, j);
                }
            if (g->requires_grad) g->accumulate(dg);
            if (b->requires_grad) b->accumulate(db);
        }
        if (!x->requires_grad) return;
        // dx = rstd * (dxhat - mean(dxhat) - xhat * mean(dxhat .* xhat))
        Matrix dx(R, C);
        for (size_t i = 0; i < R; ++i) {
            float m1 = 0.0f, m2 = 0.0f;
            for (size_t j = 0; j < C; ++j) {
                const float dxh = dY(i, j) * g->data(0, j);
                m1 += dxh;
                m2 += dxh * (*xhat)(i, j);
            }
            m1 /= static_cast<float>(C);
            m2 /= static_cast<float>(C);
            for (size_t j = 0; j < C; ++j) {
                const float dxh = dY(i, j) * g->data(0, j);
                dx(i, j) = (*rstd)[i] * (dxh - m1 - (*xhat)(i, j) * m2);
            }
        }
        x->accumulate(dx);
    });
}

Var embedding(const Var& table, const std::vector<int>& ids) {
    const size_t d = table->data.cols();
    Matrix out(ids.size(), d);
    for (size_t i = 0; i < ids.size(); ++i) {
        if (ids[i] < 0 || static_cast<size_t>(ids[i]) >= table->data.rows()) {
            throw std::runtime_error("embedding: id out of range");
        }
        for (size_t j = 0; j < d; ++j) out(i, j) = table->data(ids[i], j);
    }
    return record(std::move(out), {table}, [ids](Variable* self) {
        const Var& t = self->parents[0];
        if (!t->requires_grad) return;
        // Scatter-add straight into the table's grad. Going through
        // accumulate() would allocate a dense [vocab, d] temp per backward
        // -- 154 MB for GPT-2's wte -- for a handful of touched rows.
        if (t->grad.rows() == 0) t->grad = Matrix(t->data.rows(), t->data.cols());
        for (size_t i = 0; i < ids.size(); ++i)
            for (size_t j = 0; j < t->grad.cols(); ++j)
                t->grad(ids[i], j) += self->grad(i, j);
    });
}

Var cross_entropy(const Var& logits, const std::vector<int>& targets) {
    const size_t R = logits->data.rows(), C = logits->data.cols();
    if (targets.size() != R) {
        throw std::runtime_error("cross_entropy: one target per row");
    }
    // Cache the softmax for the fused backward -- the same
    // (P - onehot)/N contract as softmax_cross_entropy_grad_kernel.
    auto P = std::make_shared<Matrix>(logits->data);
    P->apply_softmax();
    Matrix out(1, 1);
    double nll = 0.0;
    for (size_t i = 0; i < R; ++i) {
        if (targets[i] < 0 || static_cast<size_t>(targets[i]) >= C) {
            throw std::runtime_error("cross_entropy: target out of range");
        }
        nll -= std::log(std::max((*P)(i, targets[i]), 1e-12f));
    }
    out(0, 0) = static_cast<float>(nll / R);
    return record(std::move(out), {logits}, [P, targets](Variable* self) {
        const Var& l = self->parents[0];
        if (!l->requires_grad) return;
        const float g = self->grad(0, 0) / static_cast<float>(P->rows());
        Matrix dl = *P;
        for (size_t i = 0; i < dl.rows(); ++i) dl(i, targets[i]) -= 1.0f;
        l->accumulate(dl * g);
    });
}

Var mul_row(const Var& x, const Var& r) {
    if (r->data.rows() != 1 || r->data.cols() != x->data.cols()) {
        throw std::runtime_error("mul_row: r must be [1, cols(x)]");
    }
    Matrix out = x->data;
    for (size_t i = 0; i < out.rows(); ++i)
        for (size_t j = 0; j < out.cols(); ++j) out(i, j) *= r->data(0, j);
    return record(std::move(out), {x, r}, [](Variable* self) {
        const Var& x = self->parents[0];
        const Var& r = self->parents[1];
        if (x->requires_grad) {
            Matrix dx = self->grad;
            for (size_t i = 0; i < dx.rows(); ++i)
                for (size_t j = 0; j < dx.cols(); ++j) dx(i, j) *= r->data(0, j);
            x->accumulate(dx);
        }
        if (r->requires_grad) {
            Matrix dr(1, r->data.cols());
            for (size_t i = 0; i < self->grad.rows(); ++i)
                for (size_t j = 0; j < self->grad.cols(); ++j)
                    dr(0, j) += self->grad(i, j) * x->data(i, j);
            r->accumulate(dr);
        }
    });
}

Var silu(const Var& x) {
    Matrix out = x->data;
    for (size_t i = 0; i < out.rows(); ++i)
        for (size_t j = 0; j < out.cols(); ++j) {
            const float v = out(i, j);
            out(i, j) = v / (1.0f + std::exp(-v));
        }
    return record(std::move(out), {x}, [](Variable* self) {
        const Var& x = self->parents[0];
        if (!x->requires_grad) return;
        // d/dx [x s(x)] = s(x) (1 + x (1 - s(x)))
        Matrix dx = self->grad;
        for (size_t i = 0; i < dx.rows(); ++i)
            for (size_t j = 0; j < dx.cols(); ++j) {
                const float v = x->data(i, j);
                const float s = 1.0f / (1.0f + std::exp(-v));
                dx(i, j) *= s * (1.0f + v * (1.0f - s));
            }
        x->accumulate(dx);
    });
}

Var rmsnorm(const Var& x, const Var& w) {
    const size_t R = x->data.rows(), C = x->data.cols();
    if (w->data.rows() != 1 || w->data.cols() != C) {
        throw std::runtime_error("rmsnorm: w must be [1, cols(x)]");
    }
    // RMS normalization: x / RMS(x) * w, where RMS = sqrt(mean(x^2))
    // No mean centering, no bias. Stores RMS inverse for backward.
    auto rms_inv = std::make_shared<std::vector<float>>(R);
    Matrix out(R, C);
    const float eps = 1e-5f;
    for (size_t i = 0; i < R; ++i) {
        float rms_sq = 0.0f;
        for (size_t j = 0; j < C; ++j) rms_sq += x->data(i, j) * x->data(i, j);
        rms_sq /= static_cast<float>(C);
        (*rms_inv)[i] = 1.0f / std::sqrt(rms_sq + eps);
        for (size_t j = 0; j < C; ++j)
            out(i, j) = x->data(i, j) * (*rms_inv)[i] * w->data(0, j);
    }
    return record(std::move(out), {x, w}, [rms_inv](Variable* self) {
        const Var& x = self->parents[0];
        const Var& w = self->parents[1];
        const size_t R = self->grad.rows(), C = self->grad.cols();
        if (w->requires_grad) {
            Matrix dw(1, C);
            for (size_t i = 0; i < R; ++i)
                for (size_t j = 0; j < C; ++j)
                    dw(0, j) += self->grad(i, j) * x->data(i, j) * (*rms_inv)[i];
            w->accumulate(dw);
        }
        if (!x->requires_grad) return;
        // RMSNorm gradient: y_ij = x_ij * w_j * rms_inv_i
        // dL/dx_ik = rms_inv_i * [dY_ik * w_k - x_ik * rms_inv_i^2 * sum_j(dY_ij * w_j * x_ij) / n]
        Matrix dx(R, C);
        for (size_t i = 0; i < R; ++i) {
            float term = 0.0f;
            for (size_t j = 0; j < C; ++j)
                term += self->grad(i, j) * w->data(0, j) * x->data(i, j);
            const float ri2 = (*rms_inv)[i] * (*rms_inv)[i];
            const float n_inv = 1.0f / static_cast<float>(C);
            for (size_t j = 0; j < C; ++j)
                dx(i, j) = (*rms_inv)[i] * (self->grad(i, j) * w->data(0, j) -
                    x->data(i, j) * ri2 * term * n_inv);
        }
        x->accumulate(dx);
    });
}

Var apply_rope(const Var& qk, const std::vector<int>& pos, float theta_base,
               size_t head_dim) {
    const size_t T = qk->data.rows(), d3 = qk->data.cols();
    if (d3 % 3 != 0) {
        throw std::runtime_error("apply_rope: cols must be divisible by 3");
    }
    const size_t d = d3 / 3;
    if (head_dim % 2 != 0 || head_dim > d) {
        throw std::runtime_error("apply_rope: head_dim must be even and <= d");
    }
    // Cache angles for backward (position and frequency basis)
    auto pos_cache = std::make_shared<std::vector<int>>(pos);
    Matrix out = qk->data;
    // Apply RoPE to q and k via complex rotations on adjacent dimension pairs
    // RoPE operates on pairs (x[2j], x[2j+1]) as complex number rotations
    for (size_t i = 0; i < T; ++i) {
        const float m = static_cast<float>(pos[i]);
        // Apply to q (cols 0..d) and k (cols d..2d); skip v (cols 2d..3d)
        for (size_t start = 0; start < 2 * d; start += d) {
            for (size_t dim = 0; dim < head_dim; dim += 2) {
                const float inv_freq = 1.0f / std::pow(theta_base,
                    static_cast<float>(dim) / head_dim);
                const float theta = m * inv_freq;
                const float cos_t = std::cos(theta);
                const float sin_t = std::sin(theta);
                // Rotate (x[dim], x[dim+1]) pair
                const float x0 = out(i, start + dim);
                const float x1 = out(i, start + dim + 1);
                out(i, start + dim) = x0 * cos_t - x1 * sin_t;
                out(i, start + dim + 1) = x0 * sin_t + x1 * cos_t;
            }
        }
    }
    return record(std::move(out), {qk}, [pos_cache, theta_base, head_dim,
                                        d3](Variable* self) {
        const Var& qk = self->parents[0];
        if (!qk->requires_grad) return;
        const size_t T = self->grad.rows(), d = d3 / 3;
        Matrix dqk = self->grad;
        // Backward: apply inverse rotation (negative theta)
        for (size_t i = 0; i < T; ++i) {
            const float m = static_cast<float>((*pos_cache)[i]);
            for (size_t start = 0; start < 2 * d; start += d) {
                for (size_t dim = 0; dim < head_dim; dim += 2) {
                    const float inv_freq = 1.0f / std::pow(theta_base,
                        static_cast<float>(dim) / head_dim);
                    const float theta = -m * inv_freq;   // Negative for inverse
                    const float cos_t = std::cos(theta);
                    const float sin_t = std::sin(theta);
                    const float dy0 = dqk(i, start + dim);
                    const float dy1 = dqk(i, start + dim + 1);
                    dqk(i, start + dim) = dy0 * cos_t - dy1 * sin_t;
                    dqk(i, start + dim + 1) = dy0 * sin_t + dy1 * cos_t;
                }
            }
        }
        qk->accumulate(dqk);
    });
}

// Phase 3a: Kimi Linear attention (O(n*d²) vs O(n²*d) standard attention)
Var kimi_attention(const Var& q, const Var& k, const Var& v, bool causal) {
    using kimi::KimiLinearAttention;

    // The class backward recomputes CAUSAL prefix sums; a non-causal
    // forward under grad would get silently wrong gradients. Fail loudly
    // until the full-sum backward exists.
    if (!causal && grad_enabled() &&
        (q->requires_grad || k->requires_grad || v->requires_grad)) {
        throw std::runtime_error(
            "kimi_attention: non-causal backward not implemented; wrap in "
            "NoGrad for inference or use causal=true");
    }

    size_t head_dim = q->data.cols();
    KimiLinearAttention kimi(head_dim);

    // Forward: linear-time attention
    Matrix out = kimi.forward(q->data, k->data, v->data, causal);

    // Backward: compute gradients w.r.t. q, k, v through the attention
    return record(std::move(out), {q, k, v}, [kimi = std::move(kimi)](Variable* self) {
        const Var& q_var = self->parents[0];
        const Var& k_var = self->parents[1];
        const Var& v_var = self->parents[2];

        if (!q_var->requires_grad && !k_var->requires_grad && !v_var->requires_grad) {
            return;
        }

        // Backward through Kimi Linear: compute gradients
        auto [grad_q, grad_k, grad_v] = kimi.backward(
            self->grad, q_var->data, k_var->data, v_var->data, self->data);

        if (q_var->requires_grad) q_var->accumulate(grad_q);
        if (k_var->requires_grad) k_var->accumulate(grad_k);
        if (v_var->requires_grad) v_var->accumulate(grad_v);
    });
}

Var dropout(const Var& x, float p, unsigned long long seed) {
    if (p < 0.0f || p >= 1.0f) {
        throw std::runtime_error("dropout: p must be in [0, 1)");
    }
    if (p == 0.0f) return x;
    const float keep = 1.0f - p, inv_keep = 1.0f / keep;
    // The mask is a pure function of (seed, element index); backward
    // replays the same generator instead of storing a mask matrix.
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    Matrix out = x->data;
    for (size_t i = 0; i < out.rows(); ++i)
        for (size_t j = 0; j < out.cols(); ++j)
            out(i, j) = (u(rng) < keep) ? out(i, j) * inv_keep : 0.0f;
    return record(std::move(out), {x}, [p, seed](Variable* self) {
        const Var& x = self->parents[0];
        if (!x->requires_grad) return;
        const float keep = 1.0f - p, inv_keep = 1.0f / keep;
        std::mt19937_64 rng(seed);
        std::uniform_real_distribution<float> u(0.0f, 1.0f);
        Matrix dx = self->grad;
        for (size_t i = 0; i < dx.rows(); ++i)
            for (size_t j = 0; j < dx.cols(); ++j)
                dx(i, j) = (u(rng) < keep) ? dx(i, j) * inv_keep : 0.0f;
        x->accumulate(dx);
    });
}

float clip_grad_norm(const std::vector<Var>& params, float max_norm) {
    double sq = 0.0;
    for (const auto& p : params) {
        const Matrix& g = p->grad;
        if (g.rows() == 0) continue;   // never accumulated
        for (size_t i = 0; i < g.rows(); ++i)
            for (size_t j = 0; j < g.cols(); ++j)
                sq += static_cast<double>(g(i, j)) * g(i, j);
    }
    const float total = static_cast<float>(std::sqrt(sq));
    if (total > max_norm && total > 0.0f) {
        const float scale = max_norm / total;
        for (const auto& p : params) {
            Matrix& g = p->grad;
            for (size_t i = 0; i < g.rows(); ++i)
                for (size_t j = 0; j < g.cols(); ++j)
                    g(i, j) *= scale;
        }
    }
    return total;
}

}  // namespace ops
}  // namespace microtorch
