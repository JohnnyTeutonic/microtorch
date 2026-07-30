#pragma once
// Phase 1b: Module/Parameter/state_dict and the layer zoo, composed
// entirely from ops::* so no layer owns any new calculus -- a layer's
// backward is correct because the tape's ops are gradchecked, which is the
// whole point of doing 1a first.
//
// Naming follows the torch convention (dotted paths, "weight"/"bias") so
// that 1c's safetensors loader maps HF checkpoints onto modules by name
// with no translation table beyond an optional prefix strip.
#include <cmath>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "microtorch/ops.hpp"

namespace microtorch {
namespace nn {

class Module {
public:
    virtual ~Module() = default;

    // Dotted-path collection over the registration tree.
    std::vector<std::pair<std::string, Var>> named_parameters() const;
    std::vector<Var> parameters() const;
    std::map<std::string, Matrix> state_dict() const;
    // strict: every entry must land on a parameter and every parameter must
    // be hit -- loading a frontier checkpoint should fail loudly, not
    // half-load (missing_ok lists names allowed to stay untouched).
    void load_state_dict(const std::map<std::string, Matrix>& sd,
                         bool strict = true,
                         const std::vector<std::string>& missing_ok = {});

    void train() { set_training(true); }
    void eval() { set_training(false); }
    bool training() const { return training_; }

protected:
    Var reg(const std::string& name, Matrix init);            // register param
    template <typename M, typename... A>
    std::shared_ptr<M> mod(const std::string& name, A&&... a) {  // register child
        auto m = std::make_shared<M>(std::forward<A>(a)...);
        children_.emplace_back(name, m);
        return m;
    }

private:
    void collect(const std::string& prefix,
                 std::vector<std::pair<std::string, Var>>& out) const;
    void set_training(bool t);
    std::vector<std::pair<std::string, Var>> params_;
    std::vector<std::pair<std::string, std::shared_ptr<Module>>> children_;
    bool training_ = true;
};

class Linear : public Module {
public:
    // W stored [in, out], y = x W (+ b). This is ALSO HF-GPT-2's Conv1D
    // storage order, so its checkpoints load without transposition.
    Linear(size_t in, size_t out, bool bias = true, unsigned seed = 0);
    Var forward(const Var& x) const;
    Var W, b;   // b empty when bias=false
};

class LayerNorm : public Module {
public:
    explicit LayerNorm(size_t d, float eps = 1e-5f);
    Var forward(const Var& x) const;
    Var weight, bias;
    float eps;
};

class Embedding : public Module {
public:
    Embedding(size_t vocab, size_t d, unsigned seed = 0);
    Var forward(const std::vector<int>& ids) const;
    Var weight;
};

// Pre-LN self-attention (GPT-2 layout): fused qkv projection, heads split
// by slice_cols, output projection. Everything differentiable is a
// composition of checked ops. `causal` (default true) adds the additive
// -1e9 mask; DiT passes false, since patches attend bidirectionally.
class CausalSelfAttention : public Module {
public:
    CausalSelfAttention(size_t d, size_t n_heads, unsigned seed = 0,
                        bool causal = true);
    Var forward(const Var& x) const;   // x: [T, d]
    std::shared_ptr<Linear> c_attn, c_proj;
    size_t H, dk;
    bool causal;
};

class KimiLinearAttention : public Module {  // Phase 3a: O(n*d²) linear-time attention
public:
    KimiLinearAttention(size_t d, size_t n_heads, unsigned seed = 0,
                        bool causal = true);
    Var forward(const Var& x) const;   // x: [T, d] -> [T, d]
    std::shared_ptr<Linear> c_attn, c_proj;
    size_t H, dk;
    bool causal;
    // Note: forward() internally calls ops::kimi_attention() instead of
    // standard scaled-dot-product; q,k,v projections are identical to
    // CausalSelfAttention, only the attention mechanism differs
};

class MLP : public Module {
public:
    MLP(size_t d, size_t hidden, unsigned seed = 0);
    Var forward(const Var& x) const;
    std::shared_ptr<Linear> c_fc, c_proj;
};

class Block : public Module {   // pre-LN transformer block, GPT-2 wiring
public:
    Block(size_t d, size_t n_heads, unsigned seed = 0);
    Var forward(const Var& x) const;
    std::shared_ptr<LayerNorm> ln_1, ln_2;
    std::shared_ptr<CausalSelfAttention> attn;
    std::shared_ptr<MLP> mlp;
};

struct GPT2Config {
    size_t vocab = 50257, n_ctx = 1024, d = 768, n_layers = 12, n_heads = 12;
};

class GPT2 : public Module {
public:
    explicit GPT2(const GPT2Config& cfg, unsigned seed = 0);
    Var forward(const std::vector<int>& ids) const;   // -> logits [T, vocab]
    GPT2Config cfg;
    std::shared_ptr<Embedding> wte, wpe;
    std::vector<std::shared_ptr<Block>> h;
    std::shared_ptr<LayerNorm> ln_f;
};

// ---- optimizers (over the parameter list, matrices and rows alike) ----
class SGD {
public:
    SGD(std::vector<Var> params, float lr, float momentum = 0.0f);
    void step();
    void zero_grad();
    float lr;

private:
    std::vector<Var> params_;
    std::vector<Matrix> vel_;
    float mu_;
};

class AdamW {
public:
    AdamW(std::vector<Var> params, float lr = 1e-3f, float beta1 = 0.9f,
          float beta2 = 0.999f, float eps = 1e-8f, float weight_decay = 0.01f);
    void step();
    void zero_grad();
    float lr;

private:
    std::vector<Var> params_;
    std::vector<Matrix> m_, v_;
    float b1_, b2_, eps_, wd_;
    long t_ = 0;
};

// Dropout as a module so `training()` decides train/eval behavior the way
// torch.nn.Dropout does; eval mode is the identity. Each forward draws a
// fresh op seed from the module's own counter, so two calls in one step
// get independent masks but a fixed module seed keeps runs reproducible.
class Dropout : public Module {
public:
    explicit Dropout(float p, unsigned long long seed = 0)
        : p_(p), next_seed_(seed) {}
    Var forward(const Var& x) const;

private:
    float p_;
    mutable unsigned long long next_seed_;
};

// ---- LR schedulers (mirror torch.optim.lr_scheduler; templated on the
// optimizer because SGD/AdamW share only a public `lr` field, not a base) --

// Linear warmup for `warmup` steps then cosine decay to min_lr over
// `total` steps -- the schedule every GPT-family training recipe uses.
template <typename Opt>
class CosineWarmupLR {
public:
    CosineWarmupLR(Opt& opt, size_t warmup, size_t total, float min_lr = 0.0f)
        : opt_(opt), base_lr_(opt.lr), warmup_(warmup), total_(total),
          min_lr_(min_lr) {}
    void step() {
        ++t_;
        if (warmup_ > 0 && t_ <= warmup_) {
            opt_.lr = base_lr_ * static_cast<float>(t_) / warmup_;
            return;
        }
        const float progress =
            total_ > warmup_
                ? static_cast<float>(t_ - warmup_) / (total_ - warmup_)
                : 1.0f;
        const float clamped = progress > 1.0f ? 1.0f : progress;
        opt_.lr = min_lr_ + 0.5f * (base_lr_ - min_lr_) *
                                (1.0f + std::cos(3.14159265358979f * clamped));
    }
    size_t current_step() const { return t_; }

private:
    Opt& opt_;
    float base_lr_;
    size_t warmup_, total_, t_ = 0;
    float min_lr_;
};

// Multiply lr by gamma every step_size steps (torch's StepLR).
template <typename Opt>
class StepLR {
public:
    StepLR(Opt& opt, size_t step_size, float gamma = 0.1f)
        : opt_(opt), base_lr_(opt.lr), step_size_(step_size), gamma_(gamma) {}
    void step() {
        ++t_;
        float lr = base_lr_;
        for (size_t k = 0; k < t_ / step_size_; ++k) lr *= gamma_;
        opt_.lr = lr;
    }

private:
    Opt& opt_;
    float base_lr_;
    size_t step_size_, t_ = 0;
    float gamma_;
};

}  // namespace nn
}  // namespace microtorch
