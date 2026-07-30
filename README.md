# microtorch

A compact C++17 deep-learning library with a gradchecked autograd tape, HuggingFace
checkpoint compatibility, and implementations of attention mechanisms that mainstream
frameworks don't ship yet.

**~8K lines of core code. No CUDA required. Builds in under two minutes.**

## Why this exists

PyTorch is 4M+ lines; understanding *why* your gradient is wrong means reading dispatcher
internals. microtorch takes the opposite bet: every operation is a readable forward +
hand-derived backward pair, every backward is verified against finite differences, and the
whole tape fits in [one header](include/microtorch/autograd.hpp).

That makes it three things at once:

1. **A working training stack** — load a HF GPT-2 or Qwen checkpoint, fine-tune it,
   save it back as safetensors.
2. **A research vehicle** — novel mechanisms (linear attention, selective computation,
   state-space models) land here in days, not framework-release cycles.
3. **A reference implementation** — if you want to know what RoPE or a cross-entropy
   backward *actually does*, the answer is one file away, not forty.

## Feature matrix

| Component | Status | Notes |
|---|---|---|
| Reverse-mode autograd | ✅ | DAG tape, topological sort, `NoGrad` scope |
| Gradient verification | ✅ | Every op finite-difference checked in CI |
| Layers | ✅ | Linear, LayerNorm, RMSNorm, Embedding, Attention, MLP, Dropout |
| Full GPT-2 | ✅ | Logit-parity verified against HF checkpoint |
| Llama-family ops | ✅ | RMSNorm + RoPE; Qwen 1.5-1.8B load verified |
| Optimizers | ✅ | SGD (momentum), AdamW |
| LR schedulers | ✅ | Cosine-with-warmup, StepLR |
| Grad clipping | ✅ | Global-norm (`clip_grad_norm`) |
| Checkpoint IO | ✅ | safetensors load **and** save (HF round-trip) |
| GGUF export | ✅ | `export_gguf_llama`: state_dict → .gguf for [tinyllama.cpp](#the-training--inference-pipeline) |
| Cross-entropy loss | ✅ | Fused softmax backward |
| Python bindings | ✅ | pybind11, numpy interop (`-DMICROTORCH_BUILD_PYTHON=ON`) |
| LoRA | ✅ | `LoRALinear`: frozen base + rank-r adapters, adapter-only state_dict, `merged_weight()` |
| Quantization | ✅ | int8 blockwise (absmax/block), `QLinear` ~3.7x smaller weights |
| QLoRA | ✅ | `QLoRALinear`: quantized frozen base + trainable adapters |
| CUDA dispatch seam | ✅ | `device::matmul` → `cuda::matmul` (`-DMICROTORCH_CUDA=ON`); gradcheck/nn/LoRA suites **pass on a T4** (2026-07-30) |
| **Kimi linear attention** | ✅ | O(n·d²) vs O(n²·d); drop-in `KimiLinearAttention` |
| **Cerebellum selective gating** | ✅ | Prediction-residual gating; skips compute on routine tokens |
| **Mamba / S4 state-space** | ✅ | Trainable through time: `ssm_scan` tape op, BPTT FD-gradchecked on all five inputs (hardware-parallel scan still roadmap) |
| **Surprise-routed density (SRD)** | 🧪 | Novel research: per-query exact/linear routing by prediction residual — falsifier passed at 5-6σ twice, gate concentrates on retrieval sites ([SPARSE_ATTENTION.md](SPARSE_ATTENTION.md)) |
| GPU kernels | 🧪 | matmul dispatches to transformer_core CUDA (T4-validated); phase B = resident device tensors + more ops |

## Quick start

```bash
git clone git@github.com:JohnnyTeutonic/microtorch.git
cd microtorch
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
ctest --output-on-failure     # gradchecks + unit + integration tests
```

### Train something (C++)

```cpp
#include "microtorch/nn.hpp"
using namespace microtorch;

nn::GPT2Config cfg;                 // 124M defaults
nn::GPT2 model(cfg, /*seed=*/42);

nn::AdamW opt(model.parameters(), /*lr=*/3e-4f);
nn::CosineWarmupLR<nn::AdamW> sched(opt, /*warmup=*/100, /*total=*/10000);

for (auto [ids, targets] : batches) {
    Var logits = model.forward(ids);
    Var loss = ops::cross_entropy(logits, targets);
    opt.zero_grad();
    backward(loss);
    ops::clip_grad_norm(model.parameters(), 1.0f);
    opt.step();
    sched.step();
}
save_safetensors("ckpt.safetensors", model.state_dict());
```

### Python

```bash
cmake .. -DMICROTORCH_BUILD_PYTHON=ON && make _microtorch
```

```python
import numpy as np, _microtorch as mt

x = mt.tensor(np.random.randn(8, 256).astype(np.float32), requires_grad=True)
attn = mt.nn.KimiLinearAttention(d=256, n_heads=4)
loss = mt.ops.mean(attn(x))
mt.backward(loss)
print(x.grad.shape)        # (8, 256)
```

### Load a HuggingFace checkpoint

```cpp
auto sd = load_safetensors("gpt2/model.safetensors");
nn::GPT2 model(cfg);
model.load_state_dict(sd, /*strict=*/true);   // fails loudly on any mismatch
```

## The novel mechanisms

### Kimi linear attention — [kimi_linear.hpp](include/microtorch/kimi_linear.hpp)

Standard attention pays O(n²·d) to build the full attention matrix. The linear-attention
family replaces softmax with a feature map φ(x) = elu(x)+1 and reassociates the product,
paying O(n·d²) instead:

```
out_i = φ(q_i)ᵀ · Σ_{j≤i} φ(k_j) v_jᵀ  /  φ(q_i)ᵀ · Σ_{j≤i} φ(k_j)
```

The cumulative sums preserve causality without a mask. `nn::KimiLinearAttention` is
interface-identical to `nn::CausalSelfAttention` — swap one line to try it. Measured on
CPU: ~8.9x faster forward+backward at seq=16, converging to ~1.1x as n approaches d
(the crossover regime the complexity analysis predicts).

### Cerebellum-inspired selective gating — [cerebellum.hpp](include/microtorch/cerebellum.hpp)

Motivated by cerebellar prediction-error filtering: a small predictor (d → d/4 → d)
learns what "routine" hidden states look like; the gate
`σ(‖x − predict(x)‖)` mixes the expensive layer's output with the identity path
per token. Surprising tokens get full compute, routine tokens skip it. Wraps any
layer via `std::function` — no changes to the wrapped code.

### Mamba / S4 state-space — [mamba.hpp](include/microtorch/mamba.hpp)

Discrete state-space recurrence `x[t+1] = A·x[t] + B·u[t]` as a drop-in sequence
backbone: O(1) memory per generated token instead of a growing KV cache. The current
implementation is the sequential-scan baseline; the hardware-parallel scan is the next
milestone (see [DESIGN.md](DESIGN.md)).

## Repository layout

```
include/microtorch/   public headers (one concern per header)
  autograd.hpp        the tape: Variable, backward(), NoGrad
  ops.hpp             op set — every op forward + audited backward
  nn.hpp              Module, layers, optimizers, schedulers
  kimi_linear.hpp     phase 3a: linear attention
  cerebellum.hpp      phase 3b: selective gating
  mamba.hpp           phase 3c: state-space models
  safetensors.hpp     HF checkpoint load/save
src/                  implementations
tests/                gradchecks + unit tests (all in CI)
tools/                parity checkers, benchmarks, HF download scripts
python/               pybind11 bindings
docs/                 Doxygen config (make docs)
```

## Verification philosophy

Nothing merges without:

1. **Finite-difference gradcheck** — every op's backward vs numerical gradient.
2. **Parity tests** — GPT-2 logits match the HF reference end-to-end; Qwen tensors
   load and map onto modules by name with zero translation tables.
3. **Round-trip tests** — save → load → bit-identical.

CI runs the full suite (plus cppcheck, clang-format, Valgrind) on every push —
see [.github/workflows](.github/workflows).

## The training → inference pipeline

microtorch is the training half of a two-engine pipeline:

```
HF checkpoint ──load_safetensors──► microtorch (fine-tune on the tape)
                                        │
                              export_gguf_llama(path, state_dict, cfg)
                                        │
                                        ▼
                              .gguf ──► tinyllama.cpp (inference engine)
```

```cpp
#include "microtorch/gguf.hpp"

gguf::LlamaExportConfig cfg;
cfg.embedding_length = 2048; cfg.block_count = 24;
cfg.feed_forward_length = 5504; cfg.head_count = 16;
cfg.vocab_size = 151936; cfg.tokens = vocab;
gguf::export_gguf_llama("model.gguf", model.state_dict(), cfg);
```

The GGUF writer is the alignment-audited implementation from `transformer_core`
(the tensor-data section starts at `align_up(header_end, 32)` — the exact detail
that once turned a working model into word salad, now regression-tested in
`test_gguf_export`).

## Fine-tuning on a budget: LoRA / QLoRA

```cpp
#include "microtorch/quant.hpp"

// Frozen base + rank-8 adapters; only A and B train.
nn::LoRALinear lora(W_pretrained, /*rank=*/8, /*alpha=*/16.0f);
nn::AdamW opt(lora.parameters(), 1e-3f);      // 2 tensors, not the base
// ... train ...
Matrix W_final = lora.merged_weight();        // fold in for zero-cost inference

// QLoRA: the base lives as int8 blocks (~3.7x smaller), adapters in fp32.
nn::QLoRALinear qlora(W_pretrained, bias, /*rank=*/8, /*alpha=*/16.0f);
```

Properties enforced by `test_lora_quant`: adapter output is *exactly* the base at
init (B=0), gradients touch only A/B, `merged_weight()` matches the adapter forward
to 1e-4, int8 round-trip error is bounded by half a quantization step.

## CUDA

The dispatch seam is live: every matmul routes through `device::matmul`
([device.hpp](include/microtorch/device.hpp)). Default builds are CPU-only and
bit-identical to before. `-DMICROTORCH_CUDA=ON` compiles transformer_core's kernel
tree and `device::set(Device::CUDA)` (or `MICROTORCH_DEVICE=cuda`) dispatches to
`CudaMatrix::matmul`. Validation runs the same gradcheck suite on GPU —
[tools/colab_cuda_validate.sh](tools/colab_cuda_validate.sh). Phase B (resident
device memory instead of per-call round trips) is the next CUDA milestone.

## Paper-to-architecture: the arXiv fetcher

```bash
pip install requests
python papers/fetch.py 1706.03762                     # Attention Is All You Need
python papers/fetch.py 2302.13971 --emit-cpp llama.cpp --json llama.json
```

`papers/fetch.py` downloads a paper's **LaTeX source** from arXiv, sweeps prose and
model-size tables for the architecture hyperparameters, and emits a normalized
config — every extracted value carries an **evidence snippet** from the paper, and
anything it can't resolve is listed as unresolved rather than guessed. `--emit-cpp`
generates compilable microtorch code (GPT-2-family `GPT2Config` or, when it detects
RMSNorm/RoPE, a Llama-family `LlamaExportConfig`).

Validated live against: *Attention Is All You Need* (d_model=512, N=6, h=8,
d_ff=2048, sinusoidal), *LLaMA* (4096/32/32 from its model-size table +
RMSNorm/SwiGLU/RoPE), *TinyLlama* (22 layers, 32 heads, vocab 32000). Offline
fixture tests: `python papers/test_fetch.py` — no network needed in CI.

This is the constrained config-delta approach: most transformer papers are deltas
over a known skeleton, so extraction is pattern-matching with provenance, not
free-form code generation.

## Roadmap

- arXiv fetcher v2: per-variant instantiation, GQA/MoE fields, HF-config cross-check
- CUDA phase B: resident device tensors (params uploaded once, activations on-device)
- int4/NF4 quantization (QLoRA paper's datatype; int8 is the current base)
- Parallel scan for Mamba (training-speed parity with attention)
- Paper-to-architecture pipeline: pull a paper's LaTeX source from arXiv, extract the
  architecture config (dims, heads, norm type, activation), instantiate the model
- **Sparse attention research phase**: survey the current literature and attempt
  original variants — the long-horizon flagship goal

## License

MIT — see [LICENSE](LICENSE).
