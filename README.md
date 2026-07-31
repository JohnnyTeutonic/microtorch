# microtorch

**Paste an arXiv ID. Get a model you can chat with. Every hyperparameter
carries the snippet of the paper it came from.**

```
Paste an arXiv ID
       ↓
Architecture extracted        papers/fetch.py — every value carries its
       ↓                      evidence snippet from the paper's LaTeX;
Review the hyperparameters    anything unresolved is reported, never guessed
       ↓
Generate the model            nn::Llama / GPT-2 / Kimi / Mamba / SRD
       ↓
Train                         mtstudio run spec.json — one declarative file
       ↓                      drives arch, data, schedule, export, serving
Monitor in Studio             live loss curve + per-module gradient glow
       ↓                      in the browser, tailing the run
Export GGUF / safetensors     byte-exact, vocabulary embedded
       ↓
Launch inference              tinyllama.cpp serves it — a separate engine
       ↓
Share the results             a run is one spec file
```

An educational yet research-capable LLM framework that exposes every major
component of the modern transformer training stack in readable C++ — while
remaining compatible with Hugging Face checkpoints.

**~8.7K lines of core code. No CUDA required (T4-validated when you want it).
Builds in under two minutes.**

## What makes this different

Minimal autograd engines are a well-populated genre. Four things here are not:

1. **Papers in, running models out — with provenance.** `papers/fetch.py` pulls a
   paper's actual LaTeX source, extracts the architecture, and attaches an
   **evidence snippet to every extracted value**. Fields it cannot resolve are
   reported as unresolved rather than silently guessed. This is constrained
   config-delta extraction with citations, not free-form generation.
2. **Falsifiers ship inside the modules.** Novel mechanisms carry the experiment
   designed to kill them — `SurpriseRoutedAttention::shuffle_predictor` feeds the
   router a permuted input so the gate keeps its distribution but loses its
   information. When a result fails, [the negative gets
   published](SPARSE_ATTENTION.md), not buried.
3. **The loop actually closes, across two engines.** A model trained on this tape
   exports to byte-exact GGUF and produces coherent English inside
   `tinyllama.cpp` — a *separately written* inference engine. Both halves of the
   pipeline are in this stack.
4. **Verification is the gate, not the afterthought.** Every op is
   finite-difference checked in CI; GPT-2 logits match the HF reference
   end-to-end; safetensors round-trips are bit-identical; the GGUF writer's
   32-byte alignment is regression-tested because getting it wrong once turned a
   working model into word salad.

## Why it exists

PyTorch is 4M+ lines; understanding *why* your gradient is wrong means reading
dispatcher internals. microtorch takes the opposite bet: every operation is a
readable forward + hand-derived backward pair, every backward is verified against
finite differences, and the whole tape fits in
[one header](include/microtorch/autograd.hpp).

That makes it three things at once:

1. **A working training stack** — load a HF GPT-2 or Qwen checkpoint, fine-tune
   it, save it back as safetensors.
2. **A research vehicle** — novel mechanisms (linear attention, selective
   computation, state-space models) land here in days, not framework-release
   cycles.
3. **A reference implementation** — if you want to know what RoPE or a
   cross-entropy backward *actually does*, the answer is one file away, not forty.

## Quick start

```bash
git clone git@github.com:JohnnyTeutonic/microtorch.git
cd microtorch
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
ctest --output-on-failure     # gradchecks + unit + integration tests
```

## The studio — spec in, chatting model out

One JSON file describes the whole lifecycle; `mtstudio` executes it.

```bash
./mtstudio plan specs/tinystories-llama.json   # dry-run: resolve and print
./mtstudio run  specs/tinystories-llama.json   # train, eval, export, serve
./mtstudio serve /tmp/mtstudio_llama3k 8080    # live dashboard on a running job
```

That spec trains a Llama-family model (RMSNorm + RoPE + SwiGLU, tied embeddings)
on TinyStories with early stopping, checkpoint/resume and a JSONL event stream,
then exports safetensors **and** a GGUF with the vocabulary embedded. Feeding
that GGUF straight into tinyllama.cpp (the serve command it prints):

> **prompt:** once upon a time
> **model (300 training steps, d=128):** she was a little big he had very time
> tree the girl and day a little big time they were so play to the man and said
>
> **same architecture, 3,000 steps (val loss 4.64 → 3.68):** there was a little
> girl named timmy he had a big hug and said goodbye to the park with her mom
> and she started to play outside in the sky but it was too late that they were
> playing together all day

Three hundred steps buys the word distribution; three thousand buys syntax —
clause chains, narrative connectives, attempted coreference (pronoun drift and
all). Both runs are CPU-only.

### The dashboard — `studio/index.html`

A single self-contained HTML file, no build step and no dependencies. Drop an
`events.jsonl` on it, or point it at a live run with `mtstudio serve` and it
polls every two seconds while training proceeds.

- **Loss chart** — train curve, validation points, and (for SRD runs) the mean
  gate on its own 0–1 axis as a dashed overlay.
- **Gradient glow** — per-module gradient L2 norms, taken pre-clip, drawn as
  bars. Colour is not absolute magnitude: it is `log10(norm / that module's own
  running median)`, so a module is red when *it* is hot relative to its own
  history and navy when it is fading. Vanishing and exploding gradients become
  visible per-layer while the run is still going.
- **Run stats** — step, train loss, best val, grad norm, mean gate, early-stop
  trigger.
- **Spec builder** — a form that writes the very same `spec.json` that
  `mtstudio run` consumes, previews the JSON live, downloads it, and prints the
  exact run and serve commands. Preset dropdown covers `llama-tiny`,
  `gpt2-nano`, `gpt2-small`, `kimi-tiny`, `srd-tiny`.

### Custom configuration — the spec format

Every field below is optional and falls back to a sane default. Use
`arch.preset` for a known configuration, or `arch.custom` to dimension a model
yourself and pick the attention mechanism.

```jsonc
{
  "name": "tinystories-llama-3k",
  "arch": {
    "preset": "llama-tiny",           // or omit and use "custom" below
    "custom": {                       // overrides the preset field-by-field
      "d": 256, "layers": 4, "heads": 8,
      "attention": "srd"              // exact | kimi | srd
    }
  },
  "data": {
    "corpus":    "data/train.txt",    // raw text
    "vocab":     "releases/chat7b.gguf",  // vocabulary lifted from a GGUF
    "vocab_cap": 4096,                // truncate to the top-N tokens
    "T":         256                  // context length
  },
  "train": {
    "steps": 3000, "lr": 3e-3, "clip": 1.0,
    "accum": 2,                       // true gradient accumulation
    "eval_every": 100,
    "checkpoint_every": 100,          // resume picks up from here
    "gradmap_every": 10,              // per-module grad-norm event cadence
    "early_stopping": { "patience": 8, "min_delta": 0.003 }
  },
  "export": { "formats": ["safetensors", "gguf"] },
  "serve":  { "on_finish": true },    // print the tinyllama serve command
  "out_dir": "/tmp/mtstudio_llama3k"
}
```

The event stream (`out_dir/events.jsonl`) is the contract between trainer and
UI — `start`, `step` (loss, grad_norm, optional per-module `grads` and SRD
`gate`), `eval`, `early_stop`, `export`, `done`. Anything that can read JSONL
can consume a run.

## Feature matrix

| Component | Status | Notes |
|---|---|---|
| Reverse-mode autograd | ✅ | DAG tape, topological sort, `NoGrad` scope |
| Gradient verification | ✅ | Every op finite-difference checked in CI |
| Layers | ✅ | Linear, LayerNorm, RMSNorm, Embedding, Attention, MLP, Dropout |
| Full GPT-2 | ✅ | Logit-parity verified against HF checkpoint |
| Llama-family ops | ✅ | RMSNorm + RoPE + SwiGLU; Qwen 1.5-1.8B load verified |
| Optimizers | ✅ | SGD (momentum), AdamW |
| LR schedulers | ✅ | Cosine-with-warmup, StepLR |
| Grad clipping | ✅ | Global-norm (`clip_grad_norm`) |
| Checkpoint IO | ✅ | safetensors load **and** save (HF round-trip) |
| GGUF export | ✅ | `export_gguf_llama`: state_dict → .gguf for tinyllama.cpp |
| Cross-entropy loss | ✅ | Fused softmax backward |
| Python bindings | ✅ | pybind11, numpy interop (`-DMICROTORCH_BUILD_PYTHON=ON`) |
| **Run studio** | ✅ | Declarative spec → train/eval/export/serve; `plan` dry-run; resume |
| **Live dashboard** | ✅ | Loss + val + gate chart, per-module gradient glow, spec builder |
| LoRA | ✅ | `LoRALinear`: frozen base + rank-r adapters, `merged_weight()` |
| Quantization | ✅ | int8 blockwise (absmax/block), `QLinear` ~3.7x smaller weights |
| QLoRA | ✅ | `QLoRALinear`: quantized frozen base + trainable adapters |
| CUDA dispatch seam | ✅ | `device::matmul` → `cuda::matmul`; suites **pass on a T4** |
| **Kimi linear attention** | ✅ | O(n·d²) vs O(n²·d); drop-in `KimiLinearAttention` |
| **Cerebellum selective gating** | ✅ | Prediction-residual gating; skips compute on routine tokens |
| **Mamba / S4 state-space** | ✅ | Trainable through time: `ssm_scan` tape op, BPTT FD-gradchecked |
| **Surprise-routed density (SRD)** | 🧪 | Falsifier passed 5-6σ twice; gate concentrates on retrieval sites (5x replicated). Recall claim **failed replication and the negative is published** — [SPARSE_ATTENTION.md](SPARSE_ATTENTION.md) |
| GPU kernels | 🧪 | matmul dispatches to transformer_core CUDA; phase B = resident tensors |

## Train something (C++)

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

Standard attention pays O(n²·d) to build the full attention matrix. The
linear-attention family replaces softmax with a feature map φ(x) = elu(x)+1 and
reassociates the product, paying O(n·d²) instead:

```
out_i = φ(q_i)ᵀ · Σ_{j≤i} φ(k_j) v_jᵀ  /  φ(q_i)ᵀ · Σ_{j≤i} φ(k_j)
```

The cumulative sums preserve causality without a mask. `nn::KimiLinearAttention`
is interface-identical to `nn::CausalSelfAttention` — swap one line to try it.
Measured on CPU: ~8.9x faster forward+backward at seq=16, converging to ~1.1x as
n approaches d (the crossover regime the complexity analysis predicts).

### Cerebellum-inspired selective gating — [cerebellum.hpp](include/microtorch/cerebellum.hpp)

Motivated by cerebellar prediction-error filtering: a small predictor
(d → d/4 → d) learns what "routine" hidden states look like; the gate
`σ(‖x − predict(x)‖)` mixes the expensive layer's output with the identity path
per token. Surprising tokens get full compute, routine tokens skip it. Wraps any
layer via `std::function` — no changes to the wrapped code.

### Mamba / S4 state-space — [mamba.hpp](include/microtorch/mamba.hpp)

Discrete state-space recurrence `x[t+1] = A·x[t] + B·u[t]` as a drop-in sequence
backbone: O(1) memory per generated token instead of a growing KV cache.
Trainable through time — `ssm_scan` is a tape op with BPTT verified against
finite differences on all five inputs. The hardware-parallel scan is the next
milestone (see [DESIGN.md](DESIGN.md)).

### Surprise-routed density — [srd.hpp](include/microtorch/srd.hpp)

Research, and currently a partial negative. Per-query density routed by
prediction residual rather than attention-affinity scores: one shared qkv
projection feeds both an exact and a linear path, and the gate
`σ(scale·rms(x − predict(x)) + bias)` blends them per query. Training is soft and
fully differentiable; inference hardens the gate to `g > τ`, giving
O(ρn²d + nd²) for surprise rate ρ. `mean(gate())` is exposed so density can be
priced directly in the loss.

The falsifier (`shuffle_predictor`) permutes the predictor's view of the input,
preserving the gate's distribution while destroying its alignment. It passed at
5-6σ twice, and the gate demonstrably concentrates on retrieval-critical
positions across five runs. **The recall-performance claim did not replicate
across seeds and that failure is written up in full**, including what survived
and what the next pre-registered test is.

## Verification philosophy

Nothing merges without:

1. **Finite-difference gradcheck** — every op's backward vs central differences,
   with the measured error printed, not merely asserted.
2. **Parity tests** — GPT-2 logits match the HF reference end-to-end; Qwen
   tensors load and map onto modules by name with zero translation tables.
3. **Round-trip tests** — save → load → bit-identical.
4. **Falsifiers for research claims** — every novel mechanism ships with the
   experiment that would kill it, and the result is published either way.

CI runs the full suite (plus cppcheck, clang-format, Valgrind) on every push —
see [.github/workflows](.github/workflows). API documentation is generated by
Doxygen (`doxygen docs/Doxyfile` → docs/html/) and published to GitHub Pages by
the docs workflow.

## Paper-to-architecture: the arXiv fetcher

```bash
pip install requests
python papers/fetch.py 1706.03762                     # Attention Is All You Need
python papers/fetch.py 2302.13971 --emit-cpp llama.cpp --json llama.json
```

`papers/fetch.py` downloads a paper's **LaTeX source** from arXiv, sweeps prose
and model-size tables for the architecture hyperparameters, and emits a
normalized config — every extracted value carries an **evidence snippet** from
the paper, and anything it cannot resolve is listed as unresolved rather than
guessed. `--emit-cpp` generates compilable microtorch code (GPT-2-family
`GPT2Config` or, when it detects RMSNorm/RoPE, a Llama-family
`LlamaExportConfig`).

Validated live against: *Attention Is All You Need* (d_model=512, N=6, h=8,
d_ff=2048, sinusoidal), *LLaMA* (4096/32/32 from its model-size table +
RMSNorm/SwiGLU/RoPE), *TinyLlama* (22 layers, 32 heads, vocab 32000). Offline
fixture tests: `python papers/test_fetch.py` — no network needed in CI.

This is the constrained config-delta approach: most transformer papers are deltas
over a known skeleton, so extraction is pattern-matching with provenance, not
free-form code generation.

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
init (B=0), gradients touch only A/B, `merged_weight()` matches the adapter
forward to 1e-4, int8 round-trip error is bounded by half a quantization step.

## CUDA

The dispatch seam is live: every matmul routes through `device::matmul`
([device.hpp](include/microtorch/device.hpp)). Default builds are CPU-only and
bit-identical to before. `-DMICROTORCH_CUDA=ON` compiles transformer_core's
kernel tree and `device::set(Device::CUDA)` (or `MICROTORCH_DEVICE=cuda`)
dispatches to `cuda::matmul`. Validation runs the same gradcheck suite on GPU —
[tools/colab_cuda_validate.sh](tools/colab_cuda_validate.sh). Phase B (resident
device memory instead of per-call round trips) is the next CUDA milestone.

## Repository layout

```
include/microtorch/   public headers (one concern per header)
  autograd.hpp        the tape: Variable, backward(), NoGrad
  ops.hpp             op set — every op forward + audited backward
  nn.hpp              Module, layers, optimizers, schedulers
  llama.hpp           Llama-family: RMSNorm + RoPE + SwiGLU, HF-native names
  kimi_linear.hpp     phase 3a: linear attention
  cerebellum.hpp      phase 3b: selective gating
  mamba.hpp           phase 3c: state-space models
  srd.hpp             research: surprise-routed density + its falsifier
  gguf.hpp            GGUF export (alignment-audited)
  quant.hpp           int8 blockwise, LoRA, QLoRA
  safetensors.hpp     HF checkpoint load/save
src/                  implementations
tools/                mtstudio (the run driver), parity checkers, benchmarks
studio/               the dashboard — one self-contained HTML file
specs/                example run specs
papers/               arXiv → architecture fetcher + offline fixture tests
tests/                gradchecks + unit tests (all in CI)
python/               pybind11 bindings
docs/                 Doxygen config (make docs)
```

## Roadmap

- arXiv fetcher v2: per-variant instantiation, GQA/MoE fields, HF-config
  cross-check
- CUDA phase B: resident device tensors (params uploaded once, activations
  on-device)
- int4/NF4 quantization (QLoRA paper's datatype; int8 is the current base)
- Parallel scan for Mamba (training-speed parity with attention)
- Technique transfer from open-weight frontier reports — attention residuals,
  KDA, Muon optimizer ([TECH_TRANSFER.md](TECH_TRANSFER.md))
- **Sparse attention research phase**: survey the current literature and attempt
  original variants — the long-horizon flagship goal

## License

MIT — see [LICENSE](LICENSE).
