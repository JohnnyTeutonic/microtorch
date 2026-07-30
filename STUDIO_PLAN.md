# microtorch Studio — Master Plan

*2026-07-30. The competitive thesis, the build order, and tonight's first moves.*

## 1. Thesis

Feature-for-feature saturation against PyTorch/HF is a losing race. The winning
race is the one ComfyUI proved (~80K stars wrapping an engine that already
existed): make the **pipeline** visible, composable, and clickable. Every UI
player today (Ollama, LM Studio, ComfyUI) wraps **inference only**. Nobody has
built the ComfyUI of **training**.

microtorch Studio: compose an architecture → pick a corpus → **Train** → watch
the loss live → **Chat with the model you just made**. One box. We already own
every stage: autograd trainer, LoRA/QLoRA fine-tuning, int8 quantization
(Q4/Q8 enums already in the GGUF writer), byte-exact safetensors/GGUF export,
and tinyllama.cpp's HTTP server (server.cpp + www/) for the inference link.

Two hooks no incumbent can copy quickly:
1. **Research mechanisms as dropdown architectures** — Kimi linear, Mamba
   (BPTT-trainable as of 2026-07-30), SRD sparse attention. "Train with
   attention variants that exist nowhere else."
2. **Paste an arXiv ID into the architecture box** — papers/fetch.py populates
   the config from the paper's LaTeX. A launch-demo moment nobody has.

Positioning line: **the glass-box ML studio** — every block in the UI links to
its readable source and the gradcheck that verifies it. Smallness becomes the
feature.

## 2. Inventory (what exists, with receipts)

| Stage | Status | Receipt |
|---|---|---|
| Train (pretrain) | ✅ | GPT-2 class, optimizers, schedulers, clip, dropout; suites 9/9 |
| Fine-tune | ✅ | LoRALinear/QLoRALinear, adapter-only checkpoints; test_lora_quant |
| Quantize | ✅ int8 (int4/NF4 roadmap) | quantize_int8, ~3.7x, half-step error bound |
| Export | ✅ | safetensors + GGUF; chat7b round-trip byte-identical |
| Serve | ✅ (via tinyllama.cpp) | generation verified from our re-export |
| HF import | ✅ | GPT-2 logit parity; Qwen load verified |
| Paper→config | ✅ | papers/fetch.py, 3 papers validated |
| Config precedent | ✅ | transformer_cpp config/*.json + config/architectures/ presets (llama/vanilla/custom) |

## 3. Gaps found in tonight's audit

1. **Early stopping is config-only in transformer_cpp.** `early_stopping_patience/
   threshold` exist in config.hpp:93-94 and transformer_config.json, but only
   hyperparameter_tuner.cpp reads them — the main training loop never honors
   patience. Fix: patience tracking around the validation-loss checkpoint in the
   trainer loop; microtorch's M1 driver gets early stopping natively from day one.
2. **The microtorch-TRAINED → serve leg is untested.** gguf_roundtrip proved
   trained-transformer_cpp weights survive our exporter byte-exactly and
   generate in tinyllama.cpp. Nobody has yet trained a model IN microtorch and
   chatted with it. This is M1's acceptance test, not a side item.
3. Studio-relevant misc: vocab/tokenizer story for from-scratch training (reuse
   the word-level GGUF vocab path as default; BPE later).

## 4. M1 — the run-spec CLI (the studio's engine; UI comes later)

One JSON file describes the whole lifecycle; one driver executes it.

```jsonc
// spec.json (draft schema v0)
{
  "name": "my-tinystories-gpt",
  "arch": {                        // exactly one of:
    "preset": "gpt2-small",        //  gpt2-{nano,small}, llama-tiny, kimi-tiny,
                                   //  mamba-tiny, srd-tiny ...
    "arxiv": "2302.13971",         //  or: populate from a paper
    "custom": {"d": 256, "layers": 4, "heads": 8, "vocab": 8192,
                "attention": "exact|kimi|srd", "norm": "layernorm|rmsnorm"}
  },
  "data": {"corpus": "tinystories|wikitext|/path/to.txt",
            "vocab": "chat7b|/path/to.gguf", "vocab_cap": 8192, "T": 256},
  "train": {"steps": 3000, "lr": 3e-4, "schedule": "cosine", "warmup": 100,
             "clip": 1.0, "early_stopping": {"patience": 5, "eval_every": 200},
             "checkpoint_every": 250},
  "finetune": {"base": "path/to/ckpt.safetensors", "lora_rank": 8,
                "quant_base_bits": 8},          // optional
  "export": {"formats": ["gguf", "safetensors"], "quant_bits": 32},
  "serve": {"on_finish": true, "engine": "tinyllama", "port": 8080}
}
```

Driver: `mtstudio run spec.json` (new tools/mtstudio.cpp) —
- validates the spec, prints the resolved plan, then executes stage by stage;
- training emits a JSONL event stream (`step`, `loss`, `val_loss`, `lr`,
  `gate_mean` for SRD) to stdout AND a file — this stream IS the UI's data feed
  later, so it's designed now;
- checkpoint/resume built in (the srd_parity pattern, already proven through
  VM reclaims);
- early stopping honored natively;
- on finish: export per spec, then (serve.on_finish) launch tinyllama.cpp on
  the exported GGUF and print the chat URL.

Acceptance test (closes gap 2): spec with gpt2-nano on TinyStories →
`mtstudio run` → generation from tinyllama.cpp is coherent. That transcript
goes in the README.

## 5. M2 — the UI veneer

A single-page local web app (no build system if avoidable; one HTML file the
driver can serve). It has NO logic of its own: dropdowns write a spec, the run
panel streams the JSONL events, Chat links to the tinyllama URL.

Pages/panels:
- **Build**: architecture picker (presets incl. research mechanisms | custom
  dims | arXiv ID box), corpus dropdown, hyperparams with sane defaults.
  Absorb transformer_cpp's config/architectures/*.json as importable presets.
- **Train**: live loss curve (canvas + the JSONL stream), gate histogram when
  SRD is selected, stop/early-stop status, checkpoint list.
- **Finetune**: base-checkpoint picker, LoRA rank, dataset upload (labelled
  text), quant toggle.
- **Export/Serve**: format checkboxes, quant bits, "Open chat" button.
- Every block links to source + its gradcheck (glass-box).

## 6. M3 — distribution

- `pip install microtorch` (pybind11 wheel; cibuildwheel in CI).
- README GIFs of the full loop; 90-second demo video (paste arXiv ID → train →
  chat is the money shot).
- Recipe sharing: specs are single files; a specs/ gallery in-repo.
- Launch surface is Jonathan's call (no conference constraint applies to
  software releases; HN Show-HN is the norm for this genre).

## 7. Sequencing

Tonight / tomorrow (light work day):
1. M1 skeleton: spec parser + resolved-plan printer + train stage for presets
   (gpt2-nano first), JSONL event stream, early stopping.
2. Acceptance test end-to-end (gap 2) with a short run.
3. transformer_cpp early-stopping fix (gap 1) — small, self-contained.

Next (more time arriving after the offer lands):
4. Fine-tune + quant + export stages in the driver; arXiv + custom arch paths.
5. M2 single-file UI over the event stream.
6. M3 wheels + demo assets.

Parallel research track (unchanged, SPARSE_ATTENTION.md): needle amendment
rerun (T=128, longer budget), multi-seed, then the paper-spine conversation.

## 8. Open decisions (deliberately deferred)

- BPE tokenizer training in-box vs word-level default (start word-level).
- UI framework: leaning single-file HTML+JS served by the driver; revisit only
  if the canvas plotting gets painful.
- Whether mtstudio absorbs transformer_cpp's trainer as a backend for big runs
  or stays pure-microtorch (start pure; the config-preset import keeps the
  bridge open).
