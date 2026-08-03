#!/usr/bin/env python3
"""The HF-export receipt: transformers must generate ARGMAX-IDENTICAL
continuations to `mtstudio sample --topk 1` on the same prompt.

    python tools/hf_export_verify.py HF_DIR OUT_DIR [--spec SPEC]
                                     [--prompt "..."] [--tokens N]

Same parity standard ember.cpp serving is held to: two independently
written stacks (this repo's tape vs transformers/torch) reading the
same weights must agree token-for-token under greedy decoding. Special
tokens are suppressed on the HF side because the paperkiln sampler bans
them (sampler hygiene at small vocab caps).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hf_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--spec", help="spec.json for mtstudio sample "
                                   "(default: <out_dir>/spec.json guess)")
    ap.add_argument("--prompt", default="once upon a time")
    ap.add_argument("--tokens", type=int, default=8)
    ap.add_argument("--mtstudio", default=os.path.expanduser("~/mtrel/mtstudio"))
    args = ap.parse_args()

    import torch
    from tokenizers import Tokenizer
    from transformers import AutoModelForCausalLM

    tok = Tokenizer.from_file(os.path.join(args.hf_dir, "tokenizer.json"))
    vocab = tok.get_vocab()
    inv = {i: t for t, i in vocab.items()}
    model = AutoModelForCausalLM.from_pretrained(args.hf_dir)
    model.eval()

    # Manual greedy loop with the SAME special-token ban the paperkiln
    # sampler applies — generate()'s eos/pad semantics stay out of it.
    ids = tok.encode(args.prompt).ids
    banned = {vocab[t] for t in ("<unk>", "<s>", "</s>", "<pad>") if t in vocab}
    words = []
    with torch.no_grad():
        for _ in range(args.tokens):
            logits = model(torch.tensor([ids])).logits[0, -1]
            for b in banned:
                logits[b] = -1e30
            nxt = int(torch.argmax(logits))
            ids.append(nxt)
            words.append(inv.get(nxt, "?"))
    hf_text = args.prompt + " " + " ".join(words)

    spec = args.spec or "/tmp/hf_verify_spec.json"
    cmd = [args.mtstudio, "sample", spec, "--prompt", args.prompt,
           "--tokens", str(args.tokens), "--topk", "1"]
    mt_text = subprocess.run(cmd, capture_output=True, text=True,
                             check=True).stdout.strip().splitlines()[-1]

    print(f"HF:       {hf_text}")
    print(f"mtstudio: {mt_text}")
    if hf_text.split() == mt_text.split():
        print(f"HF-PARITY-OK: {args.tokens} greedy tokens argmax-identical "
              "across transformers and the paperkiln tape")
        return 0
    print("HF-PARITY-MISMATCH")
    return 1


if __name__ == "__main__":
    sys.exit(main())
