"""Dump HF GPT-2's float32 logits for a fixed token sequence.

The output JSON is the reference side of tools/gpt2_parity.cpp (phase 1c).
Token ids are hard-coded: the parity test needs A sequence, not a tokenizer.
"""
import json
import sys

import torch
from transformers import GPT2LMHeadModel

OUT = sys.argv[1] if len(sys.argv) > 1 else "gpt2_reference.json"
TOKENS = [464, 3290, 318, 257, 922, 3290, 13, 632]   # 8 positions

model = GPT2LMHeadModel.from_pretrained(
    "openai-community/gpt2", torch_dtype=torch.float32)
model.eval()
with torch.no_grad():
    logits = model(torch.tensor([TOKENS])).logits[0]  # [T, 50257]

json.dump({"tokens": TOKENS, "rows": logits.shape[0], "cols": logits.shape[1],
           "logits": [float(x) for x in logits.reshape(-1)]}, open(OUT, "w"))
print("wrote %s  (%d x %d)" % (OUT, logits.shape[0], logits.shape[1]))
