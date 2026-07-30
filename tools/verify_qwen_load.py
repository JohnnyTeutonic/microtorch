#!/usr/bin/env python3
"""Verify Qwen model loads correctly from safetensors"""

import sys
import json
from pathlib import Path

def load_safetensors_header(filepath):
    """Read safetensors header to verify format"""
    with open(filepath, "rb") as f:
        # Read 8-byte LE length
        length_bytes = f.read(8)
        if len(length_bytes) < 8:
            return None
        length = int.from_bytes(length_bytes, "little")

        # Read header JSON
        header_bytes = f.read(length)
        try:
            header = json.loads(header_bytes.decode("utf-8"))
            return header
        except:
            return None

def verify_qwen_model(model_path, config_path):
    """Verify Qwen model structure"""
    model_path = Path(model_path)
    config_path = Path(config_path)

    if not model_path.exists():
        print(f"ERROR: model.safetensors not found at {model_path}")
        return False

    if not config_path.exists():
        print(f"ERROR: config.json not found at {config_path}")
        return False

    # Load config
    with open(config_path) as f:
        config = json.load(f)

    print("=== Qwen Model Verification ===")
    print(f"Config: {config_path.name}")
    print(f"Model: {model_path.name}")
    print()

    # Extract config details
    hidden_size = config.get("hidden_size", "?")
    num_layers = config.get("num_hidden_layers", "?")
    num_heads = config.get("num_attention_heads", "?")
    vocab_size = config.get("vocab_size", "?")

    print(f"[OK] Config loaded:")
    print(f"  hidden_size: {hidden_size}")
    print(f"  num_hidden_layers: {num_layers}")
    print(f"  num_attention_heads: {num_heads}")
    print(f"  vocab_size: {vocab_size}")
    print()

    # Verify safetensors format
    print(f"[OK] Loading model.safetensors...")
    header = load_safetensors_header(model_path)

    if header is None:
        print("ERROR: Failed to read safetensors header")
        return False

    # Count tensors and check architecture
    tensor_count = len(header)
    print(f"[OK] Safetensors header valid: {tensor_count} tensors")

    # Check for expected Qwen/Llama layers
    has_embed = any("embed" in k for k in header.keys())
    has_norm = any("norm" in k.lower() for k in header.keys())
    has_attn = any("attn" in k.lower() for k in header.keys())
    has_mlp = any("mlp" in k.lower() for k in header.keys())

    print(f"[OK] Architecture check:")
    print(f"  Embeddings: {'YES' if has_embed else 'NO'}")
    print(f"  Normalization: {'YES' if has_norm else 'NO'}")
    print(f"  Attention: {'YES' if has_attn else 'NO'}")
    print(f"  MLP layers: {'YES' if has_mlp else 'NO'}")
    print()

    if has_embed and has_norm and has_attn and has_mlp:
        print("[OK] Qwen model structure verified!")
        print()
        print(f"Phase 2c: Qwen 1.5-1.8B model ready for parity testing")
        print(f"Next step: Run qwen_parity tool in WSL to verify logit parity")
        print(f"  cd microtorch/build_wsl && make qwen_parity")
        print(f"  ./qwen_parity {model_path} {config_path}")
        return True
    else:
        print("[WARN] Model structure incomplete")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: verify_qwen_load.py <model.safetensors> <config.json>")
        sys.exit(1)

    success = verify_qwen_model(sys.argv[1], sys.argv[2])
    sys.exit(0 if success else 1)
