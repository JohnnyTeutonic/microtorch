#!/usr/bin/env python3
"""Download Qwen model from HuggingFace and prepare for parity testing."""

import os
import sys
from pathlib import Path

def download_qwen():
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError:
        print("Installing huggingface_hub...")
        os.system(f"{sys.executable} -m pip install huggingface-hub -q")
        from huggingface_hub import hf_hub_download, snapshot_download

    hf_token = os.environ.get("HF_API_KEY")
    if not hf_token:
        print("ERROR: HF_API_KEY not set")
        return False

    # Try Qwen models, smallest first (faster training/testing)
    model_variants = [
        "Qwen/Qwen1.5-1.8B",      # Smallest, fastest
        "Qwen/Qwen-7B",
        "Qwen/Qwen1.5-7B",
        "Qwen/Qwen1.5-32B",
        "Qwen/Qwen2-7B",
        "meta-llama/Llama-2-7b",  # Fallback to Llama if Qwen fails
    ]

    model_id = None
    print(f"Searching for available model (authenticated: {bool(hf_token)})...")

    for variant in model_variants:
        try:
            from huggingface_hub import model_info
            print(f"  Checking {variant}...", end=" ", flush=True)
            info = model_info(variant, token=hf_token)
            model_id = variant
            print("[OK]")
            print(f"[OK] Found model: {variant}")
            break
        except Exception as e:
            print(f"[SKIP]")
            continue

    if not model_id:
        print(f"\nERROR: Could not find any model variant.")
        print(f"Tried: {', '.join(model_variants)}")
        return False

    model_dir = Path(__file__).parent.parent / "models" / "llama-test"
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nDownloading {model_id}...")
    print("This may take a few minutes depending on model size...")

    try:
        snapshot_download(
            model_id,
            cache_dir=str(model_dir),
            token=hf_token,
        )
        print(f"\n✓ Download complete: {model_dir}")

        # Find the safetensors and config files
        safetensors_file = None
        config_file = None
        for root, dirs, files in os.walk(model_dir):
            for f in files:
                if f == "model.safetensors":
                    safetensors_file = os.path.join(root, f)
                elif f == "config.json":
                    config_file = os.path.join(root, f)

        if safetensors_file and config_file:
            print(f"\n[OK] Found model.safetensors: {safetensors_file}")
            print(f"[OK] Found config.json: {config_file}")
            print(f"\nNext step:")
            print(f"  ./build_wsl/qwen_parity '{safetensors_file}' '{config_file}'")
            return True
        else:
            print(f"ERROR: Could not find model.safetensors or config.json in {model_dir}")
            return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False

if __name__ == "__main__":
    success = download_qwen()
    sys.exit(0 if success else 1)
