#!/bin/bash
# Verify Qwen 1.5-1.8B logit parity
# Run this once download_qwen.py completes

set -e

MODEL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/models/llama-test/models--Qwen--Qwen1.5-1.8B/snapshots"

# Find the snapshot directory (should be only one)
SNAPSHOT=$(find "$MODEL_DIR" -maxdepth 1 -type d | head -1)

if [ -z "$SNAPSHOT" ] || [ ! -f "$SNAPSHOT/config.json" ]; then
    echo "ERROR: Qwen model not found. Run download_qwen.py first."
    exit 1
fi

SAFETENSORS="$SNAPSHOT/model.safetensors"
CONFIG="$SNAPSHOT/config.json"

if [ ! -f "$SAFETENSORS" ]; then
    echo "ERROR: model.safetensors not found at $SAFETENSORS"
    echo "Download may still be in progress. Check:"
    echo "  find '$MODEL_DIR' -name '*.safetensors'"
    exit 1
fi

echo "=== Qwen 1.5-1.8B Parity Verification ==="
echo "Model: $SNAPSHOT"
echo "Safetensors: $SAFETENSORS"
echo "Config: $CONFIG"
echo ""

# Run qwen_parity tool
BUILD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/build_wsl"
QWEN_PARITY="$BUILD_DIR/qwen_parity"

if [ ! -f "$QWEN_PARITY" ]; then
    echo "ERROR: qwen_parity tool not found at $QWEN_PARITY"
    echo "Build with: cd build_wsl && make qwen_parity"
    exit 1
fi

echo "Running qwen_parity..."
"$QWEN_PARITY" "$SAFETENSORS" "$CONFIG"

echo ""
echo "✓ Verification complete!"
