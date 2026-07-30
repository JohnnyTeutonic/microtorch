#!/usr/bin/env bash
# VM-side CUDA-seam validation (one-shot, T4). The contract: the SAME
# gradcheck suites that gate the CPU path must pass with device::matmul
# dispatching to transformer_core's CudaMatrix::matmul.
#   /content/cvd_src.zip  -> microtorch + transformer_cpp (incl. src/cuda)
# Writes /content/cvd.log and /content/CVD_DONE (PASS) or CVD_FAIL.
set -uo pipefail
cd /content
echo "=== cvd boot $(date) ==="
nvidia-smi -L || true
unzip -oq cvd_src.zip -d /content/src

mkdir -p /content/cvd_build && cd /content/cvd_build
cmake /content/src/microtorch -DCMAKE_BUILD_TYPE=Release \
      -DMICROTORCH_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 2>&1 | tail -4
make test_gradcheck test_nn test_lora_quant -j"$(nproc)" 2>&1 | tail -4 \
    || { echo "BUILD FAILED"; touch /content/CVD_FAIL; exit 1; }

fail=0
for t in test_gradcheck test_nn test_lora_quant; do
    echo "--- $t (MICROTORCH_DEVICE=cuda) ---"
    if MICROTORCH_DEVICE=cuda ./$t; then
        echo "$t: PASS"
    else
        echo "$t: FAIL"
        fail=1
    fi
done

if [ "$fail" -eq 0 ]; then
    echo "=== CVD PASS $(date) ==="
    touch /content/CVD_DONE
else
    echo "=== CVD FAIL $(date) ==="
    touch /content/CVD_FAIL
fi
