#!/usr/bin/env bash
# VM-side SRD long-context runner (launched detached by srdlc_boot.py).
# Idempotent: unzip -o, resume from relayed checkpoint, skip finished work.
#
# Layout on the VM:
#   /content/srdlc_src.zip      microtorch + transformer_cpp sources
#   /content/chat7b.gguf        vocab source
#   /content/corpus.txt         TinyStories slice
#   /content/srd_ckpt/          live checkpoint dir (SRD_CKPT_DIR)
#   /content/srd_ckpt.zip       relayed snapshot (pull/upload unit)
#   /content/srd_lc.csv         loss curves
#   /content/srd_lc.log         this script's log
#   /content/SRDLC_DONE         completion marker
set -uo pipefail
cd /content

TOTAL_STEPS=300
CHUNK=10
T=256
D=128
CAP=4096

echo "=== srdlc boot $(date) ==="
if [ -f SRDLC_DONE ]; then echo "already done"; exit 0; fi

unzip -oq srdlc_src.zip -d /content/src
# Restore a relayed checkpoint from a previous VM, if the supervisor
# pushed one (upload puts it at /content/restore_srd_ckpt.zip).
mkdir -p /content/srd_ckpt
if [ -f restore_srd_ckpt.zip ] && [ ! -f /content/srd_ckpt/state.txt ]; then
    unzip -oq restore_srd_ckpt.zip -d /content/srd_ckpt
    echo "restored checkpoint: $(cat /content/srd_ckpt/state.txt 2>/dev/null)"
fi

# Build (CPU only; ~2-3 min, cached across boot retries on the same VM).
if [ ! -x /content/build/srd_parity ]; then
    mkdir -p /content/build && cd /content/build
    cmake /content/src/microtorch -DCMAKE_BUILD_TYPE=Release >/dev/null 2>&1
    make srd_parity -j"$(nproc)" 2>&1 | tail -2
    cd /content
fi
[ -x /content/build/srd_parity ] || { echo "BUILD FAILED"; exit 1; }

export SRD_CKPT_DIR=/content/srd_ckpt
done_steps=$(cat /content/srd_ckpt/state.txt 2>/dev/null || echo 0)
echo "starting at step ${done_steps}/${TOTAL_STEPS}"

target=$done_steps
while [ "$target" -lt "$TOTAL_STEPS" ]; do
    target=$((target + CHUNK))
    [ "$target" -gt "$TOTAL_STEPS" ] && target=$TOTAL_STEPS
    /content/build/srd_parity /content/chat7b.gguf /content/corpus.txt \
        "$target" "$T" "$D" /content/srd_lc.csv "$CAP" \
        || { echo "CHUNK FAILED at target $target"; exit 1; }
    # Atomic-ish snapshot for the relay: zip to temp, then rename.
    ( cd /content/srd_ckpt && zip -q ../srd_ckpt_new.zip ./* )
    mv -f /content/srd_ckpt_new.zip /content/srd_ckpt.zip
    echo "chunk done through $target / $TOTAL_STEPS ($(date +%H:%M:%S))"
done

touch /content/SRDLC_DONE
echo "=== SRDLC COMPLETE $(date) ==="
