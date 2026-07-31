#!/usr/bin/env bash
# The SCREEN-RECORDING orchestrator (companion to RECORDING_GUIDE.md).
# Same pipeline as demo.sh, but paced for a human recording that toggles
# between this terminal and a browser: it starts the LIVE dashboard
# server before training begins, prints big SWITCH cues, and waits for
# you at each scene boundary so the recording never rushes you.
#
#   tools/demo_record.sh CORPUS.txt VOCAB.gguf [TINYLLAMA_BIN]
#
# Scenes (also the storyboard for the edit):
#   1  terminal   paper -> architecture with evidence
#   2  browser    the diff-to-paper page (file opens from Windows)
#   3  terminal   the spec + training starts
#   4  browser    http://localhost:8080 — loss curve + node-graph LIVE
#   5  terminal   Atlas row + chat with the model
set -e
CORPUS=${1:?usage: demo_record.sh CORPUS.txt VOCAB.gguf [TINYLLAMA_BIN]}
VOCAB=${2:?usage: demo_record.sh CORPUS.txt VOCAB.gguf [TINYLLAMA_BIN]}
TL=${3:-}
MT=$(command -v ./mtstudio || command -v build/mtstudio || echo "$HOME/mtrel/mtstudio")
OUT=/tmp/mtdemo_rec
STEPS=${DEMO_STEPS:-300}
cue()  { printf "\n\033[1;33m>>> %s\033[0m\n" "$*"; }
step() { printf "\n\033[1;36m== %s ==\033[0m\n" "$*"; }
pause() { read -rp $'\033[1;33m[enter to continue]\033[0m '; }
cleanup() { kill "${SERVE_PID:-}" 2>/dev/null || true; }
trap cleanup EXIT

step "SCENE 1 — paper -> architecture, every value with its evidence"
python3 papers/fetch.py 1706.03762 --json /tmp/mtdemo_arch.json \
    --emit-html "$OUT.paper.html" 2>/dev/null || true
cue "SCENE 2: open this file in your BROWSER now:"
echo "        file://wsl.localhost/Ubuntu$OUT.paper.html"
echo "        (or from Windows: \\\\wsl.localhost\\Ubuntu$OUT.paper.html)"
cue "hover a field — the evidence snippet highlights. Come back when done."
pause

step "SCENE 3 — one spec, then training starts"
rm -rf "$OUT"; mkdir -p "$OUT"
cat > /tmp/mtdemo_rec_spec.json <<EOF
{
  "name": "demo",
  "arch": {"preset": "llama-tiny"},
  "data": {"corpus": "$CORPUS", "vocab": "$VOCAB", "vocab_cap": 4096, "T": 128},
  "train": {"steps": $STEPS, "batch": 4, "eval_every": 25, "checkpoint_every": 100,
             "gradmap_every": 5},
  "export": {"formats": ["safetensors", "gguf"]},
  "out_dir": "$OUT"
}
EOF
"$MT" plan /tmp/mtdemo_rec_spec.json
"$MT" serve "$OUT" 8080 > /dev/null 2>&1 &
SERVE_PID=$!
cue "SCENE 4: open http://localhost:8080 in your browser."
cue "training starts on [enter]; SWITCH TO THE BROWSER as soon as it does —"
cue "the loss curve and the node-graph glow fill in live (polls every 2s)."
pause
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} OMP_WAIT_POLICY=PASSIVE
"$MT" run /tmp/mtdemo_rec_spec.json | grep --line-buffered -E '"eval"|"done"|"export"'

cue "SCENE 5: switch BACK TO THIS TERMINAL for the payoff."
pause
step "the run is now an Atlas data point"
python3 tools/atlas_extract.py "$OUT" | head -20

step "chat with it — a separately written engine reads the GGUF"
if [ -n "$TL" ] && [ -x "$TL" ]; then
  "$TL" "$OUT/demo.gguf" "$OUT/demo.gguf" 4 prompt "once upon a time" \
      --max-tokens 40 -ngl 0 --top-k 1 --raw-prompt 2>/dev/null | tail -2
fi
step "done — paper to chatting model, one stack, every claim receipted"
