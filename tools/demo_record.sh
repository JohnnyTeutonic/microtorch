#!/usr/bin/env bash
# The SCREEN-RECORDING orchestrator (companion to RECORDING_GUIDE.md).
# Everything rides ONE localhost server — the diff-to-paper page AND the
# live dashboard — so there are no file:// URLs to fight from Windows,
# and the browser tabs are auto-opened for you via Windows interop.
# Training runs in the BACKGROUND while the dashboard fills in live; the
# terminal streams the milestones meanwhile.
#
#   tools/demo_record.sh CORPUS.txt VOCAB.gguf [TINYLLAMA_BIN]
#
# Scenes (the storyboard for the edit):
#   1  terminal   paper -> architecture with evidence
#   2  browser    http://localhost:8080/paper.html  (auto-opened)
#   3  terminal   the spec; training LAUNCHES in the background
#   4  browser    http://localhost:8080/            (auto-opened) — loss
#                 curve + node-graph filling in LIVE while you watch
#   5  terminal   Atlas row + chat with the model
#
# VSCode single-window variant: instead of an external browser, Ctrl+
# Shift+P -> "Simple Browser: Show" -> paste the URL. The whole demo then
# lives inside one VSCode window (terminal pane + browser tab), which
# Game Bar (Win+Alt+R) records as a single app. See RECORDING_GUIDE.md.
set -e
cd "$(dirname "$0")/.."   # repo root, wherever the script is launched from

# Arguments optional: defaults point at the standard sibling-repo layout
# (override by passing paths, or via DEMO_CORPUS / DEMO_VOCAB / DEMO_TL).
_SIB=$(cd .. 2>/dev/null && pwd)/transformer_cpp
CORPUS=${1:-${DEMO_CORPUS:-$_SIB/data/tinystories-txt/train-0-small.txt}}
VOCAB=${2:-${DEMO_VOCAB:-$_SIB/releases/chat7b.gguf}}
TL=${3:-${DEMO_TL:-$HOME/tlbuild/tinyllama}}
[ -f "$CORPUS" ] || { echo "corpus not found: $CORPUS"; \
  echo "usage: demo_record.sh [CORPUS.txt] [VOCAB.gguf] [TINYLLAMA_BIN]"; exit 1; }
[ -f "$VOCAB" ] || { echo "vocab gguf not found: $VOCAB"; \
  echo "usage: demo_record.sh [CORPUS.txt] [VOCAB.gguf] [TINYLLAMA_BIN]"; exit 1; }
MT=$(command -v ./mtstudio || command -v build/mtstudio || echo "$HOME/mtrel/mtstudio")
OUT=/tmp/mtdemo_rec
STEPS=${DEMO_STEPS:-300}
cue()  { printf "\n\033[1;33m>>> %s\033[0m\n" "$*"; }
step() { printf "\n\033[1;36m== %s ==\033[0m\n" "$*"; }
pause() { read -rp $'\033[1;33m[enter to continue]\033[0m '; }
# Inside VSCode's integrated terminal (TERM_PROGRAM=vscode) nothing is
# spawned externally — the whole demo lives in one Simple Browser tab.
# Elsewhere, URLs auto-open in the Windows default browser via interop.
IN_VSCODE=""
[ "${TERM_PROGRAM:-}" = "vscode" ] && IN_VSCODE=1
openurl() {
  if [ -n "$IN_VSCODE" ]; then
    return 0  # the one-tab instruction is printed once, up front
  fi
  (cd /mnt/c && /mnt/c/Windows/System32/cmd.exe /c start "$1") \
    > /dev/null 2>&1 || cue "open manually: $1"
}
cleanup() { kill "${SERVE_PID:-}" "${RUN_PID:-}" 2>/dev/null || true; }
trap cleanup EXIT

rm -rf "$OUT"; mkdir -p "$OUT"

step "SCENE 1 — paper -> architecture, every value with its evidence"
python3 papers/fetch.py 1706.03762 --json /tmp/mtdemo_arch.json \
    --emit-html "$OUT/paper.html" 2>/dev/null || true

# The one-tab page: paper view and live dashboard behind an in-page
# toggle, so a VSCode Simple Browser tab never needs a companion.
cat > "$OUT/demo.html" <<'HTML'
<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<title>microtorch demo</title><style>
  * { box-sizing:border-box; margin:0; }
  body { background:#0d1117; height:100vh; display:flex; flex-direction:column;
         font:13px ui-monospace,Consolas,monospace; }
  nav { padding:.45rem .8rem; border-bottom:1px solid #30363d; }
  button { background:#161b22; color:#c9d1d9; border:1px solid #30363d;
           border-radius:6px; padding:.3rem .9rem; font:inherit; cursor:pointer;
           margin-right:.5rem; }
  button.on { background:#1f6feb; border-color:#1f6feb; color:#fff; }
  iframe { flex:1; border:0; width:100%; }
</style></head><body>
<nav>
  <button id="bp" class="on">paper &rarr; architecture</button>
  <button id="bd">live dashboard</button>
</nav>
<iframe id="fr" src="/paper.html"></iframe>
<script>
  const fr = document.getElementById('fr');
  const bp = document.getElementById('bp'), bd = document.getElementById('bd');
  bp.onclick = () => { fr.src = '/paper.html'; bp.classList.add('on'); bd.classList.remove('on'); };
  bd.onclick = () => { fr.src = '/';           bd.classList.add('on'); bp.classList.remove('on'); };
</script></body></html>
HTML

"$MT" serve "$OUT" 8080 > /dev/null 2>&1 &
SERVE_PID=$!
sleep 1
if [ -n "$IN_VSCODE" ]; then
  cue "ONE TAB, INSIDE VSCODE — do this once, now:"
  cue "  Ctrl+Shift+P  ->  'Simple Browser: Show'  ->  paste:"
  echo ""
  echo "        http://localhost:8080/demo.html"
  echo ""
  cue "Drag that tab into a split beside this terminal. The [paper] and"
  cue "[dashboard] buttons at its top switch views — no app switching ever."
fi
cue "SCENE 2: the diff-to-paper page (hover a field, evidence lights up)"
openurl "http://localhost:8080/demo.html"
pause

step "SCENE 3 — one spec describes the lifecycle; training launches NOW"
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
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} OMP_WAIT_POLICY=PASSIVE
"$MT" run /tmp/mtdemo_rec_spec.json > /tmp/mtdemo_rec_run.log 2>&1 &
RUN_PID=$!
if [ -n "$IN_VSCODE" ]; then
  cue "SCENE 4: training is RUNNING — click [live dashboard] in the demo tab."
else
  cue "SCENE 4: training is RUNNING — the dashboard is opening now."
  openurl "http://localhost:8080/"
fi
cue "Watch the loss curve descend and the node-graph glow (updates every 2s)."
echo "     (milestones stream here meanwhile:)"
tail --pid="$RUN_PID" -n +1 -f /tmp/mtdemo_rec_run.log \
  | grep --line-buffered -E '"eval"|"done"|"export"' || true
wait "$RUN_PID" 2>/dev/null || true

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
