# Recording the demo on Windows — from zero

Two artifacts, two tools. The terminal-only GIF already exists
(`docs/media/demo.gif`, auto-recorded with asciinema — re-generate any time,
see the end). This guide is for the RICH version: a screen recording that
toggles between the terminal and the browser, showing the live dashboard.

## The one-line summary

Run `tools/demo_record.sh` in a WSL terminal, record the screen with the
**Windows Snipping Tool** (built in, nothing to install), and follow the
script's yellow `>>>` cues — it pauses at every scene change and waits for
Enter, so the recording can never run away from you.

---

## 1. Set the stage (5 minutes, once)

1. **Terminal**: open Windows Terminal → a WSL/Ubuntu tab. Make the font big
   enough to film: Ctrl+Plus a couple of times (14-16pt reads well at 1080p).
2. **Browser**: open a window with ONE empty tab, positioned so you can
   Alt+Tab between terminal and browser cleanly. Close everything personal —
   the recording captures whatever is on screen.
3. **Do a silent dry run first** (nobody records a demo in one take):

   ```bash
   cd /mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/microtorch
   DEMO_STEPS=100 bash tools/demo_record.sh \
     /mnt/c/.../transformer_cpp/data/tinystories-txt/train-0-small.txt \
     /mnt/c/.../transformer_cpp/releases/chat7b.gguf \
     ~/tlbuild/tinyllama
   ```

   (100 steps ≈ 1 minute of training; the real take can use 300 ≈ 5 min,
   which you'll speed up in the edit — see §4.)

## 2. Recording — Snipping Tool (recommended, no install)

Windows 11's Snipping Tool records a REGION of the screen, and captures
whatever appears in it — including you Alt+Tabbing between windows. That is
exactly the "toggle between Claude Code and my browser" requirement.

1. Press **Win + Shift + R** (or open *Snipping Tool* → the video-camera
   icon → New).
2. Drag a rectangle over the area both windows will occupy (easiest: make
   terminal and browser both full-screen and drag the whole screen).
3. Press **Start**. A 3-2-1 countdown runs, then it records.
4. Do the demo, Alt+Tabbing at the script's cues.
5. Click **Stop** (square button, top bar). The MP4 opens for preview —
   **Save As** somewhere sensible.

### Alternative A — record ONLY the browser tab (your "fix to one tab" case)

If training runs long and you want a segment that is just the dashboard:

1. Click the browser window so it is focused.
2. Press **Win + Alt + R** (Xbox Game Bar). It records the FOCUSED WINDOW
   only — Alt+Tab away and it keeps filming the browser, which is exactly
   what you want while training runs.
3. **Win + Alt + R** again to stop. Clips land in `Videos\Captures`.

Caveat: Game Bar refuses to record the Desktop/Explorer; it wants an app
window. It records one window per clip — for the toggling scenes use the
Snipping Tool method instead.

### Alternative B — OBS Studio (only if you want scene-switching polish)

Free, heavier, the streamer's tool: install from obsproject.com, add a
*Display Capture* source, press Record. Worth it only if you plan to record
often or want hotkey scene switches between terminal-only and browser-only
layouts. For a first recording, Snipping Tool is the right call.

## 3. The take itself

`tools/demo_record.sh` is the storyboard; its yellow `>>>` lines tell you
when to switch. The scene list:

| Scene | Where | What the viewer sees |
|---|---|---|
| 1 | terminal | arXiv fetch: architecture extracted with evidence per field |
| 2 | browser | the diff-to-paper page; hover a row, evidence highlights |
| 3 | terminal | the one-file spec; `plan` resolves it; training launches |
| 4 | browser | `localhost:8080` — loss curve descending, node-graph glowing, live |
| 5 | terminal | the Atlas row, then the model CHATTING through tinyllama.cpp |

The script waits for Enter at every boundary. Take your time; dead seconds
disappear in the edit.

## 4. The edit — Clipchamp (built into Windows 11)

1. Open **Clipchamp** (Start menu), *Create a new video*, drag the MP4 in.
2. Trim the fumbles: split (S key) at the cut points, delete the middles.
3. The long training stretch: split at its start and end, click the middle
   segment, and set **Speed** to 4x-8x — the loss curve visibly descending
   at 8x is exactly the effect you want.
4. Export 1080p. For a README-embeddable GIF of a short segment, upload the
   MP4 to ezgif.com/video-to-gif, or just link the MP4 (GitHub plays MP4 in
   issues/releases; the README keeps the asciinema GIF).

## 5. Regenerating the terminal GIF (no screen needed)

The lightweight README GIF is fully scripted — no human, no screen:

```bash
pip3 install --user --break-system-packages asciinema   # once
curl -sL -o ~/.local/bin/agg \
  https://github.com/asciinema/agg/releases/latest/download/agg-x86_64-unknown-linux-gnu
chmod +x ~/.local/bin/agg                                # once

DEMO_STEPS=120 DEMO_VERBOSE=1 python3 -m asciinema rec /tmp/demo.cast \
  -c "bash tools/demo.sh <corpus> <vocab> <tinyllama>" --overwrite -q
# (if recorded without a tty, patch the header size — see
#  tools/demo.sh history — then:)
~/.local/bin/agg /tmp/demo.cast docs/media/demo.gif \
  --idle-time-limit 1.5 --speed 2 --font-size 12 --fps-cap 10 --theme monokai
```

Current README GIF: 120 steps, every-5th-step heartbeat, 1.5 MB.
