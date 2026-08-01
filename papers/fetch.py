#!/usr/bin/env python3
"""arXiv -> microtorch architecture fetcher.

    python papers/fetch.py 1706.03762                # print arch summary
    python papers/fetch.py 2302.13971 --json arch.json --emit-cpp model.cpp

Downloads the paper's LaTeX source (arxiv.org/e-print/<id>), sweeps the
detexed text and tabular rows for architecture hyperparameters, and emits a
normalized config plus (optionally) compilable microtorch C++.

This is the CONSTRAINED config-delta approach: most transformer papers are
deltas over a known skeleton (dims, depth, heads, norm flavor, activation,
position encoding), so extraction is keyword-driven pattern matching with
per-field evidence strings -- not free-form code generation. Fields the
sweep cannot resolve are reported as unresolved, never guessed silently.

Requires: requests (pip install requests). Nothing else.
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
import tarfile
from dataclasses import dataclass, field

UA = "microtorch-paper-fetcher/0.1 (+https://github.com/JohnnyTeutonic/microtorch)"


# --------------------------------------------------------------------------
# fetch + detex
# --------------------------------------------------------------------------

def fetch_source(arxiv_id: str) -> str:
    """Return the concatenated .tex source of a paper."""
    import gzip

    import requests

    url = f"https://arxiv.org/e-print/{arxiv_id}"
    r = requests.get(url, headers={"User-Agent": UA}, timeout=60)
    r.raise_for_status()
    blob = r.content

    # e-print is either a tar.gz of sources or a single gzipped .tex.
    texts: list[str] = []
    try:
        with tarfile.open(fileobj=io.BytesIO(blob), mode="r:*") as tf:
            for m in tf.getmembers():
                if m.name.endswith(".tex"):
                    f = tf.extractfile(m)
                    if f:
                        texts.append(f.read().decode("utf-8", errors="replace"))
    except tarfile.ReadError:
        try:
            texts.append(gzip.decompress(blob).decode("utf-8", errors="replace"))
        except OSError:
            texts.append(blob.decode("utf-8", errors="replace"))
    if not texts:
        raise RuntimeError(f"no .tex files found in e-print for {arxiv_id}")
    return "\n".join(texts)


def detex(tex: str) -> str:
    """Light cleanup: drop comments, collapse whitespace, keep math text."""
    tex = re.sub(r"(?<!\\)%.*", "", tex)              # comments
    tex = re.sub(r"\\(text|mathrm|mathit|mathbf|textbf|textit|emph)\{([^{}]*)\}",
                 r"\2", tex)                            # unwrap simple macros
    tex = tex.replace("~", " ").replace("$", "")        # inline-math fences
    tex = tex.replace("\\(", "").replace("\\)", "")
    return re.sub(r"[ \t]+", " ", tex)


# --------------------------------------------------------------------------
# extraction
# --------------------------------------------------------------------------

@dataclass
class Finding:
    value: int | str
    evidence: str
    # Flavor fields only (norm/activation/positional): contribution-vs-
    # mention scoring. verdict: "used" (clear winner), "contested"
    # (runner-up too close — report both, never auto-apply). Numeric
    # fields keep verdict None.
    verdict: str | None = None
    score: float | None = None
    runner_up: dict | None = None


@dataclass
class Arch:
    arxiv_id: str
    title: str | None = None
    fields: dict[str, Finding] = field(default_factory=dict)
    unresolved: list[str] = field(default_factory=list)
    variants: list[dict[str, int]] = field(default_factory=list)
    # Flavor values seen in the text but judged mention-only (baselines,
    # related work): reported for transparency, never applied.
    mentions: dict[str, list] = field(default_factory=dict)


def _num(s: str) -> int:
    s = s.replace(",", "").lower()
    if s.endswith("k"):
        return int(float(s[:-1]) * 1000)
    return int(float(s))


# Per-field prose/table patterns, strongest first. Each entry: regex with
# ONE capturing number group. Table rows ("Layers & 32") are matched by the
# generic `& value` alternates.
NUM = r"([\d][\d,\.]*k?)"
PATTERNS: dict[str, list[str]] = {
    "d_model": [
        rf"d_?\{{?\\?(?:text|mathrm)?\{{?model\}}?\}}?\s*(?:=|of|:|&)\s*\$?{NUM}",
        rf"d_\{{model\}}\s*=\s*{NUM}",
        rf"(?:model|hidden|embedding)[ -](?:dimension(?:ality)?|size|width)\s*(?:of|=|:|is|&)?\s*\$?{NUM}",
        rf"dimension\s*\(?d_?(?:model)?\)?\s*(?:of|=|:|&)\s*{NUM}",
        # ALiBi phrasing: "16 transformer layers of dimension 1024"
        rf"layers\s+of\s+dimension\s+{NUM}",
        rf"\bdim(?:ension)?\s*&\s*{NUM}",
        rf"\bhidden\s*&\s*{NUM}",
    ],
    "n_layers": [
        rf"N\s*=\s*{NUM}\s+identical\s+layers",
        rf"stack\s+of\s+\$?N\s*=\s*\$?{NUM}",
        rf"{NUM}\s+(?:identical\s+|transformer\s+|decoder\s+|hidden\s+)?layers\b",
        rf"(?:n[ _]?layers?|\#?\s*layers?|depth)\s*(?:=|of|:|&)\s*\$?{NUM}",
        # Primer phrasing: "(d_{model}=768, d_{ff}=3072, L=12)" — capital
        # L after a delimiter, so a lone variable L elsewhere can't match.
        rf"[\(,]\s*L\s*=\s*{NUM}\)?",
        rf"\blayers?\s*&\s*{NUM}",
    ],
    "n_heads": [
        rf"h\s*=\s*{NUM}\s+(?:parallel\s+)?(?:attention\s+)?(?:heads|layers)",
        rf"{NUM}\s+(?:parallel\s+)?attention\s+heads",
        rf"(?:n[ _]?heads?|\#?\s*heads?|attention\s+heads?)\s*(?:=|of|:|&)\s*\$?{NUM}",
        rf"\bn?\s*heads?\s*&\s*{NUM}",
    ],
    "n_kv_heads": [
        rf"{NUM}\s+(?:key[- /]value|KV)\s+heads",
        rf"(?:n[ _]?kv[ _]?heads?|kv\s+heads?)\s*(?:=|of|:|&)\s*{NUM}",
    ],
    "d_ff": [
        rf"d_?\{{?\\?(?:text|mathrm)?\{{?ff\}}?\}}?\s*(?:=|of|:|&)\s*\$?{NUM}",
        rf"(?:feed[- ]?forward|ffn|inner|intermediate)[ -](?:dimension(?:ality)?|size|layer)\s*(?:of|=|:|is|&)?\s*\$?{NUM}",
        rf"\bffn?\s*(?:dim|size)?\s*&\s*{NUM}",
    ],
    "vocab_size": [
        rf"vocab(?:ulary)?\s*(?:size)?\s*(?:of|=|:|is|&)?\s*\$?{NUM}",
        rf"{NUM}\s*(?:BPE|word[- ]?piece|sentence[- ]?piece)?\s*(?:token\s+)?vocabulary",
    ],
    "context_length": [
        rf"context\s+(?:length|window|size)\s*(?:of|=|:|is|&)?\s*\$?{NUM}",
        rf"sequence\s+length\s*(?:of|=|:|is|&)?\s*\$?{NUM}",
        rf"{NUM}[- ]token\s+context",
    ],
}

FLAVOR = {
    "norm": [("rmsnorm", r"RMS[-\s]?[Nn]orm"),
             ("layernorm", r"[Ll]ayer[-\s]?[Nn]orm")],
    "activation": [("swiglu", r"SwiGLU"), ("geglu", r"GeGLU"),
                   ("gelu", r"GELU"), ("silu", r"SiLU|swish"),
                   ("relu", r"ReLU")],
    "positional": [("rope", r"RoPE|[Rr]otary\s+(?:position|embedding)"),
                   ("alibi", r"ALiBi"),
                   ("sinusoidal", r"sinusoid"),
                   ("learned", r"learned\s+position")],
}

# ---- contribution-vs-mention scoring for the flavor fields -------------
# A paper MENTIONS many alternatives (baselines, related work, "such as"
# lists); it USES one. First-match extraction confused the two (observed:
# Primer's activation is squared ReLU, but SwiGLU matched from a
# related-work sentence; ALiBi's own paper matched the rotary BASELINE it
# compares against). Every candidate match is scored by explainable cues
# in its context window plus the section it sits in; the field's verdict
# is "used" only when a clear winner exists. Weights are validated by
# papers/flavor_bench.py (AUROC over ground-truth-labeled real papers).
USE_CUES = [
    (r"\bwe\s+(?:use|used|adopt|employ|apply|train|choose|chose|opt)\b", 3.0),
    (r"\b(?:add|adds|adding|incorporate[sd]?|swap(?:ped)?\s+in)\b", 2.0),
    (r"\b(?:uses|using|adopts|employs|equipped\s+with|is\s+applied|are\s+applied)\b", 2.0),
    (r"\bwe\s+(?:introduce|propose)\b", 3.0),
    (r"\bfollowed\s+by\b", 2.0),
    (r"\benhancements?\b|\bmodifications?\b|\bour\s+(?:model|architecture|implementation)\b",
     2.0),
]
MENTION_CUES = [
    (r"\bsuch\s+as\b|\be\.g\.|\bfor\s+(?:example|instance)\b", -3.0),
    (r"\bcompared?\s+(?:to|with|against)\b|\bbaselines?\b|\balternatives?\b", -3.0),
    (r"\binstead\s+of\b|\bunlike\b|\brather\s+than\b", -2.0),
    (r"\bprior\s+work\b|\bprevious\s+work\b|\bexisting\b|\brelated\s+work\b", -2.0),
    (r"\bimproves?\s+(?:over|upon|on)\b|\boutperform", -2.5),
    (r"\bover\s+the\b", -1.5),  # "a lead in perplexity over the sinusoidal model"
    (r"\bfuture\s+work\b|\bcould\b|\bmight\b", -1.0),
    # Ablation phrasing: tried-and-compared is not the architecture.
    (r"\bexperimented\s+with\b|\bwe\s+also\s+tr(?:ied|y)\b|\bablations?\b|"
     r"\bnearly\s+identical\b|\bno\s+significant\b|\bvariants?\b", -2.5),
]
# Attribution cues that read as related-work ONLY when no usage verb is
# nearby: "we use rotary embeddings introduced by [cite]" is a usage
# statement with attribution, not a mention.
WEAK_MENTION_CUES = [
    (r"\bproposed\s+by\b|\bintroduced\s+by\b|\bsuggested\b", -1.0),
]
SECTION_WEIGHTS = {"method": 1.5, "abstract": 1.0, "other": 0.0, "related": -3.0}
WIN_THRESHOLD = 1.5   # winner must clear this to be "used"
WIN_MARGIN = 1.5      # ... and beat the runner-up by this, else "contested"

SEC_RE = re.compile(r"\\(?:section|subsection|chapter)\*?\s*\{([^{}]*)\}")


def classify_section(title: str) -> str:
    t = title.lower()
    if re.search(r"related|background|prior|previous\s+work|comparison", t):
        return "related"
    if re.search(r"method|model|architecture|approach|setup|training|implementation|design",
                 t):
        return "method"
    return "other"


def split_sections(tex: str) -> list[tuple[str, str]]:
    """[(section_class, detexed_text)] — text before the first \\section
    (title/abstract territory) is classed 'abstract'."""
    parts = []
    last, cls = 0, "abstract"
    for m in SEC_RE.finditer(tex):
        parts.append((cls, tex[last:m.start()]))
        cls, last = classify_section(m.group(1)), m.end()
    parts.append((cls, tex[last:]))
    return [(c, detex(t)) for c, t in parts if t.strip()]


def score_match(seg_text: str, m: re.Match, sec_class: str) -> tuple[float, list[str]]:
    """Score one candidate occurrence; returns (score, cue names) so every
    number is explainable."""
    before = seg_text[max(0, m.start() - 120):m.start()]
    after = seg_text[m.end():m.end() + 80]
    window = before + " " + after
    score = SECTION_WEIGHTS[sec_class]
    cues = [f"section:{sec_class}"]
    used_cue = False
    for pat, w in USE_CUES:
        if re.search(pat, window, flags=re.IGNORECASE):
            score += w
            used_cue = True
            cues.append(pat[:24])
    for pat, w in MENTION_CUES:
        if re.search(pat, window, flags=re.IGNORECASE):
            score += w
            cues.append(pat[:24])
    # Config/table ROW only (tight window): "norm & layernorm". A '&'
    # somewhere else in a 200-char window is just tabular noise.
    if "&" in seg_text[max(0, m.start() - 15):m.end() + 15]:
        score += 2.0
        used_cue = True
        cues.append("config-row")
    # A modified variant is NOT the base flavor: Primer's contribution is
    # "squared/squaring ReLU", which must not count as relu.
    if re.search(r"(?:squar(?:ed|ing)|leaky|parametric|gated|approximate[ds]?)\s*$",
                 before, flags=re.IGNORECASE):
        score -= 4.0
        cues.append("modified-variant")
    # Plus-compound baseline naming: "Transformer+GELU" is a named
    # comparison config, not this paper's choice.
    if re.search(r"\+\s*$", before):
        score -= 2.5
        cues.append("plus-compound")
    # Possessive attribution: "the Transformer's ReLU" belongs to that
    # architecture, not to this paper.
    if re.search(r"[A-Za-z0-9]'s\s*$", before):
        score -= 3.0
        cues.append("possessive-attribution")
    # Attribution / citation density only count against a candidate when
    # nothing in the window asserts usage.
    if not used_cue:
        for pat, w in WEAK_MENTION_CUES:
            if re.search(pat, window, flags=re.IGNORECASE):
                score += w
                cues.append(pat[:24])
        if len(re.findall(r"\\cite[tp]?\{|\\citet\b|\\citep\b", window)) >= 2:
            score -= 1.5
            cues.append("cite-dense")
    # "replace X with Y": Y (preceded by 'with') is used, X (followed by
    # 'with ...') is the thing REMOVED — sign depends on which side the
    # match sits, so handle it here rather than in the window cues.
    if re.search(r"replace[sd]?\b[^.]{0,60}$", before, flags=re.IGNORECASE):
        if re.search(r"with\s+(?:the\s+)?$", before, flags=re.IGNORECASE):
            score += 3.0
            cues.append("replace-target")
        elif re.search(r"^\s*(?:\w+\s+){0,3}(?:with|by)\b", after, flags=re.IGNORECASE):
            score -= 3.0
            cues.append("replace-source")
    return score, cues


def score_flavors(sections: list[tuple[str, str]]) -> dict[str, list[dict]]:
    """Per flavor field: candidates sorted by score (best first), each
    {value, score, evidence, cues, n}."""
    out: dict[str, list[dict]] = {}
    for fieldname, flavors in FLAVOR.items():
        cands = []
        for value, pat in flavors:
            best, best_ev, best_cues, n = None, "", [], 0
            others = [p for v, p in flavors if v != value]
            for sec_class, seg in sections:
                for m in re.finditer(pat, seg):
                    n += 1
                    sc, cues = score_match(seg, m, sec_class)
                    # Enumeration: another alternative of the SAME field
                    # within a tight radius means a comparison list
                    # ("the sinusoidal, rotary and T5 bias models").
                    near = seg[max(0, m.start() - 45):m.end() + 45]
                    if any(re.search(p, near) for p in others):
                        sc -= 1.5
                        cues = cues + ["enumeration"]
                    if best is None or sc > best:
                        snippet = seg[max(0, m.start() - 30):m.end() + 30]
                        best, best_ev, best_cues = sc, " ".join(snippet.split()), cues
            if n:
                # repetition: contributions recur (setup, tables, results)
                best += min(n - 1, 3) * 0.5
                cands.append({"value": value, "score": round(best, 2),
                              "evidence": best_ev, "cues": best_cues, "n": n})
        if cands:
            out[fieldname] = sorted(cands, key=lambda c: -c["score"])
    return out


# Header-keyword -> field map for model-size tables (LLaMA Table 2 style:
# one column per hyperparameter, one row per model size).
HEADER_MAP = [
    ("d_model", r"^(dimension|dim|d model|d_model|hidden(?: size)?|width)$"),
    ("n_heads", r"^(n ?heads?|heads?|attention heads?)$"),
    ("n_kv_heads", r"^(n ?kv ?heads?|kv ?heads?)$"),
    ("n_layers", r"^(n ?layers?|layers?|depth|blocks?)$"),
    ("d_ff", r"^(d ?ff|ffn(?: dim| size)?|intermediate(?: size)?|inner)$"),
    ("vocab_size", r"^(vocab(?:ulary)?(?: size)?)$"),
    ("context_length", r"^(context(?: length| window)?|seq(?:uence)? ?len(?:gth)?)$"),
]


def parse_tables(text: str) -> list[dict[str, int]]:
    """Extract per-model-size rows from tabular environments whose header
    names at least two known hyperparameter columns."""
    configs: list[dict[str, int]] = []
    for tab in re.findall(r"\\begin\{tabular\}.*?\\end\{tabular\}", text,
                          flags=re.DOTALL):
        rows = [r for r in re.split(r"\\\\", tab) if "&" in r]
        if len(rows) < 2:
            continue

        def cells(row: str) -> list[str]:
            out = []
            for c in row.split("&"):
                c = re.sub(r"\\[a-zA-Z]+(\[[^\]]*\])?(\{[^{}]*\})?", " ", c)
                out.append(re.sub(r"[^a-zA-Z0-9,\. ]", " ", c).strip().lower())
            return out

        # Find the header row: the first row mapping >= 2 known columns.
        colmap: dict[int, str] = {}
        header_idx = -1
        for ri, row in enumerate(rows[:3]):
            cm = {}
            for ci, cell in enumerate(cells(row)):
                for fieldname, pat in HEADER_MAP:
                    if re.match(pat, cell):
                        cm[ci] = fieldname
                        break
            if len(cm) >= 2:
                colmap, header_idx = cm, ri
                break
        if header_idx < 0:
            continue

        for row in rows[header_idx + 1:]:
            cfg: dict[str, int] = {}
            for ci, cell in enumerate(cells(row)):
                if ci not in colmap:
                    continue
                m = re.search(r"\d[\d,\.]*k?", cell)
                if not m:
                    continue
                try:
                    cfg[colmap[ci]] = _num(m.group(0))
                except ValueError:
                    pass
            if len(cfg) >= 2:
                configs.append(cfg)
    return configs


def extract(arxiv_id: str, tex: str) -> Arch:
    text = detex(tex)
    arch = Arch(arxiv_id=arxiv_id)

    m = re.search(r"\\title\s*(?:\[[^\]]*\])?\s*\{([^{}]+)", tex)
    if m:
        arch.title = re.sub(r"\\[a-zA-Z]+", "", m.group(1)).strip()

    for fieldname, pats in PATTERNS.items():
        for pat in pats:
            m = re.search(pat, text, flags=re.IGNORECASE)
            if m:
                try:
                    v = _num(m.group(1))
                except ValueError:
                    continue
                # Sanity windows keep table noise out (e.g. a year matched
                # as d_model).
                lo, hi = {
                    "d_model": (64, 65536), "n_layers": (1, 1000),
                    "n_heads": (1, 512), "n_kv_heads": (1, 512),
                    "d_ff": (128, 1 << 20), "vocab_size": (1000, 2_000_000),
                    "context_length": (64, 1 << 24),
                }[fieldname]
                if not (lo <= v <= hi):
                    continue
                snippet = text[max(0, m.start() - 40):m.end() + 20]
                arch.fields[fieldname] = Finding(v, " ".join(snippet.split()))
                break
        if fieldname not in arch.fields:
            arch.unresolved.append(fieldname)

    # Flavor fields go through contribution-vs-mention scoring: a value
    # is only ASSERTED ("used") when a clear winner exists; close calls
    # are "contested" (both reported); mention-only matches stay
    # unresolved with the mentions listed — reported, never guessed.
    flavor_scores = score_flavors(split_sections(tex))
    for fieldname in FLAVOR:
        cands = flavor_scores.get(fieldname, [])
        if not cands:
            arch.unresolved.append(fieldname)
            continue
        top = cands[0]
        second = cands[1] if len(cands) > 1 else None
        if top["score"] >= WIN_THRESHOLD and (
                second is None or top["score"] - second["score"] >= WIN_MARGIN):
            arch.fields[fieldname] = Finding(
                top["value"], top["evidence"], verdict="used", score=top["score"])
        elif top["score"] >= WIN_THRESHOLD:
            arch.fields[fieldname] = Finding(
                top["value"], top["evidence"], verdict="contested",
                score=top["score"],
                runner_up={"value": second["value"], "score": second["score"],
                           "evidence": second["evidence"]})
        else:
            arch.unresolved.append(fieldname)
        arch.mentions[fieldname] = [
            {"value": c["value"], "score": c["score"]} for c in cands]

    # Model-size tables fill whatever prose left unresolved. Row 0 (the
    # smallest listed model) is taken as THE config; others are variants.
    tables = parse_tables(text)
    if tables:
        chosen = tables[0]
        for fieldname, v in chosen.items():
            if fieldname not in arch.fields:
                arch.fields[fieldname] = Finding(
                    v, f"model-size table row 1 of {len(tables)}")
                if fieldname in arch.unresolved:
                    arch.unresolved.remove(fieldname)
        arch.variants = tables
    return arch


# --------------------------------------------------------------------------
# emit
# --------------------------------------------------------------------------

def to_json(arch: Arch) -> dict:
    def field_json(f: Finding) -> dict:
        d: dict = {"value": f.value, "evidence": f.evidence}
        if f.verdict:
            d["verdict"] = f.verdict
            d["score"] = f.score
        if f.runner_up:
            d["runner_up"] = f.runner_up
        return d
    return {
        "arxiv_id": arch.arxiv_id,
        "title": arch.title,
        "fields": {k: field_json(f) for k, f in arch.fields.items()},
        "unresolved": arch.unresolved,
        "mentions": arch.mentions,
        "variants": arch.variants,
    }


def emit_html(arch: Arch) -> str:
    """The diff-to-paper split view (STUDIO_PLAN section 10): extracted
    config on the left, each field's verbatim evidence snippet from the
    paper's LaTeX on the right, linked by hover/click highlighting.
    Self-contained file, studio dark theme, no dependencies — the visual
    proof that every value has a citation and nothing was guessed."""
    import html as _html

    rows, snips = [], []
    for i, (k, f) in enumerate(arch.fields.items()):
        v = _html.escape(str(f.value))
        ev = _html.escape(f.evidence)
        status = "extracted" if not f.verdict else f.verdict
        rows.append(
            f'<tr class="fld" data-i="{i}"><td class="k">{_html.escape(k)}</td>'
            f'<td class="v">{v}</td><td class="st ok">{status}</td></tr>')
        extra = ""
        if f.runner_up:
            ru = f.runner_up
            extra = (f'<div class="ev">contested with <b>'
                     f'{_html.escape(str(ru["value"]))}</b> '
                     f'({ru["score"]}): &ldquo;'
                     f'{_html.escape(ru.get("evidence", ""))}&rdquo;</div>')
        snips.append(
            f'<div class="snip" data-i="{i}"><div class="sk">{_html.escape(k)}'
            f' = {v}</div><div class="ev">&ldquo;{ev}&rdquo;</div>{extra}</div>')
    for k in arch.unresolved:
        rows.append(
            f'<tr><td class="k">{_html.escape(k)}</td><td class="v">&mdash;</td>'
            f'<td class="st un">unresolved &mdash; reported, not guessed</td></tr>')
    title = _html.escape(arch.title or f"arXiv:{arch.arxiv_id}")
    variants = ""
    if arch.variants:
        variants = ("<div class='note'>model-size table: " +
                    _html.escape(json.dumps(arch.variants)) + "</div>")
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>diff-to-paper — {title}</title>
<style>
  :root {{ --bg:#0d1117; --panel:#161b22; --border:#30363d; --text:#c9d1d9;
           --dim:#8b949e; --accent:#58a6ff; --ok:#3fb950; --un:#f0883e; }}
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ background:var(--bg); color:var(--text);
          font:14px/1.5 ui-monospace,Consolas,monospace; padding:1.2rem; }}
  h1 {{ font-size:1.05rem; margin-bottom:.2rem; }}
  h1 .accent {{ color:var(--accent); }}
  #sub {{ color:var(--dim); margin-bottom:1rem; }}
  .cols {{ display:flex; gap:1rem; align-items:flex-start; }}
  .panel {{ background:var(--panel); border:1px solid var(--border);
            border-radius:8px; padding:1rem; flex:1; }}
  .panel h2 {{ font-size:.85rem; color:var(--dim); text-transform:uppercase;
               letter-spacing:.06em; margin-bottom:.6rem; }}
  table {{ width:100%; border-collapse:collapse; }}
  td {{ padding:.3rem .5rem; border-bottom:1px solid var(--border); }}
  .k {{ color:var(--dim); }} .v {{ color:var(--text); }}
  .st.ok {{ color:var(--ok); font-size:.78rem; }}
  .st.un {{ color:var(--un); font-size:.78rem; }}
  tr.fld {{ cursor:pointer; }}
  tr.hl {{ background:#1f6feb22; }}
  .snip {{ border-left:3px solid var(--border); padding:.4rem .7rem;
           margin-bottom:.7rem; }}
  .snip.hl {{ border-left-color:var(--accent); background:#1f6feb14; }}
  .sk {{ color:var(--accent); font-size:.8rem; margin-bottom:.2rem; }}
  .ev {{ color:var(--text); }}
  .note {{ color:var(--dim); font-size:.78rem; margin-top:.8rem; }}
</style></head><body>
<h1>diff-to-paper <span class="accent">{title}</span></h1>
<div id="sub">arXiv:{arch.arxiv_id} &middot; every value below carries its
evidence; unresolved fields are reported, never guessed</div>
<div class="cols">
  <div class="panel"><h2>extracted architecture</h2>
    <table>{''.join(rows)}</table>{variants}</div>
  <div class="panel"><h2>evidence from the paper's LaTeX</h2>
    {''.join(snips)}</div>
</div>
<script>
  const on = (i, add) => document.querySelectorAll(`[data-i="${{i}}"]`)
    .forEach(el => el.classList.toggle('hl', add));
  document.querySelectorAll('.fld,.snip').forEach(el => {{
    el.addEventListener('mouseenter', () => on(el.dataset.i, true));
    el.addEventListener('mouseleave', () => on(el.dataset.i, false));
    el.addEventListener('click', () => {{
      const tgt = document.querySelector(
        `.snip[data-i="${{el.dataset.i}}"]`);
      if (tgt) tgt.scrollIntoView({{behavior:'smooth', block:'center'}});
    }});
  }});
</script>
</body></html>
"""


def emit_cpp(arch: Arch) -> str:
    g = {k: f.value for k, f in arch.fields.items()}
    llama_family = g.get("norm") == "rmsnorm" or g.get("positional") == "rope"
    lines = [
        "// Generated by papers/fetch.py from arXiv:" + arch.arxiv_id,
        f"// {arch.title or '(title not found)'}",
        "// Unresolved fields (verify by hand): " +
        (", ".join(arch.unresolved) or "none"),
        '#include "microtorch/nn.hpp"',
        "",
        "using namespace microtorch;",
        "",
    ]
    if llama_family:
        lines += [
            "// Llama-family (RMSNorm/RoPE): pair with gguf.hpp's",
            "// LlamaExportConfig and the phase-2b ops (rmsnorm, apply_rope).",
            "gguf::LlamaExportConfig make_config() {",
            "    gguf::LlamaExportConfig cfg;",
            f"    cfg.embedding_length   = {g.get('d_model', 0)};",
            f"    cfg.block_count        = {g.get('n_layers', 0)};",
            f"    cfg.head_count         = {g.get('n_heads', 0)};",
            f"    cfg.head_count_kv      = {g.get('n_kv_heads', g.get('n_heads', 0))};",
            f"    cfg.feed_forward_length= {g.get('d_ff', 0)};",
            f"    cfg.vocab_size         = {g.get('vocab_size', 0)};",
            f"    cfg.context_length     = {g.get('context_length', 2048)};",
            "    return cfg;",
            "}",
        ]
        lines.insert(3, '#include "microtorch/gguf.hpp"')
    else:
        lines += [
            "nn::GPT2 make_model(unsigned seed = 0) {",
            "    nn::GPT2Config cfg;",
            f"    cfg.d        = {g.get('d_model', 768)};",
            f"    cfg.n_layers = {g.get('n_layers', 12)};",
            f"    cfg.n_heads  = {g.get('n_heads', 12)};",
            f"    cfg.vocab    = {g.get('vocab_size', 50257)};",
            f"    cfg.n_ctx    = {g.get('context_length', 1024)};",
            "    return nn::GPT2(cfg, seed);",
            "}",
        ]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("arxiv_id")
    ap.add_argument("--json", help="write normalized arch config here")
    ap.add_argument("--emit-cpp", help="write microtorch C++ here")
    ap.add_argument("--emit-html", help="write the diff-to-paper split view here")
    ap.add_argument("--tex", help="parse a local .tex file instead of fetching")
    args = ap.parse_args()

    tex = (open(args.tex, encoding="utf-8", errors="replace").read()
           if args.tex else fetch_source(args.arxiv_id))
    arch = extract(args.arxiv_id, tex)

    print(f"arXiv:{arch.arxiv_id}  {arch.title or ''}".strip())
    for k, f in arch.fields.items():
        tag = f" ({f.verdict} {f.score})" if f.verdict else ""
        print(f"  {k:15} = {f.value!s:8}{tag}  [{f.evidence[:60]}]")
        if f.runner_up:
            print(f"  {'':15}   vs {f.runner_up['value']} "
                  f"({f.runner_up['score']}) — contested, verify by hand")
    if arch.unresolved:
        print(f"  unresolved: {', '.join(arch.unresolved)}")
    dropped = {k: v for k, v in arch.mentions.items() if k in arch.unresolved}
    if dropped:
        summary = "; ".join(
            f"{k}: " + ", ".join(f"{c['value']}({c['score']})" for c in v)
            for k, v in dropped.items())
        print(f"  mention-only (not applied): {summary}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(to_json(arch), fh, indent=2)
        print(f"wrote {args.json}")
    if args.emit_cpp:
        with open(args.emit_cpp, "w", encoding="utf-8") as fh:
            fh.write(emit_cpp(arch))
        print(f"wrote {args.emit_cpp}")
    if args.emit_html:
        with open(args.emit_html, "w", encoding="utf-8") as fh:
            fh.write(emit_html(arch))
        print(f"wrote {args.emit_html}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
