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


@dataclass
class Arch:
    arxiv_id: str
    title: str | None = None
    fields: dict[str, Finding] = field(default_factory=dict)
    unresolved: list[str] = field(default_factory=list)
    variants: list[dict[str, int]] = field(default_factory=list)


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
        rf"\bdim(?:ension)?\s*&\s*{NUM}",
        rf"\bhidden\s*&\s*{NUM}",
    ],
    "n_layers": [
        rf"N\s*=\s*{NUM}\s+identical\s+layers",
        rf"stack\s+of\s+\$?N\s*=\s*\$?{NUM}",
        rf"{NUM}\s+(?:identical\s+|transformer\s+|decoder\s+|hidden\s+)?layers\b",
        rf"(?:n[ _]?layers?|\#?\s*layers?|depth)\s*(?:=|of|:|&)\s*\$?{NUM}",
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

    for fieldname, flavors in FLAVOR.items():
        for value, pat in flavors:
            m = re.search(pat, text)
            if m:
                snippet = text[max(0, m.start() - 30):m.end() + 30]
                arch.fields[fieldname] = Finding(value, " ".join(snippet.split()))
                break
        if fieldname not in arch.fields:
            arch.unresolved.append(fieldname)

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
    return {
        "arxiv_id": arch.arxiv_id,
        "title": arch.title,
        "fields": {k: {"value": f.value, "evidence": f.evidence}
                   for k, f in arch.fields.items()},
        "unresolved": arch.unresolved,
        "variants": arch.variants,
    }


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
    ap.add_argument("--tex", help="parse a local .tex file instead of fetching")
    args = ap.parse_args()

    tex = (open(args.tex, encoding="utf-8", errors="replace").read()
           if args.tex else fetch_source(args.arxiv_id))
    arch = extract(args.arxiv_id, tex)

    print(f"arXiv:{arch.arxiv_id}  {arch.title or ''}".strip())
    for k, f in arch.fields.items():
        print(f"  {k:15} = {f.value!s:8}  [{f.evidence[:70]}]")
    if arch.unresolved:
        print(f"  unresolved: {', '.join(arch.unresolved)}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(to_json(arch), fh, indent=2)
        print(f"wrote {args.json}")
    if args.emit_cpp:
        with open(args.emit_cpp, "w", encoding="utf-8") as fh:
            fh.write(emit_cpp(arch))
        print(f"wrote {args.emit_cpp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
