"""Offline tests for papers/fetch.py — no network, fixture LaTeX only.

    python papers/test_fetch.py
"""
import sys

from fetch import Arch, detex, emit_cpp, extract, parse_tables

FIXTURE_PROSE = r"""
\title{A Tiny Transformer}
We stack $N=4$ identical layers with $d_{\text{model}}=256$ and employ
$h=8$ parallel attention layers, or heads. The inner-layer has
dimensionality $d_{ff}=1024$. We use a vocabulary of 8K tokens and a
context length of 512 tokens. Sub-layers are followed by layer
normalization with a ReLU activation. % comment should vanish
"""

FIXTURE_TABLE = r"""
We use the RMSNorm normalizing function with SwiGLU and rotary positional
embeddings (RoPE).
\begin{tabular}{cccc}
params & dimension & n heads & n layers \\
1.3B & 2048 & 16 & 24 \\
7B & 4096 & 32 & 32 \\
\end{tabular}
"""


def test_prose() -> None:
    arch = extract("0000.00000", FIXTURE_PROSE)
    got = {k: f.value for k, f in arch.fields.items()}
    assert got["d_model"] == 256, got
    assert got["n_layers"] == 4, got
    assert got["n_heads"] == 8, got
    assert got["d_ff"] == 1024, got
    assert got["vocab_size"] == 8000, got
    assert got["context_length"] == 512, got
    assert got["norm"] == "layernorm", got
    assert got["activation"] == "relu", got
    assert arch.title == "A Tiny Transformer", arch.title
    assert "% comment" not in detex(FIXTURE_PROSE)
    cpp = emit_cpp(arch)
    assert "cfg.d        = 256" in cpp and "GPT2" in cpp
    print("prose extraction ok:", got)


def test_table() -> None:
    arch = extract("0000.00001", FIXTURE_TABLE)
    got = {k: f.value for k, f in arch.fields.items()}
    assert got["norm"] == "rmsnorm", got
    assert got["activation"] == "swiglu", got
    assert got["positional"] == "rope", got
    # Row 1 (smallest model) is the chosen config; both rows are variants.
    assert got["d_model"] == 2048, got
    assert got["n_heads"] == 16, got
    assert got["n_layers"] == 24, got
    assert len(arch.variants) == 2, arch.variants
    assert arch.variants[1]["d_model"] == 4096
    cpp = emit_cpp(arch)
    assert "LlamaExportConfig" in cpp and "cfg.block_count        = 24" in cpp
    print("table extraction ok:", got, "variants:", len(arch.variants))


FIXTURE_COMPACT = r"""
The model has 16 transformer layers of dimension 1024. Each decoder is
110M parameters ($d_{model}=768$, $d_{ff}=3072$, L=12).
"""


def test_compact_notation() -> None:
    # Real phrasings that used to slip through: ALiBi's "layers of
    # dimension 1024" and Primer's "(d_model=768, d_ff=3072, L=12)".
    arch = extract("0000.00003", FIXTURE_COMPACT)
    got = {k: f.value for k, f in arch.fields.items()}
    assert got["n_layers"] == 16, got   # first match wins (prose order)
    assert got["d_ff"] == 3072, got
    assert got["d_model"] in (768, 1024), got
    # The L=12 pattern alone, without the ALiBi sentence:
    arch2 = extract("0000.00004", r"Each decoder is ($d_{model}=768$, L=12).")
    got2 = {k: f.value for k, f in arch2.fields.items()}
    assert got2["n_layers"] == 12, got2
    print("compact-notation extraction ok:", got, got2)


FIXTURE_MENTION_ONLY = r"""
Prior work has explored alternatives such as SwiGLU \cite{a} \cite{b},
compared against strong baselines. Our search discovers squaring ReLU
activations, a novel modification.
"""

FIXTURE_REPLACE = r"""
We replace LayerNorm with RMSNorm in every block of our model.
"""

FIXTURE_CONTESTED = r"""
We use LayerNorm for the encoder blocks. We use RMSNorm for the decoder
blocks.
"""


def test_contribution_vs_mention() -> None:
    # Mention-only: nothing asserted, mentions reported.
    a1 = extract("0000.00005", FIXTURE_MENTION_ONLY)
    assert "activation" in a1.unresolved, a1.fields
    assert any(c["value"] == "swiglu" for c in a1.mentions.get("activation", []))
    # Replacement: the target is used, the source is not.
    a2 = extract("0000.00006", FIXTURE_REPLACE)
    f = a2.fields["norm"]
    assert f.value == "rmsnorm" and f.verdict == "used", (f.value, f.verdict)
    # Symmetric usage: contested, runner-up carried, nothing asserted
    # silently.
    a3 = extract("0000.00007", FIXTURE_CONTESTED)
    f3 = a3.fields.get("norm")
    assert f3 is not None and f3.verdict == "contested", f3
    assert f3.runner_up is not None
    print("contribution-vs-mention ok: mention-only abstains, "
          f"replace->{a2.fields['norm'].value}, symmetric->contested")


def test_unresolved_reported() -> None:
    arch = extract("0000.00002", r"A paper with no architecture at all.")
    assert "d_model" in arch.unresolved and "norm" in arch.unresolved
    assert not arch.fields
    print("unresolved reporting ok")


def test_emit_html() -> None:
    from fetch import emit_html
    arch = extract("0000.00000", FIXTURE_PROSE)
    html = emit_html(arch)
    # Every extracted field appears with its evidence; unresolved fields
    # are labelled as such; the page is self-contained (no external refs).
    for k, f in arch.fields.items():
        assert k in html, f"field {k} missing from html"
        assert str(f.value) in html
    for k in arch.unresolved:
        assert k in html
    assert "unresolved &mdash; reported, not guessed" in html or not arch.unresolved
    assert "http" not in html.split("</style>")[1], "external reference leaked"
    assert "diff-to-paper" in html
    print("emit_html ok "
          f"({len(arch.fields)} fields, {len(arch.unresolved)} unresolved)")


if __name__ == "__main__":
    test_compact_notation()
    test_contribution_vs_mention()
    test_prose()
    test_table()
    test_unresolved_reported()
    test_emit_html()
    print("\n[PASS] all fetcher tests")
    sys.exit(0)
