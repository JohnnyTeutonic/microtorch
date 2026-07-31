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
    test_prose()
    test_table()
    test_unresolved_reported()
    test_emit_html()
    print("\n[PASS] all fetcher tests")
    sys.exit(0)
