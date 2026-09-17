"""The documentation stays buildable: every relative link resolves, nav is complete."""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_build_docs():
    spec = importlib.util.spec_from_file_location("build_docs", ROOT / "scripts" / "build_docs.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_every_relative_link_in_the_docs_resolves():
    bd = _load_build_docs()
    pages = bd.page_map()
    broken = []
    for src, dst in pages.items():
        _, bad = bd.rewrite_links(src.read_text(encoding="utf-8"), src, dst, pages)
        broken += [(str(src.relative_to(ROOT)), b) for b in bad]
    assert broken == [], f"broken links: {broken}"


def test_link_rewriting_maps_into_the_site_tree():
    bd = _load_build_docs()
    pages = bd.page_map()
    readme = ROOT / "README.md"
    text, bad = bd.rewrite_links("[x](docs/claim.md) [y](benchmarks/README.md) [z](benchmarks/context_rot/README.md#setup)",
                                 readme, pages[readme], pages)
    assert bad == []
    assert "(claim.md)" in text and "(benchmarks/index.md)" in text and "(benchmarks/context_rot.md#setup)" in text
    bench = ROOT / "benchmarks" / "context_rot" / "README.md"
    text, _ = bd.rewrite_links("[e](../../docs/evidence.md)", bench, pages[bench], pages)
    assert "(../evidence.md)" in text
    # source files that are not pages become GitHub links, not 404s
    text, bad = bd.rewrite_links("[src](src/agent_memory/core.py)", readme, pages[readme], pages)
    assert bad == [] and "github.com/XLORD-oss/Agent-Memory/blob/" in text


def test_nav_lists_every_doc_page_exactly_once():
    nav_text = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    nav_text = nav_text[nav_text.index("nav:"):]
    listed = re.findall(r":\s*([\w./-]+\.md)\s*$", nav_text, re.M)
    assert len(listed) == len(set(listed)), "duplicate nav entries"
    expected = {p.name for p in (ROOT / "docs").glob("*.md")} | {
        "index.md", "benchmarks/index.md", "benchmarks/context_rot.md", "benchmarks/sycophancy.md",
        "benchmarks/fidelity.md", "benchmarks/token_cost.md",
    }
    assert set(listed) == expected, f"nav/doc mismatch: missing={expected - set(listed)} extra={set(listed) - expected}"


@pytest.mark.skipif(importlib.util.find_spec("mkdocs") is None, reason="mkdocs not installed (pip install -e '.[docs]')")
def test_site_builds_strict(tmp_path):
    import subprocess

    r = subprocess.run([sys.executable, "scripts/build_docs.py"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert (ROOT / "site" / "index.html").exists()
    assert (ROOT / "site" / "FRAMEWORK" / "index.html").exists()


def test_every_page_passes_the_gfm_lint():
    bd = _load_build_docs()
    problems = []
    for src in bd.page_map():
        problems += [(str(src.relative_to(ROOT)), m) for m in bd.lint_gfm(src.read_text(encoding="utf-8"))]
    assert problems == [], problems


def test_gfm_lint_catches_the_known_failure_modes():
    bd = _load_build_docs()
    assert bd.lint_gfm("text\n| a | b |\n|---|---|\n") == ["line 2: table must be preceded by a blank line (GFM)"]
    assert any("code span" in m for m in bd.lint_gfm("\n| `x \\| y` | b |\n|---|---|\n"))
    assert any("never closed" in m for m in bd.lint_gfm("```python\nprint(1)\n"))
    assert any("cells" in m for m in bd.lint_gfm("\n| a | b |\n|---|---|\n| 1 | 2 | 3 |\n"))
    # inside a fence, pipes and tables are ignored
    assert bd.lint_gfm("```\ntext\n| a | b |\n```\n") == []
