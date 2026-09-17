#!/usr/bin/env python3
"""Build the documentation site from the repo's Markdown.

    python scripts/build_docs.py            # assemble + build ./site (static HTML)
    python scripts/build_docs.py --serve    # live preview on 0.0.0.0:8000
    python scripts/build_docs.py --check    # assemble + verify links only, no build

The repo keeps its docs where developers expect them (``README.md``, ``docs/``,
``benchmarks/*/README.md``). MkDocs wants one tree. This script:

1. copies those files into ``site_src/`` (git-ignored),
2. rewrites relative links so they resolve inside the site
   (``docs/claim.md`` from the README → ``claim.md``; ``../docs/kaggle.md`` from a
   benchmark README → ``../kaggle.md``; ``benchmarks/context_rot/README.md`` →
   ``benchmarks/context_rot.md``),
3. fails if any relative link points at a file that is not in the site — so a
   broken cross-reference breaks the build instead of shipping a 404,
4. **lints every page against GitHub-Flavored-Markdown rules** that Python-
   Markdown tolerates but GitHub, VS Code and most viewers do not (tables that
   start without a blank line, pipes inside code spans in table cells, fences
   that never close, ragged table rows) — and fails on any of them,
5. runs ``mkdocs build --strict``.

Requires ``pip install -e ".[docs]"``.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "site_src"

# (source path relative to repo root) -> (destination relative to site_src)
PAGES = {
    "README.md": "index.md",
    "benchmarks/README.md": "benchmarks/index.md",
    "benchmarks/context_rot/README.md": "benchmarks/context_rot.md",
    "benchmarks/sycophancy/README.md": "benchmarks/sycophancy.md",
    "benchmarks/fidelity/README.md": "benchmarks/fidelity.md",
    "benchmarks/token_cost/README.md": "benchmarks/token_cost.md",
}

LINK = re.compile(r"(!?\[[^\]]*\]\()([^)\s]+)(\))")
EXTERNAL = ("http://", "https://", "mailto:", "#")


def page_map() -> dict[Path, Path]:
    """Every source file that becomes a page, mapped to its site path."""
    pages = {ROOT / src: SRC / dst for src, dst in PAGES.items()}
    for md in sorted((ROOT / "docs").glob("*.md")):
        pages[md] = SRC / md.name
    return pages


def rewrite_links(text: str, src: Path, dst: Path, pages: dict[Path, Path]) -> tuple[str, list[str]]:
    """Point relative links at their site locations; report the ones that resolve nowhere."""
    broken: list[str] = []

    def fix(m: re.Match) -> str:
        target = m.group(2)
        if target.startswith(EXTERNAL):
            return m.group(0)
        path_part, _, anchor = target.partition("#")
        if not path_part:
            return m.group(0)
        resolved = (src.parent / path_part).resolve()
        if resolved in pages:
            new_target = Path(_relpath(pages[resolved], dst.parent))
            out = new_target.as_posix() + (f"#{anchor}" if anchor else "")
            return f"{m.group(1)}{out}{m.group(3)}"
        if resolved.exists():
            # a real file that is not a page (source code, a script): link to GitHub
            rel = resolved.relative_to(ROOT).as_posix()
            kind = "tree" if resolved.is_dir() else "blob"
            url = f"https://github.com/XLORD-oss/Agent-Memory/{kind}/arena/01a02e08-agent-memory/{rel}"
            return f"{m.group(1)}{url}{m.group(3)}"
        broken.append(target)
        return m.group(0)

    return LINK.sub(fix, text), broken


def _relpath(target: Path, start: Path) -> str:
    import os

    return os.path.relpath(target, start)


def lint_gfm(text: str) -> list[str]:
    """Problems that render fine in Python-Markdown but break in GFM viewers.

    Returns human-readable messages with 1-based line numbers. Kept
    deliberately small: each rule corresponds to a failure actually seen.
    """
    problems: list[str] = []
    lines = text.splitlines()
    in_fence = False
    fence_open_line = 0
    table_cols = None
    for i, line in enumerate(lines, start=1):
        stripped = line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            fence_open_line = i if in_fence else 0
            table_cols = None
            continue
        if in_fence:
            continue
        if line.startswith("|"):
            prev = lines[i - 2] if i >= 2 else ""
            if not prev.startswith("|") and prev.strip():
                problems.append(f"line {i}: table must be preceded by a blank line (GFM)")
            cells = line.strip().strip("|").split("|")
            for c in cells:
                if c.count("`") % 2:
                    problems.append(f"line {i}: pipe inside a code span splits the table cell in GFM: {c.strip()[:50]!r}")
                    break
            row_cols = len(cells)
            if table_cols is None:
                table_cols = row_cols
            elif row_cols != table_cols and not set(stripped) <= set("|-: "):
                problems.append(f"line {i}: table row has {row_cols} cells, header has {table_cols}")
        else:
            table_cols = None
    if in_fence:
        problems.append(f"line {fence_open_line}: code fence never closed")
    return problems


def assemble() -> int:
    if SRC.exists():
        shutil.rmtree(SRC)
    SRC.mkdir()
    pages = page_map()
    problems = 0
    for src, dst in pages.items():
        text = src.read_text(encoding="utf-8")
        for msg in lint_gfm(text):
            print(f"[docs] GFM in {src.relative_to(ROOT)}: {msg}", file=sys.stderr)
            problems += 1
        text, broken = rewrite_links(text, src, dst, pages)
        for b in broken:
            print(f"[docs] BROKEN LINK in {src.relative_to(ROOT)}: {b}", file=sys.stderr)
            problems += 1
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(text, encoding="utf-8")
    # anything else under docs/ (images, plots) travels along
    for extra in (ROOT / "docs").iterdir():
        if extra.is_file() and extra.suffix.lower() in (".png", ".svg", ".jpg", ".jpeg", ".gif"):
            shutil.copy2(extra, SRC / extra.name)
    print(f"[docs] assembled {len(pages)} pages into {SRC.relative_to(ROOT)}/")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--serve", action="store_true", help="live preview instead of a static build")
    ap.add_argument("--check", action="store_true", help="assemble and verify links only")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    problems = assemble()
    if problems:
        print(f"[docs] {problems} problem(s) — fix them before building.", file=sys.stderr)
        sys.exit(1)
    if args.check:
        print("[docs] links and GFM lint OK")
        return

    mkdocs = [sys.executable, "-m", "mkdocs"]
    if args.serve:
        cmd = mkdocs + ["serve", "-a", f"0.0.0.0:{args.port}"]
    else:
        cmd = mkdocs + ["build", "--strict", "--clean"]
    print("[docs]", " ".join(cmd))
    sys.exit(subprocess.call(cmd, cwd=ROOT))


if __name__ == "__main__":
    main()
