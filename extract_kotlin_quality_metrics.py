#!/usr/bin/env python3
"""
Extract Kotlin code-quality markers per snippet.

Outputs a CSV keyed by snippet_id with:
- NCL, CC, LLOC, LOC, SLOC, MI
- optional tool-derived counts from detekt/ktlint if available
"""

import argparse
import math
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
_REPO_ROOT = BASE.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from repo_relative_path import as_repo_relative
DB_PATH = BASE / "db.sqlite3"
OUTPUT_DIR = BASE / "analysis_output"
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / "kotlin_quality_metrics.csv"

LLMS = {"chatgpt", "gemini", "deepseek", "github-copilot", "qwen", "baidu", "falcon3"}

_CC_KEYWORDS = re.compile(
    r"\b(if|else\s+if|when|for|while|do|catch|case|&&|\|\||\?:|try)\b"
)
_TOKEN_RE = re.compile(r"[A-Za-z_]\w*|\d+|==|!=|<=|>=|&&|\|\||[{}()[\].,;:+\-*/%<>]")


def get_snippet_tags(conn) -> Dict[int, set]:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT s.id, t.name
        FROM snippets s
        JOIN snippets_tags st ON s.id = st.snippet_id
        JOIN tags t ON t.id = st.tag_id
        """
    )
    out = defaultdict(set)
    for sid, tag in cur.fetchall():
        out[int(sid)].add(str(tag))
    return dict(out)


def resolve_metadata(snippet_tags: Dict[int, set]) -> Dict[int, dict]:
    def get_llm(tags: set):
        for llm in LLMS:
            if llm in tags or any(t.startswith(llm + "-") for t in tags):
                return llm
        return None

    def get_prompt_type(tags: set):
        for t in tags:
            if t.endswith("-friendly"):
                return "friendly"
            if t.endswith("-invasive"):
                return "invasive"
        return "neutral"

    def get_permission(tags: set):
        skip = LLMS | {"kotlin", "xml", "java, xml", "kottlin", "kotkin", "andorid", "friendly"}
        skip |= {t for t in tags if "-friendly" in t or "-invasive" in t}
        perms = [p for p in tags if p not in skip]
        return perms[0] if perms else None

    return {
        sid: {
            "llm": get_llm(tags),
            "permission": get_permission(tags),
            "prompt_type": get_prompt_type(tags),
        }
        for sid, tags in snippet_tags.items()
    }


def split_lines(code: str) -> List[str]:
    return (code or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")


def count_comment_metrics(code: str):
    lines = split_lines(code)
    loc = len(lines)
    ncl = 0
    sloc = 0
    lloc = 0
    in_block = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        is_comment_line = False
        if in_block:
            is_comment_line = True
            if "*/" in stripped:
                in_block = False
        elif stripped.startswith("//"):
            is_comment_line = True
        elif stripped.startswith("/*"):
            is_comment_line = True
            if "*/" not in stripped:
                in_block = True
        if is_comment_line:
            ncl += 1
            continue
        sloc += 1
        # Approximate logical LOC: semicolons, control declarations, or closing expressions.
        if (
            stripped.endswith(";")
            or stripped.endswith(")")
            or stripped.endswith("}")
            or re.match(r"^(if|for|while|when|return|throw|val|var|fun)\b", stripped)
        ):
            lloc += 1
    if sloc > 0 and lloc == 0:
        lloc = sloc
    return loc, sloc, lloc, ncl


def estimate_cc(code: str) -> int:
    c = len(_CC_KEYWORDS.findall(code or ""))
    return max(1, c + 1)


def estimate_halstead_volume(code: str) -> float:
    tokens = _TOKEN_RE.findall(code or "")
    if not tokens:
        return 0.0
    n = len(set(tokens))
    N = len(tokens)
    if n <= 1:
        return float(N)
    return float(N * math.log2(n))


def maintainability_index(code: str, cc: int, sloc: int, ncl: int) -> float:
    hv = max(estimate_halstead_volume(code), 1.0)
    sloc_safe = max(sloc, 1)
    comment_ratio = min(max(ncl / sloc_safe, 0.0), 1.0)
    comment_term = 50.0 * math.sin(math.sqrt(2.4 * comment_ratio))
    mi_raw = 171.0 - 5.2 * math.log(hv) - 0.23 * float(cc) - 16.2 * math.log(sloc_safe) + comment_term
    return max(0.0, min(100.0, (mi_raw * 100.0) / 171.0))


def run_tool(command: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(command, capture_output=True, text=True, check=False)


def extract_tool_counts(tmp_dir: Path):
    detekt = shutil.which("detekt")
    ktlint = shutil.which("ktlint")
    detekt_counts: Dict[str, int] = {}
    ktlint_counts: Dict[str, int] = {}

    if detekt:
        for f in tmp_dir.glob("*.kt"):
            res = run_tool([detekt, "--input", str(f), "--build-upon-default-config"])
            body = (res.stdout or "") + "\n" + (res.stderr or "")
            detekt_counts[f.name] = len(re.findall(r"\bwarning\b|\berror\b", body, flags=re.IGNORECASE))

    if ktlint:
        for f in tmp_dir.glob("*.kt"):
            res = run_tool([ktlint, str(f)])
            body = (res.stdout or "") + "\n" + (res.stderr or "")
            # Typical format: file.kt:line:col: message
            ktlint_counts[f.name] = len(re.findall(rf"{re.escape(str(f))}:\d+:\d+:", body))

    return bool(detekt), bool(ktlint), detekt_counts, ktlint_counts


def main():
    parser = argparse.ArgumentParser(description="Extract Kotlin complexity and maintainability metrics.")
    parser.add_argument("--db", type=Path, default=DB_PATH, help="Path to db.sqlite3")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH, help="Output CSV path")
    args = parser.parse_args()

    db_path = args.db if args.db.is_absolute() else (BASE / args.db)
    out_path = args.output if args.output.is_absolute() else (BASE / args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT id, code, title, description, language FROM snippets")
    rows = cur.fetchall()
    metadata = resolve_metadata(get_snippet_tags(conn))
    conn.close()

    with tempfile.TemporaryDirectory(prefix="kotlin_metrics_") as td:
        tmp_dir = Path(td)
        record_rows = []
        for sid, code, title, description, language in rows:
            sid = int(sid)
            code = code or ""
            loc, sloc, lloc, ncl = count_comment_metrics(code)
            cc = estimate_cc(code)
            mi = maintainability_index(code, cc, sloc, ncl)
            meta = metadata.get(sid, {})
            # Prompt text is inconsistently stored; use description first, then title fallback.
            prompt_text = ((description or "").strip() or (title or "").strip())
            prompt_length_chars = len(prompt_text)
            prompt_length_words = len(prompt_text.split())
            fname = f"snippet_{sid}.kt"
            (tmp_dir / fname).write_text(code, encoding="utf-8")
            record_rows.append(
                {
                    "snippet_id": sid,
                    "llm": meta.get("llm"),
                    "prompt_type": meta.get("prompt_type"),
                    "permission": meta.get("permission"),
                    "language": language,
                    "prompt_length_chars": prompt_length_chars,
                    "prompt_length_words": prompt_length_words,
                    "LOC": loc,
                    "SLOC": sloc,
                    "LLOC": lloc,
                    "NCL": ncl,
                    "CC": cc,
                    "MI": mi,
                    "_tool_file": fname,
                }
            )

        detekt_available, ktlint_available, detekt_counts, ktlint_counts = extract_tool_counts(tmp_dir)

    out_df = pd.DataFrame(record_rows)
    out_df["detekt_findings"] = out_df["_tool_file"].map(detekt_counts).fillna(0).astype(int)
    out_df["ktlint_violations"] = out_df["_tool_file"].map(ktlint_counts).fillna(0).astype(int)
    out_df["detekt_available"] = detekt_available
    out_df["ktlint_available"] = ktlint_available
    out_df = out_df.drop(columns=["_tool_file"])
    out_df.to_csv(out_path, index=False)
    print(f"Wrote metrics: {as_repo_relative(out_path, _REPO_ROOT)}")
    print(f"Rows: {len(out_df)} | detekt_available={detekt_available} | ktlint_available={ktlint_available}")


if __name__ == "__main__":
    main()
