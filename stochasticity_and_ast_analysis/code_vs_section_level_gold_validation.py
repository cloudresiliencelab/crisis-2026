#!/usr/bin/env python3
"""
Gold-set validation: code-level (per Cxxx) vs section-level (per codebook category).

Section-level (micro across snippets): for each snippet, let G = sections implied by
gold codes, P = sections implied by predicted codes (via codebook mapping).
  TP_sec = |P ∩ G|,  FP_sec = |P \\ G|,  FN_sec = |G \\ P|
Summed over all snippets, then precision/recall/F1 on those totals.

Code-level: same counting as validate_gold_set (all sections), TP/FP/FN over codes.

Outputs:
  code_vs_section_level_report.txt
  code_vs_section_level_report.csv  (one row per matcher)

Run from project root:
  python stochasticity_and_ast_analysis/code_vs_section_level_gold_validation.py
"""

from __future__ import annotations

import csv
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

BASE = Path(__file__).resolve().parent
PROJECT_ROOT = BASE.parent
sys.path.insert(0, str(PROJECT_ROOT))

from extend_annotations import (
    build_code_to_section,
    get_applicable_codes,
    get_applicable_codes_hybrid,
    parse_codebook,
)
from repo_relative_path import as_repo_relative

CODEBOOK_PATH = PROJECT_ROOT / "Codebook_en.csv"
DB_PATH = PROJECT_ROOT / "db.sqlite3"
GROUND_TRUTH_PATH = PROJECT_ROOT / "gold_set_validation" / "ground_truth.csv"
OUT_TXT = BASE / "code_vs_section_level_report.txt"
OUT_CSV = BASE / "code_vs_section_level_report.csv"


def load_ground_truth(path: Path) -> List[dict]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            sid_str = (row.get("snippet_id") or "").strip()
            gt_str = (row.get("applicable_codes_ground_truth") or "").strip()
            if not sid_str or not gt_str:
                continue
            try:
                snippet_id = int(sid_str)
            except ValueError:
                continue
            gt_codes = [c.strip() for c in gt_str.replace(",", ";").split(";") if c.strip()]
            if not gt_codes:
                continue
            rows.append({
                "snippet_id": snippet_id,
                "applicable_codes_ground_truth": gt_codes,
            })
    return rows


def codes_to_sections(codes: Set[str], code_to_section: Dict[str, str]) -> Set[str]:
    out: Set[str] = set()
    for c in codes:
        sec = code_to_section.get(c)
        if sec:
            out.add(sec)
        else:
            # Unknown label (should not happen for valid codebook ids)
            out.add(f"(unknown:{c})")
    return out


def metrics(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def aggregate_code_level(pred: Set[str], gold: Set[str]) -> Tuple[int, int, int]:
    tp = len(pred & gold)
    fp = len(pred - gold)
    fn = len(gold - pred)
    return tp, fp, fn


def aggregate_section_level(pred: Set[str], gold: Set[str], code_to_section: Dict[str, str]) -> Tuple[int, int, int]:
    P = codes_to_sections(pred, code_to_section)
    G = codes_to_sections(gold, code_to_section)
    tp = len(P & G)
    fp = len(P - G)
    fn = len(G - P)
    return tp, fp, fn


def run_matcher(name: str, gold_rows: List[dict], match_fn, section_to_patterns: dict, all_sections: Set[str], code_to_section: Dict[str, str]):
    cur = sqlite3.connect(DB_PATH).cursor()

    c_tp = c_fp = c_fn = 0
    s_tp = s_fp = s_fn = 0

    for row in gold_rows:
        sid = row["snippet_id"]
        cur.execute("SELECT code FROM snippets WHERE id = ?", (sid,))
        r = cur.fetchone()
        if r is None:
            continue
        code = (r[0] or "") or ""
        gold = set(row["applicable_codes_ground_truth"])
        pred = set(match_fn(code, all_sections, section_to_patterns))

        t1, f1, n1 = aggregate_code_level(pred, gold)
        c_tp += t1
        c_fp += f1
        c_fn += n1

        t2, f2, n2 = aggregate_section_level(pred, gold, code_to_section)
        s_tp += t2
        s_fp += f2
        s_fn += n2

    cur.connection.close()

    cp, cr, cf1 = metrics(c_tp, c_fp, c_fn)
    sp, sr, sf1 = metrics(s_tp, s_fp, s_fn)

    return {
        "matcher": name,
        "code_tp": c_tp,
        "code_fp": c_fp,
        "code_fn": c_fn,
        "code_precision": cp,
        "code_recall": cr,
        "code_f1": cf1,
        "section_tp": s_tp,
        "section_fp": s_fp,
        "section_fn": s_fn,
        "section_precision": sp,
        "section_recall": sr,
        "section_f1": sf1,
        "delta_precision": sp - cp,
        "delta_recall": sr - cr,
        "delta_f1": sf1 - cf1,
    }


def main() -> int:
    if not GROUND_TRUTH_PATH.exists():
        print(f"Missing {as_repo_relative(GROUND_TRUTH_PATH, PROJECT_ROOT)}", file=sys.stderr)
        return 1

    gold_rows = load_ground_truth(GROUND_TRUTH_PATH)
    section_to_patterns = parse_codebook(CODEBOOK_PATH)
    all_sections = set(section_to_patterns.keys())
    code_to_section = build_code_to_section(section_to_patterns)

    rows_out = [
        run_matcher(
            "regex_all_sections",
            gold_rows,
            get_applicable_codes,
            section_to_patterns,
            all_sections,
            code_to_section,
        ),
        run_matcher(
            "hybrid_all_sections",
            gold_rows,
            lambda code, secs, pat: get_applicable_codes_hybrid(code, secs, pat),
            section_to_patterns,
            all_sections,
            code_to_section,
        ),
    ]

    lines = [
        "=" * 76,
        "Gold set (46 snippets): code-level vs section-level validation",
        "=" * 76,
        "",
        "Code-level: TP = |pred ∩ gold|, FP = |pred - gold|, FN = |gold - pred| over codes.",
        "Section-level: map each code to its codebook section; per snippet",
        "  TP_sec = |P_sections ∩ G_sections|, FP_sec = |P - G|, FN_sec = |G - P|;",
        "then sum TP/FP/FN across snippets (micro over section slots).",
        "",
    ]

    for r in rows_out:
        lines.extend([
            f"--- {r['matcher']} ---",
            f"  Code-level:    P={r['code_precision']:.4f}  R={r['code_recall']:.4f}  F1={r['code_f1']:.4f}"
            f"   TP={r['code_tp']} FP={r['code_fp']} FN={r['code_fn']}",
            f"  Section-level: P={r['section_precision']:.4f}  R={r['section_recall']:.4f}  F1={r['section_f1']:.4f}"
            f"   TP={r['section_tp']} FP={r['section_fp']} FN={r['section_fn']}",
            f"  Δ (section − code):  ΔP={r['delta_precision']:+.4f}  ΔR={r['delta_recall']:+.4f}  ΔF1={r['delta_f1']:+.4f}",
            "",
        ])

    lines.append("Interpretation: section-level metrics are usually higher when wrong codes stay")
    lines.append("within the same category as a gold code, or when recall improves via")
    lines.append("partial category overlap; they are not comparable to code-level numbers.")
    lines.append("=" * 76)

    report = "\n".join(lines)
    OUT_TXT.write_text(report, encoding="utf-8")

    fieldnames = list(rows_out[0].keys())
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows_out:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items()})

    print(report)
    print(
        f"\nWrote {as_repo_relative(OUT_TXT, PROJECT_ROOT)} and {as_repo_relative(OUT_CSV, PROJECT_ROOT)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
