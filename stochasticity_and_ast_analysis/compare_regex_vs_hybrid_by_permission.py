#!/usr/bin/env python3
"""
Compare regex-only vs hybrid (masked + AST + proximity) matcher on the gold set,
with metrics broken down by ground_truth permission tag.

Outputs:
  - hybrid_vs_regex_by_permission.csv
  - hybrid_vs_regex_by_permission.txt

Run from project root:
  python stochasticity_and_ast_analysis/compare_regex_vs_hybrid_by_permission.py
"""

from __future__ import annotations

import csv
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

BASE = Path(__file__).resolve().parent
PROJECT_ROOT = BASE.parent
sys.path.insert(0, str(PROJECT_ROOT))

from extend_annotations import (
    get_applicable_codes,
    get_applicable_codes_hybrid,
    parse_codebook,
)
from repo_relative_path import as_repo_relative

CODEBOOK_PATH = PROJECT_ROOT / "Codebook_en.csv"
DB_PATH = PROJECT_ROOT / "db.sqlite3"
GROUND_TRUTH_PATH = PROJECT_ROOT / "gold_set_validation" / "ground_truth.csv"
OUT_CSV = BASE / "hybrid_vs_regex_by_permission.csv"
OUT_TXT = BASE / "hybrid_vs_regex_by_permission.txt"


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
                "permission": (row.get("permission") or "").strip().lower(),
            })
    return rows


def metrics(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def main() -> int:
    if not GROUND_TRUTH_PATH.exists():
        print(f"Missing {as_repo_relative(GROUND_TRUTH_PATH, PROJECT_ROOT)}", file=sys.stderr)
        return 1

    gold_rows = load_ground_truth(GROUND_TRUTH_PATH)
    n_by_perm: Dict[str, int] = defaultdict(int)
    for row in gold_rows:
        n_by_perm[row["permission"] or "(unknown)"] += 1

    section_to_patterns = parse_codebook(CODEBOOK_PATH)
    all_sections = set(section_to_patterns.keys())

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # permission -> tp, fp, fn for regex and hybrid
    agg_r: Dict[str, List[int]] = defaultdict(lambda: [0, 0, 0])
    agg_h: Dict[str, List[int]] = defaultdict(lambda: [0, 0, 0])

    for row in gold_rows:
        sid = row["snippet_id"]
        perm = row["permission"] or "(unknown)"
        cur.execute("SELECT code FROM snippets WHERE id = ?", (sid,))
        r = cur.fetchone()
        if r is None:
            continue
        code = (r["code"] or "") or ""
        gold = set(row["applicable_codes_ground_truth"])

        pred_r = set(get_applicable_codes(code, all_sections, section_to_patterns))
        pred_h = set(get_applicable_codes_hybrid(code, all_sections, section_to_patterns))

        tp_r = len(pred_r & gold)
        fp_r = len(pred_r - gold)
        fn_r = len(gold - pred_r)
        tp_h = len(pred_h & gold)
        fp_h = len(pred_h - gold)
        fn_h = len(gold - pred_h)

        ar = agg_r[perm]
        ar[0] += tp_r
        ar[1] += fp_r
        ar[2] += fn_r
        ah = agg_h[perm]
        ah[0] += tp_h
        ah[1] += fp_h
        ah[2] += fn_h

    conn.close()

    lines = [
        "=" * 72,
        "Gold set: regex vs hybrid — micro-averaged by permission tag",
        "=" * 72,
        f"n snippets in ground truth: {len(gold_rows)}",
        "",
    ]

    # Overall (all sections) — same as validate_gold_set aggregation
    tot_r = [sum(x[i] for x in agg_r.values()) for i in range(3)]
    tot_h = [sum(x[i] for x in agg_h.values()) for i in range(3)]
    pr, rr, fr = metrics(*tot_r)
    ph, rh, fh = metrics(*tot_h)
    lines.extend([
        "--- Overall (all codebook sections) ---",
        f"  Regex:  P={pr:.4f} R={rr:.4f} F1={fr:.4f}  TP={tot_r[0]} FP={tot_r[1]} FN={tot_r[2]}",
        f"  Hybrid: P={ph:.4f} R={rh:.4f} F1={fh:.4f}  TP={tot_h[0]} FP={tot_h[1]} FN={tot_h[2]}",
        f"  Delta P: {ph - pr:+.4f}  Delta R: {rh - rr:+.4f}  Delta F1: {fh - fr:+.4f}",
        "",
        "--- By permission (gold-set tag) ---",
        f"{'Permission':<22} {'n':>3} {'P_reg':>8} {'P_hyb':>8} {'ΔP':>8} {'R_reg':>8} {'R_hyb':>8} {'ΔR':>8} {'F1_reg':>8} {'F1_hyb':>8} {'ΔF1':>8}",
        "-" * 72,
    ])

    csv_rows = []
    perms_sorted = sorted(agg_r.keys(), key=lambda p: p.lower())
    for perm in perms_sorted:
        tr, frg, fnr = agg_r[perm]
        th, fph, fnh = agg_h[perm]
        prm, rrm, f1rm = metrics(tr, frg, fnr)
        phm, rhm, f1hm = metrics(th, fph, fnh)
        n_snips = None  # count snippets per perm - need separate counter
        csv_rows.append({
            "permission": perm,
            "n_snippets": n_by_perm.get(perm, 0),
            "tp_regex": tr,
            "fp_regex": frg,
            "fn_regex": fnr,
            "precision_regex": round(prm, 4),
            "recall_regex": round(rrm, 4),
            "f1_regex": round(f1rm, 4),
            "tp_hybrid": th,
            "fp_hybrid": fph,
            "fn_hybrid": fnh,
            "precision_hybrid": round(phm, 4),
            "recall_hybrid": round(rhm, 4),
            "f1_hybrid": round(f1hm, 4),
            "delta_precision": round(phm - prm, 4),
            "delta_recall": round(rhm - rrm, 4),
            "delta_f1": round(f1hm - f1rm, 4),
        })

    for row in csv_rows:
        perm = row["permission"]
        n = row["n_snippets"]
        prm = row["precision_regex"]
        phm = row["precision_hybrid"]
        dpr = row["delta_precision"]
        rrm = row["recall_regex"]
        rhm = row["recall_hybrid"]
        dr = row["delta_recall"]
        f1r = row["f1_regex"]
        f1h = row["f1_hybrid"]
        df1 = row["delta_f1"]
        lines.append(
            f"{perm:<22} {n:>3} {prm:>8.4f} {phm:>8.4f} {dpr:>+8.4f} "
            f"{rrm:>8.4f} {rhm:>8.4f} {dr:>+8.4f} {f1r:>8.4f} {f1h:>8.4f} {df1:>+8.4f}"
        )

    lines.append("")
    lines.append("Micro-averaging: TP/FP/FN summed within each permission across snippets, then P/R/F1.")
    lines.append("Interpretation: improves where ΔP/ΔF1 > 0 for that permission tag.")
    lines.append("=" * 72)

    OUT_TXT.write_text("\n".join(lines), encoding="utf-8")

    fieldnames = [
        "permission", "n_snippets", "tp_regex", "fp_regex", "fn_regex",
        "precision_regex", "recall_regex", "f1_regex",
        "tp_hybrid", "fp_hybrid", "fn_hybrid",
        "precision_hybrid", "recall_hybrid", "f1_hybrid",
        "delta_precision", "delta_recall", "delta_f1",
    ]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(csv_rows)

    print("\n".join(lines))
    print(
        f"\nWrote {as_repo_relative(OUT_TXT, PROJECT_ROOT)} and {as_repo_relative(OUT_CSV, PROJECT_ROOT)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
