#!/usr/bin/env python3
"""
Validate the automated pattern matcher against manual ground truth.

Reads gold_set_validation/ground_truth.csv: for each row with snippet_id and
applicable_codes_ground_truth filled, loads the snippet from the DB, runs the
same pattern matching as annotate_all_snippets.py, and compares predicted vs
ground-truth codes. Validation is run twice: (1) all codebook sections (same as
full-dataset matcher), (2) permission-restricted (only sections for the snippet's
permission). Writes both metric sets to gold_set_validation/validation_report.txt
and gold_set_validation/validation_report.csv.

Run from the reproducibility folder (this script’s directory):
  python3 validate_gold_set.py
  python3 validate_gold_set.py --hybrid   # optional AST-assisted patterns (see stochasticity_and_ast_analysis/)
"""

import argparse
import csv
import sqlite3
from pathlib import Path
from typing import Callable, List, Set

from repo_relative_path import as_repo_relative

BASE = Path(__file__).resolve().parent
CODEBOOK_PATH = BASE / "Codebook_en.csv"
DB_PATH = BASE / "db.sqlite3"
GOLD_SET_DIR = BASE / "gold_set_validation"
GROUND_TRUTH_PATH = GOLD_SET_DIR / "ground_truth.csv"
REPORT_TXT_PATH = GOLD_SET_DIR / "validation_report.txt"
REPORT_CSV_PATH = GOLD_SET_DIR / "validation_report.csv"

from permission_sections import PERMISSION_TO_SECTIONS

# Reuse the same matching logic as the main pipeline
from extend_annotations import parse_codebook, get_applicable_codes, get_applicable_codes_hybrid


def load_ground_truth(path: Path) -> list[dict]:
    """Load ground_truth.csv; return list of rows with snippet_id and applicable_codes_ground_truth."""
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
                "permission": (row.get("permission") or "").strip(),
                "prompt_type": (row.get("prompt_type") or "").strip(),
                "notes": (row.get("notes") or "").strip(),
            })
    return rows


def _matcher(use_hybrid: bool) -> Callable[[str, Set[str], dict], List[str]]:
    if use_hybrid:
        def match(snippet_code: str, sections: Set[str], section_to_patterns: dict) -> List[str]:
            return get_applicable_codes_hybrid(snippet_code, sections, section_to_patterns, use_ast=True)
        return match
    return get_applicable_codes


def run_validation(
    *,
    use_hybrid: bool = False,
    ground_truth_path: Path | None = None,
    report_txt_path: Path | None = None,
    report_csv_path: Path | None = None,
) -> int:
    gt_path = ground_truth_path or GROUND_TRUTH_PATH
    out_txt = report_txt_path or REPORT_TXT_PATH
    out_csv = report_csv_path or REPORT_CSV_PATH

    if not gt_path.exists():
        print(f"Error: {gt_path} not found. Fill gold_set_validation/ground_truth.csv first.")
        return 1

    gold_rows = load_ground_truth(gt_path)
    if not gold_rows:
        print("Error: No rows with both snippet_id and applicable_codes_ground_truth found.")
        return 1

    section_to_patterns = parse_codebook(CODEBOOK_PATH)
    all_sections = set(section_to_patterns.keys())
    match_fn = _matcher(use_hybrid)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    total_tp_all = total_fp_all = total_fn_all = 0
    total_tp_restricted = total_fp_restricted = total_fn_restricted = 0
    per_snippet = []

    for row in gold_rows:
        sid = row["snippet_id"]
        cur.execute("SELECT code FROM snippets WHERE id = ?", (sid,))
        r = cur.fetchone()
        if r is None:
            per_snippet.append({
                "snippet_id": sid,
                "error": "snippet not in DB",
                "precision": None,
                "recall": None,
                "f1": None,
                "precision_restricted": None,
                "recall_restricted": None,
                "f1_restricted": None,
                "n_predicted": None,
                "n_gold": None,
                "tp": None,
                "fp": None,
                "fn": None,
                "n_predicted_restricted": None,
                "tp_restricted": None,
                "fp_restricted": None,
                "fn_restricted": None,
            })
            continue
        code = (r["code"] or "") or ""
        gold = set(row["applicable_codes_ground_truth"])

        # Pass 1: all sections (same as full-dataset matcher)
        predicted_all = set(match_fn(code, all_sections, section_to_patterns))
        tp_all = len(predicted_all & gold)
        fp_all = len(predicted_all - gold)
        fn_all = len(gold - predicted_all)
        total_tp_all += tp_all
        total_fp_all += fp_all
        total_fn_all += fn_all
        prec_all = tp_all / (tp_all + fp_all) if (tp_all + fp_all) > 0 else 0.0
        rec_all = tp_all / (tp_all + fn_all) if (tp_all + fn_all) > 0 else 0.0
        f1_all = 2 * prec_all * rec_all / (prec_all + rec_all) if (prec_all + rec_all) > 0 else 0.0

        # Pass 2: permission-restricted sections
        permission_key = (row.get("permission") or "").strip().lower()
        sections_restricted = set(PERMISSION_TO_SECTIONS.get(permission_key, list(all_sections)))
        predicted_restricted = set(match_fn(code, sections_restricted, section_to_patterns))
        tp_r = len(predicted_restricted & gold)
        fp_r = len(predicted_restricted - gold)
        fn_r = len(gold - predicted_restricted)
        total_tp_restricted += tp_r
        total_fp_restricted += fp_r
        total_fn_restricted += fn_r
        prec_r = tp_r / (tp_r + fp_r) if (tp_r + fp_r) > 0 else 0.0
        rec_r = tp_r / (tp_r + fn_r) if (tp_r + fn_r) > 0 else 0.0
        f1_r = 2 * prec_r * rec_r / (prec_r + rec_r) if (prec_r + rec_r) > 0 else 0.0

        per_snippet.append({
            "snippet_id": sid,
            "error": None,
            "precision": round(prec_all, 4),
            "recall": round(rec_all, 4),
            "f1": round(f1_all, 4),
            "n_predicted": len(predicted_all),
            "n_gold": len(gold),
            "tp": tp_all,
            "fp": fp_all,
            "fn": fn_all,
            "precision_restricted": round(prec_r, 4),
            "recall_restricted": round(rec_r, 4),
            "f1_restricted": round(f1_r, 4),
            "n_predicted_restricted": len(predicted_restricted),
            "tp_restricted": tp_r,
            "fp_restricted": fp_r,
            "fn_restricted": fn_r,
        })

    conn.close()

    # Overall (micro-averaged) metrics — all sections
    denom_p_all = total_tp_all + total_fp_all
    denom_r_all = total_tp_all + total_fn_all
    overall_precision_all = total_tp_all / denom_p_all if denom_p_all > 0 else 0.0
    overall_recall_all = total_tp_all / denom_r_all if denom_r_all > 0 else 0.0
    overall_f1_all = (
        2 * overall_precision_all * overall_recall_all / (overall_precision_all + overall_recall_all)
        if (overall_precision_all + overall_recall_all) > 0
        else 0.0
    )
    # Overall — permission-restricted
    denom_p_r = total_tp_restricted + total_fp_restricted
    denom_r_r = total_tp_restricted + total_fn_restricted
    overall_precision_restricted = total_tp_restricted / denom_p_r if denom_p_r > 0 else 0.0
    overall_recall_restricted = total_tp_restricted / denom_r_r if denom_r_r > 0 else 0.0
    overall_f1_restricted = (
        2 * overall_precision_restricted * overall_recall_restricted
        / (overall_precision_restricted + overall_recall_restricted)
        if (overall_precision_restricted + overall_recall_restricted) > 0
        else 0.0
    )

    mode = "regex + AST hybrid" if use_hybrid else "regex/string"
    # Write report
    lines = [
        "=" * 60,
        "Gold-set validation: automated pattern matcher vs manual ground truth",
        "=" * 60,
        f"Matcher mode: {mode}",
        f"Ground truth file: {as_repo_relative(gt_path, BASE)}",
        f"Codebook: {as_repo_relative(CODEBOOK_PATH, BASE)}",
        f"Snippets in validation: {len(per_snippet)}",
        "",
        "--- Overall (micro-averaged, all sections) ---",
        f"  Precision: {overall_precision_all:.4f}",
        f"  Recall:    {overall_recall_all:.4f}",
        f"  F1:        {overall_f1_all:.4f}",
        f"  TP: {total_tp_all}, FP: {total_fp_all}, FN: {total_fn_all}",
        "",
        "--- Overall (micro-averaged, permission-restricted) ---",
        f"  Precision: {overall_precision_restricted:.4f}",
        f"  Recall:    {overall_recall_restricted:.4f}",
        f"  F1:        {overall_f1_restricted:.4f}",
        f"  TP: {total_tp_restricted}, FP: {total_fp_restricted}, FN: {total_fn_restricted}",
        "",
        "--- Per-snippet (all sections) ---",
    ]
    for s in per_snippet:
        if s["error"]:
            lines.append(f"  snippet_id={s['snippet_id']}: {s['error']}")
        else:
            lines.append(
                f"  snippet_id={s['snippet_id']}: P={s['precision']:.4f} R={s['recall']:.4f} F1={s['f1']:.4f} "
                f"(n_pred={s['n_predicted']}, n_gold={s['n_gold']}, TP={s['tp']}, FP={s['fp']}, FN={s['fn']})"
            )
    lines.append("")
    lines.append("=" * 60)

    report_text = "\n".join(lines)
    out_txt.write_text(report_text, encoding="utf-8")
    print(report_text)
    print(f"\nReport written to {as_repo_relative(out_txt, BASE)}")

    # Per-snippet CSV (all + restricted columns)
    csv_fieldnames = [
        "snippet_id", "precision", "recall", "f1",
        "n_predicted", "n_gold", "tp", "fp", "fn",
        "precision_restricted", "recall_restricted", "f1_restricted",
        "n_predicted_restricted", "tp_restricted", "fp_restricted", "fn_restricted",
        "error",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fieldnames, delimiter=";")
        writer.writeheader()
        for s in per_snippet:
            writer.writerow({
                "snippet_id": s["snippet_id"],
                "precision": s.get("precision") or "",
                "recall": s.get("recall") or "",
                "f1": s.get("f1") or "",
                "n_predicted": s.get("n_predicted") or "",
                "n_gold": s.get("n_gold") or "",
                "tp": s.get("tp") or "",
                "fp": s.get("fp") or "",
                "fn": s.get("fn") or "",
                "precision_restricted": s.get("precision_restricted") or "",
                "recall_restricted": s.get("recall_restricted") or "",
                "f1_restricted": s.get("f1_restricted") or "",
                "n_predicted_restricted": s.get("n_predicted_restricted") or "",
                "tp_restricted": s.get("tp_restricted") or "",
                "fp_restricted": s.get("fp_restricted") or "",
                "fn_restricted": s.get("fn_restricted") or "",
                "error": s.get("error") or "",
            })
    print(f"Per-snippet metrics written to {as_repo_relative(out_csv, BASE)}")

    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Gold-set validation for the pattern matcher.")
    ap.add_argument(
        "--hybrid",
        action="store_true",
        help="Use regex/string matching plus AST-assisted rules (Kotlin tree-sitter).",
    )
    args = ap.parse_args()
    return run_validation(use_hybrid=args.hybrid)


if __name__ == "__main__":
    raise SystemExit(main())
