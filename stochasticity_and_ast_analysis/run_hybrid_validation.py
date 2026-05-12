#!/usr/bin/env python3
"""
Compare gold-set validation metrics: regex-only vs regex+AST hybrid matcher.

Writes to this folder:
  - regex_validation_report.txt / .csv (copy for comparison)
  - hybrid_validation_report.txt / .csv
  - hybrid_vs_regex_summary.txt

Run from the reproducibility folder:
  python3 stochasticity_and_ast_analysis/run_hybrid_validation.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
PROJECT_ROOT = BASE.parent
sys.path.insert(0, str(PROJECT_ROOT))

from validate_gold_set import run_validation
from repo_relative_path import as_repo_relative


def _extract_overall(txt: str, label: str) -> dict[str, float]:
    """Parse Precision/Recall/F1 from validation report block."""
    out: dict[str, float] = {}
    block = label
    for line in txt.splitlines():
        if block in line or ("Precision:" in line and "--- Overall" in txt):
            pass
    sec = txt.split("--- Overall (micro-averaged, all sections) ---")
    if len(sec) > 1:
        chunk = sec[1].split("---")[0]
        m = re.search(r"Precision:\s*([\d.]+)", chunk)
        if m:
            out["precision_all"] = float(m.group(1))
        m = re.search(r"Recall:\s*([\d.]+)", chunk)
        if m:
            out["recall_all"] = float(m.group(1))
        m = re.search(r"F1:\s*([\d.]+)", chunk)
        if m:
            out["f1_all"] = float(m.group(1))
        m = re.search(r"TP:\s*(\d+),\s*FP:\s*(\d+),\s*FN:\s*(\d+)", chunk)
        if m:
            out["tp"] = int(m.group(1))
            out["fp"] = int(m.group(2))
            out["fn"] = int(m.group(3))
    return out


def main() -> int:
    regex_txt = BASE / "regex_validation_report.txt"
    regex_csv = BASE / "regex_validation_report.csv"
    hybrid_txt = BASE / "hybrid_validation_report.txt"
    hybrid_csv = BASE / "hybrid_validation_report.csv"

    rc1 = run_validation(use_hybrid=False, report_txt_path=regex_txt, report_csv_path=regex_csv)
    rc2 = run_validation(use_hybrid=True, report_txt_path=hybrid_txt, report_csv_path=hybrid_csv)
    if rc1 or rc2:
        return max(rc1, rc2)

    t1 = regex_txt.read_text(encoding="utf-8")
    t2 = hybrid_txt.read_text(encoding="utf-8")
    m1 = _extract_overall(t1, "")
    m2 = _extract_overall(t2, "")

    lines = [
        "=" * 60,
        "Gold set: regex-only vs regex+AST hybrid",
        "=" * 60,
        "",
        "--- Regex-only (all sections) ---",
        str(m1),
        "",
        "--- Hybrid ---",
        str(m2),
        "",
        "Notes: Hybrid adds Kotlin AST-assisted detection for selected codes",
        "(see pattern_detection_map.csv) on top of signature matching.",
        "=" * 60,
    ]
    summary_path = BASE / "hybrid_vs_regex_summary.txt"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWritten {as_repo_relative(summary_path, PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
