#!/usr/bin/env python3
"""
Identify gold-set snippets where predicted thematic sections != gold sections,
and aggregate recurring extra/missed section "clusters".

Uses the same matcher as main validation: regex get_applicable_codes, all sections.

Outputs:
  thematic_mismatch_report.txt
  thematic_mismatch_by_snippet.csv
  thematic_mismatch_pattern_counts.csv

Run from project root:
  python stochasticity_and_ast_analysis/thematic_mismatch_analysis.py
"""

from __future__ import annotations

import csv
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

BASE = Path(__file__).resolve().parent
PROJECT_ROOT = BASE.parent
sys.path.insert(0, str(PROJECT_ROOT))

from extend_annotations import build_code_to_section, get_applicable_codes, parse_codebook
from repo_relative_path import as_repo_relative

CODEBOOK_PATH = PROJECT_ROOT / "Codebook_en.csv"
DB_PATH = PROJECT_ROOT / "db.sqlite3"
GROUND_TRUTH_PATH = PROJECT_ROOT / "gold_set_validation" / "ground_truth.csv"
OUT_REPORT = BASE / "thematic_mismatch_report.txt"
OUT_SNIPPETS = BASE / "thematic_mismatch_by_snippet.csv"
OUT_PATTERNS = BASE / "thematic_mismatch_pattern_counts.csv"


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
                "permission_tag": (row.get("permission") or "").strip().lower(),
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
            out.add(f"(unknown:{c})")
    return out


def main() -> int:
    rows = load_ground_truth(GROUND_TRUTH_PATH)
    section_to_patterns = parse_codebook(CODEBOOK_PATH)
    all_sections = set(section_to_patterns.keys())
    code_to_section = build_code_to_section(section_to_patterns)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    extra_section_counts: Counter = Counter()
    missed_section_counts: Counter = Counter()
    pattern_counts: Counter = Counter()
    snippet_records = []

    for row in rows:
        sid = row["snippet_id"]
        cur.execute("SELECT code FROM snippets WHERE id = ?", (sid,))
        r = cur.fetchone()
        if r is None:
            continue
        code_text = (r["code"] or "") or ""
        gold = set(row["applicable_codes_ground_truth"])
        pred = set(get_applicable_codes(code_text, all_sections, section_to_patterns))

        G = codes_to_sections(gold, code_to_section)
        P = codes_to_sections(pred, code_to_section)

        extra = P - G
        missed = G - P

        if extra or missed:
            key = (
                ";".join(sorted(extra)) if extra else "(none)",
                ";".join(sorted(missed)) if missed else "(none)",
            )
            pattern_counts[key] += 1
            for s in extra:
                extra_section_counts[s] += 1
            for s in missed:
                missed_section_counts[s] += 1

            snippet_records.append({
                "snippet_id": sid,
                "permission_tag": row["permission_tag"],
                "gold_sections": ";".join(sorted(G)),
                "pred_sections": ";".join(sorted(P)),
                "extra_themes": ";".join(sorted(extra)) if extra else "",
                "missed_themes": ";".join(sorted(missed)) if missed else "",
                "n_gold_codes": len(gold),
                "n_pred_codes": len(pred),
            })

    conn.close()

    # Pair-level confusion: when exactly one extra and one missed (simple directional confusion)
    pair_counts: Counter = Counter()
    for rec in snippet_records:
        ex = [x for x in rec["extra_themes"].split(";") if x]
        ms = [x for x in rec["missed_themes"].split(";") if x]
        for e in ex:
            for m in ms:
                pair_counts[(e, m)] += 1

    lines = [
        "=" * 76,
        "Thematic (section-level) mismatches on gold set — regex matcher, all sections",
        "=" * 76,
        f"Snippets with perfect thematic agreement (P_sections == G_sections): "
        f"{len(rows) - len(snippet_records)} / {len(rows)}",
        f"Snippets with at least one extra or missed theme: {len(snippet_records)}",
        "",
        "--- Sections most often predicted but not in gold (spurious themes) ---",
    ]
    for sec, c in extra_section_counts.most_common():
        lines.append(f"  {sec}: {c}")

    lines.extend(["", "--- Sections most often in gold but not predicted (missed themes) ---"])
    for sec, c in missed_section_counts.most_common():
        lines.append(f"  {sec}: {c}")

    lines.extend(["", "--- Recurring (extra_themes, missed_themes) patterns ---" ])
    for (ex_key, ms_key), c in pattern_counts.most_common(25):
        lines.append(f"  [{c}x] extra={{ {ex_key} }}  missed={{ {ms_key} }}")

    lines.extend(["", "--- Directional pairs (extra_section -> missed_section), snippet counts ---"])
    for (e, m), c in pair_counts.most_common(20):
        lines.append(f"  {e}  vs  {m}: {c}")

    lines.extend(["", "=" * 76])

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")

    with open(OUT_SNIPPETS, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "snippet_id", "permission_tag", "gold_sections", "pred_sections",
                "extra_themes", "missed_themes", "n_gold_codes", "n_pred_codes",
            ],
        )
        w.writeheader()
        for rec in snippet_records:
            w.writerow(rec)

    with open(OUT_PATTERNS, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["extra_themes_key", "missed_themes_key", "n_snippets"])
        for (ex_key, ms_key), c in pattern_counts.most_common():
            w.writerow([ex_key, ms_key, c])

    print("\n".join(lines))
    print(
        "\nWrote "
        f"{as_repo_relative(OUT_REPORT, PROJECT_ROOT)}, "
        f"{as_repo_relative(OUT_SNIPPETS, PROJECT_ROOT)}, "
        f"{as_repo_relative(OUT_PATTERNS, PROJECT_ROOT)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
