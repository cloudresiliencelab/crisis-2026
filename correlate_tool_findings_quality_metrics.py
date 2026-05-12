#!/usr/bin/env python3
"""
Spearman correlations: detekt_findings / ktlint_violations vs quality markers.

Markers: NCL, CC, LLOC, LOC, SLOC, MI (same as paper).

Outputs:
  - Full-corpus correlation (all snippets, pairwise complete).
  - Per-LLM Spearman + Fisher-z aggregate across models (k=7).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BASE = Path(__file__).resolve().parent
_REPO_ROOT = BASE.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from repo_relative_path import as_repo_relative
METRICS_PATH = BASE / "analysis_output" / "kotlin_quality_metrics.csv"
OUT_FULL = BASE / "analysis_output" / "tool_findings_quality_correlations_full.csv"
OUT_BY_MODEL = BASE / "analysis_output" / "tool_findings_quality_correlations_by_model.csv"
OUT_AGG = BASE / "analysis_output" / "tool_findings_quality_correlations_aggregated.csv"

TOOL_COLS = ["detekt_findings", "ktlint_violations"]
QUALITY_MARKERS = ["NCL", "CC", "LLOC", "LOC", "SLOC", "MI"]


def _safe_atanh(r: float) -> float:
    r = max(min(float(r), 0.999999), -0.999999)
    return float(np.arctanh(r))


def fisher_aggregate(rs: list[float]) -> dict[str, float]:
    if not rs:
        return {
            "k": 0,
            "avg_r": np.nan,
            "se_z": np.nan,
            "se_r_delta": np.nan,
            "n_pos": 0,
            "n_neg": 0,
        }
    z_vals = np.array([_safe_atanh(r) for r in rs], dtype=float)
    k = int(len(z_vals))
    z_bar = float(np.mean(z_vals))
    sd = float(np.std(z_vals, ddof=1)) if k > 1 else 0.0
    se_z = float(sd / math.sqrt(k)) if k > 0 else np.nan
    r_bar = float(np.tanh(z_bar))
    se_r = abs(1.0 - r_bar**2) * se_z if np.isfinite(se_z) else np.nan
    rs_arr = np.array(rs, dtype=float)
    return {
        "k": k,
        "avg_r": r_bar,
        "se_z": se_z,
        "se_r_delta": se_r,
        "n_pos": int(np.sum(rs_arr > 0)),
        "n_neg": int(np.sum(rs_arr < 0)),
    }


def spearman_pair(x: pd.Series, y: pd.Series) -> tuple[float, float, int]:
    sub = pd.concat([x, y], axis=1).dropna()
    n = len(sub)
    if n < 3:
        return np.nan, np.nan, n
    if sub.iloc[:, 0].nunique() < 2 or sub.iloc[:, 1].nunique() < 2:
        return np.nan, np.nan, n
    r, p = stats.spearmanr(sub.iloc[:, 0].values, sub.iloc[:, 1].values)
    return float(r), float(p) if np.isfinite(p) else np.nan, n


def main() -> None:
    df = pd.read_csv(METRICS_PATH)
    rows_full: list[dict] = []
    rows_model: list[dict] = []
    rows_agg: list[dict] = []

    for tool in TOOL_COLS:
        for marker in QUALITY_MARKERS:
            r, p, n = spearman_pair(df[tool], df[marker])
            rows_full.append(
                {
                    "tool_metric": tool,
                    "quality_marker": marker,
                    "n_pairs": n,
                    "r_spearman": r,
                    "p_value": p,
                }
            )

            per_llm_rs: list[float] = []
            for llm, grp in df.groupby("llm", dropna=True):
                r_m, p_m, n_m = spearman_pair(grp[tool], grp[marker])
                rows_model.append(
                    {
                        "tool_metric": tool,
                        "quality_marker": marker,
                        "llm": llm,
                        "n_pairs": n_m,
                        "r_spearman": r_m,
                        "p_value": p_m,
                    }
                )
                if np.isfinite(r_m):
                    per_llm_rs.append(r_m)

            agg = fisher_aggregate(per_llm_rs)
            rows_agg.append(
                {
                    "tool_metric": tool,
                    "quality_marker": marker,
                    "k_models": agg["k"],
                    "avg_r_fisher_z": agg["avg_r"],
                    "se_r": agg["se_r_delta"],
                    "n_pos_models": agg["n_pos"],
                    "n_neg_models": agg["n_neg"],
                }
            )

    pd.DataFrame(rows_full).to_csv(OUT_FULL, index=False)
    pd.DataFrame(rows_model).to_csv(OUT_BY_MODEL, index=False)
    pd.DataFrame(rows_agg).to_csv(OUT_AGG, index=False)

    print(f"Wrote {as_repo_relative(OUT_FULL, _REPO_ROOT)}")
    print(f"Wrote {as_repo_relative(OUT_BY_MODEL, _REPO_ROOT)}")
    print(f"Wrote {as_repo_relative(OUT_AGG, _REPO_ROOT)}")


if __name__ == "__main__":
    main()
