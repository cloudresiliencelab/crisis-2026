#!/usr/bin/env python3
"""
Aggregate prompt-characteristic × quality-marker correlations.

Computes per-LLM Spearman correlations and Fisher-z aggregated summaries.
"""

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats

BASE = Path(__file__).resolve().parent
_REPO_ROOT = BASE.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from repo_relative_path import as_repo_relative
METRICS_PATH = BASE / "analysis_output" / "kotlin_quality_metrics.csv"
OUTPUT_PATH = BASE / "analysis_output" / "prompt_quality_correlation_summary.csv"

QUALITY_MARKERS = ["NCL", "CC", "LLOC", "LOC", "SLOC", "MI"]


def _safe_atanh(r: float) -> float:
    r = max(min(float(r), 0.999999), -0.999999)
    return float(np.arctanh(r))


def fisher_aggregate(rs: List[float]) -> Dict[str, float]:
    if not rs:
        return {
            "k": 0,
            "avg_r": np.nan,
            "avg_z": np.nan,
            "se_z": np.nan,
            "se_r_delta": np.nan,
            "n_pos": 0,
            "n_neg": 0,
            "n_zero": 0,
        }
    z_vals = np.array([_safe_atanh(r) for r in rs], dtype=float)
    k = int(len(z_vals))
    z_bar = float(np.mean(z_vals))
    sd = float(np.std(z_vals, ddof=1)) if k > 1 else 0.0
    se_z = float(sd / math.sqrt(k)) if k > 0 else np.nan
    r_bar = float(np.tanh(z_bar))
    # Delta method: var(r) ~ (1 - r^2)^2 var(z)
    se_r = abs(1.0 - r_bar ** 2) * se_z if np.isfinite(se_z) else np.nan
    rs_arr = np.array(rs, dtype=float)
    return {
        "k": k,
        "avg_r": r_bar,
        "avg_z": z_bar,
        "se_z": se_z,
        "se_r_delta": se_r,
        "n_pos": int(np.sum(rs_arr > 0)),
        "n_neg": int(np.sum(rs_arr < 0)),
        "n_zero": int(np.sum(rs_arr == 0)),
    }


def one_cell_correlations(df: pd.DataFrame, x_col: str, y_col: str) -> List[dict]:
    rows = []
    for llm, grp in df.groupby("llm", dropna=True):
        sub = grp[[x_col, y_col]].dropna()
        if len(sub) < 3:
            continue
        if sub[x_col].nunique() < 2 or sub[y_col].nunique() < 2:
            continue
        r, p = stats.spearmanr(sub[x_col].values, sub[y_col].values)
        if np.isnan(r):
            continue
        rows.append(
            {
                "llm": llm,
                "n_pairs": int(len(sub)),
                "r_spearman": float(r),
                "p_value": float(p) if np.isfinite(p) else np.nan,
            }
        )
    return rows


def build_prompt_characteristics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["prompt_type_friendly"] = (out["prompt_type"] == "friendly").astype(int)
    out["prompt_type_invasive"] = (out["prompt_type"] == "invasive").astype(int)
    out["prompt_type_ordinal"] = out["prompt_type"].map({"friendly": 0, "neutral": 1, "invasive": 2})
    return out


def main():
    parser = argparse.ArgumentParser(description="Aggregate quality correlations by prompt characteristics.")
    parser.add_argument("--metrics", type=Path, default=METRICS_PATH, help="Input metrics CSV")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH, help="Output summary CSV")
    args = parser.parse_args()

    metrics_path = args.metrics if args.metrics.is_absolute() else (BASE / args.metrics)
    output_path = args.output if args.output.is_absolute() else (BASE / args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(metrics_path)
    df = build_prompt_characteristics(df)

    prompt_vars = [
        "prompt_length_chars",
        "prompt_length_words",
        "prompt_type_ordinal",
        "prompt_type_friendly",
        "prompt_type_invasive",
    ]
    prompt_vars = [c for c in prompt_vars if c in df.columns]
    quality_markers = [c for c in QUALITY_MARKERS if c in df.columns]

    per_model_records = []
    summary_records = []

    for prompt_var in prompt_vars:
        for marker in quality_markers:
            model_rows = one_cell_correlations(df, prompt_var, marker)
            for r in model_rows:
                per_model_records.append(
                    {
                        "prompt_characteristic": prompt_var,
                        "quality_marker": marker,
                        **r,
                    }
                )
            agg = fisher_aggregate([r["r_spearman"] for r in model_rows])
            summary_records.append(
                {
                    "prompt_characteristic": prompt_var,
                    "quality_marker": marker,
                    "method": "spearman_fisher_z",
                    "k_models": agg["k"],
                    "avg_correlation_r": agg["avg_r"],
                    "std_error_z": agg["se_z"],
                    "std_error_r_delta": agg["se_r_delta"],
                    "n_positive": agg["n_pos"],
                    "n_negative": agg["n_neg"],
                    "n_zero": agg["n_zero"],
                }
            )

    summary_df = pd.DataFrame(summary_records)
    per_model_df = pd.DataFrame(per_model_records)
    summary_df.to_csv(output_path, index=False)
    per_model_out = output_path.with_name("prompt_quality_correlation_by_model.csv")
    per_model_df.to_csv(per_model_out, index=False)
    print(f"Wrote summary: {as_repo_relative(output_path, _REPO_ROOT)}")
    print(f"Wrote per-model correlations: {as_repo_relative(per_model_out, _REPO_ROOT)}")


if __name__ == "__main__":
    main()
