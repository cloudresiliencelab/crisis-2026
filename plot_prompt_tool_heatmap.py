#!/usr/bin/env python3
"""
Create a heatmap for Kotlin tool findings by prompt type.

Rows:
  - detekt_findings
  - ktlint_violations

Columns:
  - friendly
  - neutral
  - invasive

Cell annotation: mean +/- SE and sample size.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Headless-safe matplotlib before pyplot
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent
_REPO_ROOT = ROOT.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from repo_relative_path import as_repo_relative

DEFAULT_PAPER_FIG_DIR = ROOT.parent / "_ESEM____PrivLLMs" / "figures"
DEFAULT_BUNDLE_FIG_DIR = ROOT / "figures"
CSV_PATH = ROOT / "analysis_output" / "kotlin_quality_metrics.csv"
STEM = "fig_prompt_tool_heatmap"

PROMPT_ORDER = ["friendly", "neutral", "invasive"]
METRICS = ["detekt_findings", "ktlint_violations"]
METRIC_LABELS = {
    "detekt_findings": "Detekt Findings",
    "ktlint_violations": "ktlint Violations",
}


def mean_se(values: pd.Series) -> tuple[float, float]:
    arr = values.dropna().to_numpy(dtype=float)
    if len(arr) == 0:
        return np.nan, np.nan
    mean = float(np.mean(arr))
    if len(arr) == 1:
        return mean, 0.0
    se = float(np.std(arr, ddof=1) / np.sqrt(len(arr)))
    return mean, se


def build_figure(df: pd.DataFrame):
    df = df[df["prompt_type"].isin(PROMPT_ORDER)].copy()

    means = pd.DataFrame(index=METRICS, columns=PROMPT_ORDER, dtype=float)
    ses = pd.DataFrame(index=METRICS, columns=PROMPT_ORDER, dtype=float)
    ns = pd.DataFrame(index=METRICS, columns=PROMPT_ORDER, dtype=int)

    for metric in METRICS:
        for prompt in PROMPT_ORDER:
            vals = df.loc[df["prompt_type"] == prompt, metric]
            m, se = mean_se(vals)
            means.loc[metric, prompt] = m
            ses.loc[metric, prompt] = se
            ns.loc[metric, prompt] = int(vals.dropna().shape[0])

    color_vals = means.copy()
    for metric in METRICS:
        row = color_vals.loc[metric, :]
        mu = float(np.nanmean(row.to_numpy(dtype=float)))
        sigma = float(np.nanstd(row.to_numpy(dtype=float), ddof=0))
        if sigma > 0:
            color_vals.loc[metric, :] = (row - mu) / sigma
        else:
            color_vals.loc[metric, :] = 0.0

    annot = means.copy().astype(str)
    for metric in METRICS:
        for prompt in PROMPT_ORDER:
            annot.loc[metric, prompt] = (
                f"{means.loc[metric, prompt]:.2f} +/- {ses.loc[metric, prompt]:.2f}\n"
                f"(n={ns.loc[metric, prompt]})"
            )

    plot_df = color_vals.rename(index=METRIC_LABELS).rename(
        columns={k: k.title() for k in PROMPT_ORDER}
    )
    annot_df = annot.rename(index=METRIC_LABELS).rename(
        columns={k: k.title() for k in PROMPT_ORDER}
    )

    sns.set_style("white")
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    hm = sns.heatmap(
        plot_df,
        cmap="coolwarm",
        center=0.0,
        vmin=-1.6,
        vmax=1.6,
        linewidths=0.7,
        linecolor="white",
        annot=annot_df,
        fmt="",
        annot_kws={"fontsize": 9},
        cbar_kws={"label": "Row-standardized score (z)"},
        ax=ax,
    )

    ax.set_title(
        "Kotlin Static Analysis by Prompt Type\n"
        "(Mean +/- SE across snippets)",
        fontsize=13,
        pad=12,
    )
    ax.set_xlabel("Prompt Type", fontsize=11)
    ax.set_ylabel("Tool Metric", fontsize=11)
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)
    hm.collections[0].colorbar.ax.tick_params(labelsize=9)

    fig.tight_layout()
    return fig


def save_figure(fig, path_png: Path, write_pdf: bool) -> None:
    path_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_png, bbox_inches="tight", format="png")
    if write_pdf:
        fig.savefig(path_png.with_suffix(".pdf"), bbox_inches="tight", format="pdf")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prompt type x detekt/ktlint heatmap.")
    p.add_argument(
        "--csv",
        type=Path,
        default=CSV_PATH,
        help="kotlin_quality_metrics.csv path",
    )
    p.add_argument(
        "--paper-dir",
        type=Path,
        default=DEFAULT_PAPER_FIG_DIR,
        help="Paper figures directory (PNG/PDF basename fig_prompt_tool_heatmap)",
    )
    p.add_argument(
        "--bundle-dir",
        type=Path,
        default=DEFAULT_BUNDLE_FIG_DIR,
        help="Reproducibility bundle figures directory (same filenames)",
    )
    p.add_argument(
        "--pdf",
        action="store_true",
        help="Also write PDF next to PNG in each output directory",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = args.csv if args.csv.is_absolute() else (ROOT / args.csv)
    if not csv_path.exists():
        raise SystemExit(f"Missing metrics CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    fig = build_figure(df)

    out_dirs = [args.paper_dir, args.bundle_dir]
    written: list[str] = []
    for d in out_dirs:
        png = d / f"{STEM}.png"
        save_figure(fig, png, args.pdf)
        written.append(str(png))
        if args.pdf:
            written.append(str(png.with_suffix(".pdf")))

    plt.close(fig)
    for w in written:
        print(f"Wrote {as_repo_relative(Path(w), _REPO_ROOT)}")


if __name__ == "__main__":
    main()
