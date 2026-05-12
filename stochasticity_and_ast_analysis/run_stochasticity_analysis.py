#!/usr/bin/env python3
"""
Synthetic stochasticity analysis (no new LLM calls).

1. Ranks the 19 permission groups by dispersion (variance) of per-snippet metrics.
2. Selects a stratified subset (high / median / low variance).
3. Bootstraps within selected permissions to approximate sampling variability.
4. Fits linear mixed models (random intercept by permission) on stacked snippets.

Outputs (this folder):
  - permission_dispersion.csv
  - selected_permissions.txt
  - stochasticity_report.txt
  - stochasticity_by_cell.csv
  - mixed_effects_summary.txt
  - figures/stochasticity_bootstrap_sd.pdf

Run from the reproducibility folder:
  python3 stochasticity_and_ast_analysis/run_stochasticity_analysis.py
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
PROJECT_ROOT = BASE.parent
sys.path.insert(0, str(PROJECT_ROOT))

from repo_relative_path import as_repo_relative

ANNOTATIONS_PATH = PROJECT_ROOT / "annotations_all_snippets.csv"
CODEBOOK_PATH = PROJECT_ROOT / "Codebook_en.csv"

OUT_DISPERSION = BASE / "permission_dispersion.csv"
OUT_SELECTED = BASE / "selected_permissions.txt"
OUT_REPORT = BASE / "stochasticity_report.txt"
OUT_BY_CELL = BASE / "stochasticity_by_cell.csv"
OUT_MIXED = BASE / "mixed_effects_summary.txt"
FIG_DIR = BASE / "figures"

RNG_SEED = 42
N_BOOTSTRAP = 200
N_PERM_HIGH = 2
N_PERM_MED = 1
N_PERM_LOW = 2


def load_codebook_severity(path: Path) -> Dict[str, float]:
    code_to_severity: Dict[str, float] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=";")
        next(reader)
        for row in reader:
            if not row:
                continue
            code = (row[0] or "").strip()
            if not code or not re.match(r"^C\d+", code):
                continue
            sev = (row[4] or "").strip() if len(row) > 4 else ""
            try:
                code_to_severity[code] = float(sev)
            except ValueError:
                code_to_severity[code] = np.nan
    return code_to_severity


def load_annotations(path: Path) -> pd.DataFrame:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            sid = (row.get("snippet_id") or "").strip()
            try:
                sid = int(sid)
            except ValueError:
                continue
            codes_str = (row.get("applicable_codes") or "").strip().strip('"')
            codes = [c.strip() for c in codes_str.split(";") if c.strip()] if codes_str else []
            rows.append({
                "snippet_id": sid,
                "applicable_codes": codes,
                "llm": (row.get("llm") or "").strip(),
                "permission": (row.get("permission") or "").strip().lower(),
                "prompt_type": (row.get("prompt_type") or "").strip(),
            })
    return pd.DataFrame(rows)


def per_snippet_metrics(df: pd.DataFrame, code_to_severity: Dict[str, float]) -> pd.DataFrame:
    out = []
    for _, row in df.iterrows():
        codes = row["applicable_codes"] or []
        sevs = [code_to_severity.get(c, np.nan) for c in codes]
        sevs = [s for s in sevs if not np.isnan(s)]
        n_sev4 = sum(1 for s in sevs if s >= 4.0)
        out.append({
            "snippet_id": row["snippet_id"],
            "permission": row["permission"],
            "prompt_type": row["prompt_type"],
            "llm": row["llm"],
            "n_applicable_codes": len(codes),
            "n_severity4": n_sev4,
            "max_severity": float(max(sevs)) if sevs else np.nan,
        })
    return pd.DataFrame(out)


def select_stratified_permissions(disp_df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    disp_sorted = disp_df.sort_values("score_var", ascending=False).reset_index(drop=True)
    n = len(disp_sorted)
    high = disp_sorted.loc[: N_PERM_HIGH - 1, "permission"].tolist() if n >= N_PERM_HIGH else disp_sorted["permission"].tolist()
    low = disp_sorted.loc[n - N_PERM_LOW :, "permission"].tolist() if n >= N_PERM_LOW else []
    mid_idx = max(0, n // 2)
    med = [disp_sorted.loc[mid_idx, "permission"]]

    selected: List[str] = []
    for p in high + med + low:
        if p and p not in selected:
            selected.append(p)

    labels = []
    for tag, plist in [("high_variance", high), ("median", med), ("low_variance", low)]:
        for p in plist:
            labels.append(f"{tag}:{p}")
    return labels, selected


def bootstrap_prompt_type_means(df_perm: pd.DataFrame, n_boot: int, rng: np.random.Generator) -> pd.DataFrame:
    """Bootstrap-resample rows with replacement; compute mean n_codes and mean max_severity per prompt_type."""
    pt_order = ["friendly", "neutral", "invasive"]
    rows_out = []
    n_rows = len(df_perm)
    idx = np.arange(n_rows)
    for b in range(n_boot):
        samp = rng.choice(idx, size=n_rows, replace=True)
        sample_df = df_perm.iloc[samp]
        for pt in pt_order:
            sub = sample_df[sample_df["prompt_type"] == pt]
            rows_out.append({
                "bootstrap_iter": b,
                "prompt_type": pt,
                "mean_n_codes": float(sub["n_applicable_codes"].mean()) if len(sub) else np.nan,
                "mean_max_severity": float(sub["max_severity"].mean()) if len(sub) else np.nan,
                "n_rows": len(sub),
            })
    return pd.DataFrame(rows_out)


def main() -> int:
    if not ANNOTATIONS_PATH.exists():
        print(f"Missing {as_repo_relative(ANNOTATIONS_PATH, PROJECT_ROOT)}", file=sys.stderr)
        return 1

    code_to_sev = load_codebook_severity(CODEBOOK_PATH)
    df_raw = load_annotations(ANNOTATIONS_PATH)
    df = per_snippet_metrics(df_raw, code_to_sev)

    disp_rows = []
    for perm, g in df.groupby("permission"):
        if len(g) < 2:
            continue
        v_codes = float(np.var(g["n_applicable_codes"], ddof=1))
        ok_max = g["max_severity"].notna().sum()
        v_max = (
            float(np.nanvar(g["max_severity"], ddof=1))
            if ok_max > 1
            else 0.0
        )
        disp_rows.append({
            "permission": perm,
            "n_snippets": len(g),
            "var_n_codes": v_codes,
            "var_max_severity": v_max,
            "score_var": v_codes + v_max,
            "sd_n_codes": float(np.std(g["n_applicable_codes"], ddof=1)),
            "sd_max_severity": float(np.nanstd(g["max_severity"], ddof=1)),
        })
    disp_df = pd.DataFrame(disp_rows).sort_values("score_var", ascending=False)
    BASE.mkdir(parents=True, exist_ok=True)
    disp_df.to_csv(OUT_DISPERSION, index=False)

    labels, selected = select_stratified_permissions(disp_df)
    OUT_SELECTED.write_text(
        "\n".join(
            ["Stratified permission selection (by var(n_codes)+var(max_severity)):", ""]
            + labels
            + ["", "Unique selected (analysis):", ""]
            + selected
        ),
        encoding="utf-8",
    )

    rng = np.random.default_rng(RNG_SEED)
    cell_rows = []
    boot_summaries: List[Tuple[str, pd.DataFrame]] = []

    for perm in selected:
        df_p = df[df["permission"] == perm].copy()
        if len(df_p) == 0:
            continue
        boot_df = bootstrap_prompt_type_means(df_p, N_BOOTSTRAP, rng)
        boot_df["permission"] = perm
        boot_summaries.append((perm, boot_df))

        # SD across bootstrap iterations for each prompt_type (distribution of cell-level bootstrap means)
        for pt in ["friendly", "neutral", "invasive"]:
            sub = boot_df[boot_df["prompt_type"] == pt]
            sd_c = float(sub["mean_n_codes"].std(ddof=1)) if len(sub) > 1 else 0.0
            sd_m = float(sub["mean_max_severity"].std(ddof=1)) if len(sub) > 1 else 0.0
            cell_rows.append({
                "permission": perm,
                "aggregate": "bootstrap_sd_of_prompt_type_mean",
                "prompt_type": pt,
                "llm": "",
                "sd_mean_n_codes_across_iterations": sd_c,
                "sd_mean_max_severity_across_iterations": sd_m,
            })

        # Per (prompt_type, llm) cell: bootstrap SD of the sample mean across iterations
        # when resampling rows with replacement (21 snippets per permission)
        cell_keys = df_p[["prompt_type", "llm"]].drop_duplicates()
        n_rows = len(df_p)
        for _, ck in cell_keys.iterrows():
            pt, lm = ck["prompt_type"], ck["llm"]
            means_c = []
            means_m = []
            for _ in range(N_BOOTSTRAP):
                samp = rng.choice(np.arange(n_rows), size=n_rows, replace=True)
                sample_df = df_p.iloc[samp]
                sub = sample_df[(sample_df["prompt_type"] == pt) & (sample_df["llm"] == lm)]
                means_c.append(float(sub["n_applicable_codes"].mean()) if len(sub) else np.nan)
                means_m.append(float(sub["max_severity"].mean()) if len(sub) else np.nan)
            cell_rows.append({
                "permission": perm,
                "aggregate": "cell_bootstrap_sd",
                "prompt_type": pt,
                "llm": lm,
                "sd_mean_n_codes_across_iterations": float(np.nanstd(means_c, ddof=1)),
                "sd_mean_max_severity_across_iterations": float(np.nanstd(means_m, ddof=1)),
            })

    pd.DataFrame(cell_rows).to_csv(OUT_BY_CELL, index=False)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        nfig = max(1, len(boot_summaries))
        fig, axes = plt.subplots(1, nfig, figsize=(4 * nfig, 3.5), squeeze=False)
        for ax, (perm, sdf) in zip(axes[0], boot_summaries):
            pts = ["friendly", "neutral", "invasive"]
            sds = []
            for pt in pts:
                col = sdf[sdf["prompt_type"] == pt]["mean_n_codes"]
                sds.append(float(col.std(ddof=1)) if len(col) > 1 else 0.0)
            ax.bar(range(len(pts)), sds, tick_label=[p.capitalize() for p in pts])
            ax.set_title(perm[:30])
            ax.set_ylabel("SD of bootstrap mean n_codes")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "stochasticity_bootstrap_sd.pdf", bbox_inches="tight")
        plt.close()
    except Exception:
        pass

    df_mix = df[df["permission"].isin(selected)].copy()
    df_mix = df_mix.dropna(subset=["prompt_type", "llm", "permission"])
    df_mix_sev = df_mix.dropna(subset=["max_severity"])
    lines_mixed = []
    try:
        import statsmodels.formula.api as smf
        model = smf.mixedlm(
            "n_applicable_codes ~ C(prompt_type) + C(llm)",
            df_mix,
            groups=df_mix["permission"],
        )
        fit = model.fit(method="lbfgs", maxiter=500)
        lines_mixed.append(str(fit.summary()))
        if len(df_mix_sev) >= 10:
            model2 = smf.mixedlm(
                "max_severity ~ C(prompt_type) + C(llm)",
                df_mix_sev,
                groups=df_mix_sev["permission"],
            )
            fit2 = model2.fit(method="lbfgs", maxiter=500)
            lines_mixed.append(str(fit2.summary()))
        else:
            lines_mixed.append("MixedLM (max_severity): skipped (insufficient non-null rows).")
    except Exception as e:
        lines_mixed.append(f"MixedLM failed: {e}")

    OUT_MIXED.write_text("\n\n".join(lines_mixed), encoding="utf-8")

    report_lines = [
        "=" * 60,
        "Synthetic stochasticity analysis (bootstrap resampling; no new LLM calls)",
        "=" * 60,
        f"Annotations: {as_repo_relative(ANNOTATIONS_PATH, PROJECT_ROOT)}",
        f"Bootstrap iterations per permission: {N_BOOTSTRAP}",
        f"RNG seed: {RNG_SEED}",
        "",
        "--- Selected permissions (stratified) ---",
        "\n".join(selected),
        "",
        "--- Dispersion table (top 5 by score_var) ---",
        disp_df.head(5).to_string(index=False),
        "",
        "--- Interpretation ---",
        "Joint bootstrap resamples the N snippets within each permission with replacement.",
        "SD columns quantify instability of bootstrap means for prompt types and cells.",
        "MixedLM: random intercept by permission on stacked snippets from selected groups.",
        "=" * 60,
    ]
    OUT_REPORT.write_text("\n".join(report_lines), encoding="utf-8")
    print("\n".join(report_lines))
    print(
        "\nWrote: "
        f"{as_repo_relative(OUT_REPORT, PROJECT_ROOT)}, "
        f"{as_repo_relative(OUT_DISPERSION, PROJECT_ROOT)}, "
        f"{as_repo_relative(OUT_BY_CELL, PROJECT_ROOT)}, "
        f"{as_repo_relative(OUT_MIXED, PROJECT_ROOT)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
