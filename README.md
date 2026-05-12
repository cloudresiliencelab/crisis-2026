# Reproducibility

This folder contains raw data and a single entry script to reproduce the quantitative results and figures for the ESEM paper (PrivLLMs).

## Quick start

From this directory:

```bash
python3 run_analysis.py
```

Recommended environment variables for headless runs (CI, clusters):

```bash
export MPLBACKEND=Agg
export MPLCONFIGDIR="$(pwd)/.mpl-cache"
mkdir -p .mpl-cache
python3 run_analysis.py
```

### Primary outputs

| Output | Description |
|--------|-------------|
| `report.txt` | Full numeric summary: pattern statistics, regressions, gold validation (if `ground_truth.csv` present), Kotlin pipeline logs, **detekt/ktlint vs quality-marker Spearman** block, **sanity lines for tab:kotlin-quality-corr**, tool/heatmap subprocess logs |
| `figures/*.pdf` | Main paper-style figures (means, heatmaps, top codes, etc.) |
| `figures/fig_prompt_tool_heatmap.png` (and `.pdf` if seaborn run succeeds) | Mean detekt/ktlint counts by prompt type (duplicate copy under `_ESEM____PrivLLMs/figures/` with the same filenames) |
| `analysis_output/kotlin_quality_metrics.csv` | Per-snippet Kotlin markers (`NCL`, `CC`, `LLOC`, `LOC`, `SLOC`, `MI`, `detekt_findings`, `ktlint_violations`) |
| `analysis_output/prompt_quality_correlation_summary.csv` | Fisher-z aggregated prompt × marker correlations |
| `analysis_output/prompt_quality_correlation_by_model.csv` | Per-LLM Spearman correlations |
| `analysis_output/tool_findings_quality_correlations_full.csv` | Pooled Spearman: detekt/ktlint vs quality markers |
| `analysis_output/tool_findings_quality_correlations_by_model.csv` | Per-LLM tool × marker Spearman |
| `analysis_output/tool_findings_quality_correlations_aggregated.csv` | Fisher-z aggregate for tool × marker (where \(k>1\)) |

### Kotlin quality pipeline (also invoked by `run_analysis.py`)

```bash
python3 extract_kotlin_quality_metrics.py
python3 aggregate_prompt_quality_correlations.py
python3 correlate_tool_findings_quality_metrics.py
python3 plot_prompt_tool_heatmap.py --pdf
```

## Dependencies

- Python 3.10+ recommended
- `pandas`, `numpy`, `scipy`, `matplotlib`, `seaborn` (see `requirements.txt` in this folder)
- `statsmodels` (optional; Poisson / negative binomial in `report.txt`)
- `scikit-learn` (optional; weighted Cohen’s \(\kappa\); otherwise a fallback implementation is used)
- `detekt` and `ktlint` (**optional**): when both are on `PATH`, `kotlin_quality_metrics.csv` includes non-zero `detekt_findings` / `ktlint_violations` consistent with the paper’s Kotlin layer. Without them, columns may be zeroed; re-run after installing tools to match published counts.

## Contents

- `Codebook_en.csv` – codebook (patterns, severity, valence)
- `db.sqlite3` – snippet database (399 snippets)
- `ground_truth.csv` – gold-set labels for `run_analysis.py` (optional; validation block skipped if missing)
- `gold_set_validation/ground_truth.csv` – same gold set in the layout expected by `validate_gold_set.py`
- `annotations_all_snippets.csv` – per-snippet applicable codes for stochasticity / robustness scripts
- `requirements.txt` – Python dependencies (includes tree-sitter stack for hybrid matching)
- `run_analysis.py` – full pipeline
- `extract_kotlin_quality_metrics.py` – Kotlin quality markers
- `aggregate_prompt_quality_correlations.py` – prompt × marker Spearman + Fisher-\(z\) summary
- `correlate_tool_findings_quality_metrics.py` – detekt/ktlint × marker correlations
- `plot_prompt_tool_heatmap.py` – prompt-type × tool heatmap (PNG/PDF to `figures/` and paper `figures/`)
- `validate_gold_set.py`, `extend_annotations.py`, `permission_sections.py`, `repo_relative_path.py` – matcher + gold validation
- `stochasticity_and_ast_analysis/` – stochasticity proxy, hybrid matcher, AST helpers, and auxiliary gold-set analyses

The pipeline above reproduces **main RQ statistics**, Kotlin correlations, tool correlations, and the prompt×tool heatmap from this folder’s database.


- **Bootstrap + linear mixed models (stochasticity proxy)** — uses `annotations_all_snippets.csv`, `Codebook_en.csv`, and `db.sqlite3` included here.

  ```bash
  python3 stochasticity_and_ast_analysis/run_stochasticity_analysis.py
  ```

- **Hybrid (regex + AST) gold validation** — uses `gold_set_validation/ground_truth.csv`, `Codebook_en.csv`, and `db.sqlite3`.

  ```bash
  python3 stochasticity_and_ast_analysis/run_hybrid_validation.py
  python3 validate_gold_set.py --help
  ```
