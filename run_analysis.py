#!/usr/bin/env python3
"""
Reproducibility script: from raw data (Codebook_en.csv, db.sqlite3, optional ground_truth.csv)
in this folder, recomputes all numbers and figures from the paper.
Output: report.txt, figures/*.pdf. No verification messages—results only.
Dependencies: pandas, numpy, scipy, matplotlib, seaborn (heatmap), statsmodels (optional for Poisson/NB).
"""

import csv
import os
import re
import sqlite3
import subprocess
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Optional, Any

import pandas as pd
import numpy as np
from scipy import stats

BASE = Path(__file__).resolve().parent
CODEBOOK_PATH = BASE / "Codebook_en.csv"
DB_PATH = BASE / "db.sqlite3"
GROUND_TRUTH_PATH = BASE / "ground_truth.csv"
REPORT_PATH = BASE / "report.txt"
FIGURES_DIR = BASE / "figures"
ANALYSIS_OUTPUT_DIR = BASE / "analysis_output"
ANALYSIS_OUTPUT_DIR.mkdir(exist_ok=True)

LLM_TO_REGION = {
    "chatgpt": "USA",
    "gemini": "USA",
    "github-copilot": "USA",
    "baidu": "China",
    "deepseek": "China",
    "qwen": "China",
    "falcon3": "UAE",
}



_LOCATION_REQUEST_SUBSTRINGS = [
    "requestPermissions", "requestLocationUpdates", "getLastKnownLocation",
    "FusedLocationProviderClient", "checkSelfPermission",
    "ACCESS_FINE_LOCATION", "ACCESS_COARSE_LOCATION",
]
_RATIONALE_OPTIN_SUBSTRINGS = [
    "shouldShowRequestPermissionRationale", "Rationale", "Snackbar", "AlertDialog",
    "explain", "why we need", "purpose", "reason for", "before requesting", "reason to use",
]


def _has_location_request_without_rationale(snippet_code: str) -> bool:
    code_lower = snippet_code.lower()
    has_request = any(s.lower() in code_lower for s in _LOCATION_REQUEST_SUBSTRINGS)
    has_rationale = any(s.lower() in code_lower for s in _RATIONALE_OPTIN_SUBSTRINGS)
    return bool(has_request and not has_rationale)


def extract_signatures_from_example(example: str, code_name: str = "") -> list:
    sigs = []
    text = (example or "").replace('""', '"')
    if code_name:
        name_clean = code_name.strip()
        if re.match(r"^[A-Z][A-Z0-9_]+$", name_clean):
            sigs.append(name_clean)
        if "permission" in name_clean.lower() or "_" in name_clean:
            sigs.append(name_clean)
    if not text.strip() or text.strip() in ("–", "-"):
        return list(dict.fromkeys(sigs))
    for m in re.finditer(r"android\.permission\.([A-Z_0-9]+)", text, re.IGNORECASE):
        tok = m.group(1).upper()
        sigs.append("android.permission." + tok)
        sigs.append(tok)
    for m in re.finditer(r"uses-permission\s+android:name=\"([^\"]+)\"", text, re.IGNORECASE):
        sigs.append(m.group(1))
    for m in re.finditer(r"android\.hardware\.(\w+)", text, re.IGNORECASE):
        sigs.append("android.hardware." + m.group(1))
    for pattern, min_len in [
        (r"getLastKnownLocation|requestLocationUpdates|FusedLocationProviderClient", 5),
        (r"PRIORITY_HIGH_ACCURACY|PRIORITY_BALANCED_POWER_ACCURACY", 5),
        (r"getFilesDir|getExternalFilesDir|openFileOutput|MODE_PRIVATE|filesDir|cacheDir", 5),
        (r"READ_EXTERNAL_STORAGE|WRITE_EXTERNAL_STORAGE|MANAGE_EXTERNAL_STORAGE", 5),
        (r"requestLegacyExternalStorage", 5),
        (r"Log\.d\s*\(|Log\.i\s*\(|Log\.e\s*\(", 5),
        (r"SharedPreferences|EncryptedSharedPreferences", 5),
        (r"sendBroadcast\s*\(|LocalBroadcastManager", 5),
        (r"UUID\.randomUUID|nameUUIDFromBytes|ANDROID_ID", 5),
        (r"Build\.(MODEL|MANUFACTURER|FINGERPRINT|HARDWARE)", 5),
        (r"READ_PHONE_STATE|TelephonyManager|\.imei|line1Number", 5),
        (r"listenUsingRfcommWithServiceRecord|BLUETOOTH_SCAN|BLUETOOTH_CONNECT|BLUETOOTH_ADMIN|BLUETOOTH\b", 5),
        (r"device\.address|device\.name", 5),
        (r"registerReceiver\s*\(|BroadcastReceiver", 5),
        (r"startForeground\s*\(|createBond\s*\(", 5),
        (r"SYSTEM_ALERT_WINDOW|TYPE_APPLICATION_OVERLAY|FLAG_NOT_FOCUSABLE", 5),
        (r"RECEIVE_BOOT_COMPLETED", 5),
        (r"ACCESS_NOTIFICATION_POLICY|setInterruptionFilter|WRITE_SETTINGS", 5),
        (r"SET_SYSTEM_TIME|SET_DATETIME", 5),
        (r"READ_SMS|RECEIVE_SMS|SmsManager|SMS_RECEIVED", 5),
        (r"READ_PHONE_NUMBERS|READ_CONTACTS|READ_CALL_LOG|CALL_PHONE|ANSWER_PHONE_CALLS", 5),
        (r"WRITE_CALL_LOG|loadCallLogs", 5),
        (r"Intent\.ACTION_PICK|ContactsContract", 5),
        (r"RECORD_AUDIO|AudioRecord|DIRECTORY_MUSIC", 5),
        (r"CLIPBOARD_READ|primaryClip|addPrimaryClipChangedListener", 5),
        (r"Settings\.Secure\.ANDROID_ID|Formatter\.formatIpAddress|getActiveNetworkInfo", 5),
        (r"PACKAGE_USAGE_STATS|HttpLoggingInterceptor|setLevel\s*\(\s*BODY", 5),
        (r"BIND_ACCESSIBILITY_SERVICE|canRetrieveWindowContent|performGlobalAction|dispatchGesture", 5),
        (r"typeAllMask|accessibilityEventTypes", 5),
        (r"BODY_SENSORS|TYPE_ACCELEROMETER|WAKE_LOCK", 5),
        (r"CAMERA", 3),
        (r"exported\s*=\s*[\"']true[\"']", 5),
        (r"FileOutputStream|FileWriter|getExternalFilesDir\s*\(", 5),
        (r"/storage/emulated|emulated/0", 5),
    ]:
        for m in re.finditer(pattern, text):
            s = (m.group(0) if m.lastindex is None else m.group(1) or m.group(0)).strip()
            if len(s) >= min_len and s not in sigs:
                sigs.append(s)
    return list(dict.fromkeys(sigs))


def parse_codebook(path: Path) -> dict:
    section_to_patterns = {}
    current_section = ""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=";")
        next(reader)
        for row in reader:
            if not row:
                continue
            code = (row[0] or "").strip()
            if code and not re.match(r"^C\d+", code):
                current_section = code
                if current_section not in section_to_patterns:
                    section_to_patterns[current_section] = []
                continue
            if not code.startswith("C") or not code[1:].isdigit():
                continue
            code_name = (row[6] or "").strip() if len(row) > 6 else ""
            example = (row[10] or "").strip() if len(row) > 10 else ""
            sigs = extract_signatures_from_example(example, code_name)
            if code_name and code_name not in sigs:
                sigs.append(code_name)
            if not sigs and example and example not in ("–", "-"):
                short = re.sub(r"\s+", " ", example.replace('""', '"'))[:60]
                if len(short) > 12:
                    sigs.append(short)
            if current_section:
                section_to_patterns.setdefault(current_section, []).append((code, sigs))
    return section_to_patterns


def get_applicable_codes(
    snippet_code: str,
    sections: set,
    section_to_patterns: dict,
) -> list:
    applicable = []
    for section in sections:
        patterns = section_to_patterns.get(section, [])
        for code_id, sigs in patterns:
            if not sigs:
                continue
            for sig in sigs:
                if sig in snippet_code:
                    applicable.append(code_id)
                    break
                if len(sig) > 3 and sig.upper() in snippet_code.upper():
                    applicable.append(code_id)
                    break
        if section == "Location" and _has_location_request_without_rationale(snippet_code):
            applicable.append("C008")
            applicable.append("C009")
    return sorted(set(applicable))


# --- Snippet metadata and annotation ---

def get_snippet_tags(conn) -> dict:
    cur = conn.cursor()
    cur.execute("""
        SELECT s.id, t.name as tag_name
        FROM snippets s
        JOIN snippets_tags st ON s.id = st.snippet_id
        JOIN tags t ON t.id = st.tag_id
    """)
    snippet_tags = defaultdict(set)
    for (sid, tag) in cur.fetchall():
        snippet_tags[sid].add(tag)
    return dict(snippet_tags)


def resolve_metadata(snippet_tags: dict) -> dict:
    llms = {"chatgpt", "gemini", "deepseek", "github-copilot", "qwen", "baidu", "falcon3"}

    def get_llm(tags):
        for llm in llms:
            if llm in tags or any(t.startswith(llm + "-") for t in tags):
                return llm
        return None

    def get_prompt_type(tags):
        for t in tags:
            if t.endswith("-friendly"):
                return "friendly"
            if t.endswith("-invasive"):
                return "invasive"
        return "neutral"

    def get_permission(tags):
        skip = llms | {"kotlin", "xml", "java, xml", "kottlin", "kotkin", "andorid"}
        skip |= {t for t in tags if "-friendly" in t or "-invasive" in t}
        perms = [p for p in tags if p not in skip]
        return perms[0] if perms else None

    return {
        sid: {"llm": get_llm(tags), "permission": get_permission(tags), "prompt_type": get_prompt_type(tags)}
        for sid, tags in snippet_tags.items()
    }


def annotate_all_snippets(conn, section_to_patterns: dict) -> pd.DataFrame:
    all_sections = set(section_to_patterns.keys())
    cur = conn.cursor()
    cur.execute("SELECT id, code FROM snippets")
    rows = []
    metadata = resolve_metadata(get_snippet_tags(conn))
    for (sid, code) in cur.fetchall():
        code = code or ""
        meta = metadata.get(sid, {})
        applicable = get_applicable_codes(code, all_sections, section_to_patterns)
        rows.append({
            "snippet_id": sid,
            "applicable_codes": applicable,
            "llm": meta.get("llm"),
            "permission": meta.get("permission"),
            "prompt_type": meta.get("prompt_type"),
        })
    return pd.DataFrame(rows)


# --- Codebook IRR and taxonomy ---

def _cohen_kappa_quadratic(y1, y2):
    try:
        from sklearn.metrics import cohen_kappa_score
        return float(cohen_kappa_score(y1, y2, weights="quadratic"))
    except ImportError:
        y1, y2 = np.asarray(y1), np.asarray(y2)
        n = len(y1)
        if n == 0:
            return np.nan
        labels = np.unique(np.r_[y1, y2])
        n_cat = len(labels)
        label_to_idx = {int(l): i for i, l in enumerate(labels)}
        w = np.ones((n_cat, n_cat))
        for i in range(n_cat):
            for j in range(n_cat):
                w[i, j] = 1 - ((i - j) ** 2) / ((n_cat - 1) ** 2) if n_cat > 1 else 1.0
        O = np.zeros((n_cat, n_cat))
        for i in range(n):
            a = label_to_idx.get(int(y1[i]), 0)
            b = label_to_idx.get(int(y2[i]), 0)
            O[a, b] += 1
        Po = np.sum(w * O) / n
        Pe = 0
        for i in range(n_cat):
            for j in range(n_cat):
                Pe += w[i, j] * (O[i, :].sum() / n) * (O[:, j].sum() / n)
        if Pe >= 1:
            return 0.0
        return float((Po - Pe) / (1 - Pe))


def load_codebook_for_irr_and_taxonomy(path: Path) -> Tuple[pd.DataFrame, dict]:
    """Returns (df with Code, Section, Valence, Severity_Oskar, Severity_Amine, Severity_Final), section_to_patterns."""
    rows = []
    current_section = ""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=";")
        next(reader)
        for row in reader:
            if not row:
                continue
            code = (row[0] or "").strip()
            if code and not re.match(r"^C\d+", code):
                current_section = code
                continue
            if not code or not re.match(r"^C\d+", code):
                continue
            valence = (row[1] or "").strip() if len(row) > 1 else ""
            sev_oskar = (row[2] or "").strip() if len(row) > 2 else ""
            sev_amine = (row[3] or "").strip() if len(row) > 3 else ""
            sev_final = (row[4] or "").strip() if len(row) > 4 else ""
            def to_num(s):
                try:
                    return float(s)
                except ValueError:
                    return np.nan
            rows.append({
                "Code": code,
                "Section": current_section,
                "Valence": valence,
                "Severity_Oskar": to_num(sev_oskar),
                "Severity_Amine": to_num(sev_amine),
                "Severity_Final": to_num(sev_final),
            })
    return pd.DataFrame(rows), None


def irr_and_taxonomy(codebook_df: pd.DataFrame) -> Tuple[dict, dict]:
    out_irr = {}
    valid = codebook_df[["Severity_Oskar", "Severity_Amine"]].dropna()
    valid = valid[(valid["Severity_Oskar"].notna()) & (valid["Severity_Amine"].notna())]
    if len(valid) >= 2:
        o = valid["Severity_Oskar"].round().astype(int).clip(1, 4).values
        a = valid["Severity_Amine"].round().astype(int).clip(1, 4).values
        out_irr["n"] = len(valid)
        out_irr["Cohen_Kappa_weighted_quadratic"] = _cohen_kappa_quadratic(o, a)
        out_irr["Exact_agreement_rate"] = (o == a).mean()
    else:
        out_irr["n"] = len(valid)
        out_irr["Cohen_Kappa_weighted_quadratic"] = None
        out_irr["Exact_agreement_rate"] = None

    out_tax = {}
    out_tax["n_patterns"] = len(codebook_df)
    out_tax["Severity_Final_distribution"] = codebook_df["Severity_Final"].dropna().round().astype(int).value_counts().sort_index().to_dict()
    out_tax["Valence_distribution"] = codebook_df["Valence"].value_counts().to_dict()
    out_tax["Patterns_per_section"] = codebook_df.groupby("Section").size().to_dict()
    return out_irr, out_tax


# --- Load codebook severity/valence for analysis ---

def load_codebook_severity(path: Path) -> dict:
    code_to_severity = {}
    current_section = ""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=";")
        next(reader)
        for row in reader:
            if not row:
                continue
            code = (row[0] or "").strip()
            if not code:
                continue
            if not re.match(r"^C\d+", code):
                current_section = code
                continue
            sev = (row[4] or "").strip() if len(row) > 4 else ""
            try:
                code_to_severity[code] = float(sev)
            except ValueError:
                code_to_severity[code] = np.nan
    return code_to_severity


def load_codebook_valence(path: Path) -> dict:
    code_to_valence = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=";")
        next(reader)
        for row in reader:
            if not row:
                continue
            code = (row[0] or "").strip()
            if not code or not re.match(r"^C\d+", code):
                continue
            valence = (row[1] or "").strip() if len(row) > 1 else ""
            if valence in ("Positive", "Negative"):
                code_to_valence[code] = valence
    return code_to_valence


def load_snippet_code_lengths(db_path: Path) -> Dict[int, int]:
    if not db_path.exists():
        return {}
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT id, length(code) AS code_len FROM snippets")
    id_to_len = {int(r[0]): int(r[1]) for r in cur.fetchall()}
    conn.close()
    return id_to_len


def pattern_level_analyses(
    df: pd.DataFrame, code_to_severity: dict, code_to_valence: Optional[dict] = None
) -> dict:
    code_to_valence = code_to_valence or {}
    results = []
    for _, row in df.iterrows():
        codes = row.get("applicable_codes") or []
        severities = [code_to_severity.get(c, np.nan) for c in codes]
        severities = [s for s in severities if not np.isnan(s)]
        n_severity4 = sum(1 for s in severities if s >= 4)
        n_positive = sum(1 for c in codes if code_to_valence.get(c) == "Positive")
        n_negative = sum(1 for c in codes if code_to_valence.get(c) == "Negative")
        results.append({
            "snippet_id": row["snippet_id"],
            "llm": row.get("llm") or None,
            "prompt_type": row.get("prompt_type") or None,
            "permission": row.get("permission") or None,
            "n_applicable_codes": len(codes),
            "n_severity4": n_severity4,
            "n_positive": n_positive,
            "n_negative": n_negative,
            "max_severity": max(severities) if severities else np.nan,
            "mean_severity": float(np.mean(severities)) if severities else np.nan,
        })
    res_df = pd.DataFrame(results)
    res_df["region"] = res_df["llm"].map(LLM_TO_REGION)
    out = {}
    for group in ["prompt_type", "llm", "permission", "region"]:
        valid = res_df[res_df[group].notna()]
        if len(valid) == 0:
            continue
        agg = valid.groupby(group).agg(
            n_snippets=("snippet_id", "count"),
            mean_n_codes=("n_applicable_codes", "mean"),
            mean_n_severity4=("n_severity4", "mean"),
            mean_max_severity=("max_severity", "mean"),
            mean_mean_severity=("mean_severity", "mean"),
            mean_n_positive=("n_positive", "mean"),
            mean_n_negative=("n_negative", "mean"),
        ).round(3)
        out[f"by_{group}"] = agg.to_dict(orient="index")
    out["per_snippet_df"] = res_df
    return out


def dunn_posthoc_bonferroni(
    res_df: pd.DataFrame,
    value_col: str,
    group_col: str = "prompt_type",
    order: List[str] = None,
) -> Dict[Tuple[str, str], float]:
    """Dunn's post-hoc test with Bonferroni correction. Returns (group_a, group_b) -> adjusted p-value."""
    order = order or ["friendly", "neutral", "invasive"]
    df = res_df[[group_col, value_col]].dropna()
    if len(df) < 6:
        return {}
    ranks = df[value_col].rank()
    n = len(df)
    n_pairs = 3  # friendly-neutral, friendly-invasive, neutral-invasive
    from scipy.stats import norm
    results = {}
    for i, ga in enumerate(order):
        for gb in order[i + 1 :]:
            a = df[df[group_col] == ga][value_col].values
            b = df[df[group_col] == gb][value_col].values
            if len(a) == 0 or len(b) == 0:
                continue
            ra = ranks[df[group_col] == ga].mean()
            rb = ranks[df[group_col] == gb].mean()
            se = np.sqrt((n * (n + 1) / 12.0) * (1.0 / len(a) + 1.0 / len(b)))
            if se <= 0:
                continue
            z = abs(ra - rb) / se
            p = 2.0 * (1.0 - norm.cdf(z))
            p_adj = min(1.0, p * n_pairs)
            results[(ga, gb)] = p_adj
    return results


def bootstrap_ci(values: np.ndarray, stat="mean", n_bootstrap: int = 2000, ci: float = 0.95) -> Tuple[float, float, float]:
    n = len(values)
    if n == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(42)
    if stat == "mean":
        point = float(np.nanmean(values))
        boot = rng.choice(values, size=(n_bootstrap, n), replace=True)
        boot_means = np.nanmean(boot, axis=1)
    else:
        point = float(np.nanmean(values.astype(float)))
        boot = rng.choice(values.astype(float), size=(n_bootstrap, n), replace=True)
        boot_means = np.nanmean(boot, axis=1)
    lo = float(np.percentile(boot_means, (1 - ci) / 2 * 100))
    hi = float(np.percentile(boot_means, (1 + ci) / 2 * 100))
    return (point, lo, hi)


def pattern_frequency(df: pd.DataFrame, top_n: int = 25) -> List[Tuple[str, int]]:
    counter = Counter()
    for codes in df["applicable_codes"]:
        for c in codes or []:
            counter[c] += 1
    return counter.most_common(top_n)


def plot_figures(agg_out: dict, df: pd.DataFrame, fig_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    def fig_path(name: str) -> Path:
        return fig_dir / f"{name}.pdf"

    by_pt = agg_out.get("by_prompt_type") or {}
    order_pt = ["friendly", "neutral", "invasive"]
    labels_pt = [x.capitalize() for x in order_pt]
    x = np.arange(len(labels_pt))
    width = 0.35
    n_codes_pt = [by_pt.get(p, {}).get("mean_n_codes", 0) for p in order_pt]
    max_sev_pt = [by_pt.get(p, {}).get("mean_max_severity", 0) for p in order_pt]
    fig, ax1 = plt.subplots(figsize=(4.5, 3.2))
    ax1.bar(x - width / 2, n_codes_pt, width, label="Mean no. codes", color="steelblue")
    ax1.set_ylabel("Mean number of applicable codes")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels_pt)
    ax1.legend(loc="upper left")
    ax2 = ax1.twinx()
    ax2.bar(x + width / 2, max_sev_pt, width, label="Mean max severity", color="coral", alpha=0.8)
    ax2.set_ylabel("Mean max severity (1--4)")
    ax2.set_ylim(0, 4.5)
    ax2.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(fig_path("fig_prompt_type"), bbox_inches="tight")
    plt.close()

    if by_pt and any(by_pt.get(p, {}).get("mean_rate_per_1k_chars") is not None for p in order_pt):
        rate_pt = [by_pt.get(p, {}).get("mean_rate_per_1k_chars", 0) or 0 for p in order_pt]
        fig_rate, ax_rate = plt.subplots(figsize=(4, 3))
        ax_rate.bar(labels_pt, rate_pt, color="steelblue")
        ax_rate.set_ylabel("Mean patterns per 1000 chars")
        ax_rate.set_xlabel("Prompt type")
        fig_rate.tight_layout()
        fig_rate.savefig(fig_path("fig_rate_per_1k_chars_by_prompt_type"), bbox_inches="tight")
        plt.close()

    by_llm = agg_out.get("by_llm") or {}
    llm_order = ["chatgpt", "deepseek", "falcon3", "gemini", "github-copilot", "qwen", "baidu"]
    llm_labels = ["ChatGPT", "DeepSeek", "Falcon", "Gemini", "Copilot", "Qwen", "Baidu"]
    n_codes_llm = [by_llm.get(l, {}).get("mean_n_codes", 0) for l in llm_order]
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.bar(llm_labels, n_codes_llm, color="steelblue")
    ax.set_ylabel("Mean number of applicable codes")
    ax.set_xlabel("LLM")
    plt.xticks(rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(fig_path("fig_mean_codes_by_llm"), bbox_inches="tight")
    plt.close()

    by_region = agg_out.get("by_region") or {}
    region_order = ["USA", "China", "UAE"]
    x_r = np.arange(len(region_order))
    width_r = 0.35
    n_codes_r = [by_region.get(r, {}).get("mean_n_codes", 0) for r in region_order]
    max_sev_r = [by_region.get(r, {}).get("mean_max_severity", 0) for r in region_order]
    fig_r, ax1_r = plt.subplots(figsize=(4, 3.2))
    ax1_r.bar(x_r - width_r / 2, n_codes_r, width_r, color="steelblue")
    ax1_r.set_ylabel("Mean number of applicable codes")
    ax1_r.set_xticks(x_r)
    ax1_r.set_xticklabels(region_order)
    ax2_r = ax1_r.twinx()
    ax2_r.bar(x_r + width_r / 2, max_sev_r, width_r, color="coral", alpha=0.8)
    ax2_r.set_ylabel("Mean max severity (1--4)")
    ax2_r.set_ylim(0, 4.5)
    fig_r.tight_layout()
    fig_r.savefig(fig_path("fig_mean_codes_and_severity_by_region"), bbox_inches="tight")
    plt.close()

    by_perm = agg_out.get("by_permission") or {}
    perms_sorted = sorted(by_perm.keys(), key=lambda p: by_perm[p].get("mean_max_severity", 0), reverse=True)
    sev_perm = [by_perm[p].get("mean_max_severity", 0) for p in perms_sorted]
    fig, ax = plt.subplots(figsize=(5, 6))
    ax.barh(range(len(perms_sorted)), sev_perm, color="coral", alpha=0.8)
    ax.set_yticks(range(len(perms_sorted)))
    ax.set_yticklabels(perms_sorted, fontsize=9)
    ax.set_xlabel("Mean max severity (1--4)")
    ax.set_xlim(0, 4.5)
    fig.tight_layout()
    fig.savefig(fig_path("fig_mean_max_severity_by_permission"), bbox_inches="tight")
    plt.close()

    top = pattern_frequency(df, 15)
    codes_top = [c for c, _ in top]
    counts_top = [n for _, n in top]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.barh(range(len(codes_top)), counts_top, color="steelblue")
    ax.set_yticks(range(len(codes_top)))
    ax.set_yticklabels(codes_top)
    ax.set_xlabel("Number of snippets with pattern")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(fig_path("fig_top_pattern_codes"), bbox_inches="tight")
    plt.close()

    res_df = agg_out.get("per_snippet_df")
    if res_df is not None and len(res_df) > 0:
        pt_order = ["friendly", "neutral", "invasive"]
        pt_labels = ["Friendly", "Neutral", "Invasive"]
        pivot_llm_pt = res_df.groupby(["llm", "prompt_type"])["max_severity"].mean().unstack(fill_value=np.nan)
        pivot_llm_pt = pivot_llm_pt.reindex(index=llm_order, columns=pt_order)
        mat = pivot_llm_pt.values
        fig, ax = plt.subplots(figsize=(4, 4.5))
        im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=2.5, vmax=4.0)
        ax.set_xticks(range(len(pt_labels)))
        ax.set_xticklabels(pt_labels)
        ax.set_yticks(range(len(llm_labels)))
        ax.set_yticklabels(llm_labels)
        ax.set_xlabel("Prompt type")
        ax.set_ylabel("LLM")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9, color="black" if v < 3.3 else "white")
        plt.colorbar(im, ax=ax, label="Mean max severity")
        fig.tight_layout()
        fig.savefig(fig_path("fig_heatmap_llm_prompt_type"), bbox_inches="tight")
        plt.close()

        perm_order = sorted(res_df["permission"].dropna().unique(), key=lambda p: res_df[res_df["permission"] == p]["max_severity"].mean(), reverse=True)
        pivot_perm_pt = res_df.groupby(["permission", "prompt_type"])["max_severity"].mean().unstack(fill_value=np.nan)
        pivot_perm_pt = pivot_perm_pt.reindex(index=perm_order, columns=pt_order)
        mat2 = pivot_perm_pt.values
        fig2, ax2 = plt.subplots(figsize=(4.5, 7))
        im2 = ax2.imshow(mat2, aspect="auto", cmap="YlOrRd", vmin=2.2, vmax=4.0)
        ax2.set_xticks(range(len(pt_labels)))
        ax2.set_xticklabels(pt_labels)
        ax2.set_yticks(range(len(perm_order)))
        ax2.set_yticklabels(perm_order, fontsize=8)
        ax2.set_xlabel("Prompt type")
        ax2.set_ylabel("Permission")
        plt.colorbar(im2, ax=ax2, label="Mean max severity")
        fig2.tight_layout()
        fig2.savefig(fig_path("fig_heatmap_permission_prompt_type"), bbox_inches="tight")
        plt.close()

        pivot_n = res_df.groupby(["llm", "prompt_type"])["n_applicable_codes"].mean().unstack(fill_value=np.nan)
        pivot_n = pivot_n.reindex(index=llm_order, columns=pt_order)
        mat_n = pivot_n.values
        fig3, ax3 = plt.subplots(figsize=(4, 4.5))
        im3 = ax3.imshow(mat_n, aspect="auto", cmap="Blues", vmin=0, vmax=12)
        ax3.set_xticks(range(len(pt_labels)))
        ax3.set_xticklabels(pt_labels)
        ax3.set_yticks(range(len(llm_labels)))
        ax3.set_yticklabels(llm_labels)
        ax3.set_xlabel("Prompt type")
        ax3.set_ylabel("LLM")
        for i in range(mat_n.shape[0]):
            for j in range(mat_n.shape[1]):
                v = mat_n[i, j]
                if not np.isnan(v):
                    ax3.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=9, color="white" if v > 6 else "black")
        plt.colorbar(im3, ax=ax3, label="Mean no. applicable codes")
        fig3.tight_layout()
        fig3.savefig(fig_path("fig_heatmap_llm_prompt_type_n_codes"), bbox_inches="tight")
        plt.close()


def run_kotlin_quality_correlation_pipeline(base_dir: Path) -> List[str]:
    logs: List[str] = []
    extractor = base_dir / "extract_kotlin_quality_metrics.py"
    aggregator = base_dir / "aggregate_prompt_quality_correlations.py"
    if not extractor.exists() or not aggregator.exists():
        logs.append("Kotlin quality-correlation pipeline skipped: helper scripts not found.")
        return logs
    try:
        r1 = subprocess.run(
            ["python3", str(extractor)],
            cwd=str(base_dir),
            capture_output=True,
            text=True,
            check=False,
        )
        logs.append(f"extract_kotlin_quality_metrics.py exit={r1.returncode}")
        if r1.stdout.strip():
            logs.append(f"  {r1.stdout.strip().splitlines()[-1]}")
        if r1.returncode != 0 and r1.stderr.strip():
            logs.append(f"  stderr: {r1.stderr.strip().splitlines()[-1]}")
        r2 = subprocess.run(
            ["python3", str(aggregator)],
            cwd=str(base_dir),
            capture_output=True,
            text=True,
            check=False,
        )
        logs.append(f"aggregate_prompt_quality_correlations.py exit={r2.returncode}")
        if r2.stdout.strip():
            logs.append(f"  {r2.stdout.strip().splitlines()[-1]}")
        if r2.returncode != 0 and r2.stderr.strip():
            logs.append(f"  stderr: {r2.stderr.strip().splitlines()[-1]}")
    except Exception as exc:
        logs.append(f"Kotlin quality-correlation pipeline failed: {exc}")
    return logs


def run_correlate_tool_findings_pipeline(base_dir: Path) -> List[str]:
    logs: List[str] = []
    script = base_dir / "correlate_tool_findings_quality_metrics.py"
    metrics_csv = base_dir / "analysis_output" / "kotlin_quality_metrics.csv"
    if not script.exists():
        logs.append("correlate_tool_findings_quality_metrics.py not found; skipped.")
        return logs
    if not metrics_csv.exists():
        logs.append("kotlin_quality_metrics.csv missing; tool correlations skipped.")
        return logs
    try:
        r = subprocess.run(
            ["python3", str(script)],
            cwd=str(base_dir),
            capture_output=True,
            text=True,
            check=False,
        )
        logs.append(f"correlate_tool_findings_quality_metrics.py exit={r.returncode}")
        if r.stdout.strip():
            for line in r.stdout.strip().splitlines():
                logs.append(f"  {line}")
        if r.returncode != 0 and r.stderr.strip():
            logs.append(f"  stderr: {r.stderr.strip().splitlines()[-1]}")
    except Exception as exc:
        logs.append(f"correlate_tool_findings_quality_metrics failed: {exc}")
    return logs


def run_plot_prompt_tool_heatmap(base_dir: Path) -> List[str]:
    logs: List[str] = []
    script = base_dir / "plot_prompt_tool_heatmap.py"
    metrics_csv = base_dir / "analysis_output" / "kotlin_quality_metrics.csv"
    if not script.exists():
        logs.append("plot_prompt_tool_heatmap.py not found; skipped.")
        return logs
    if not metrics_csv.exists():
        logs.append("kotlin_quality_metrics.csv missing; heatmap skipped.")
        return logs
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("MPLCONFIGDIR", str(base_dir / ".mpl-cache"))
    try:
        r = subprocess.run(
            ["python3", str(script), "--pdf"],
            cwd=str(base_dir),
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
        logs.append(f"plot_prompt_tool_heatmap.py exit={r.returncode}")
        if r.stdout.strip():
            for line in r.stdout.strip().splitlines():
                logs.append(f"  {line}")
        if r.returncode != 0 and r.stderr.strip():
            logs.append(f"  stderr: {r.stderr.strip()}")
    except Exception as exc:
        logs.append(f"plot_prompt_tool_heatmap failed: {exc}")
    return logs


def append_tool_and_kotlin_sanity_sections(base_dir: Path, report_path: Path) -> None:
    """Append tool Spearman table and Kotlin tab:kotlin-quality-corr sanity lines to report.txt."""
    blocks: List[str] = []

    full_csv = base_dir / "analysis_output" / "tool_findings_quality_correlations_full.csv"
    if full_csv.exists():
        tdf = pd.read_csv(full_csv)
        lines = ["\n=== Tool findings vs quality markers (Spearman, full corpus) ==="]
        for _, row in tdf.iterrows():
            r_v = row["r_spearman"]
            p_v = row["p_value"]
            rs = f"{float(r_v):.6f}" if pd.notna(r_v) else "nan"
            ps = f"{float(p_v):.6e}" if pd.notna(p_v) else "nan"
            lines.append(
                f"  {row['tool_metric']} x {row['quality_marker']}: "
                f"n={row['n_pairs']}, rho={rs}, p={ps}"
            )
        blocks.append("\n".join(lines))

    summary_csv = base_dir / "analysis_output" / "prompt_quality_correlation_summary.csv"
    pairs = [
        ("prompt_type_ordinal", "LLOC"),
        ("prompt_type_ordinal", "CC"),
        ("prompt_type_ordinal", "MI"),
        ("prompt_type_invasive", "LLOC"),
        ("prompt_type_invasive", "CC"),
        ("prompt_type_friendly", "LLOC"),
        ("prompt_type_friendly", "CC"),
        ("prompt_length_words", "LOC"),
        ("prompt_length_words", "SLOC"),
    ]
    if summary_csv.exists():
        sdf = pd.read_csv(summary_csv)
        lines = [
            "\n=== Kotlin correlation table sanity "
            "(compare to tab:kotlin-quality-corr; rounded in paper) ==="
        ]
        for pc, qm in pairs:
            sub = sdf[(sdf["prompt_characteristic"] == pc) & (sdf["quality_marker"] == qm)]
            if len(sub) == 1:
                row = sub.iloc[0]
                lines.append(
                    f"  {pc} x {qm}: r_bar={row['avg_correlation_r']:.6f}, "
                    f"SE_r_delta={row['std_error_r_delta']:.6f}, "
                    f"k={int(float(row['k_models']))}, pos={int(row['n_positive'])}, neg={int(row['n_negative'])}"
                )
            else:
                lines.append(f"  {pc} x {qm}: row missing (n_matches={len(sub)})")
        blocks.append("\n".join(lines))

    if blocks:
        with report_path.open("a", encoding="utf-8") as fp:
            fp.write("\n".join(blocks) + "\n")


def main():
    FIGURES_DIR.mkdir(exist_ok=True)
    report = []

    report.append("=== Codebook IRR and taxonomy ===")
    codebook_df, _ = load_codebook_for_irr_and_taxonomy(CODEBOOK_PATH)
    irr, tax = irr_and_taxonomy(codebook_df)
    if irr.get("Cohen_Kappa_weighted_quadratic") is not None:
        report.append(f"Cohen's kappa (quadratic weights): {irr['Cohen_Kappa_weighted_quadratic']:.2f}")
        report.append(f"Exact agreement rate: {irr['Exact_agreement_rate']*100:.1f}%")
    report.append(f"n_patterns: {tax['n_patterns']}")
    report.append(f"Valence: {tax['Valence_distribution']}")
    report.append(f"Severity Final: {tax['Severity_Final_distribution']}")
    report.append(f"Patterns per section (top): {dict(sorted(tax['Patterns_per_section'].items(), key=lambda x: -x[1])[:8])}")

    report.append("\n=== Gold-set validation ===")
    if GROUND_TRUTH_PATH.exists():
        gold_rows = []
        with open(GROUND_TRUTH_PATH, newline="", encoding="utf-8") as f:
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
                gt_codes = [c.strip() for c in gt_str.split(";") if c.strip()]
                if not gt_codes:
                    continue
                gold_rows.append({"snippet_id": snippet_id, "applicable_codes_ground_truth": gt_codes})
        if gold_rows:
            section_to_patterns = parse_codebook(CODEBOOK_PATH)
            all_sections = set(section_to_patterns.keys())
            conn = sqlite3.connect(DB_PATH)
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            tp = fp = fn = 0
            for row in gold_rows:
                cur.execute("SELECT code FROM snippets WHERE id = ?", (row["snippet_id"],))
                r = cur.fetchone()
                if r is None:
                    continue
                code = (r["code"] or "") or ""
                gold = set(row["applicable_codes_ground_truth"])
                pred = set(get_applicable_codes(code, all_sections, section_to_patterns))
                tp += len(pred & gold)
                fp += len(pred - gold)
                fn += len(gold - pred)
            conn.close()
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            report.append(f"Gold set: n={len(gold_rows)}; precision={prec:.2f}, recall={rec:.2f}, F1={f1:.2f}; TP={tp}, FP={fp}, FN={fn}")
    else:
        report.append("ground_truth.csv not found; validation skipped.")

    report.append("\n=== Full-dataset pattern analysis ===")
    conn = sqlite3.connect(DB_PATH)
    section_to_patterns = parse_codebook(CODEBOOK_PATH)
    df = annotate_all_snippets(conn, section_to_patterns)
    conn.close()

    report.append(f"Total snippets: {len(df)}")
    n_with_codes = (df["applicable_codes"].apply(len) > 0).sum()
    report.append(f"Snippets with at least one applicable code: {n_with_codes}")
    report.append(f"Snippets with zero applicable codes: {len(df) - n_with_codes}")

    n_codes = df["applicable_codes"].apply(len)
    report.append("\n--- Distribution of number of applicable codes per snippet ---")
    report.append(f"  mean: {n_codes.mean():.2f}")
    report.append(f"  median: {n_codes.median():.0f}")
    report.append(f"  min: {n_codes.min()}, max: {n_codes.max()}")
    for q in [0.25, 0.5, 0.75]:
        report.append(f"  {int(q*100)}%: {n_codes.quantile(q):.0f}")

    report.append("\n--- Top 25 most frequent pattern codes ---")
    for code, count in pattern_frequency(df, 25):
        report.append(f"  {code}: {count}")

    code_to_severity = load_codebook_severity(CODEBOOK_PATH)
    code_to_valence = load_codebook_valence(CODEBOOK_PATH)
    agg_out = pattern_level_analyses(df, code_to_severity, code_to_valence)
    res_df = agg_out.pop("per_snippet_df", None)

    id_to_len = load_snippet_code_lengths(DB_PATH)
    if res_df is not None and id_to_len:
        res_df["code_length"] = res_df["snippet_id"].map(id_to_len)
        res_df["rate_per_1k_chars"] = np.where(
            (res_df["code_length"] > 0),
            res_df["n_applicable_codes"] / (res_df["code_length"] / 1000.0),
            np.nan,
        )
        valid_rate = res_df["rate_per_1k_chars"].notna()
        report.append("\n--- Length-normalized pattern density (patterns per 1000 chars) ---")
        for group_col, label in [("prompt_type", "by prompt_type"), ("llm", "by LLM"), ("region", "by region")]:
            if group_col not in res_df.columns:
                continue
            report.append(f"  {label}:")
            for grp_name, grp in res_df[valid_rate].groupby(group_col, dropna=False):
                mean_r = grp["rate_per_1k_chars"].mean()
                n = len(grp)
                report.append(f"    {grp_name}: n={n}, mean_rate_per_1k_chars={mean_r:.3f}")
                key = f"by_{group_col}"
                if key in agg_out and grp_name in agg_out[key]:
                    agg_out[key][grp_name]["mean_rate_per_1k_chars"] = round(mean_r, 3)

        report.append("\n--- Mean code length (characters) by prompt_type ---")
        for pt in ["friendly", "neutral", "invasive"]:
            grp = res_df[res_df["prompt_type"] == pt]
            if len(grp) > 0:
                mean_chars = grp["code_length"].mean()
                report.append(f"  {pt}: n={len(grp)}, mean_code_length={mean_chars:.0f}")

        report.append("\n--- Inferential tests (Kruskal-Wallis) ---")
        order_pt = ["friendly", "neutral", "invasive"]
        k_groups = 3
        n_total = len(res_df)
        groups_n = [res_df[res_df["prompt_type"] == p]["n_applicable_codes"].values for p in order_pt]
        groups_n = [g for g in groups_n if len(g) > 0]
        if len(groups_n) >= 2:
            h_n, p_n = stats.kruskal(*groups_n)
            eta_sq_n = (h_n - (k_groups - 1)) / (n_total - k_groups) if (n_total - k_groups) > 0 else np.nan
            report.append(f"  n_applicable_codes by prompt_type: H={h_n:.4f}, p={p_n:.2e}, eta^2_H={eta_sq_n:.2f}")
        groups_sev = [res_df[res_df["prompt_type"] == p]["max_severity"].dropna().values for p in order_pt]
        groups_sev = [g for g in groups_sev if len(g) > 0]
        if len(groups_sev) >= 2:
            h_s, p_s = stats.kruskal(*groups_sev)
            eta_sq_s = (h_s - (k_groups - 1)) / (n_total - k_groups) if (n_total - k_groups) > 0 else np.nan
            report.append(f"  max_severity by prompt_type: H={h_s:.4f}, p={p_s:.2e}, eta^2_H={eta_sq_s:.2f}")
        report.append("\n--- Dunn's post-hoc (Bonferroni) for prompt_type ---")
        dunn_n = dunn_posthoc_bonferroni(res_df, "n_applicable_codes", "prompt_type", order_pt)
        for (ga, gb), p_adj in sorted(dunn_n.items()):
            report.append(f"  n_applicable_codes {ga} vs {gb}: p_adj={p_adj:.4f}")
        dunn_s = dunn_posthoc_bonferroni(res_df, "max_severity", "prompt_type", order_pt)
        for (ga, gb), p_adj in sorted(dunn_s.items()):
            report.append(f"  max_severity {ga} vs {gb}: p_adj={p_adj:.4f}")
        llm_order = ["chatgpt", "gemini", "github-copilot", "baidu", "deepseek", "qwen", "falcon3"]
        groups_llm = [res_df[res_df["llm"] == l]["n_applicable_codes"].values for l in llm_order]
        groups_llm = [g for g in groups_llm if len(g) > 0]
        if len(groups_llm) >= 2:
            h_l, p_l = stats.kruskal(*groups_llm)
            report.append(f"  n_applicable_codes by LLM: H={h_l:.4f}, p={p_l:.2e}")

        if res_df is not None and "code_length" in res_df.columns:
            try:
                reg_df = res_df[["n_applicable_codes", "code_length", "prompt_type", "llm", "permission"]].dropna(
                    subset=["n_applicable_codes", "code_length", "prompt_type", "llm", "permission"]
                )
                if len(reg_df) > 50:
                    Y = reg_df["n_applicable_codes"].values.astype(float)
                    X_pt = pd.get_dummies(reg_df["prompt_type"], prefix="pt", drop_first=True)
                    X_llm = pd.get_dummies(reg_df["llm"], prefix="llm", drop_first=True)
                    X_perm = pd.get_dummies(reg_df["permission"], prefix="perm", drop_first=True)
                    X = np.column_stack([
                        np.ones(len(reg_df)),
                        reg_df["code_length"].values.astype(float),
                        X_pt.values,
                        X_llm.values,
                        X_perm.values,
                    ])
                    k = X.shape[1] - 1
                    n = X.shape[0]
                    beta = np.linalg.lstsq(X, Y, rcond=None)[0]
                    Y_hat = X @ beta
                    ss_res = np.sum((Y - Y_hat) ** 2)
                    ss_tot = np.sum((Y - np.mean(Y)) ** 2)
                    r_sq = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0
                    r_sq_adj = 1.0 - (1 - r_sq) * (n - 1) / (n - k - 1) if (n - k - 1) > 0 else r_sq
                    f_stat = (r_sq / k) / ((1 - r_sq) / (n - k - 1)) if (1 - r_sq) > 0 and (n - k - 1) > 0 else 0
                    f_pvalue = 1 - stats.f.cdf(f_stat, k, n - k - 1)
                    report.append("\n--- OLS regression ---")
                    report.append(f"  R-squared: {r_sq:.4f}, Adj. R-squared: {r_sq_adj:.4f}")
                    report.append(f"  F-statistic: {f_stat:.4f}, p-value: {f_pvalue:.2e}")

                    try:
                        import statsmodels.api as sm
                        from statsmodels.genmod.generalized_linear_model import GLM
                        from statsmodels.genmod.families import Poisson, NegativeBinomial
                        exog_df = pd.concat([
                            reg_df[["code_length"]],
                            pd.get_dummies(reg_df["prompt_type"], prefix="pt", drop_first=True),
                            pd.get_dummies(reg_df["llm"], prefix="llm", drop_first=True),
                            pd.get_dummies(reg_df["permission"], prefix="perm", drop_first=True),
                        ], axis=1)
                        exog = sm.add_constant(exog_df).astype(float)
                        glm_poisson = GLM(Y.astype(float), exog, family=Poisson()).fit()
                        report.append("\n--- Poisson regression (log-link): coefficients and p-values ---")
                        for name in ["code_length", "pt_neutral", "pt_invasive"]:
                            if name in glm_poisson.params.index:
                                coef, pv = glm_poisson.params[name], glm_poisson.pvalues[name]
                                report.append(f"  {name}: coef={coef:.4f}, p={pv:.4f}")
                        try:
                            glm_nb = GLM(Y.astype(float), exog, family=NegativeBinomial(alpha=1.0)).fit()
                            report.append("\n--- Negative binomial regression (log-link): coefficients and p-values ---")
                            for name in ["code_length", "pt_neutral", "pt_invasive"]:
                                if name in glm_nb.params.index:
                                    coef, pv = glm_nb.params[name], glm_nb.pvalues[name]
                                    report.append(f"  {name}: coef={coef:.4f}, p={pv:.4f}")
                        except Exception:
                            pass
                    except ImportError:
                        pass
            except Exception as e:
                report.append(f"\n--- OLS regression skipped: {e} ---")

    if res_df is not None and "n_positive" in res_df.columns and "n_negative" in res_df.columns:
        report.append("\n--- Valence of matched patterns ---")
        report.append("  by prompt_type:")
        for pt in ["friendly", "neutral", "invasive"]:
            grp = res_df[res_df["prompt_type"] == pt]
            if len(grp) > 0:
                mp, mn = grp["n_positive"].mean(), grp["n_negative"].mean()
                report.append(f"    {pt}: n={len(grp)}, mean_n_positive={mp:.3f}, mean_n_negative={mn:.3f}")
        report.append("  by LLM:")
        for llm in ["chatgpt", "gemini", "github-copilot", "baidu", "deepseek", "qwen", "falcon3"]:
            grp = res_df[res_df["llm"] == llm]
            if len(grp) > 0:
                mp, mn = grp["n_positive"].mean(), grp["n_negative"].mean()
                report.append(f"    {llm}: n={len(grp)}, mean_n_positive={mp:.3f}, mean_n_negative={mn:.3f}")

    if res_df is not None and "n_severity4" in res_df.columns:
        report.append("\n--- Severity-4 patterns (mean per snippet) ---")
        report.append("  by prompt_type:")
        for pt in ["friendly", "neutral", "invasive"]:
            grp = res_df[res_df["prompt_type"] == pt]
            if len(grp) > 0:
                report.append(f"    {pt}: n={len(grp)}, mean_n_severity4={grp['n_severity4'].mean():.3f}")
        report.append("  by LLM:")
        for llm in ["chatgpt", "gemini", "github-copilot", "baidu", "deepseek", "qwen", "falcon3"]:
            grp = res_df[res_df["llm"] == llm]
            if len(grp) > 0:
                report.append(f"    {llm}: n={len(grp)}, mean_n_severity4={grp['n_severity4'].mean():.3f}")

    if res_df is not None and "max_severity" in res_df.columns:
        res_df["high_risk"] = res_df["max_severity"] >= 4
        report.append("\n--- High-risk snippets (max_severity >= 4), proportion with 95% CI ---")
        for group_col, label in [("prompt_type", "by prompt_type"), ("llm", "by LLM"), ("region", "by region")]:
            if group_col not in res_df.columns:
                continue
            report.append(f"  {label}:")
            for grp_name, grp in res_df.groupby(group_col, dropna=False):
                vals = grp["high_risk"].values.astype(float)
                p, lo, hi = bootstrap_ci(vals, stat="mean")
                report.append(f"    {grp_name}: n={len(grp)}, prop_high_risk={p:.3f} [95% CI: {lo:.3f}, {hi:.3f}]")
        report.append("\n--- 95% CIs for mean n_codes and mean max_severity by prompt_type ---")
        for grp_name, grp in res_df.groupby("prompt_type", dropna=False):
            for col, name in [("n_applicable_codes", "mean_n_codes"), ("max_severity", "mean_max_severity")]:
                p, lo, hi = bootstrap_ci(grp[col].values)
                report.append(f"  {grp_name} {name}: {p:.3f} [95% CI: {lo:.3f}, {hi:.3f}]")
        report.append("\n--- 95% CIs by region ---")
        for grp_name, grp in res_df.groupby("region", dropna=False):
            for col, name in [("n_applicable_codes", "mean_n_codes"), ("max_severity", "mean_max_severity")]:
                p, lo, hi = bootstrap_ci(grp[col].values)
                report.append(f"  {grp_name} {name}: {p:.3f} [95% CI: {lo:.3f}, {hi:.3f}]")
        agg_out["per_snippet_df"] = res_df

    report.append("\n  by prompt_type:")
    for k, v in (agg_out.get("by_prompt_type") or {}).items():
        report.append(f"    {k}: {v}")
    report.append("\n  by LLM:")
    for k, v in sorted((agg_out.get("by_llm") or {}).items()):
        report.append(f"    {k}: {v}")
    report.append("\n  by region:")
    for k, v in sorted((agg_out.get("by_region") or {}).items()):
        report.append(f"    {k}: {v}")
    report.append("\n  by permission:")
    for k, v in sorted((agg_out.get("by_permission") or {}).items()):
        report.append(f"    {k}: {v}")

    report_text = "\n".join(report)
    REPORT_PATH.write_text(report_text, encoding="utf-8")

    agg_out["per_snippet_df"] = res_df
    try:
        plot_figures(agg_out, df, FIGURES_DIR)
    except Exception as exc:
        with REPORT_PATH.open("a", encoding="utf-8") as fp:
            fp.write(f"\n\n=== Figure generation failed: {exc} ===\n")

    kotlin_logs = run_kotlin_quality_correlation_pipeline(BASE)
    if kotlin_logs:
        with REPORT_PATH.open("a", encoding="utf-8") as fp:
            fp.write("\n\n=== Kotlin quality-correlation pipeline ===\n")
            for line in kotlin_logs:
                fp.write(line + "\n")

    tool_logs = run_correlate_tool_findings_pipeline(BASE)
    heatmap_logs = run_plot_prompt_tool_heatmap(BASE)
    if tool_logs or heatmap_logs:
        with REPORT_PATH.open("a", encoding="utf-8") as fp:
            fp.write("\n\n=== Tool findings and heatmap pipeline ===\n")
            for line in tool_logs:
                fp.write(line + "\n")
            for line in heatmap_logs:
                fp.write(line + "\n")

    append_tool_and_kotlin_sanity_sections(BASE, REPORT_PATH)


if __name__ == "__main__":
    main()
