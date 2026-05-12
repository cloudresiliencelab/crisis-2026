#!/usr/bin/env python3
"""
Extend annotations.csv with applicable codebook codes (C001, C002, ...) by
matching snippet code against codebook pattern definitions (Example, Code Name).
Uses sections from annotations (rater1/rater2) to restrict which codebook patterns to check.
"""

import csv
import re
import sqlite3
from pathlib import Path

BASE = Path(__file__).resolve().parent

from repo_relative_path import as_repo_relative

CODEBOOK_PATH = BASE / "Codebook_en.csv"
ANNOTATIONS_PATH = BASE / "annotations.csv"
ANNOTATIONS_EXTENDED_PATH = BASE / "annotations_extended.csv"
DB_PATH = BASE / "db.sqlite3"


def normalize_section(name: str) -> str:
    """Map annotation section names to codebook section names."""
    n = (name or "").strip()
    if not n:
        return ""
    # Annotation style: Storage/File, Body_Sensor, Call_Logs, Disturb_Mode, etc.
    n = n.replace("_", " ")
    if n.lower() == "storage/file" or n == "Storage/File":
        return "Storage / File"
    return n


def extract_signatures_from_example(example: str, code_name: str = "") -> list[str]:
    """Extract searchable signatures from the Example column (permissions, APIs, keywords).
    Signature extraction is hybrid: (1) Codebook-derived: android.permission.*, uses-permission,
    Example/code-name substrings. (2) Fixed list of known Android/privacy tokens and regexes
    for patterns that lack sufficient material in the codebook. If the codebook gains new patterns,
    the fixed list may need updating or the pattern's Example field should be filled.
    """
    sigs = []
    # Normalize: CSV often has "" for one quote
    text = (example or "").replace('""', '"')
    if code_name:
        # Code Name is often the permission or API identifier
        name_clean = code_name.strip()
        if re.match(r"^[A-Z][A-Z0-9_]+$", name_clean):
            sigs.append(name_clean)
        if "permission" in name_clean.lower() or "_" in name_clean:
            sigs.append(name_clean)
    if not text.strip() or text.strip() in ("–", "-"):
        return list(dict.fromkeys(sigs))
    # android.permission.XXX
    for m in re.finditer(r"android\.permission\.([A-Z_0-9]+)", text, re.IGNORECASE):
        tok = m.group(1).upper()
        sigs.append("android.permission." + tok)
        sigs.append(tok)
    # uses-permission (alternate)
    for m in re.finditer(r"uses-permission\s+android:name=\"([^\"]+)\"", text, re.IGNORECASE):
        sigs.append(m.group(1))
    # uses-feature
    for m in re.finditer(r"android\.hardware\.(\w+)", text, re.IGNORECASE):
        sigs.append("android.hardware." + m.group(1))
    # Distinctive API/code substrings
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


def parse_codebook(path: Path) -> dict[str, list[tuple[str, list[str]]]]:
    """Parse Codebook.csv into section -> [(code_id, signatures), ...]."""
    section_to_patterns: dict[str, list[tuple[str, list[str]]]] = {}
    current_section = ""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=";")
        header = next(reader)
        for row in reader:
            if not row:
                continue
            code = (row[0] or "").strip()
            # Section header: row starts with a name that is not C + digits
            if code and not re.match(r"^C\d+", code):
                current_section = code  # e.g. "Location", "Storage / File"
                if current_section not in section_to_patterns:
                    section_to_patterns[current_section] = []
                continue
            if not code.startswith("C") or not code[1:].isdigit():
                continue
            code_name = (row[6] or "").strip() if len(row) > 6 else ""
            example = (row[10] or "").strip() if len(row) > 10 else ""
            # Build signatures: from Example and Code Name
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


def load_annotation_sections(rater1: str, rater2: str) -> set[str]:
    """Parse rater1 and rater2 cells into a set of normalized section names."""
    sections = set()
    for cell in (rater1 or "", rater2 or ""):
        for part in cell.split(";"):
            n = normalize_section(part)
            if n:
                sections.add(n)
    return sections


# C008 (no opt-in before request) / C009 (no user explanation before permission):
# Detect location/permission request without rationale/opt-in (codebook Example is "–", so no signatures).
_LOCATION_REQUEST_SUBSTRINGS = [
    "requestPermissions",
    "requestLocationUpdates",
    "getLastKnownLocation",
    "FusedLocationProviderClient",
    "checkSelfPermission",
    "ACCESS_FINE_LOCATION",
    "ACCESS_COARSE_LOCATION",
]
_RATIONALE_OPTIN_SUBSTRINGS = [
    "shouldShowRequestPermissionRationale",
    "Rationale",
    "Snackbar",
    "AlertDialog",
    "explain",
    "why we need",
    "purpose",
    "reason for",
    "before requesting",
    "reason to use",
]


def _has_location_request_without_rationale(snippet_code: str) -> bool:
    """True if snippet shows location/permission request but no clear rationale/opt-in (C008/C009)."""
    code_lower = snippet_code.lower()
    has_request = any(s.lower() in code_lower for s in _LOCATION_REQUEST_SUBSTRINGS)
    has_rationale = any(s.lower() in code_lower for s in _RATIONALE_OPTIN_SUBSTRINGS)
    return bool(has_request and not has_rationale)


def build_code_to_section(
    section_to_patterns: dict[str, list[tuple[str, list[str]]]],
) -> dict[str, str]:
    """Map codebook id -> section name."""
    m: dict[str, str] = {}
    for sec, plist in section_to_patterns.items():
        for code_id, _ in plist:
            m[code_id] = sec
    return m


def get_applicable_codes(
    snippet_code: str,
    sections: set[str],
    section_to_patterns: dict[str, list[tuple[str, list[str]]]],
) -> list[str]:
    """Return list of codebook code IDs that apply to this snippet for the given sections."""
    code_upper = snippet_code  # search case-insensitively for permissions
    applicable = []
    for section in sections:
        patterns = section_to_patterns.get(section, [])
        for code_id, sigs in patterns:
            if not sigs:
                continue
            for sig in sigs:
                if sig in snippet_code or sig in code_upper:
                    applicable.append(code_id)
                    break
                # Try case-insensitive for permission-like strings
                if len(sig) > 3 and sig.upper() in snippet_code.upper():
                    applicable.append(code_id)
                    break
        # C008/C009 (Location): no Example in codebook; use heuristic: request without rationale
        if section == "Location" and _has_location_request_without_rationale(snippet_code):
            applicable.append("C008")
            applicable.append("C009")
    return sorted(set(applicable))


def get_applicable_codes_masked(
    snippet_code: str,
    sections: set[str],
    section_to_patterns: dict[str, list[tuple[str, list[str]]]],
    prose_spans: list[tuple[int, int]],
) -> list[str]:
    """
    Like get_applicable_codes, but signature hits count only if they occur outside
    prose spans (comments / multiline strings). Location heuristic C008/C009 uses
    comment-masked text so rationale keywords in comments do not suppress detection.
    """
    from stochasticity_and_ast_analysis.ast_matcher import (
        find_sig_in_code_only,
        mask_prose_string,
    )

    masked_for_heuristic = mask_prose_string(snippet_code, prose_spans)
    applicable: list[str] = []
    for section in sections:
        patterns = section_to_patterns.get(section, [])
        for code_id, sigs in patterns:
            if not sigs:
                continue
            for sig in sigs:
                if find_sig_in_code_only(snippet_code, sig, prose_spans):
                    applicable.append(code_id)
                    break
                if len(sig) > 3 and find_sig_in_code_only(
                    snippet_code, sig, prose_spans, case_insensitive=True
                ):
                    applicable.append(code_id)
                    break
        if section == "Location" and _has_location_request_without_rationale(masked_for_heuristic):
            applicable.append("C008")
            applicable.append("C009")
    return sorted(set(applicable))


def get_applicable_codes_hybrid(
    snippet_code: str,
    sections: set[str],
    section_to_patterns: dict[str, list[tuple[str, list[str]]]],
    use_ast: bool = True,
    apply_comment_mask: bool = True,
    apply_proximity_rules: bool = True,
) -> list[str]:
    """
    Regex/string matcher plus optional AST-assisted rules for selected patterns.

    When ``apply_comment_mask`` is True (default), signatures must not appear only
    inside comments or multiline string literals (see ast_matcher). When
    ``apply_proximity_rules`` is True, optional same-function filters tighten C003/C059.
    """
    try:
        from stochasticity_and_ast_analysis.ast_matcher import (
            apply_proximity_filters,
            codes_allowed_for_sections,
            collect_prose_byte_spans,
            detect_ast_pattern_codes,
            parse_kotlin_tree,
        )
    except ImportError:
        return get_applicable_codes(snippet_code, sections, section_to_patterns)

    tree = parse_kotlin_tree(snippet_code) if apply_comment_mask or apply_proximity_rules or use_ast else None
    prose_spans = collect_prose_byte_spans(tree) if tree else []

    if apply_comment_mask:
        regex_codes = set(
            get_applicable_codes_masked(snippet_code, sections, section_to_patterns, prose_spans)
        )
    else:
        regex_codes = set(get_applicable_codes(snippet_code, sections, section_to_patterns))

    if not use_ast:
        merged = apply_proximity_filters(
            regex_codes, snippet_code, tree, prose_spans, enabled=apply_proximity_rules
        )
        return sorted(merged)

    code_to_sec = build_code_to_section(section_to_patterns)
    ast_hits = codes_allowed_for_sections(
        detect_ast_pattern_codes(snippet_code, prose_spans, tree),
        sections,
        code_to_sec,
    )
    merged = regex_codes | set(ast_hits)
    merged = apply_proximity_filters(
        merged, snippet_code, tree, prose_spans, enabled=apply_proximity_rules
    )
    return sorted(merged)


def main():
    section_to_patterns = parse_codebook(CODEBOOK_PATH)
    # Normalized section name mapping: annotation format -> codebook format
    # (already done in load_annotation_sections via normalize_section)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    rows_out = []
    with open(ANNOTATIONS_PATH, newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=";")
        header = next(reader)
        rows_out.append(header + ["applicable_codes"])

        for row in reader:
            if len(row) < 2:
                rows_out.append(row + [""])
                continue
            snippet_id_str = row[0].strip()
            try:
                snippet_id = int(snippet_id_str)
            except ValueError:
                rows_out.append(row + [""])
                continue
            rater1 = row[1].strip() if len(row) > 1 else ""
            rater2 = row[3].strip() if len(row) > 3 else ""
            sections = load_annotation_sections(rater1, rater2)
            cur.execute("SELECT code FROM snippets WHERE id = ?", (snippet_id,))
            r = cur.fetchone()
            snippet_code = (r["code"] or "") if r else ""
            applicable = get_applicable_codes(snippet_code, sections, section_to_patterns)
            row_extended = row + [";".join(applicable)]
            rows_out.append(row_extended)

    conn.close()

    with open(ANNOTATIONS_EXTENDED_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerows(rows_out)

    print(
        f"Wrote {as_repo_relative(ANNOTATIONS_EXTENDED_PATH, BASE)} with {len(rows_out) - 1} rows and applicable_codes column."
    )


if __name__ == "__main__":
    main()
