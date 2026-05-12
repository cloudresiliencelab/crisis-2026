"""
AST-assisted detection for a subset of codebook patterns (Kotlin).

Uses tree-sitter Kotlin grammar for:
  (2) Ignoring matches inside comments and string literals (prose masking).
  (4) Optional same-function proximity filters for high-impact overlay / location codes.

Complements regex/string matching in extend_annotations.get_applicable_codes.
"""

from __future__ import annotations

import re
from typing import Callable, Dict, List, Optional, Set, Tuple

# Lazy parser
_PARSER = None
_LANGUAGE = None

# Kotlin grammar: comments always; multiline / raw strings often tutorial prose.
# We deliberately do *not* mask plain string_literal so manifest-style permission
# strings in Kotlin remain visible to matchers (FP reduction relies on proximity).
_PROSE_NODE_TYPES = frozenset({
    "line_comment",
    "block_comment",
    "multiline_string_literal",
})


def _get_parser():
    global _PARSER, _LANGUAGE
    if _PARSER is not None:
        return _PARSER, _LANGUAGE
    try:
        import tree_sitter_kotlin as tsk
        from tree_sitter import Language, Parser

        _LANGUAGE = Language(tsk.language())
        _PARSER = Parser(_LANGUAGE)
    except Exception:
        _PARSER = False  # type: ignore
        _LANGUAGE = None
    return _PARSER, _LANGUAGE


def parse_kotlin_tree(snippet_code: str):
    """Return tree-sitter Tree or None if parser unavailable / parse error."""
    parser, _ = _get_parser()
    if parser is False or parser is None:
        return None
    try:
        return parser.parse(snippet_code.encode("utf-8"))
    except Exception:
        return None


def merge_intervals(spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not spans:
        return []
    spans = sorted(spans)
    out = [spans[0]]
    for s, e in spans[1:]:
        ls, le = out[-1]
        if s <= le:
            out[-1] = (ls, max(le, e))
        else:
            out.append((s, e))
    return out


def collect_prose_byte_spans(tree) -> List[Tuple[int, int]]:
    """Byte ranges [start, end) that are comments or string literals."""
    spans: List[Tuple[int, int]] = []

    def visit(node) -> None:
        if node.type in _PROSE_NODE_TYPES:
            spans.append((node.start_byte, node.end_byte))
        for child in node.children:
            visit(child)

    visit(tree.root_node)
    return merge_intervals(spans)


def interval_overlaps_any(lo: int, hi: int, intervals: List[Tuple[int, int]]) -> bool:
    for a, b in intervals:
        if hi > a and lo < b:
            return True
    return False


def mask_prose_bytes(src: bytes, prose_spans: List[Tuple[int, int]]) -> bytes:
    """Replace prose regions with ASCII spaces (same length) to preserve offsets."""
    out = bytearray(src)
    for s, e in prose_spans:
        if 0 <= s < e <= len(out):
            out[s:e] = b" " * (e - s)
    return bytes(out)


def mask_prose_string(snippet_code: str, prose_spans: List[Tuple[int, int]]) -> str:
    b = mask_prose_bytes(snippet_code.encode("utf-8"), prose_spans)
    return b.decode("utf-8", errors="replace")


def find_sig_in_code_only(
    snippet_code: str,
    sig: str,
    prose_spans: List[Tuple[int, int]],
    case_insensitive: bool = False,
) -> bool:
    """True if some occurrence of sig lies entirely outside prose spans (byte-accurate)."""
    src_b = snippet_code.encode("utf-8")
    if case_insensitive:
        sig_b = sig.encode("utf-8").lower()
        hay = src_b.lower()
    else:
        sig_b = sig.encode("utf-8")
        hay = src_b
    if not sig_b:
        return False
    start = 0
    while True:
        i = hay.find(sig_b, start)
        if i < 0:
            return False
        end = i + len(sig_b)
        if not interval_overlaps_any(i, end, prose_spans):
            return True
        start = i + 1


def collect_function_declaration_spans(tree) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []

    def visit(node) -> None:
        if node.type == "function_declaration":
            spans.append((node.start_byte, node.end_byte))
        for child in node.children:
            visit(child)

    visit(tree.root_node)
    return spans


def _spans_intersecting_fragment(
    global_prose: List[Tuple[int, int]], frag_lo: int, frag_hi: int
) -> List[Tuple[int, int]]:
    """Prose spans relative to fragment [frag_lo, frag_hi), as offsets into fragment bytes."""
    out: List[Tuple[int, int]] = []
    for a, b in global_prose:
        lo = max(a, frag_lo)
        hi = min(b, frag_hi)
        if lo < hi:
            out.append((lo - frag_lo, hi - frag_lo))
    return merge_intervals(out)


def _fragment_has_tokens(
    fragment: bytes,
    local_prose: List[Tuple[int, int]],
    tokens: List[str],
    any_of: bool = True,
) -> bool:
    """Any/all tokens must appear in code-only (non-prose) regions of fragment."""
    checks = []
    for tok in tokens:
        checks.append(find_sig_in_code_only(fragment.decode("utf-8", errors="replace"), tok, local_prose))
    if any_of:
        return any(checks)
    return all(checks)


def apply_proximity_filters(
    codes: Set[str],
    snippet_code: str,
    tree,
    prose_spans: List[Tuple[int, int]],
    enabled: bool = True,
) -> Set[str]:
    """
    Optional (4): tighten selected severity patterns using same-function co-occurrence.

    - C059: keep only if TYPE_APPLICATION_OVERLAY appears in code together with
      at least one usage indicator (WindowManager, addView, LayoutParams) or
      SYSTEM_ALERT_WINDOW in the *same* function_declaration body (or file-level
      fragment outside any function, treated as one bucket).
    - C003: keep only if ACCESS_BACKGROUND_LOCATION appears in code in the same
      function as a location API (requestLocationUpdates, FusedLocationProviderClient,
      getLastKnownLocation, requestPermissions).
    """
    if not enabled or tree is None:
        return set(codes)
    out = set(codes)
    src_b = snippet_code.encode("utf-8")
    funcs = collect_function_declaration_spans(tree)

    def buckets() -> List[Tuple[int, int]]:
        if not funcs:
            return [(0, len(src_b))]
        return funcs

    # --- C059 proximity ---
    if "C059" in out:
        ok = False
        for lo, hi in buckets():
            frag = src_b[lo:hi]
            loc_prose = _spans_intersecting_fragment(prose_spans, lo, hi)
            has_overlay = find_sig_in_code_only(
                frag.decode("utf-8", errors="replace"),
                "TYPE_APPLICATION_OVERLAY",
                loc_prose,
            )
            if not has_overlay:
                continue
            usage = _fragment_has_tokens(
                frag,
                loc_prose,
                ["WindowManager", "addView", "LayoutParams", "SYSTEM_ALERT_WINDOW"],
                any_of=True,
            )
            if usage:
                ok = True
                break
        if not ok:
            out.discard("C059")

    # --- C003 proximity ---
    if "C003" in out:
        ok = False
        loc_toks = [
            "requestLocationUpdates",
            "FusedLocationProviderClient",
            "getLastKnownLocation",
            "requestPermissions",
        ]
        bg = "ACCESS_BACKGROUND_LOCATION"
        for lo, hi in buckets():
            frag = src_b[lo:hi]
            loc_prose = _spans_intersecting_fragment(prose_spans, lo, hi)
            frag_s = frag.decode("utf-8", errors="replace")
            if not find_sig_in_code_only(frag_s, bg, loc_prose):
                continue
            if any(find_sig_in_code_only(frag_s, t, loc_prose) for t in loc_toks):
                ok = True
                break
        if not ok:
            out.discard("C003")

    return out


def _walk(node, visit: Callable[..., None], source: bytes, depth: int = 0) -> None:
    visit(node, source)
    for child in node.children:
        _walk(child, visit, source, depth + 1)


def _node_text(node, source: bytes) -> str:
    return source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def detect_overlay_codes(source: str, prose_spans: List[Tuple[int, int]]) -> Set[str]:
    """C058: SYSTEM_ALERT_WINDOW; C059: TYPE_APPLICATION_OVERLAY (code spans only)."""
    found: Set[str] = set()
    if find_sig_in_code_only(source, "SYSTEM_ALERT_WINDOW", prose_spans) or find_sig_in_code_only(
        source, "android.permission.SYSTEM_ALERT_WINDOW", prose_spans
    ):
        found.add("C058")
    if find_sig_in_code_only(source, "TYPE_APPLICATION_OVERLAY", prose_spans):
        found.add("C059")
    return found


def detect_location_codes_ast(source: str, tree, prose_spans: List[Tuple[int, int]]) -> Set[str]:
    """
    C003: ACCESS_BACKGROUND_LOCATION in code (identifier / string literal body parsed as AST).
    C006: call_expression containing requestLocationUpdates in code.
    """
    found: Set[str] = set()
    src_b = source.encode("utf-8")

    def visit(node, src: bytes) -> None:
        nonlocal found
        t = node.type
        txt = _node_text(node, src)
        if t == "string_literal":
            inner = txt.strip('"').strip("'")
            if "ACCESS_BACKGROUND_LOCATION" in inner or "android.permission.ACCESS_BACKGROUND_LOCATION" in inner:
                found.add("C003")
        elif t == "identifier" and txt == "ACCESS_BACKGROUND_LOCATION":
            found.add("C003")
        elif t == "call_expression" and "requestLocationUpdates" in txt:
            found.add("C006")

    _walk(tree.root_node, visit, src_b)
    return found


def detect_ast_pattern_codes(
    snippet_code: str,
    prose_spans: Optional[List[Tuple[int, int]]] = None,
    tree=None,
) -> List[str]:
    """
    Return sorted unique code IDs from AST-focused rules.
    If prose_spans is None, parses once and computes spans.
    """
    found: Set[str] = set()
    tree = tree or parse_kotlin_tree(snippet_code)
    if tree is None:
        prose_spans = prose_spans or []
        found |= detect_overlay_codes(snippet_code, prose_spans)
        if "ACCESS_BACKGROUND_LOCATION" in snippet_code:
            found.add("C003")
        if re.search(r"requestLocationUpdates\s*\(", snippet_code):
            found.add("C006")
        return sorted(found)

    if prose_spans is None:
        prose_spans = collect_prose_byte_spans(tree)

    found |= detect_overlay_codes(snippet_code, prose_spans)
    try:
        found |= detect_location_codes_ast(snippet_code, tree, prose_spans)
    except Exception:
        pass
    return sorted(found)


def codes_allowed_for_sections(
    codes: List[str],
    sections: Set[str],
    code_to_section: Dict[str, str],
) -> List[str]:
    out = []
    for c in codes:
        sec = code_to_section.get(c)
        if sec and sec in sections:
            out.append(c)
    return out
