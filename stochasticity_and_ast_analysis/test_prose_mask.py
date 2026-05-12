#!/usr/bin/env python3
"""Quick sanity checks for comment masking + proximity (run: python -m pytest or python file)."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from extend_annotations import parse_codebook, get_applicable_codes_hybrid


def test_comment_no_false_overlay():
    code = '''
// Tutorial: never use SYSTEM_ALERT_WINDOW without review
fun ok() {}
'''
    cb = parse_codebook(ROOT / "Codebook_en.csv")
    sections = {"Overlay"}
    codes = get_applicable_codes_hybrid(code, sections, cb, use_ast=True)
    assert "C058" not in codes, codes


def test_code_overlay():
    code = """
fun x() {
  val p = WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY
}
"""
    cb = parse_codebook(ROOT / "Codebook_en.csv")
    sections = {"Overlay"}
    codes = get_applicable_codes_hybrid(code, sections, cb, use_ast=True)
    assert "C059" in codes, codes


def main():
    test_comment_no_false_overlay()
    test_code_overlay()
    print("ok")


if __name__ == "__main__":
    main()
