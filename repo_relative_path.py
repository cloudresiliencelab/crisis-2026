"""
Format paths for logs and reports without leaking absolute filesystem locations.

Use repository root as anchor so outputs show paths like ``reproducibility/analysis_output/foo.csv``.
"""

from __future__ import annotations

from pathlib import Path


def as_repo_relative(path: Path, repo_root: Path) -> str:
    """Return *path* relative to *repo_root* when possible; otherwise basename only."""
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.name
