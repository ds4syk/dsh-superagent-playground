#!/usr/bin/env python
"""构建两份 DOCX：构建与预实验 / 数学缺口。"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REPO = os.path.dirname(ROOT)
sys.path.insert(0, HERE)

import md2docx  # noqa: E402

OUT = os.path.join(REPO, "deliverables")
os.makedirs(OUT, exist_ok=True)
SUMMARY = os.path.join(REPO, "docs", "visuals", "2026-11-28-qpep-batch2.png")


def build_build_doc():
    md2docx.CLEAN_MATH = True
    return md2docx.md_to_docx(
        os.path.join(ROOT, "docs", "BUILD_REPORT.md"),
        os.path.join(OUT, "QPEP_第二批构建与预实验_2026-11.docx"),
        title="QPEP 第二批：构建与预实验（2026-11）",
        subtitle="独立工作目录 qpep-batch2/ · 代码、预实验与证据 · 2026-11-28",
        image=SUMMARY,
    )


def build_gap_doc():
    md2docx.CLEAN_MATH = True
    return md2docx.md_to_docx(
        os.path.join(ROOT, "math", "MATHEMATICAL_GAPS.md"),
        os.path.join(OUT, "QPEP_第二批数学缺口_2026-11.docx"),
        title="QPEP 第二批：数学缺口（2026-11-28）",
        subtitle="16 条缺口 · 与 qpep-batch2/ 的实验对应 · 阅读副本",
        image=None,
    )


if __name__ == "__main__":
    p1 = build_build_doc()
    p2 = build_gap_doc()
    print("wrote", p1)
    print("wrote", p2)
