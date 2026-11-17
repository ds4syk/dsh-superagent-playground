# -*- coding: utf-8 -*-
"""Build the DOCX version of the FourColor Lean 4 report.

Run with the venv that has python-docx::

    /mnt/storage1/syk/.venvs/fourcolor/bin/python scripts/build_fourcolor_docx.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src.md2docx as md2docx  # noqa: E402


def main():
    md = ROOT / "report" / "fourcolor-lean.md"
    out = ROOT / "deliverables" / "四色定理Lean4形式化_2026-11.docx"
    out.parent.mkdir(parents=True, exist_ok=True)
    # The generic renderer hard-codes the SDPR footer; replace it for this report.
    original_footer = md2docx.add_footer
    md2docx.add_footer = lambda section, _text: original_footer(
        section, "四色定理的 Lean 4 形式化 · 2026-11"
    )
    md2docx.render(
        md,
        out,
        title="四色定理的 Lean 4 形式化",
        subtitle="从连续平面到内核可检查的证明：一次可复现的独立验证",
        meta="2026-11-17 · 独立复现 corun1024/4ct @ 3db71e0 · Lean v4.34.0-rc2 · Mathlib 141f6b6",
    )
    print(f"wrote {out} ({out.stat().st_size/1024/1024:.2f} MiB)")


if __name__ == "__main__":
    main()
