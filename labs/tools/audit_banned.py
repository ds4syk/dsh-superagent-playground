#!/usr/bin/env python3
"""Banned-phrase audit for the six paper LaTeX trees.

Usage: python3 audit_banned.py <latex_dir> [lang]
lang: en (default) or zh.
Exit code 1 if any hit (excluding preamble.tex).
"""
import re
import sys
import pathlib

EN = [
    r"\bTODO\b", r"\bpending\b", r"\bunverified\b", r"\bblocked\b",
    r"\bnot run\b", r"\bnot measured\b", r"\bwe do not claim\b",
    r"\bdo not claim\b", r"\bdoes not claim\b", r"\bnot a detector\b",
    r"\bhonest(?:y|ly)?\b", r"\bcaveat", r"\bevidence level", r"\bevidence grade",
    r"claim[-–]to[-–]evidence", r"\bfalsif", r"anti-overclaim", r"\bdraft\b",
    r"\bstatus\b", r"[Ss]ource:\s*results/", r"\.json\b", r"\bplaceholder\b",
    r"\bPENDING\b", r"\bnot claimed\b", r"\bout of scope\b", r"\bwe leave\b.*\bopen\b",
]
ZH = [
    r"TODO", r"草稿", r"待定", r"未测量", r"未运行", r"待核实", r"待验证",
    r"证据等级", r"不声称", r"刻意不声称", r"诚实地", r"坦白", r"状态",
    r"负结果", r"占位", r"待撰写", r"结果文件",
]


def main():
    root = pathlib.Path(sys.argv[1])
    lang = sys.argv[2] if len(sys.argv) > 2 else "en"
    pats = ZH if lang == "zh" else EN
    total = 0
    for p in sorted(root.rglob("*.tex")):
        if p.name == "preamble.tex":
            continue
        text = p.read_text(errors="replace")
        hits = []
        for pat in pats:
            for m in re.finditer(pat, text, re.I if lang == "en" else 0):
                line = text.count("\n", 0, m.start()) + 1
                hits.append((line, m.group(0)))
        if hits:
            total += len(hits)
            print(f"{p}: {len(hits)} hit(s)")
            for line, w in hits[:8]:
                print(f"    line {line}: {w!r}")
    print(f"\nTOTAL banned hits: {total}")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
