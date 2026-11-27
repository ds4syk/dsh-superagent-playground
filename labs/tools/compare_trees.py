#!/usr/bin/env python3
"""Compare the numeric content of an English and a Chinese LaTeX tree.

A faithful translation must contain the same measured numbers. This flags
numbers present in one tree but not the other (formatting-normalised).

Usage: python3 compare_trees.py <en_dir> <zh_dir>
"""
import re
import sys
import pathlib
from collections import Counter

NUM_RE = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def read_tex(root: pathlib.Path) -> str:
    return "\n".join(p.read_text(errors="replace") for p in sorted(root.rglob("*.tex")) if p.name != "preamble.tex")


def normalise(text: str) -> str:
    s = text
    s = s.replace(r"\%", "%").replace(r"\$", "$").replace("~", " ")
    s = s.replace("$", " ").replace("{,}", ",").replace("\\,", " ")
    s = s.replace("\\ ", " ").replace("\\times", " x ").replace("\\cdot", " x ")
    s = s.replace("\\approx", " ~ ").replace("\\pm", " +- ")
    s = re.sub(r"(?:x\s*)?10\s*\^\s*\{?\s*([-+]?\d+)\s*\}?", r"e\1", s)
    s = re.sub(r"e\s*([-+])\s*", r"e\1", s)
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)
    s = re.sub(r"(?i)\b(section|sec\.?|table|tab\.?|figure|fig\.?|theorem|lemma|proposition|corollary|appendix|eq\.?|equation|节|表|图)\s*~?\s*\d+(?:\.\d+)*", " REF ", s)
    return s


def numbers(text: str) -> Counter:
    out = Counter()
    for m in NUM_RE.finditer(text):
        tok = m.group(0).lstrip("-")
        if "." in tok or "e" in tok.lower():
            try:
                v = float(tok)
            except ValueError:
                continue
            if v != 0:
                out[f"{v:.12g}"] += 1
        elif len(tok) >= 3:
            out[tok] += 1
    return out


def main():
    en, zh = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    a, b = numbers(normalise(read_tex(en))), numbers(normalise(read_tex(zh)))
    only_en = sorted(k for k in a if k not in b)
    only_zh = sorted(k for k in b if k not in a)
    print(f"EN distinct: {len(a)}   ZH distinct: {len(b)}")
    print(f"\n== in EN only ({len(only_en)}) ==")
    for k in only_en:
        print(f"  {k:>16}  en_count={a[k]}")
    print(f"\n== in ZH only ({len(only_zh)}) ==")
    for k in only_zh:
        print(f"  {k:>16}  zh_count={b[k]}")


if __name__ == "__main__":
    main()
