#!/usr/bin/env python3
"""Report total pages and the page where the appendix starts (main-text length).

Usage: python3 maintext_pages.py <pdf> [<pdf> ...]
"""
import re
import subprocess
import sys


def page_texts(pdf):
    out = subprocess.run(["pdftotext", "-layout", pdf, "-"], capture_output=True, text=True).stdout
    return out.split("\f")


def main():
    for pdf in sys.argv[1:]:
        pages = page_texts(pdf)
        # drop trailing empty page artefact
        while pages and not pages[-1].strip():
            pages.pop()
        total = len(pages)
        app = None
        refs = None
        for i, p in enumerate(pages, 1):
            if app is None and re.search(r"(?i)^\s*(appendix|APPENDIX|附\s*录)\b", p, re.M) and i > 2:
                app = i
            if refs is None and re.search(r"(?i)^\s*(references|REFERENCES|参考文献)\s*$", p, re.M) and i > 2:
                refs = i
        main_pages = (app - 1) if app else ((refs - 1) if refs else total)
        print(f"{pdf}\n  total={total}  refs_start={refs}  appendix_start={app}  main_body<= {main_pages} pages")


if __name__ == "__main__":
    main()
