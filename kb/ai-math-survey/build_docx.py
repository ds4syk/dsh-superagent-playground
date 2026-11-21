#!/usr/bin/env python3
"""Build the DOCX knowledge base from the 00-06 markdown pages."""

import glob
import os
import pypandoc

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(ROOT, "deliverables", "AI数学证明调研与OpenAI722盲复现_2026-11.docx")


def main():
    files = sorted(glob.glob(os.path.join(HERE, "[0-9][0-9].md")))
    parts = []
    for p in files:
        with open(p, encoding="utf-8") as f:
            parts.append(f.read())
    text = "\n\n\\newpage\n\n".join(parts)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    pypandoc.convert_text(
        text, "docx", format="markdown", outputfile=OUT,
        extra_args=["--toc", "--toc-depth=2", "-V", "lang=zh-CN",
                    f"--resource-path={HERE}"],
    )
    print("wrote", OUT, os.path.getsize(OUT))


if __name__ == "__main__":
    main()
