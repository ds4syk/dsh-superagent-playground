"""把 kb/qpep-layer/*.md 合成 DOCX 知识库。用 base conda 的 pypandoc（含 pandoc 3.9）。

用法：~/miniconda3/bin/python scripts_build_qpep_docx.py
"""

import glob
import os

import pypandoc

root = os.path.dirname(os.path.abspath(__file__))
kb = os.path.join(root, "kb", "qpep-layer")
out = os.path.join(root, "deliverables", "QPEP_可微优化层_2026-11.docx")
os.makedirs(os.path.dirname(out), exist_ok=True)

files = [os.path.join(kb, f"{i:02d}.md") for i in range(6)]
parts = []
for p in files:
    assert os.path.exists(p), p
    with open(p, encoding="utf-8") as f:
        parts.append(f.read())
text = "\n\n\\newpage\n\n".join(parts)

pypandoc.convert_text(
    text,
    "docx",
    format="markdown",
    outputfile=out,
    extra_args=[
        "-V", "lang=zh-CN",
        f"--resource-path={root}",
    ],
)
print("wrote", out, os.path.getsize(out))
