"""Build the SuperMimic knowledge-base DOCX from kb/supermimic/*.md.

Uses pypandoc (bundled pandoc) so tables and $...$ math make it into the DOCX.
Run with the docs venv:
  /mnt/storage1/syk/VLA/.venv-docs/bin/python vla/SuperMimic/repro/build_docx.py
"""

from __future__ import annotations

import glob
import os

import pypandoc

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
KB = os.path.join(REPO, "kb", "supermimic")
OUT = os.path.join(REPO, "deliverables", "SuperMimic复现_2026-11.docx")


def main() -> None:
  files = sorted(glob.glob(os.path.join(KB, "*.md")))
  if not files:
    raise SystemExit(f"no markdown pages under {KB}")
  parts = []
  for path in files:
    with open(path, encoding="utf-8") as fh:
      parts.append(fh.read())
  text = "\n\n\\newpage\n\n".join(parts)
  os.makedirs(os.path.dirname(OUT), exist_ok=True)
  pypandoc.convert_text(
    text,
    "docx",
    format="markdown",
    outputfile=OUT,
    extra_args=[
      "-V", "lang=zh-CN",
      f"--resource-path={KB}",
    ],
  )
  print(f"wrote {OUT} ({os.path.getsize(OUT)} bytes)")


if __name__ == "__main__":
  main()
