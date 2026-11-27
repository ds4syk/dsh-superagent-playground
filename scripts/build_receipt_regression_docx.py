"""Build the receipt-regression report DOCX from the markdown report."""
import os
import pypandoc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "docs", "reports", "2026-11-27-receipt-regression.md")
OUT = os.path.join(ROOT, "deliverables", "回执功能回归诊断与修复_2026-11.docx")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
pypandoc.convert_file(SRC, "docx", format="markdown", outputfile=OUT,
                      extra_args=["--toc", "--toc-depth=2", "-V", "lang=zh-CN",
                                  f"--resource-path={os.path.dirname(SRC)}"])
print("wrote", OUT, os.path.getsize(OUT))
