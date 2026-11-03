import glob, os, subprocess, sys
import pypandoc

root = os.path.dirname(os.path.abspath(__file__))
kb = os.path.join(root, "kb")
out = os.path.join(root, "deliverables", "JEV世界模型知识库_2026-11.docx")
os.makedirs(os.path.dirname(out), exist_ok=True)
files = sorted(glob.glob(os.path.join(kb, "*.md")))
parts = []
for p in files:
    with open(p, encoding="utf-8") as f:
        parts.append(f.read())
text = "\n\n\\newpage\n\n".join(parts)
pypandoc.convert_text(text, "docx", format="markdown", outputfile=out,
                      extra_args=["--toc", "--toc-depth=2", "-V", "lang=zh-CN",
                                  f"--resource-path={kb}"])
print("wrote", out, os.path.getsize(out))
