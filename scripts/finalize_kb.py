"""Refresh figures/tables in kb/ and rebuild the DOCX."""
import os, shutil, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JEV = os.path.join(ROOT, "jev-wm")
KB = os.path.join(ROOT, "kb")
ASSETS = os.path.join(KB, "assets")
os.makedirs(ASSETS, exist_ok=True)

subprocess.run([sys.executable, "scripts/analyze.py"], cwd=JEV, check=True,
               env={**os.environ, "PYTHONPATH": "."})
subprocess.run([sys.executable, "scripts/fill_kb.py"], cwd=JEV, check=True,
               env={**os.environ, "PYTHONPATH": "."})

for fn in ["fig1_exp1_by_kind.png", "fig2_exp1_rollout.png",
           "fig3_exp2_success.png", "fig4_exp2_cost.png"]:
    p = os.path.join(JEV, "results", "figures", fn)
    if os.path.exists(p):
        shutil.copy(p, os.path.join(ASSETS, fn))

inserts = {
    "04.md": "\n![图 1：按动作类型的精确匹配](assets/fig1_exp1_by_kind.png)\n\n"
             "![图 2：开环 rollout 的误差累积](assets/fig2_exp1_rollout.png)\n",
    "05.md": "\n![图 3：闭环成功率](assets/fig3_exp2_success.png)\n\n"
             "![图 4：成本-成功率](assets/fig4_exp2_cost.png)\n",
}
for fn, block in inserts.items():
    p = os.path.join(KB, fn)
    s = open(p, encoding="utf-8").read()
    if "assets/fig" not in s:
        s = s.rstrip() + "\n" + block
        open(p, "w", encoding="utf-8").write(s)

subprocess.run([sys.executable, os.path.join(ROOT, "scripts_build_docx.py")], cwd=ROOT, check=True)
print("done")
