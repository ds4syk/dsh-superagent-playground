import os, sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from md2docx import render

ROOT = Path(__file__).resolve().parent.parent
out = ROOT / 'deliverables' / '可微QPEP_研究知识库_2026-11.docx'
out.parent.mkdir(parents=True, exist_ok=True)
render(ROOT / 'report' / 'diff_qpep_research.md', out,
       '可微全局最优几何估计层',
       'QPEP 的全局解映射、梯度正确性、分支切换与不确定性（研究知识库）',
       '2026-11 · 技术报告 · 证据分级：论文声称 / 独立复现 / 未验证')
print('wrote', out, out.stat().st_size)
