#!/usr/bin/env python3
"""Fill the results table in kb/ai-math-survey/05.md from grades_summary.json."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
KB = ROOT.parent.parent / "kb" / "ai-math-survey" / "05.md"
TITLES = {
    "007_two_point_chowla": "007 两点相关",
    "017_pi_irrationality": "017 μ(π)=2",
    "087_mahler": "087 Mahler",
    "102_basic_sdp": "102 BasicSDP",
    "159_erdos_progressions": "159 Erdős",
    "197_kaplansky_direct_finiteness": "197 Kaplansky",
    "221_mezard_parisi": "221 Mézard–Parisi",
    "271_heisenberg_magnetization": "271 Heisenberg",
    "287_free_group_factors": "287 自由群因子",
    "362_vlasov_maxwell": "362 Vlasov–Maxwell",
}
DIR = {
    "007_two_point_chowla": "真",
    "017_pi_irrationality": "真",
    "087_mahler": "真",
    "102_basic_sdp": "真",
    "159_erdos_progressions": "真",
    "197_kaplansky_direct_finiteness": "假（反例）",
    "221_mezard_parisi": "真",
    "271_heisenberg_magnetization": "真",
    "287_free_group_factors": "真",
    "362_vlasov_maxwell": "真",
}


def load():
    p = ROOT / "results" / "grades_summary.json"
    return json.loads(p.read_text()) if p.exists() else []


def grade(sid: str, grader: str):
    f = ROOT / "results" / "grades" / f"{sid.replace('/', '__')}__{grader}.json"
    if f.exists():
        return json.loads(f.read_text())
    return {}


def main():
    rows = load()
    by = {(r["session"].split("/")[0], r["session"].split("/")[1]): r for r in rows}
    lines = ["| 案例 | DeepSeek-V4-Pro | Kimi-K3 | 方向 (D/K) | 最强有效部分结果 / 主要缺口 |",
             "|---|---:|---:|---|---|"]
    for pid, title in TITLES.items():
        d = by.get(("deepseek", pid), {})
        k = by.get(("kimi", pid), {})
        ds = d.get("mean", "—")
        ks = k.get("mean", "—")
        dd = "/".join(sorted({str(v) for v in (d.get("directions") or {}).values()})) or "—"
        kd = "/".join(sorted({str(v) for v in (k.get("directions") or {}).values()})) or "—"
        gd = grade(f"deepseek/{pid}", "deepseek")
        gk = grade(f"kimi/{pid}", "deepseek")
        note = gd.get("strongest_valid_partial_result") or gk.get("strongest_valid_partial_result") or ""
        gap = (gd.get("fatal_gaps") or gk.get("fatal_gaps") or [""])[0]
        note = (note[:140] + "…") if len(note) > 140 else note
        gap = (gap[:100] + "…") if len(gap) > 100 else gap
        combo = (note + ("；缺口：" + gap if gap else "")).strip()
        lines.append(f"| {title} | {ds} | {ks} | {dd}/{kd} | {combo} |")
    lines.append("")
    lines.append("**读法**：表中 DeepSeek/Kimi 两列是**求解模型**；每个分数是两名评分者（DeepSeek/Kimi）"
                 "的均值。5 = 完整证明/反例；4 = 方向正确、主定理基本证出但有小缺口；3 = 实质性部分结果；"
                 "2 = 正确背景+非核心观察；1 = 致命错误/方向错；0 = 无有效内容。")
    block = "<!-- SCORES_START -->\n" + "\n".join(lines) + "\n<!-- SCORES_END -->"

    s = KB.read_text()
    s = re.sub(r"<!-- SCORES_START -->.*?<!-- SCORES_END -->", block, s, flags=re.S)
    full = [r for r in rows if any((v or 0) >= 4 for v in (r.get("scores") or {}).values())]
    best = [(r["session"], r.get("mean") or 0) for r in rows]
    best.sort(key=lambda x: -x[1])
    headline = (f"> **结论先行**：在信息隔离、每案例约 40k 输出 token 的预算下，"
                f"DeepSeek-V4-Pro 与 Kimi-K3 对 10 个研究级主定理的完整复现数为 "
                f"**{len(full)}/10**（评分 ≥4）；最高均分为 "
                + "、".join(f"{s} {v}" for s, v in best[:4]) +
                "。两者都能给出有价值的部分结果/路线，但多数输出把“已知定理/条件性证明”"
                "当作部分结果，完整主定理仍未证。")
    s = re.sub(r"> \*\*结论先行\*\*：.*?\n", headline + "\n", s, count=1, flags=re.S)
    KB.write_text(s)
    print(block)
    print("---")
    print(headline)


if __name__ == "__main__":
    main()
