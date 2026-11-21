#!/usr/bin/env python3
"""Render the grading table and short per-case notes as Markdown."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
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


def main():
    p = ROOT / "results" / "grades_summary.json"
    rows = json.loads(p.read_text()) if p.exists() else []
    by = {(r["session"].split("/")[0], r["session"].split("/")[1]): r for r in rows}
    print("| 案例 | DeepSeek-V4-Pro | Kimi-K3 | 方向 (D/K) | 备注 |")
    print("|---|---:|---:|---|---|")
    for pid, title in TITLES.items():
        d = by.get(("deepseek", pid), {})
        k = by.get(("kimi", pid), {})
        ds = d.get("mean", "—")
        ks = k.get("mean", "—")
        dd = "/".join(sorted({str(v) for v in (d.get("directions") or {}).values()})) or "—"
        kd = "/".join(sorted({str(v) for v in (k.get("directions") or {}).values()})) or "—"
        print(f"| {title} | {ds} | {ks} | {dd}/{kd} | |")
    print()
    for r in rows:
        sid = r["session"]
        print(f"### {sid}")
        for g in ("deepseek", "kimi"):
            f = ROOT / "results" / "grades" / f"{sid.replace('/', '__')}__{g}.json"
            if not f.exists():
                continue
            o = json.loads(f.read_text())
            print(f"- **{g}**: score={o.get('score')} direction={o.get('direction')} "
                  f"proved={o.get('main_theorem_proved')}")
            if o.get("fatal_gaps"):
                print("  - gaps: " + "; ".join(str(x) for x in o["fatal_gaps"][:3]))
            if o.get("notes"):
                print(f"  - {o['notes'][:300]}")
        print()


if __name__ == "__main__":
    main()
