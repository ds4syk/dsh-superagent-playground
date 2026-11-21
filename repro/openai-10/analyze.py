#!/usr/bin/env python3
"""Summarize transcripts: status, turns, chars, estimated tokens, STATUS line."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "sessions.csv"


def main():
    rows = []
    for m in ["deepseek", "kimi"]:
        for f in sorted((ROOT / "transcripts" / m).glob("*.json")):
            r = json.loads(f.read_text())
            turns = r.get("turns", [])
            content = ""
            for t in reversed(turns):
                if t.get("content"):
                    content = t["content"]
                    break
            status = ""
            sm = re.search(r"STATUS:\s*(COMPLETE_PROOF|COMPLETE_COUNTEREXAMPLE|PARTIAL|NO_RESOLUTION)",
                           content)
            if sm:
                status = sm.group(1)
            rows.append({
                "session": r["session"],
                "state": r.get("status"),
                "turns": len(turns),
                "content_chars": len(content),
                "reason_chars": sum(len(t.get("reasoning_content", "")) for t in turns),
                "est_tokens": round((len(content) + sum(len(t.get("reasoning_content", ""))
                                                        for t in turns)) / 3.5),
                "finish": ";".join(t.get("finish_reason") or "" for t in turns),
                "trunc": any(t.get("truncated_by_harness") for t in turns),
                "status_line": status,
                "usage_total": sum((u or {}).get("total_tokens", 0) or 0 for u in r.get("usage", [])),
            })
    OUT.parent.mkdir(exist_ok=True)
    import csv
    with OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else [])
        w.writeheader(); w.writerows(rows)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
