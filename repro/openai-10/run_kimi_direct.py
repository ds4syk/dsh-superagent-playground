#!/usr/bin/env python3
"""Reduced Kimi experiment: one direct final-answer call per problem.

Kimi-K3's thinking mode was too slow for the full 2-stage protocol in this
ticket (7-8 min per 24k-token reasoning call, plus a second stage).  This
script keeps the blind input and the same grading, but asks for a direct final
manuscript with thinking disabled.  It is explicitly a reduced protocol; the
report must say so.
"""

from __future__ import annotations

import json
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import run_sessions as rs

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "transcripts" / "kimi"

PROMPT = """Write a complete, rigorous final manuscript for the problem above. You may not use tools,
external sources, or the internet. State your final claim, prove everything you claim, and list every
gap explicitly. If you cannot resolve the problem, give the strongest rigorously proved partial result
and say what remains open. End with:
STATUS: COMPLETE_PROOF | COMPLETE_COUNTEREXAMPLE | PARTIAL | NO_RESOLUTION"""


def one(problem: str, keys: dict) -> dict:
    cfg = dict(rs.MODELS["kimi"])
    cfg["max_tokens"] = 16384
    prompt = (rs.PROMPTS / f"{problem}.md").read_text()
    msgs = [
        {"role": "system", "content": rs.SYSTEM},
        {"role": "user", "content": prompt + "\n\n---\n\n" + PROMPT},
    ]
    resp = rs.api_call(cfg, keys["MOONSHOTAI_CN_API_KEY"], msgs, cap_tokens=16384,
                       timeout=1800,
                       extra={"thinking": {"type": "disabled"},
                              "reasoning_effort": "none"})
    msg = resp["choices"][0]["message"]
    content = msg.get("content") or ""
    rec = {
        "session": f"kimi/{problem}",
        "model": cfg["model"],
        "problem": problem,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "turns": [{
            "stage": "s1_direct",
            "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "content": content,
            "reasoning_content": msg.get("reasoning_content") or "",
            "finish_reason": resp["choices"][0].get("finish_reason"),
            "latency_s": resp.get("_latency_s"),
        }],
        "usage": [resp.get("usage", {})],
        "status": "done" if content else "failed",
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "protocol": "reduced: single direct call, thinking disabled",
    }
    rs.save(OUT / f"{problem}.json", rec)
    return {"session": rec["session"], "chars": len(content), "lat": resp.get("_latency_s")}


def main():
    keys = rs.load_keys()
    OUT.mkdir(parents=True, exist_ok=True)
    rs.log(f"Kimi reduced run: {len(rs.PROBLEMS)} direct calls")
    results = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(one, p, keys): p for p in rs.PROBLEMS}
        for f in as_completed(futs):
            p = futs[f]
            try:
                r = f.result()
                results.append(r)
                rs.log(f"DONE kimi/{p}: {r}")
            except Exception as e:
                rs.log(f"FAIL kimi/{p}: {e}")
                results.append({"session": f"kimi/{p}", "error": repr(e),
                                "trace": traceback.format_exc()})
    (ROOT / "results").mkdir(exist_ok=True)
    rs.save(ROOT / "results" / "run_summary_kimi_direct.json", {
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "results": results,
    })
    rs.log("KIMI DIRECT ALL DONE")


if __name__ == "__main__":
    main()
