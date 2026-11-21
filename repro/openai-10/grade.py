#!/usr/bin/env python3
"""Grade blind-reproduction transcripts against the official resolutions.

Two graders (DeepSeek/Kimi by default) independently score each final manuscript
on a 0-5 rubric; the script stores both JSON verdicts and a consensus field.  The
grading key is never placed in a solver session; it is only used here.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import run_sessions as rs

ROOT = Path(__file__).resolve().parent
KEY = ROOT.parent.parent / "research" / "ai-math-survey" / "official_key.json"

GRADER_SYSTEM = """You are an expert mathematical referee. You will receive:
(A) the official resolution of a research problem, for grading only;
(B) a candidate manuscript written independently, without access to (A).

Judge the candidate against the exact target statement. Be strict and concrete. Do not give
credit for plausible prose, name-dropping, or a proof of a different/weaker statement. A
counterexample must be explicit and checkable. A proof must address every quantifier and
hypothesis. If the candidate claims a result but a key step is unproved, say so.

Return ONLY a JSON object with this schema:
{
  "direction": "correct" | "wrong" | "unclear",
  "main_theorem_proved": true | false,
  "score": 0-5,
  "strongest_valid_partial_result": "one sentence, or empty",
  "fatal_gaps": ["at most 5 short items"],
  "evidence_quotes": ["at most 3 short exact quotes from the candidate"],
  "notes": "one short paragraph"
}

Score rubric:
5 = complete correct proof/counterexample of the exact target;
4 = correct direction, main theorem essentially proved, one minor fixable gap;
3 = substantial rigorous partial result (key lemma/special case/route with proved steps);
2 = correct background and some correct non-core observations;
1 = fatal error, wrong direction, or conjecture presented as theorem;
0 = no meaningful mathematical content."""

GRADER_USER = """## (A) Official resolution (grading key)

Problem: {title}
Official resolution: {official}
Source: {source}

## (B) Candidate manuscript

{candidate}

## Task

Return the JSON verdict now."""


def load_key():
    return json.loads(KEY.read_text())


def final_content(rec: dict) -> str:
    turns = rec.get("turns", [])
    for t in reversed(turns):
        if t.get("content"):
            return t["content"]
    return ""


def grade_one(grader: str, sid: str, rec: dict, key: dict, keys: dict) -> dict:
    problem = rec["problem"]
    cand = final_content(rec)
    if not cand:
        return {"session": sid, "grader": grader, "score": 0,
                "direction": "unclear", "main_theorem_proved": False,
                "fatal_gaps": ["candidate produced no final content (reasoning budget exhausted)"],
                "raw": None}
    cfg = dict(rs.MODELS[grader])
    cfg["max_tokens"] = 8000
    msgs = [
        {"role": "system", "content": GRADER_SYSTEM},
        {"role": "user", "content": GRADER_USER.format(
            title=key["title"], official=key["official"], source=key["source"],
            candidate=cand[:200000])},
    ]
    txt = ""
    try:
        resp = rs.api_call(cfg, keys[cfg["key_ref"]], msgs, timeout=1800,
                           extra={"thinking": {"type": "disabled"},
                                  "reasoning_effort": "none"})
        txt = resp["choices"][0]["message"].get("content") or ""
        m = re.search(r"\{.*\}", txt, flags=re.S)
        obj = json.loads(m.group(0)) if m else {}
        obj.update({"session": sid, "grader": grader, "_usage": resp.get("usage", {})})
        return obj
    except Exception as e:
        return {"session": sid, "grader": grader, "error": repr(e), "raw": txt}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graders", default="deepseek,kimi")
    ap.add_argument("--only", default="")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    keys = rs.load_keys()
    official = load_key()
    outdir = ROOT / "results" / "grades"
    outdir.mkdir(parents=True, exist_ok=True)
    graders = [g.strip() for g in args.graders.split(",") if g.strip()]
    jobs = []
    for m in ["deepseek", "kimi"]:
        for p in rs.PROBLEMS:
            if args.only and p not in args.only:
                continue
            f = ROOT / "transcripts" / m / f"{p}.json"
            if not f.exists():
                continue
            sid = f"{m}/{p}"
            rec = json.loads(f.read_text())
            for g in graders:
                jobs.append((g, sid, rec, p))
    rs.log(f"grading {len(jobs)} session-grader pairs")
    results = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {}
        for g, sid, rec, p in jobs:
            out = outdir / f"{sid.replace('/', '__')}__{g}.json"
            if out.exists() and not args.force:
                results.append(json.loads(out.read_text()))
                continue
            futs[ex.submit(grade_one, g, sid, rec, official[p], keys)] = out
        for f in as_completed(futs):
            out = futs[f]
            try:
                obj = f.result()
            except Exception as e:
                obj = {"error": repr(e)}
            out.write_text(json.dumps(obj, ensure_ascii=False, indent=1))
            results.append(obj)
            rs.log(f"  graded {out.name}: score={obj.get('score')} err={obj.get('error')}")
    # consensus table
    by = {}
    for o in results:
        by.setdefault(o["session"], {})[o.get("grader")] = o
    rows = []
    for sid, gs in sorted(by.items()):
        scores = [g.get("score") for g in gs.values() if isinstance(g.get("score"), int)]
        rows.append({
            "session": sid,
            "scores": {k: v.get("score") for k, v in gs.items()},
            "mean": round(sum(scores) / len(scores), 2) if scores else None,
            "directions": {k: v.get("direction") for k, v in gs.items()},
            "proved": {k: v.get("main_theorem_proved") for k, v in gs.items()},
        })
    (ROOT / "results" / "grades_summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    rs.log("wrote results/grades_summary.json")


if __name__ == "__main__":
    main()
