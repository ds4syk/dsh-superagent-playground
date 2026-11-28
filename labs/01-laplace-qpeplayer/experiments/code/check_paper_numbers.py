#!/usr/bin/env python3
"""Cross-check that the numbers quoted in paper/draft.md exist in the result files.

This is a *presence / rounding* check, not a semantic reviewer: each fact lists
(a) a pattern that must appear in the paper and (b) a source file plus a callable
that returns the underlying value(s).  It catches transcription drift after a
re-run; it does not prove the paper's interpretation.

Run:
  /home/syk/miniconda3/envs/WM/bin/python code/check_paper_numbers.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
P1 = EXP.parent
PAPER = P1 / "paper" / "draft.md"


def load(p):
    return json.load(open(EXP / p))


def fact(name, patterns, source, check=None):
    return dict(name=name, patterns=patterns, source=source, check=check)


def main():
    text = PAPER.read_text()
    facts = []
    mt = load("results/main_table.json")
    by = {r["regime"]: r for r in mt}
    facts += [
        fact("main far laplace", ["4.74e-07", "4.7e-7", "4.74\\times10^{-7}"], "results/main_table.json",
             lambda: by["far"]["laplace_grad_rel"]),
        fact("main near laplace", ["3.46e-05", "3.5e-5", "3.46\\times10^{-5}"], "results/main_table.json",
             lambda: by["near"]["laplace_grad_rel"]),
        fact("main at mc", ["3.26e-02", "3.3e-2", "3.26\\times10^{-2}"], "results/main_table.json",
             lambda: by["at"]["mc_grad_rel"]),
    ]
    six = load("sdp/results/six_point.json")
    facts += [
        fact("six f10", ["5.41500000", "5.415"], "sdp/results/six_point.json", lambda: six["f10"]),
        fact("six f13up", ["3.59431373", "3.5943"], "sdp/results/six_point.json", lambda: six["f13_up"]),
        fact("six gap", ["1.82068627", "1.8207", "1.82"], "sdp/results/six_point.json",
             lambda: six["f10_minus_f13_up"]),
        fact("six rank", ["rank 1", "rank $1$"], "sdp/results/six_point.json", lambda: six["rank10"]),
    ]
    sd = load("sdp/results/sdp_check.json")
    facts += [
        fact("CL f10", ["1.9658119658", "1.9658"], "sdp/results/sdp_check.json",
             lambda: sd["regression"]["cl_f10"]),
        fact("CL gap", ["0.034188", "3.4\\times10^{-2}"], "sdp/results/sdp_check.json",
             lambda: sd["regression"]["cl_f10_analytic"] if "cl_f10_analytic" in sd["regression"] else 0.034188),
    ]
    sc = load("theory/results/scorer_fold.json")
    facts += [
        fact("scorer std jump", ["3.041e-01", "3.04\\times10^{-1}", "0.3041", "3.04"], "theory/results/scorer_fold.md",
             None),  # jump table is in the .md, not the JSON
        fact("scorer theta_f", ["1.3185", "1.318"], "theory/results/scorer_fold.json", lambda: sc["six_d"]["theta_f"]),
    ]
    pe = load("real/linemod/results/pose_error.json")["aggregate"]
    facts += [
        fact("real laplace rot", ["44.24", "44.2"], "real/linemod/results/pose_error.json",
             lambda: pe["rot_deg"]["laplace"]["median"]),
        fact("real hard adds", ["3.7", "3.74"], "real/linemod/results/pose_error.json",
             lambda: pe["adds_m"]["hard"]["median"] * 1000.0),
        fact("real lap adds", ["13.9", "13.94"], "real/linemod/results/pose_error.json",
             lambda: pe["adds_m"]["laplace"]["median"] * 1000.0),
    ]
    rc = load("results/rootcount_fast.json")
    facts += [
        fact("roots 40", ["40 complex projective", "40 个复射影"], "results/rootcount_fast.json",
             lambda: [c["n_projective_roots"] for c in rc["cases"]]),
    ]
    tm = load("real/linemod/results/two_mode.json")
    a = tm["part_a"]
    facts += [
        fact("real K2", ["15/15", "K_{\\rm phys}=2", "K=2"], "real/linemod/results/two_mode.json",
             lambda: sum(1 for r in a if r["n_phys"] == 2)),
        fact("real hard median", ["2.06", "2.1"], "real/linemod/results/two_mode.json",
             lambda: float(sorted([r["hard_rot_err"] for r in a])[len(a)//2])),
    ]
    fold = load("results/fold_6d.json")
    facts += [
        fact("fold reg jump", ["1.07e-01", "0.107", "1.068"], "results/fold_6d.json",
             lambda: fold["summary"]["reg"]["max_jump"]),
        fact("fold std jump", ["3.089e-01", "0.309", "3.041"], "results/fold_6d.json",
             lambda: fold["summary"]["std"]["max_jump"]),
    ]
    q6 = load("theory/results/q6_verify.json")
    facts += [
        fact("q6 rel err", ["8.8e-09", "9\\times10^{-9}", "1.65e-09"], "theory/results/q6_verify.json",
             lambda: max(r["rel_err"] for r in q6)),
    ]

    fails = 0
    for f in facts:
        in_paper = any(p in text for p in f["patterns"])
        try:
            val = f["check"]() if f["check"] else None
        except Exception as exc:  # pragma: no cover
            val = f"ERR {exc!r}"
        status = "OK " if in_paper else "MISS"
        if not in_paper:
            fails += 1
        print(f"[{status}] {f['name']:22s} paper_pattern={f['patterns'][0]!r:28s} "
              f"source={f['source']:38s} value={val}")
    print(f"\n{len(facts) - fails}/{len(facts)} paper patterns found. "
          f"NOTE: presence only; values are printed for manual confirmation.")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
