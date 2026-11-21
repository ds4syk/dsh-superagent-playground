#!/usr/bin/env python3
"""Contamination probes: ask each model what it remembers about the 10 topics.

These probes run in fresh sessions; their outputs are never fed back to the blind
sessions.  They estimate whether pretraining may already contain the official result.
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run_sessions as rs  # noqa: E402

TOPICS = {
    "007_two_point_chowla": "the binary corrected Elliott conjecture for ordinary averages of multiplicative functions, and the ordinary two-point Chowla conjecture",
    "017_pi_irrationality": "the exact irrationality exponent of pi (whether mu(pi)=2) and convergence of the Flint-Hills series",
    "087_mahler": "the symmetric and general geometric Mahler conjectures on volume products of convex bodies and their polars",
    "102_basic_sdp": "NP-hardness at the basic semidefinite programming threshold for fixed finite Max-CSPs, and the Unique Games Conjecture",
    "159_erdos_progressions": "Erdos's conjecture that a set of positive integers with divergent reciprocal sum contains arithmetic progressions of every finite length",
    "197_kaplansky_direct_finiteness": "Kaplansky's direct-finiteness conjecture for group algebras in positive characteristic",
    "221_mezard_parisi": "the Mezard-Parisi formula for diluted spin glasses",
    "271_heisenberg_magnetization": "spontaneous magnetization in the quantum Heisenberg ferromagnet for d>=3, and Bloch's T^{3/2} law",
    "287_free_group_factors": "isomorphism of the free group factors L(F_2) and L(F_3)",
    "362_vlasov_maxwell": "large-data global classical well-posedness for the 3-dimensional relativistic Vlasov-Maxwell system",
}

PROBE = """Answer from memory only; do not search the web and do not attempt to solve anything.
As of your knowledge cutoff, what is the current status of the following research problem?

Topic: {topic}

If you recall a specific 2025-2026 resolution, preprint, announcement, or claimed proof, state
the claimed result, the author/venue if remembered, and your confidence (low/medium/high). If you
do not have a reliable recollection, say exactly: "NO RELIABLE RECOLLECTION". Keep it under 250
words."""


def main():
    keys = rs.load_keys()
    outdir = ROOT / "contamination"
    outdir.mkdir(exist_ok=True)
    jobs = [(m, p) for m in ["deepseek", "kimi"] for p in TOPICS]
    results = []

    def one(m, p):
        cfg = dict(rs.MODELS[m])
        cfg["max_tokens"] = 4000
        msgs = [{"role": "user", "content": PROBE.format(topic=TOPICS[p])}]
        r = rs.api_call(cfg, keys[cfg["key_ref"]], msgs, timeout=600)
        msg = r["choices"][0]["message"]
        return {"model": m, "problem": p, "content": msg.get("content") or "",
                "reasoning": msg.get("reasoning_content") or "",
                "usage": r.get("usage", {})}

    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(one, m, p): (m, p) for m, p in jobs}
        for f in as_completed(futs):
            try:
                results.append(f.result())
            except Exception as e:
                results.append({"model": futs[f][0], "problem": futs[f][1], "error": repr(e)})
    (outdir / "probes.json").write_text(json.dumps(results, ensure_ascii=False, indent=1))
    for r in results:
        print(r["model"], r["problem"], (r.get("content") or r.get("error", ""))[:160].replace("\n", " "))


if __name__ == "__main__":
    main()
