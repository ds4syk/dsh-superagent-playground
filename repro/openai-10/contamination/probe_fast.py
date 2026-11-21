#!/usr/bin/env python3
"""Fast contamination probes: one short direct answer per (model, topic)."""

from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run_sessions as rs  # noqa: E402
from contamination.probe import TOPICS, PROBE  # noqa: E402


def one(model: str, problem: str, keys: dict) -> dict:
    cfg = dict(rs.MODELS[model])
    cfg["max_tokens"] = 1500
    r = rs.api_call(cfg, keys[cfg["key_ref"]],
                    [{"role": "user", "content": PROBE.format(topic=TOPICS[problem])}],
                    cap_tokens=1500, timeout=300,
                    extra={"thinking": {"type": "disabled"},
                           "reasoning_effort": "none"})
    msg = r["choices"][0]["message"]
    return {"model": model, "problem": problem, "content": msg.get("content") or ""}


def main():
    keys = rs.load_keys()
    out = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(one, m, p, keys) for m in ("deepseek", "kimi") for p in TOPICS]
        for f in as_completed(futs):
            r = f.result()
            out.append(r)
            print(r["model"], r["problem"], r["content"][:120].replace("\n", " "), flush=True)
    path = ROOT / "contamination" / "probes.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("DONE", path)


if __name__ == "__main__":
    main()
