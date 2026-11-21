#!/usr/bin/env python3
"""Blind reproduction harness for the 10 public OpenAI-math cases.

Information isolation:
  * Each session sees only its own sanitized problem statement and its own turns.
  * No official paper, reasoning trace, grading key, or other model's output is ever
    placed in a session's context.
  * The API keys are read at runtime from ~/.dsh/.credentials.yaml and are never logged.

Calls are streamed and capped by a token budget so that a model cannot spend an
unbounded amount of time in hidden reasoning.  If the cap is hit before any final
content appears, the harness asks the model to consolidate its reasoning into a
final manuscript.  This is recorded as a budget-truncated turn, not hidden.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROMPTS = ROOT / "prompts"
TRANSCRIPTS = ROOT / "transcripts"
LOGS = ROOT / "logs"
CRED = Path.home() / ".dsh" / ".credentials.yaml"

PROBLEMS = [
    "007_two_point_chowla",
    "017_pi_irrationality",
    "087_mahler",
    "102_basic_sdp",
    "159_erdos_progressions",
    "197_kaplansky_direct_finiteness",
    "221_mezard_parisi",
    "271_heisenberg_magnetization",
    "287_free_group_factors",
    "362_vlasov_maxwell",
]

MODELS = {
    "deepseek": {
        "url": "https://api.deepseek.com/chat/completions",
        "model": "deepseek-v4-pro",
        "key_ref": "DEEPSEEK_API_KEY",
        "max_tokens": 24576,
        "temperature": 0.7,
    },
    "kimi": {
        "url": "https://api.moonshot.cn/v1/chat/completions",
        "model": "kimi-k3",
        "key_ref": "MOONSHOTAI_CN_API_KEY",
        "max_tokens": 24576,
        "temperature": None,  # kimi-k3 only accepts temperature=1
    },
}

SYSTEM = """You are a research mathematician working alone. You will be given one mathematical problem.
Work only from the problem statement and standard mathematical knowledge. Do not browse the internet,
do not use external retrieval, and do not rely on recollection of recent unpublished announcements,
preprints, or news; derive the argument from first principles. Your goal is a complete, rigorous
resolution: a proof, a rigorous counterexample, or a precisely stated no-go theorem. Partial results
are acceptable only when explicitly labelled and rigorously proved. Never present a conjecture or a
heuristic as a theorem."""

ROUND1 = """Produce your best complete solution. Write a self-contained mathematical manuscript with full
definitions, statements, and proofs. Clearly separate what is proved from what is conjectural or
unproved. End with a status line:
STATUS: COMPLETE_PROOF | COMPLETE_COUNTEREXAMPLE | PARTIAL | NO_RESOLUTION"""

ROUND2 = """Act as a hostile but fair referee of your previous manuscript, then produce the final version.

1. Audit line by line: identify every unproved claim, circular step, hidden hypothesis, incorrect use
   of a theorem, and any place where the claimed direction is not established.
2. Either repair the argument or state precisely why the gap is fatal. Do not claim a resolution you
   cannot rigorously defend.
3. Output the final self-contained manuscript. Include: the final claim; whether the target is
   proved, disproved, or unresolved; complete proofs of everything claimed; a precise gap list; and a
   short verification section. End with:
STATUS: COMPLETE_PROOF | COMPLETE_COUNTEREXAMPLE | PARTIAL | NO_RESOLUTION"""

FORCE_FINAL = """Your previous turn exhausted its output budget on internal reasoning and produced no final
answer. Its internal reasoning (possibly truncated) is reproduced below. Do not reason further about
the problem: consolidate that work into a final self-contained manuscript, with a clear status line and
an explicit gap list. If the reasoning is inconclusive, say so and give the strongest rigorously proved
partial result.

--- BEGIN PREVIOUS INTERNAL REASONING (truncated) ---
{reasoning}
--- END PREVIOUS INTERNAL REASONING ---

End with:
STATUS: COMPLETE_PROOF | COMPLETE_COUNTEREXAMPLE | PARTIAL | NO_RESOLUTION"""


def load_keys() -> dict:
    import yaml

    c = yaml.safe_load(CRED.read_text())
    return {k: c["refs"][k] for k in ("DEEPSEEK_API_KEY", "MOONSHOTAI_CN_API_KEY")}


def log(msg: str):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)


def _request(cfg: dict, key: str, messages: list, cap_tokens: int, timeout: int, extra: dict | None = None):
    body = {
        "model": cfg["model"],
        "messages": messages,
        "max_tokens": cap_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if cfg.get("temperature") is not None:
        body["temperature"] = cfg["temperature"]
    if extra:
        body.update(extra)
    data = json.dumps(body, ensure_ascii=False).encode()
    headers = {"Authorization": "Bearer " + key, "Content-Type": "application/json"}
    return urllib.request.Request(cfg["url"], data=data, headers=headers)


def api_call(cfg: dict, key: str, messages: list, cap_tokens: int | None = None,
             timeout: int = 1500, extra: dict | None = None) -> dict:
    """Stream one completion, stopping early when the token budget is exhausted.

    Returns a synthetic non-streaming response with content / reasoning_content /
    finish_reason / usage, plus _latency_s and _truncated.
    """
    cap_tokens = cap_tokens or cfg.get("max_tokens", 24576)
    # conservative char budget: ~3.5 chars/token for mixed English/LaTeX
    cap_chars = int(cap_tokens * 3.5)
    last_err = None
    for attempt in range(1, 4):
        try:
            req = _request(cfg, key, messages, cap_tokens, timeout, extra)
            t0 = time.time()
            content, reasoning = [], []
            usage, finish, truncated = {}, None, False
            n_chars = 0
            with urllib.request.urlopen(req, timeout=timeout) as r:
                for raw in r:
                    line = raw.decode("utf-8", "ignore").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                    except Exception:
                        continue
                    if obj.get("usage"):
                        usage = obj["usage"]
                    chs = obj.get("choices") or []
                    if not chs:
                        continue
                    ch = chs[0]
                    delta = ch.get("delta") or {}
                    piece = delta.get("reasoning_content") or ""
                    if piece:
                        reasoning.append(piece)
                        n_chars += len(piece)
                    piece = delta.get("content") or ""
                    if piece:
                        content.append(piece)
                        n_chars += len(piece)
                    if ch.get("finish_reason"):
                        finish = ch["finish_reason"]
                    if n_chars >= cap_chars:
                        truncated = True
                        finish = finish or "truncated_by_harness"
                        break
            out = {
                "choices": [{
                    "message": {"role": "assistant",
                                "content": "".join(content),
                                "reasoning_content": "".join(reasoning)},
                    "finish_reason": finish or ("stop" if content else "length"),
                }],
                "usage": usage,
                "_latency_s": round(time.time() - t0, 1),
                "_truncated": truncated,
                "_cap_tokens": cap_tokens,
            }
            return out
        except urllib.error.HTTPError as e:
            body_txt = e.read().decode("utf-8", "ignore")[:1500]
            last_err = f"HTTP {e.code}: {body_txt}"
            if e.code not in (429, 500, 502, 503, 504):
                break
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
        if attempt < 3:
            delay = 10 * attempt
            log(f"  retry {attempt}/3 in {delay}s: {last_err}")
            time.sleep(delay)
    raise RuntimeError(f"API call failed: {last_err}")


def save(path: Path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1))
    tmp.replace(path)


def run_session(model_name: str, problem: str, keys: dict, force: bool) -> dict:
    cfg = MODELS[model_name]
    out_dir = TRANSCRIPTS / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{problem}.json"
    if out_file.exists() and not force:
        try:
            old = json.loads(out_file.read_text())
            if old.get("status") == "done":
                return {"session": f"{model_name}/{problem}", "skipped": True}
        except Exception:
            pass

    prompt = (PROMPTS / f"{problem}.md").read_text()
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": prompt + "\n\n---\n\n" + ROUND1},
    ]
    rec = {
        "session": f"{model_name}/{problem}",
        "model": cfg["model"],
        "problem": problem,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "turns": [],
        "usage": [],
        "status": "running",
    }
    save(out_file, rec)
    log(f"START {model_name}/{problem}")

    def call(stage: str, cap: int, extra: dict | None = None) -> str:
        resp = api_call(cfg, keys[cfg["key_ref"]], messages, cap_tokens=cap, extra=extra)
        msg = resp["choices"][0]["message"]
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or ""
        rec["turns"].append({
            "stage": stage,
            "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "content": content,
            "reasoning_content": reasoning,
            "finish_reason": resp["choices"][0].get("finish_reason"),
            "latency_s": resp.get("_latency_s"),
            "truncated_by_harness": resp.get("_truncated"),
            "cap_tokens": cap,
        })
        rec["usage"].append(resp.get("usage", {}))
        save(out_file, rec)
        messages.append({"role": "assistant",
                         "content": content if content else
                         "[The previous turn produced internal reasoning but no final content.]"})
        log(f"  {model_name}/{problem} {stage}: content={len(content)} chars, "
            f"reason={len(reasoning)}, finish={resp['choices'][0].get('finish_reason')}, "
            f"trunc={resp.get('_truncated')}, lat={resp.get('_latency_s')}s")
        return content

    try:
        r1_cap = cfg["max_tokens"]
        content = call("s1_solve", r1_cap)
        if not content:
            # force a consolidation of the (partial) reasoning
            last = rec["turns"][-1]
            reasoning = (last.get("reasoning_content") or "")[-80000:]
            if reasoning:
                messages.append({"role": "user",
                                 "content": FORCE_FINAL.format(reasoning=reasoning)})
                content = call("s1_force_final", 12288,
                               extra={"thinking": {"type": "disabled"},
                                      "reasoning_effort": "none"})
        # audit + final
        messages.append({"role": "user", "content": ROUND2})
        content = call("s2_final", 16384)
        if not content:
            last = rec["turns"][-1]
            reasoning = (last.get("reasoning_content") or "")[-80000:]
            if reasoning:
                messages.append({"role": "user",
                                 "content": FORCE_FINAL.format(reasoning=reasoning)})
                call("s2_force_final", 12288,
                     extra={"thinking": {"type": "disabled"},
                            "reasoning_effort": "none"})

        rec["status"] = "done"
        rec["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        save(out_file, rec)
        log(f"DONE {model_name}/{problem}")
        return {"session": f"{model_name}/{problem}", "skipped": False,
                "chars": sum(len(t.get("content", "")) for t in rec["turns"]),
                "tokens": sum(u.get("total_tokens", 0) or 0 for u in rec["usage"])}
    except Exception as e:
        rec["status"] = "failed"
        rec["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        rec["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        save(out_file, rec)
        log(f"FAIL {model_name}/{problem}: {e}")
        return {"session": f"{model_name}/{problem}", "error": str(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--problems", default=",".join(PROBLEMS))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    LOGS.mkdir(parents=True, exist_ok=True)
    keys = load_keys()
    jobs = []
    for m in [x.strip() for x in args.models.split(",") if x.strip()]:
        for p in [x.strip() for x in args.problems.split(",") if x.strip()]:
            if p not in PROBLEMS:
                raise SystemExit(f"unknown problem {p}")
            jobs.append((m, p))
    log(f"running {len(jobs)} sessions, workers={args.workers}")
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_session, m, p, keys, args.force): (m, p) for m, p in jobs}
        for f in as_completed(futs):
            try:
                results.append(f.result())
            except Exception as e:
                results.append({"session": str(futs[f]), "error": repr(e)})
    (ROOT / "results").mkdir(exist_ok=True)
    save(ROOT / "results" / "run_summary.json", {
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "results": results,
    })
    log("ALL DONE")


if __name__ == "__main__":
    main()
