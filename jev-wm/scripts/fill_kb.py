"""Fill the report results sections from result JSONs (idempotent)."""

from __future__ import annotations
import glob, json, os, re

KB = "../kb/jev-wm"
SHORT = {
    "persistence": "persistence",
    "tacit-Tacit-4B": "Tacit-4B (typed)",
    "decider-Qwen3-4B-Instruct-2507-raw": "Qwen3-4B (typed)",
    "generative-Qwen3-4B-Instruct-2507": "Qwen3-4B (gen.)",
}


def load(p):
    with open(p) as f:
        return json.load(f)


def exp1_table():
    rows = []
    for p in sorted(glob.glob("results/exp1_*.json")):
        d = load(p)
        k = d.get("exact_by_action_kind", {})
        r = d.get("rollout", {})
        k4 = r.get("exact_by_k", {}).get("4", r.get("exact_by_k", {}).get(4))
        rows.append((SHORT.get(d["backend"], d["backend"]), d, k, k4))
    if not rows:
        return "（尚无结果）\n"
    out = ["| 模型 | 一步精确匹配 | 执行判断 | 变量宏平均 | observe | simple | derive | 4 步精确匹配 | 延迟/样本 |",
           "|-|-|-|-|-|-|-|-|-|"]
    for name, d, k, k4 in rows:
        out.append(f"| {name} | {d['exact_match']:.3f} | {d['exec_accuracy']:.3f} | "
                   f"{d['macro_variable_accuracy']:.3f} | {_fmt(k.get('observe'))} | {_fmt(k.get('simple'))} | "
                   f"{_fmt(k.get('derive'))} | {_fmt(k4)} | {d.get('mean_latency_s', 0)*1000:.1f} ms |")
    return "\n".join(out) + "\n"


def _fmt(x):
    return "—" if x is None else f"{x:.3f}"


def exp2_table():
    p = "results/exp2_eval.json"
    if not os.path.exists(p):
        return "（尚无结果）\n"
    d = load(p)
    names = [n for n in ["random", "oracle", "choice", "score", "wm"] if n in d] + \
            [k for k in d if k.startswith("student")]
    out = ["| 策略 | 成功率 | 平均步数 | 每回合 JEV 决策数 | 无效动作率 |", "|-|-|-|-|-|"]
    for n in names:
        r = d[n]
        out.append(f"| {n} | {r['success_rate']:.3f} | {r['mean_steps']:.2f} | "
                   f"{r.get('decisions_per_episode', 0):.1f} | {r.get('invalid_rate', 0):.3f} |")
    return "\n".join(out) + "\n"


def replace(path, start, end, body):
    s = open(path).read()
    pat = re.compile(re.escape(start) + r".*?" + re.escape(end), re.S)
    s2 = pat.sub(start + "\n" + body + end, s)
    open(path, "w").write(s2)
    print("updated", path)


if __name__ == "__main__":
    replace(os.path.join(KB, "04.md"), "<!-- EXP1_TABLE_START -->", "<!-- EXP1_TABLE_END -->", exp1_table())
    replace(os.path.join(KB, "05.md"), "<!-- EXP2_TABLE_START -->", "<!-- EXP2_TABLE_END -->", exp2_table())
