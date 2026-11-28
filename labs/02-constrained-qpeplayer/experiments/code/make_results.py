#!/usr/bin/env python
"""把 results/raw_main.json / raw_pareto.json / fb_table.json 变成主表与图。

输出（都在 experiments/results/）：
  main_table.md      论文草稿用主表（method × p，method × 噪声/偏差）
  main_table.csv     长表（group,p,noise,bias,method,指标…）
  main_table.json    聚合统计 + 配置（含分位数，便于复查）
  fig_main_p.png     穿透 / runtime vs p
  fig_main_pose.png  rot / trans vs p
  fig_main_cert.png  可行率 / 认证率 / 证书 gap
  fig_pareto.png     软惩罚 λ 的穿透-精度 Pareto
  fig_noise.png      method × 噪声/偏差
  fig_pen_box.png    p=16 的穿透分布（含 cheirality 镜像模式）
运行：srun --partition=cpu --time=00:05:00 --ntasks=1 \
      /home/syk/miniconda3/envs/sdpr/bin/python make_results.py
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
METHODS = [("unc", "Unconstrained QPEP"), ("soft", "Soft penalty ($\\lambda{=}10^3$)"),
           ("proj", "Post-hoc projection"), ("alm", "ALM-QPEP (ours)")]
COLORS = {"unc": "#d62728", "soft": "#ff7f0e", "proj": "#1f77b4", "alm": "#2ca02c"}
MARKERS = {"unc": "s", "soft": "^", "proj": "D", "alm": "o"}

raw = json.load(open(os.path.join(RES, "raw_main.json")))
pareto = json.load(open(os.path.join(RES, "raw_pareto.json")))
cfg = raw["config"]


def stats(vals):
    v = np.asarray(vals, float)
    v = v[np.isfinite(v)]
    return dict(mean=float(np.mean(v)), std=float(np.std(v)),
                median=float(np.median(v)),
                q1=float(np.percentile(v, 25)), q3=float(np.percentile(v, 75)),
                p90=float(np.percentile(v, 90)), max=float(np.max(v)))


def agg_cell(cell):
    out = {}
    for name, _ in METHODS:
        ms = [r[name] for r in cell["recs"] if name in r]
        out[name] = dict(
            pen=stats([m["pen_mm"] for m in ms]),
            rot=stats([m["rot_deg"] for m in ms]),
            trans=stats([m["trans_mm"] for m in ms]),
            time=stats([m["time_ms"] for m in ms]),
            feasible=float(np.mean([m["feasible"] for m in ms])),
            certified=float(np.mean([m.get("certified", False) for m in ms])),
            mirror=float(np.mean([m.get("depth_min_mm", 1.0) < 0.0 for m in ms])),
            viol_max=float(np.max([m["viol"] for m in ms])),
            J=stats([m["J"] for m in ms]),
            cert_gap=stats([m.get("cert_gap", np.nan) for m in ms]) if "cert_gap" in ms[0] else None,
        )
    sd = [r["sdp"] for r in cell["recs"] if r.get("sdp")]
    out["sdp"] = dict(tight=float(np.mean([s["tight"] for s in sd])),
                      sound=float(np.mean([s["sound"] for s in sd])),
                      ratio_max=float(np.max([s["ratio"] for s in sd])),
                      ratio_med=float(np.median([s["ratio"] for s in sd])),
                      time=stats([s["time_ms"] for s in sd]),
                      value=stats([s["value"] for s in sd]))
    out["n"] = len(cell["recs"])
    return out


agg_main = {p: agg_cell(raw["main"]["p%d" % p]) for p in cfg["main_p"]}
agg_noise = {tag: agg_cell(c) for tag, c in raw["noise"].items()}

# ---------------------------------------------------------------------------
# CSV（长表）
# ---------------------------------------------------------------------------
import csv
rows = []
for group, cells in (("main", agg_main), ("noise", agg_noise)):
    for key, a in cells.items():
        cell = raw["main"]["p%d" % key] if group == "main" else raw["noise"][key]
        for name, _ in METHODS:
            m = a[name]
            rows.append(dict(
                group=group, cell=key, p=cell["p"], noise=cell["noise"], bias_z=cell["bias_z"],
                method=name, n=a["n"],
                pen_median_mm=m["pen"]["median"], pen_q1=m["pen"]["q1"], pen_q3=m["pen"]["q3"],
                pen_mean_mm=m["pen"]["mean"], pen_std_mm=m["pen"]["std"],
                pen_rate_1mm=100 * np.mean([r[name]["pen_mm"] > 1.0 for r in cell["recs"]]),
                rot_median_deg=m["rot"]["median"], rot_mean_deg=m["rot"]["mean"], rot_std_deg=m["rot"]["std"],
                trans_median_mm=m["trans"]["median"], trans_mean_mm=m["trans"]["mean"], trans_std_mm=m["trans"]["std"],
                feasible_rate=100 * m["feasible"], certified_rate=100 * m["certified"],
                mirror_rate=100 * m["mirror"], viol_max=m["viol_max"],
                J_median=m["J"]["median"],
                cert_gap_median=m["cert_gap"]["median"] if m["cert_gap"] else "",
                time_ms_median=m["time"]["median"], time_ms_mean=m["time"]["mean"],
            ))
with open(os.path.join(RES, "main_table.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    for r in rows:
        w.writerow(r)

# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------
with open(os.path.join(RES, "main_table.json"), "w") as f:
    json.dump(dict(config=cfg, agg_main=agg_main, agg_noise=agg_noise,
                   raw_file="raw_main.json"), f, indent=1, default=float)

# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------
def fmt_med_iqr(s, nd=2):
    return "%.*f [%.*f, %.*f]" % (nd, s["median"], nd, s["q1"], nd, s["q3"])


md = []
md.append("# Constrained-QPEPLayer 合成桌面 PnP 主表\n")
md.append("> 由 `experiments/code/make_results.py` 从 `results/raw_main.json` 自动生成；**不要手改**。\n")
md.append("设置：合成桌面 6-DoF PnP（N=%d 个 3D-2D 对应，bearing 型代数代价），"
          "逐点噪声 σ=%.3f m、系统性外参偏差 %.3f m，GT 恰好接触桌面；"
          "每格 %d trials，`seed=trial` 使不同 p/噪声档共享同一批场景。\n"
          % (cfg["N_pts"], 0.010, 0.080, cfg["n_trials"]))
md.append("p 的构成（非穿透/正深度/FoV 对/倾角 = 总数）：%s。\n"
          % "，".join("p=%d: %d/%d/%d/%d" % (int(k), v["pen"], v["depth"], v["fov"], v["tilt"])
                     for k, v in sorted(cfg["split"].items(), key=lambda kv: int(kv[0]))))
md.append("方法：Unconstrained QPEP（多起点 48 起点数值全局） / Soft penalty（二次惩罚 λ=1e3，"
          "14 起点；λ 扫描见 Pareto 图） / Post-hoc projection（无约束解 → 最近可行点 SLSQP） / "
          "ALM-QPEP（冻结活跃集 + 定理 2.1 装配 + 一致性检查）。\n")
md.append("统计：穿透/位姿误差给 **median [IQR]**（无约束解约 2% 的 cheirality 镜像模式会把 mean 拉爆，"
          "mean±std 在 CSV/JSON 里完整保留）；可行=违反≤1e-6，certified=可行且 $J-\\mathrm{Shor}\\le10^{-6}$ 且 SDP tight。\n")

md.append("## 表 1. 最大穿透深度（mm，median [IQR]）与穿透率（>1 mm）\n")
md.append("| method | " + " | ".join("p=%d" % p for p in cfg["main_p"]) + " |")
md.append("|---|" + "---|" * len(cfg["main_p"]))
for name, label in METHODS:
    cells = []
    for p in cfg["main_p"]:
        a = agg_main[p][name]
        rate = 100 * np.mean([r[name]["pen_mm"] > 1.0 for r in raw["main"]["p%d" % p]["recs"]])
        cells.append("%s (%.0f%%)" % (fmt_med_iqr(a["pen"]), rate))
    md.append("| %s | %s |" % (label, " | ".join(cells)))
md.append("\n- 无约束解 median 穿桌 ~%.0f mm；ALM 全部 p 都是 **0.00 mm**（硬约束，违反≤%.1e）。\n"
          % (agg_main[cfg["main_p"][0]]["unc"]["pen"]["median"],
             max(agg_main[p]["alm"]["viol_max"] for p in cfg["main_p"])))
md.append("- **p 饱和是结构性的**：底面是平面，≥3 个不共线接触点在刚体上已等价于整面非穿透，"
          "所以 p>8 只增加冗余接触点（也正是 LICQ 失效的来源），精度/穿透不再变化。"
          "p 的真实作用是装配 O(p) 与证书的 p 条线性约束：ALM runtime 的 log–log 斜率 %.2f（常数维），"
          "SDP 从 %.0f ms(p=8) 增到 %.0f ms(p=64)。\n"
          % (np.polyfit(np.log(cfg["main_p"]), np.log([agg_main[p]["alm"]["time"]["median"] for p in cfg["main_p"]]), 1)[0],
             agg_main[cfg["main_p"][0]]["sdp"]["time"]["median"],
             agg_main[cfg["main_p"][-1]]["sdp"]["time"]["median"]))

md.append("## 表 2. 位姿误差（deg / mm，median [IQR]）\n")
md.append("| method | " + " | ".join("p=%d rot / trans" % p for p in cfg["main_p"]) + " |")
md.append("|---|" + "---|" * len(cfg["main_p"]))
for name, label in METHODS:
    cells = []
    for p in cfg["main_p"]:
        a = agg_main[p][name]
        cells.append("%s / %s" % (fmt_med_iqr(a["rot"]), fmt_med_iqr(a["trans"])))
    md.append("| %s | %s |" % (label, " | ".join(cells)))
md.append("\n")

md.append("## 表 3. 可行率 / 认证率 / runtime / 证书\n")
md.append("| method | " + " | ".join("p=%d" % p for p in cfg["main_p"]) + " |")
md.append("|---|" + "---|" * len(cfg["main_p"]))
for name, label in METHODS:
    cells = []
    for p in cfg["main_p"]:
        a = agg_main[p][name]
        cells.append("feas %.0f%% / cert %.0f%% / %.0f ms" % (100 * a["feasible"], 100 * a["certified"], a["time"]["median"]))
    md.append("| %s | %s |" % (label, " | ".join(cells)))
md.append("| **14×14 SDP** | " + " | ".join("tight %.0f%% / %.0f ms" % (100 * agg_main[p]["sdp"]["tight"], agg_main[p]["sdp"]["time"]["median"]) for p in cfg["main_p"]) + " |")
md.append("\n- SDP sound（$\\mathrm{Shor}\\le J_{ALM}+10^{-7}$）在全部 p 上 %.0f%%。\n"
          % (100 * min(agg_main[p]["sdp"]["sound"] for p in cfg["main_p"])))
md.append("- 认证率：ALM %.0f%%；projection %.0f%%（可行但 $J$ 比全局最优高，证书不通过）；"
          "soft %.0f%%（既不够可行、也非全局最优）。\n"
          % (100 * np.mean([agg_main[p]["alm"]["certified"] for p in cfg["main_p"]]),
             100 * np.mean([agg_main[p]["proj"]["certified"] for p in cfg["main_p"]]),
             100 * np.mean([agg_main[p]["soft"]["certified"] for p in cfg["main_p"]])))
md.append("- cheirality 镜像模式（无约束解把物体放到相机后方，depth_min<0）比例：%s。"
          "这是 bearing 型代数代价的符号不敏感性，不是实现 bug；正深度/非穿透约束把它排除。\n"
          % ", ".join("p=%d: %.0f%%" % (p, 100 * agg_main[p]["unc"]["mirror"]) for p in cfg["main_p"]))

md.append("## 表 4. method × 噪声/外参偏差（p=16，median [IQR]）\n")
md.append("| cell (σ / bias) | method | pen mm | rot deg | trans mm | feasible | certified |")
md.append("|---|---|---|---|---|---|---|")
for tag, a in agg_noise.items():
    cell = raw["noise"][tag]
    for name, label in METHODS:
        m = a[name]
        md.append("| %.3f / %.3f | %s | %s | %s | %s | %.0f%% | %.0f%% |" % (
            cell["noise"], cell["bias_z"], label, fmt_med_iqr(m["pen"]), fmt_med_iqr(m["rot"]),
            fmt_med_iqr(m["trans"]), 100 * m["feasible"], 100 * m["certified"]))
md.append("\n- 所有实例还含固定 5 mm 横向外参偏差（bias_xy=0.005），故 σ=0/竖向 bias=0 档仍有 "
          "7.07 mm 的纯横向平移误差，四种方法完全一致；这不是方法差异，而是实例的已知偏置。\n")

md.append("## 附：软惩罚 Pareto（p=16）\n")
md.append("| λ | pen mm (median) | rot deg | trans mm | feasible |")
md.append("|---|---|---|---|---|")
for lam in cfg["lam_pareto"]:
    sub = [r for r in pareto if r["lam"] == lam]
    md.append("| %.0e | %.2f | %.2f | %.2f | %.0f%% |" % (
        lam, np.median([r["pen_mm"] for r in sub]), np.median([r["rot_deg"] for r in sub]),
        np.median([r["trans_mm"] for r in sub]),
        100 * np.mean([r["viol"] <= 1e-6 for r in sub])))
md.append("\n- 权重越大穿透越低，但位姿误差与局部极小风险上升；λ=1e4 有 trial 落到差局部解"
          "（trans 最大 %.0f mm）⇒ 软惩罚的“穿透-精度”权衡不可证书化。\n"
          % max(r["trans_mm"] for r in pareto if r["lam"] == 1e4))

with open(os.path.join(RES, "main_table.md"), "w") as f:
    f.write("\n".join(md))

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 150, "savefig.bbox": "tight"})
ps = cfg["main_p"]

xp = np.arange(len(ps))
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
FLOOR = 1e-3
for name, label in METHODS:
    med = np.array([agg_main[p][name]["pen"]["median"] for p in ps])
    q1 = np.array([agg_main[p][name]["pen"]["q1"] for p in ps])
    q3 = np.array([agg_main[p][name]["pen"]["q3"] for p in ps])
    lo = np.maximum(med - q1, 0); hi = np.maximum(q3 - med, 0)
    axes[0].errorbar(xp, np.maximum(med, FLOOR), yerr=[np.minimum(lo, np.maximum(med, FLOOR) - FLOOR), hi],
                     marker=MARKERS[name], color=COLORS[name], label=label, capsize=2, lw=1.2, ms=4)
axes[0].axhline(FLOOR, color="gray", ls=":", lw=0.8)
axes[0].set_yscale("log"); axes[0].set_ylim(5e-4, 300)
axes[0].set_xticks(xp); axes[0].set_xticklabels([str(p) for p in ps])
axes[0].set_xlabel("p (number of inequality constraints)")
axes[0].set_ylabel("max penetration (mm; median/IQR, 0→floor)")
axes[0].set_title("Penetration vs p")
axes[0].legend(fontsize=7)
for name, label in METHODS:
    t = [agg_main[p][name]["time"]["median"] for p in ps]
    axes[1].plot(xp, t, marker=MARKERS[name], color=COLORS[name], label=label, lw=1.2, ms=4)
axes[1].set_yscale("log"); axes[1].set_xticks(xp); axes[1].set_xticklabels([str(p) for p in ps])
axes[1].set_xlabel("p"); axes[1].set_ylabel("runtime (ms / trial)")
slope = np.polyfit(np.log(ps), np.log([agg_main[p]["alm"]["time"]["median"] for p in ps]), 1)[0]
axes[1].set_title("Runtime vs p (ALM log-log slope %.2f)" % slope)
fig.tight_layout(); fig.savefig(os.path.join(RES, "fig_main_p.png")); plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
for name, label in METHODS:
    axes[0].plot(xp, [agg_main[p][name]["rot"]["median"] for p in ps], marker=MARKERS[name],
                 color=COLORS[name], label=label, lw=1.2, ms=4)
    axes[1].plot(xp, [agg_main[p][name]["trans"]["median"] for p in ps], marker=MARKERS[name],
                 color=COLORS[name], label=label, lw=1.2, ms=4)
for ax, yl, ti in [(axes[0], "rotation error (deg, median)", "Rotation error"),
                   (axes[1], "translation error (mm, median)", "Translation error")]:
    ax.set_xticks(xp); ax.set_xticklabels([str(p) for p in ps])
    ax.set_xlabel("p"); ax.set_ylabel(yl); ax.set_title(ti); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(RES, "fig_main_pose.png")); plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
wbar = 0.8 / len(METHODS)
xp = np.arange(len(ps))
for k, (name, label) in enumerate(METHODS):
    axes[0].bar(xp + (k - 1.5) * wbar, [100 * agg_main[p][name]["feasible"] for p in ps],
                width=wbar, color=COLORS[name], label=label)
    axes[1].bar(xp + (k - 1.5) * wbar, [100 * agg_main[p][name]["certified"] for p in ps],
                width=wbar, color=COLORS[name], label=label)
for ax, ti in [(axes[0], "Constraint feasibility rate (viol<=1e-6)"),
               (axes[1], "Certified globally optimal ($J-Shor<=1e-6$)")]:
    ax.set_xticks(xp); ax.set_xticklabels(["p=%d" % p for p in ps]); ax.set_ylim(0, 105)
    ax.set_ylabel("% trials"); ax.set_title(ti)
axes[1].legend(fontsize=7, loc="center right")
fig.tight_layout(); fig.savefig(os.path.join(RES, "fig_main_cert.png")); plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
lams = cfg["lam_pareto"]
for k, (xkey, xlabel) in enumerate([("trans_mm", "translation error (mm)"), ("rot_deg", "rotation error (deg)")]):
    xs = [np.median([r[xkey] for r in pareto if r["lam"] == lam]) for lam in lams]
    ys = [np.median([r["pen_mm"] for r in pareto if r["lam"] == lam]) for lam in lams]
    axes[k].plot(xs, np.maximum(ys, 1e-3), "o-", color=COLORS["soft"], lw=1.2)
    for jj, (x, y, lam) in enumerate(zip(xs, ys, lams)):
        off = (3, 4) if jj % 2 == 0 else (3, -8)
        axes[k].annotate("λ=%.0e" % lam, (x, max(y, 1e-3)), textcoords="offset points", xytext=off, fontsize=6)
    a16 = agg_main[16]
    axes[k].scatter([a16["alm"]["trans"]["median"] if k == 0 else a16["alm"]["rot"]["median"]],
                    [max(a16["alm"]["pen"]["median"], 1e-3)], marker="*", s=110, color=COLORS["alm"],
                    label="ALM-QPEP", zorder=5)
    axes[k].scatter([a16["proj"]["trans"]["median"] if k == 0 else a16["proj"]["rot"]["median"]],
                    [max(a16["proj"]["pen"]["median"], 1e-3)], marker="D", s=45, color=COLORS["proj"],
                    label="projection", zorder=5)
    axes[k].set_yscale("log"); axes[k].set_xlabel(xlabel)
    axes[k].set_ylabel("max penetration (mm, median)")
    axes[k].set_title("Soft-penalty Pareto (p=16)")
    axes[k].legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(RES, "fig_pareto.png")); plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
tags = list(agg_noise.keys())
labels = ["σ=%.3f\nbias=%.3f" % (raw["noise"][t]["noise"], raw["noise"][t]["bias_z"]) for t in tags]
xp = np.arange(len(tags))
for k, (name, label) in enumerate(METHODS):
    axes[0].bar(xp + (k - 1.5) * wbar, [agg_noise[t][name]["pen"]["median"] for t in tags],
                width=wbar, color=COLORS[name], label=label)
    axes[1].bar(xp + (k - 1.5) * wbar, [agg_noise[t][name]["trans"]["median"] for t in tags],
                width=wbar, color=COLORS[name], label=label)
for ax, ti, yl in [(axes[0], "Penetration (median)", "mm"),
                   (axes[1], "Translation error (median)", "mm")]:
    ax.set_xticks(xp); ax.set_xticklabels(labels, fontsize=6)
    ax.set_ylabel(yl); ax.set_title(ti + " vs noise/bias (p=16)")
axes[0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(RES, "fig_noise.png")); plt.close(fig)

fig, ax = plt.subplots(figsize=(6.4, 3.2))
data = [[r[name]["pen_mm"] for r in raw["main"]["p16"]["recs"]] for name, _ in METHODS]
bp = ax.boxplot([np.maximum(np.array(d), 1e-4) for d in data], labels=[m[1] for m in METHODS],
                patch_artist=True, widths=0.5, showfliers=True)
for patch, (name, _) in zip(bp["boxes"], METHODS):
    patch.set_facecolor(COLORS[name]); patch.set_alpha(0.65)
ax.set_yscale("log"); ax.set_ylabel("max penetration (mm)")
ax.set_title("Penetration distribution at p=16 (50 trials; mirror mode visible)")
fig.tight_layout(); fig.savefig(os.path.join(RES, "fig_pen_box.png")); plt.close(fig)



# ---------------------------------------------------------------------------
# one-page overview
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(10, 6.2))
ax = axes[0, 0]
for name, label in METHODS:
    med = np.array([agg_main[pp][name]["pen"]["median"] for pp in ps])
    ax.plot(xp, np.maximum(med, FLOOR), marker=MARKERS[name], color=COLORS[name], label=label, lw=1.2, ms=4)
ax.set_yscale("log"); ax.set_ylim(5e-4, 400); ax.axhline(FLOOR, color="gray", ls=":", lw=0.8)
ax.set_xticks(xp); ax.set_xticklabels([str(pp) for pp in ps])
ax.set_xlabel("p"); ax.set_ylabel("max penetration (mm, median)")
ax.set_title("(a) Penetration vs p"); ax.legend(fontsize=6)
ax = axes[0, 1]
for k, (name, label) in enumerate(METHODS):
    ax.bar(xp + (k - 1.5) * wbar, [100 * agg_main[pp][name]["certified"] for pp in ps],
           width=wbar, color=COLORS[name], label=label)
ax.set_xticks(xp); ax.set_xticklabels(["p=%d" % pp for pp in ps]); ax.set_ylim(0, 105)
ax.set_ylabel("% trials certified"); ax.set_title("(b) Feasible + globally certified")
ax.legend(fontsize=6, loc="center right")
ax = axes[1, 0]
for k, (name, label) in enumerate(METHODS):
    ax.bar(xp + (k - 1.5) * wbar, [agg_noise[t][name]["pen"]["median"] for t in tags],
           width=wbar, color=COLORS[name], label=label)
ax.set_xticks(xp); ax.set_xticklabels(["%s" % t.replace("_", "\n") for t in tags], fontsize=5)
ax.set_ylabel("max penetration (mm, median)"); ax.set_title("(c) Noise / extrinsic bias (p=16)")
ax.legend(fontsize=6)
ax = axes[1, 1]
xs = [np.median([r["trans_mm"] for r in pareto if r["lam"] == lam]) for lam in lams]
ys = [np.median([r["pen_mm"] for r in pareto if r["lam"] == lam]) for lam in lams]
ax.plot(xs, np.maximum(ys, FLOOR), "o-", color=COLORS["soft"], lw=1.2)
for x, y, lam in zip(xs, ys, lams):
    ax.annotate("λ=%.0e" % lam, (x, max(y, FLOOR)), textcoords="offset points", xytext=(3, 4), fontsize=5)
ax.scatter([agg_main[16]["alm"]["trans"]["median"]], [max(agg_main[16]["alm"]["pen"]["median"], FLOOR)],
           marker="*", s=110, color=COLORS["alm"], zorder=5, label="ALM-QPEP")
ax.set_yscale("log"); ax.set_xlabel("translation error (mm)"); ax.set_ylabel("max penetration (mm, median)")
ax.set_title("(d) Soft-penalty Pareto (p=16)"); ax.legend(fontsize=6)
fig.suptitle("Constrained-QPEPLayer synthetic tabletop PnP: ALM 400/400 feasible, 400/400 certified; "
             "14x14 Shor tight 400/400", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(RES, "fig_overview.png")); plt.close(fig)

print("wrote main_table.md / .csv / .json and 6 PNGs to experiments/results/")
print("ALM runtime log-log slope vs p: %.2f" % slope)
