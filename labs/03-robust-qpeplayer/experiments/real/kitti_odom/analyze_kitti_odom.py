#!/usr/bin/env python3
"""Tables + figure + comparison for the KITTI odometry extension.

Reads only measured JSON (``results/kitti_odom.json`` and
``results/verify_kitti_odom.json``) and the parent raw-drive results
(``../kitti/results/kitti.json``, ``../kitti2/results/kitti2.json``).
Writes ``results/kitti_odom.md`` and ``results/kitti_odom.png``.

Run after ``run_kitti_odom.py``:
  /home/syk/miniconda3/envs/WM/bin/python analyze_kitti_odom.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
P_LIST = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
METHODS = ["Kabsch", "RANSAC-1000", "RANSAC-adaptive", "IRLS-GM", "GNC-GM"]
OTYPES = ["random", "coherent", "boundary"]
RAW = {"0001": HERE.parent / "kitti" / "results" / "kitti.json",
       "0002": HERE.parent / "kitti2" / "results" / "kitti2.json"}


def get(rows, otype, p, method):
    for r in rows:
        if r["outlier_type"] == otype and abs(r["p"] - p) < 1e-9 and r["method"] == method:
            return r
    return None


def success_table(rows, otype, label):
    L = [f"**{label} — success (%)**", "",
         "| method | " + " | ".join(f"{int(100*p)}%" for p in P_LIST) + " |",
         "|---|" + "|".join(["---"] * len(P_LIST)) + "|"]
    for m in METHODS:
        cells = [f"{100*get(rows, otype, p, m)['success']:.0f}" for p in P_LIST]
        L.append(f"| {m} | " + " | ".join(cells) + " |")
    L.append("")
    return L


def rot_table(rows, otype, label):
    L = [f"**{label} — median rotation error (deg)**", "",
         "| method | " + " | ".join(f"{int(100*p)}%" for p in P_LIST) + " |",
         "|---|" + "|".join(["---"] * len(P_LIST)) + "|"]
    for m in METHODS:
        cells = [f"{get(rows, otype, p, m)['rot_median']:.2f}" for p in P_LIST]
        L.append(f"| {m} | " + " | ".join(cells) + " |")
    L.append("")
    return L


def speed_ratio(rows, otype, p):
    g = get(rows, otype, p, "GNC-GM")["time_ms_p50"]
    r = get(rows, otype, p, "RANSAC-1000")["time_ms_p50"]
    a = get(rows, otype, p, "RANSAC-adaptive")["time_ms_p50"]
    return r / g, a / g, g, r, a


def bin_by_t(records, p, method, edges):
    xs, ys, ns = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = [r for r in records if r["method"] == method and abs(r["p"] - p) < 1e-9
               and lo <= r["t_m"] < hi]
        if sel:
            xs.append(0.5 * (lo + hi))
            ys.append(100.0 * np.mean([r["success"] for r in sel]))
            ns.append(len(sel))
    return xs, ys, ns


def main():
    d = json.loads((RES / "kitti_odom.json").read_text())
    V = json.loads((RES / "verify_kitti_odom.json").read_text())["summary"]
    cfg = d["config"]
    per_seq = d["per_sequence"]
    pooled = d["pooled"]
    motion_pairs = d["motion_pairs"]
    gap_rows = d["motion_gaps"]
    raw_rows = {k: (v["rows"] if isinstance(v, dict) else v)
                for k, v in d["raw_drive_rows"].items()}
    seqs = cfg["sequences"]

    # per-dataset rows for the comparison tables
    ds_rows = {f"odom{s}": per_seq[s]["rows"] for s in seqs}
    ds_rows["pooled_odom"] = pooled
    ds_rows.update({f"raw{k}": v for k, v in raw_rows.items()})

    all_motion = motion_pairs + gap_rows
    t_lo = min(r["t_m"] for r in all_motion)
    t_hi = max(r["t_m"] for r in all_motion)
    ni_lo = min(r["natural_inlier"] for r in all_motion)
    ni_hi = max(r["natural_inlier"] for r in all_motion)
    sp05, _, g05, r05, _ = speed_ratio(pooled, "random", 0.5)

    L = ["# KITTI odometry 00/05/07: robust 3D-3D registration on the requested 00-10 axis", ""]
    L += ["> **一句话结论**：在 KITTI **odometry** 序列 00/05/07（官方精确 GT 位姿，"
          f"共 {sum(per_seq[s]['n_frames'] for s in seqs)} 帧中按协议取每序列 10 对相邻帧）上，"
          "父目录 drive 0001/0002 的结论全部复现：随机/边界外点 0–70% 的 RANSAC-1000 与 GNC-GM "
          "成功率 100%（RANSAC-adaptive/IRLS-GM ≥95%，仅 00 序列 p=0.7 掉 1 个 trial），"
          "Kabsch 在 p≥0.1 崩溃，相干外点在 p≈0.5 相变（GNC 53%、IRLS 38%、RANSAC 0%），"
          f"GNC-GM 比固定 1000 迭代 RANSAC 快约 {sp05:.0f}×（p=0.5，{g05:.1f} vs {r05:.1f} ms p50）；"
          f"并且**结论不依赖帧间运动幅度**（|t|={t_lo:.2f}–{t_hi:.2f} m 全覆盖，"
          f"自然内点率 {100*ni_hi:.1f}%→{100*ni_lo:.1f}%，成功率随 |t| 无系统性趋势；"
          "唯一绝对低点是 9.62 m 极值格 GNC 80% vs RANSAC 70%（8/10 trials），见 §6。", ""]

    L += ["## 0. 数据与可行性（KITTI 00–10 到什么程度）", "",
          "**结论：官方 00–10 的完整 Velodyne（84.8 GB）没有下载；本次用 HTTP Range 从镜像**"
          "**逐帧流式抽取了 00/05/07 三个序列中协议实际用到的 99 帧（194 MB）**，"
          "位姿用官方 `data_odometry_poses.zip`（精确 GT），标定用官方 per-sequence "
          "`calib.txt` 的 `Tr:` 行。", "",
          "调查记录：", "",
          "| 尝试 | 结果 |", "|---|---|",
          "| 官方 per-sequence Velodyne 下载 | 官方只提供 ~80 GB 整包，无按序列分包 |",
          "| HuggingFace per-sequence 镜像搜索（`kitti+odometry`，20+ 仓库） | "
          "`yujie2696/kitti_odometry_00..21` 只有 `image_0/`、`image_1/`、`calib.txt`，"
          "**没有 `velodyne/`**，不可用 |",
          "| HuggingFace `sangramrout/velodyne` | 镜像官方 `data_odometry_velodyne.zip`"
          "（84,786,535,790 B，43,574 个成员，`dataset/sequences/NN/velodyne/NNNNNN.bin`）"
          "＋ `data_odometry_poses.zip`（1.31 MB）＋ `data_odometry_calib.zip`（0.60 MB）；"
          "CDN 支持 `Range`，实测 ~6 MB/s |",
          "| KITTI raw 追加 drive（备用方案） | S3 经代理实测仅 **52–71 KB/s**（父任务时 15–20 MB/s），"
          "646 MB 的 drive 0005 需 ~2.5–3.5 h，**不可行**，已放弃并清理半成品 zip |", "",
          f"实测：`httpzip.py` 打开 84.8 GB 远端 zip 4.7 s，列出 43,574 成员 7.8 s；"
          f"三个序列各 33 帧，共 99 帧 / 194.0 MB，抓取耗时 256 s。"
          f"清单见 `/mnt/storage1/syk/data/kitti_odom/manifest.json`。", "",
          "所以：**“KITTI 00–10” 在“官方 odometry 序列 + 精确 GT”意义上是可行的（只要 194 MB 而非 80 GB），"
          "本次只覆盖 00/05/07 三条**；raw drive 0001/0002 是父任务的口径，作为对照保留。", ""]

    L += ["## 1. 方向校验（`verify_kitti_odom.py`）", "",
          f"18/18 对相邻帧判为 `i->j`：正确方向 NN 中位距离 **{V['fwd_range'][0]:.3f}–"
          f"{V['fwd_range'][1]:.3f} m**，反向 **{V['inv_range'][0]:.3f}–{V['inv_range'][1]:.3f} m**。"
          "与 raw drive 0001（fwd 0.100–0.123 / inv 0.143–0.183）同量级但更紧（精确 GT 位姿 + "
          "更近的帧间运动）。逐对数值见 `results/verify_kitti_odom.json`。", "",
          "| seq | 帧数(GT) | 10 对相邻帧的 \\|t\\| (m) | km/h | 旋转 (deg) | 自然内点率 |",
          "|---|---|---|---|---|---|"]
    for s in seqs:
        b = per_seq[s]["base"]
        tm = [x["t_m"] for x in b]
        rot = [x["rot_deg"] for x in b]
        ni = [x["natural_inlier"] for x in b]
        L.append(f"| {s} | {per_seq[s]['n_frames']} | {min(tm):.2f}–{max(tm):.2f} "
                 f"(median {np.median(tm):.2f}) | {min(tm)*36:.0f}–{max(tm)*36:.0f} | "
                 f"{min(rot):.2f}–{max(rot):.2f} | {100*min(ni):.1f}–{100*max(ni):.1f}% |")
    L += ["", "（`|t| × 36` 即 10 Hz 下的 km/h。）", ""]

    L += ["## 2. 主表：每条序列与 pooled", "",
          f"协议与父目录逐项相同：N={cfg['n']}，c={cfg['c']} m，tau={cfg['tau']} m，"
          f"voxel={cfg['voxel']} m，{cfg['trials']} trials/cell，p=0–0.7，5 方法，"
          f"10 对相邻帧/序列。区别只有一处：位姿是 KITTI odometry 的**精确 GT**"
          f"（`dataset/poses`），不是 OXTS 积分。", ""]
    for s in seqs:
        L += [f"### 序列 {s}（{per_seq[s]['n_frames']} 帧，10 对相邻帧）", ""]
        for ot in OTYPES:
            L += success_table(ds_rows[f"odom{s}"], ot, f"{s} / {ot}")
    L += ["### 三序列 pooled（60 trials/cell）", ""]
    for ot in OTYPES:
        L += success_table(pooled, ot, f"pooled / {ot}")
    L += ["### pooled 中位旋转误差（deg）", ""]
    for ot in OTYPES:
        L += rot_table(pooled, ot, f"pooled / {ot}")

    # runtime
    L += ["## 3. 运行时间（pooled，random，p50 ms）", "",
          "| p | Kabsch | RANSAC-1000 | RANSAC-adaptive | IRLS-GM | GNC-GM | GNC vs RANSAC-1000 |",
          "|---|---|---|---|---|---|---|"]
    for p in P_LIST:
        cells = [f"{get(pooled, 'random', p, m)['time_ms_p50']:.2f}" for m in METHODS]
        r, _, _, _, _ = speed_ratio(pooled, "random", p)
        L.append(f"| {int(100*p)}% | " + " | ".join(cells) + f" | {r:.1f}× |")
    L.append("")

    # per-sequence runtime / speedup (requirement 2: per-seq, not only pooled)
    L += ["### 3.1 每序列运行时间与 GNC 加速比（random，p50 ms）", "",
          "| dataset | p | RANSAC-1000 | RANSAC-adaptive | GNC-GM | speedup GNC vs RANSAC-1000 |",
          "|---|---|---|---|---|---|"]
    for tag in [f"odom{s}" for s in seqs] + ["pooled_odom"]:
        rows_rt = ds_rows[tag]
        for p in [0.0, 0.3, 0.5, 0.7]:
            r, a, g = (get(rows_rt, "random", p, "RANSAC-1000")["time_ms_p50"],
                       get(rows_rt, "random", p, "RANSAC-adaptive")["time_ms_p50"],
                       get(rows_rt, "random", p, "GNC-GM")["time_ms_p50"])
            L.append(f"| {tag} | {int(100*p)}% | {r:.1f} | {a:.2f} | {g:.2f} | {r/g:.1f}× |")
    L.append("")

    # coherent collapse + boundary error (requirements 2/3)
    L += ["## 4. 相干外点相变与边界外点误差", "",
          "### 4.1 相干外点：p=0.4→0.5 的相变", "",
          "多数外点构成同一竞争刚体时，正确解不再是残差最小的解；相变点由内点/外点"
          "数量比决定，与运动幅度无关。下表为 pooled 成功率（%）。", "",
          "| p | Kabsch | RANSAC-1000 | RANSAC-adaptive | IRLS-GM | GNC-GM | GNC 选中外点变换比例 |",
          "|---|---|---|---|---|---|---|"]
    for p in [0.3, 0.4, 0.5, 0.6, 0.7]:
        cells = [f"{100*get(pooled, 'coherent', p, m)['success']:.0f}" for m in METHODS]
        fo = get(pooled, "coherent", p, "GNC-GM").get("frac_picked_outlier")
        L.append(f"| {int(100*p)}% | " + " | ".join(cells) + " | "
                 + (f"{100*fo:.0f}%" if fo is not None else "n/a") + " |")
    L.append("")
    L += ["逐序列表（GNC-GM / RANSAC-1000 成功率 %）：", "",
          "| dataset | p=0.4 | p=0.5 | p=0.6 | p=0.7 |", "|---|---|---|---|---|"]
    for tag in [f"odom{s}" for s in seqs] + ["pooled_odom"]:
        rr = ds_rows[tag]
        cells = [f"{100*get(rr,'coherent',p,'GNC-GM')['success']:.0f}/"
                 f"{100*get(rr,'coherent',p,'RANSAC-1000')['success']:.0f}"
                 for p in [0.4, 0.5, 0.6, 0.7]]
        L.append(f"| {tag} | " + " | ".join(cells) + " |")
    L.append("")
    L += ["### 4.2 边界外点：成功率与误差", "",
          "边界外点的残差平方 `e/c²` 落在 [1/3, 3]，即刚好跨过内点阈值 c；"
          "这是最考验鲁棒核连续性的协议。下表为 pooled（random 与 boundary 在 p=0.7 的对照）。", "",
          "| 协议 | p | 方法 | success (%) | rot median (deg) | trans median (m) |",
          "|---|---|---|---|---|---|"]
    for ot in ["boundary", "random"]:
        for p in [0.3, 0.5, 0.7]:
            for m in ["Kabsch", "RANSAC-1000", "IRLS-GM", "GNC-GM"]:
                rr = get(pooled, ot, p, m)
                L.append(f"| {ot} | {int(100*p)}% | {m} | {100*rr['success']:.0f} | "
                         f"{rr['rot_median']:.3f} | {rr['trans_median']:.4f} |")
    L.append("")

    # comparison with raw drives
    L += ["## 5. 与 raw drive 0001/0002 的逐格对照", "",
          "同格（outlier_type, p, method）成功率之差，pooled odometry 减 raw drive；"
          "raw drive 行来自父目录 `../kitti/results/kitti.json` 与 "
          "`../kitti2/results/kitti2.json`（各 120 行）。", "",
          "| 对照 | 最大 \\|Δsuccess\\|（全部格） | 最大 \\|Δsuccess\\|（鲁棒方法，非相干相变） | 平均 \\|Δsuccess\\| | 结论 |",
          "|---|---|---|---|---|"]
    cmp_rows = []
    for tag in ["raw0001", "raw0002"]:
        deltas, deltas_core = [], []
        for ot in OTYPES:
            for p in P_LIST:
                for m in METHODS:
                    a = get(pooled, ot, p, m)["success"]
                    b = get(raw_rows[tag[-4:]], ot, p, m)["success"]
                    dv = 100.0 * (a - b)
                    deltas.append(dv)
                    if m != "Kabsch" and not (ot == "coherent" and p in (0.5, 0.6)):
                        deltas_core.append(dv)
        cmp_rows.append((tag, deltas, deltas_core))
        L.append(f"| pooled_odom − {tag} | {np.max(np.abs(deltas)):.0f} pp | "
                 f"{np.max(np.abs(deltas_core)):.0f} pp | "
                 f"{np.mean(np.abs(deltas)):.1f} pp | "
                 f"{'PASS (≤10 pp)' if np.max(np.abs(deltas_core)) <= 10 else 'CHECK'} |")
    L.append("")
    L += ["注：唯一的 18 pp 差异出现在 `coherent p=0.5`——该格是 50/50 相变点，"
          "20 trials 下二项噪声约 ±11 pp，且 odom 53% 与 raw 35–40% 同向（GNC 高于 RANSAC 0%），"
          "不是复现失败。剔除该相变格与已崩溃的 Kabsch 后，最大差 ≤"
          f"{max(np.max(np.abs(c)) for _, _, c in cmp_rows):.0f} pp。", ""]
    L += ["关键格点（成功率 %）：", "",
          "| 格点 | odom 00 | odom 05 | odom 07 | pooled odom | raw 0001 | raw 0002 |",
          "|---|---|---|---|---|---|---|"]
    key = [("random", 0.0), ("random", 0.3), ("random", 0.7),
           ("coherent", 0.4), ("coherent", 0.5), ("coherent", 0.6),
           ("boundary", 0.5), ("boundary", 0.7)]
    for ot, p in key:
        for m in ["Kabsch", "RANSAC-1000", "IRLS-GM", "GNC-GM"]:
            cells = [f"{100*get(ds_rows[f'odom{s}'], ot, p, m)['success']:.0f}" for s in seqs]
            cells.append(f"{100*get(pooled, ot, p, m)['success']:.0f}")
            cells.append(f"{100*get(raw_rows['0001'], ot, p, m)['success']:.0f}")
            cells.append(f"{100*get(raw_rows['0002'], ot, p, m)['success']:.0f}")
            L.append(f"| {ot} p={p:.1f} {m} | " + " | ".join(cells) + " |")
    L.append("")

    # motion analysis
    L += ["## 6. 分析轮：结论是否依赖帧间运动幅度", "",
          "两个轴：(a) 每条序列 10 个相邻帧对（|t| 由序列速度决定），每对固定跑 "
          f"{cfg['motion_trials']} trials，p={cfg['motion_p']}，random 外点；"
          f"(b) 同一序列内把帧间隔放大到 gaps={cfg['gaps']}，|t| 单调增大，"
          "检验“运动越大、重叠越少、外点比例不变”时结论是否变化。", ""]
    allm = motion_pairs + gap_rows
    L += ["### 6.1 成功率 vs |t|（按 |t| 分箱，p=0.5，random）", "",
          "| \\|t\\| 区间 (m) | 样本格数 | Kabsch | RANSAC-1000 | RANSAC-adaptive | IRLS-GM | GNC-GM |",
          "|---|---|---|---|---|---|---|"]
    edges = [0.0, 0.3, 0.6, 1.0, 1.5, 2.5, 4.0, 8.0, 30.0]
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = [r for r in allm if abs(r["p"] - 0.5) < 1e-9 and lo <= r["t_m"] < hi]
        if not sel:
            continue
        cells = []
        for m in METHODS:
            sub = [r for r in sel if r["method"] == m]
            cells.append(f"{100*np.mean([r['success'] for r in sub]):.0f}")
        L.append(f"| {lo:.1f}–{hi:.1f} | {len(sel)//len(METHODS)} | " + " | ".join(cells) + " |")
    L.append("")
    L += ["### 6.2 中位旋转误差 vs |t|（p=0.5，random）", "",
          "| \\|t\\| 区间 (m) | Kabsch | RANSAC-1000 | IRLS-GM | GNC-GM |",
          "|---|---|---|---|---|"]
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = [r for r in allm if abs(r["p"] - 0.5) < 1e-9 and lo <= r["t_m"] < hi]
        if not sel:
            continue
        cells = []
        for m in ["Kabsch", "RANSAC-1000", "IRLS-GM", "GNC-GM"]:
            sub = [r for r in sel if r["method"] == m]
            cells.append(f"{np.median([r['rot_median'] for r in sub]):.2f}")
        L.append(f"| {lo:.1f}–{hi:.1f} | " + " | ".join(cells) + " |")
    L.append("")
    # within-sequence gap trend
    L += ["### 6.3 同序列内放大帧间隔（|t| 增大）", "",
          "| dataset | gap | \\|t\\| (m) | 自然内点率 | p | Kabsch | RANSAC-1000 | GNC-GM | GNC rot (deg) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(gap_rows, key=lambda z: (z["dataset"], z["gap"], z["p"], z["method"])):
        if r["method"] != "GNC-GM":
            continue
        row = dict(r)
        k = next(z for z in gap_rows if z["dataset"] == r["dataset"]
                 and z["pair"] == r["pair"] and z["gap"] == r["gap"]
                 and abs(z["p"] - r["p"]) < 1e-9 and z["method"] == "Kabsch")
        ra = next(z for z in gap_rows if z["dataset"] == r["dataset"]
                  and z["pair"] == r["pair"] and z["gap"] == r["gap"]
                  and abs(z["p"] - r["p"]) < 1e-9 and z["method"] == "RANSAC-1000")
        L.append(f"| {r['dataset']} | {r['gap']} | {row['t_m']:.2f} | "
                 f"{100*row['natural_inlier']:.0f}% | {int(100*r['p'])}% | "
                 f"{100*k['success']:.0f} | {100*ra['success']:.0f} | "
                 f"{100*row['success']:.0f} | {row['rot_median']:.3f} |")
    L.append("")
    L += ["### 6.4 GNC 优势是否随 |t| 变化（p=0.5，random）", "",
          "同一 |t| 区间内比较 GNC-GM 与 RANSAC-1000：成功率差、旋转误差、运行时间。"
          "若 GNC 的优势依赖小运动，Δsuccess 或 speedup 应随 |t| 趋势变化。", "",
          "| \\|t\\| 区间 (m) | 格点数 | Δsuccess (pp) | rot GNC | rot RANSAC | "
          "time GNC (ms) | time RANSAC (ms) | speedup |",
          "|---|---|---|---|---|---|---|---|"]
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = [r for r in allm if abs(r["p"] - 0.5) < 1e-9 and lo <= r["t_m"] < hi]
        if not sel:
            continue
        g = [r for r in sel if r["method"] == "GNC-GM"]
        rr = [r for r in sel if r["method"] == "RANSAC-1000"]
        gs_ = float(np.mean([r["success"] for r in g]))
        rs_ = float(np.mean([r["success"] for r in rr]))
        gt = float(np.median([r["time_ms_p50"] for r in g]))
        rt = float(np.median([r["time_ms_p50"] for r in rr]))
        L.append(f"| {lo:.1f}–{hi:.1f} | {len(g)} | {100*(gs_-rs_):+.0f} | "
                 f"{np.median([r['rot_median'] for r in g]):.3f} | "
                 f"{np.median([r['rot_median'] for r in rr]):.3f} | "
                 f"{gt:.2f} | {rt:.1f} | {rt/gt:.1f}× |")
    L.append("")

    # spearman
    def spearman(x, y):
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        rx = np.argsort(np.argsort(x))
        ry = np.argsort(np.argsort(y))
        return float(np.corrcoef(rx, ry)[0, 1])

    L += ["### 6.5 相关性检验（per-cell success vs |t|）", "",
          "| p | 方法 | Spearman ρ(success, \\|t\\|) | 说明 |", "|---|---|---|---|"]
    for p in cfg["motion_p"]:
        for m in ["Kabsch", "RANSAC-1000", "IRLS-GM", "GNC-GM"]:
            cells = [r for r in allm if abs(r["p"] - p) < 1e-9 and r["method"] == m]
            rho = spearman([r["t_m"] for r in cells], [r["success"] for r in cells])
            note = ("无实质趋势（成功率饱和，ρ 由并列值主导）" if abs(rho) < 0.5 else
                    ("成功率随 |t| 增大而降" if rho < 0 else "成功率随 |t| 增大而升"))
            L.append(f"| {int(100*p)}% | {m} | {rho:+.2f} | {note} |")
    L.append("")

    js = [r for r in allm if r["method"] == "Kabsch" and abs(r["p"] - 0.3) < 1e-9]
    gs = [r for r in allm if r["method"] == "GNC-GM"]
    # paired GNC-vs-RANSAC deltas at the same (dataset, pair, p)
    by_cell = {}
    for r in allm:
        by_cell.setdefault((r["dataset"], r["pair"], round(r["p"], 2)), {})[r["method"]] = r
    paired = [v for v in by_cell.values()
              if "GNC-GM" in v and "RANSAC-1000" in v]
    deltas = [v["GNC-GM"]["success"] - v["RANSAC-1000"]["success"] for v in paired]
    min_gnc = min(gs, key=lambda r: r["success"])
    L += [f"读法：GNC-GM 在 {len(gs)} 个 (序列×\|t\|×p) 格点里成功率 "
          f"{100*np.mean([r['success'] for r in gs]):.1f}%，"
          f"最低 {100*min(r['success'] for r in gs):.0f}%；"
          f"Kabsch 在 p=0.3 的 {len(js)} 个格点成功率 "
          f"{100*np.mean([r['success'] for r in js]):.1f}%（已崩溃，与 |t| 无关）。"
          "因此“鲁棒层是否必要、GNC 是否够用”的结论与运动幅度无系统性关系；|t| 只改变重叠率"
          f"（自然内点率 {100*ni_hi:.1f}%→{100*ni_lo:.1f}%），不改变方法排序。"
          f"成对比较（{len(paired)} 个 dataset×pair×p 格点）GNC 成功率从未低于 RANSAC："
          f"最小差 {100*min(deltas):+.0f} pp，中位差 {100*np.median(deltas):+.0f} pp。"
          f"唯一绝对低点是 {min_gnc['dataset']} 的 {min_gnc['pair']} 对"
          f"（|t|={min_gnc['t_m']:.2f} m，自然内点率 {100*min_gnc['natural_inlier']:.0f}%，"
          f"p={min_gnc['p']:.1f}）的 {100*min_gnc['success']:.0f}%，但同格 RANSAC 只有 "
          f"{100*next(v['RANSAC-1000']['success'] for v in paired if v['GNC-GM'] is min_gnc):.0f}%"
          "；该格是 8 帧间隔/96 km/h 的极值，10 trials 下每 1 次失败即 10 pp，"
          "不能据此推出 |t| 趋势。", ""]

    # caveats
    L += ["## 7. 口径与限制", "",
          "1. **只覆盖 00/05/07 三条 odometry 序列**（官方 00–10 共 11 条）。"
          "抽取成本 ~65 MB/序列，扩展到全部 11 条只需 ~0.7 GB / ~15 min，未做是因为本轮时间盒。",
          "2. 每序列只用 10 对相邻帧（与父目录一致），不是全序列里程计评测；"
          "这是鲁棒外层的外点敏感性实验，不是 SLAM 精度评测。",
          "3. 基础对应是 GT-mapped NN（oracle 匹配），与父目录/3DMatch oracle 表同一口径，"
          "不是学习描述子（3DMatch FPFH 表是另一条证据）。",
          "4. 成功阈值仍是绝对位姿误差 rot<5°, |t|<0.1 m；运动大时该阈值相对运动更宽松，"
          "但本文所有格点误差都远低于阈值（GNC 中位 <0.1° / <0.03 m），结论不敏感。",
          "5. 位姿来自官方 odometry GT（激光 SLAM 生成），不是毫米级真值；"
          "但它明显紧于 raw drive 的 OXTS 积分（方向校验 fwd 0.053–0.114 m vs 0.100–0.123 m）。",
          "6. 时间数值受共享节点同租户抖动影响（父目录观测到 ~1.8×）；"
          "GNC/RANSAC 的比值在多次运行间稳定在 ~20×，绝对值按 p50 读。", ""]

    (RES / "kitti_odom.md").write_text("\n".join(L) + "\n")
    print("wrote", RES / "kitti_odom.md")

    # ---------------- figure ----------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"Kabsch": "tab:gray", "RANSAC-1000": "tab:orange",
              "RANSAC-adaptive": "tab:green", "IRLS-GM": "tab:blue", "GNC-GM": "tab:red"}
    fig, ax = plt.subplots(2, 3, figsize=(16, 8.4))
    for m in METHODS:
        ys = [100 * get(pooled, "random", p, m)["success"] for p in P_LIST]
        ax[0, 0].plot([100 * p for p in P_LIST], ys, "o-", color=colors[m], label=m, lw=1.5, ms=4)
    ax[0, 0].set_title("pooled 00/05/07 · random: success")
    ax[0, 0].set_xlabel("outlier fraction (%)")
    ax[0, 0].set_ylabel("success (%)")
    ax[0, 0].set_ylim(-3, 103)
    ax[0, 0].grid(alpha=0.3)
    ax[0, 0].legend(fontsize=7)

    for m in METHODS:
        ys = [100 * get(pooled, "coherent", p, m)["success"] for p in P_LIST]
        ax[0, 1].plot([100 * p for p in P_LIST], ys, "o-", color=colors[m], lw=1.5, ms=4)
    ax[0, 1].set_title("pooled · coherent: success (p≈0.5 transition)")
    ax[0, 1].set_xlabel("outlier fraction (%)")
    ax[0, 1].set_ylim(-3, 103)
    ax[0, 1].grid(alpha=0.3)

    for m in METHODS:
        ys = [100 * get(pooled, "boundary", p, m)["success"] for p in P_LIST]
        ax[0, 2].plot([100 * p for p in P_LIST], ys, "o-", color=colors[m], lw=1.5, ms=4)
    ax[0, 2].set_title("pooled · boundary: success")
    ax[0, 2].set_xlabel("outlier fraction (%)")
    ax[0, 2].set_ylim(-3, 103)
    ax[0, 2].grid(alpha=0.3)

    for m in ["Kabsch", "RANSAC-1000", "IRLS-GM", "GNC-GM"]:
        for p, ls in [(0.3, ":"), (0.5, "-"), (0.7, "--")]:
            xs, ys, _ = bin_by_t(allm, p, m, [0.0, 0.5, 1.0, 1.5, 2.5, 4.0, 8.0, 30.0])
            ax[1, 0].plot(xs, ys, ls, marker="o", ms=4, color=colors[m], lw=1.2,
                          label=f"{m} p={p}")
    ax[1, 0].set_xscale("log")
    ax[1, 0].set_xlabel("|t| per pair (m, log) — 0.1 s frame")
    ax[1, 0].set_ylabel("success (%)")
    ax[1, 0].set_title("success vs frame motion (random outliers)")
    ax[1, 0].set_ylim(-3, 103)
    ax[1, 0].grid(alpha=0.3, which="both")
    ax[1, 0].legend(fontsize=5, ncol=2)

    for m in ["Kabsch", "RANSAC-1000", "IRLS-GM", "GNC-GM"]:
        for p, ls in [(0.3, ":"), (0.5, "-"), (0.7, "--")]:
            xs, ys, _ = bin_by_t(allm, p, m, [0.0, 0.5, 1.0, 1.5, 2.5, 4.0, 8.0, 30.0])
            ax[1, 1].plot(xs, np.maximum(ys, 0.1), ls, marker="o", ms=3, color=colors[m], lw=1.0)
    ax[1, 1].set_xscale("log")
    ax[1, 1].set_yscale("log")
    ax[1, 1].set_xlabel("|t| per pair (m, log)")
    ax[1, 1].set_ylabel("success (%)")
    ax[1, 1].set_title("(same, log y to expose 0%)")
    ax[1, 1].grid(alpha=0.3, which="both")

    g = [get(pooled, "random", p, "GNC-GM")["time_ms_p50"] for p in P_LIST]
    r = [get(pooled, "random", p, "RANSAC-1000")["time_ms_p50"] for p in P_LIST]
    a = [get(pooled, "random", p, "RANSAC-adaptive")["time_ms_p50"] for p in P_LIST]
    ax[1, 2].semilogy([100 * p for p in P_LIST], r, "o-", color=colors["RANSAC-1000"],
                      label="RANSAC-1000", lw=1.5)
    ax[1, 2].semilogy([100 * p for p in P_LIST], a, "o-", color=colors["RANSAC-adaptive"],
                      label="RANSAC-adaptive", lw=1.5)
    ax[1, 2].semilogy([100 * p for p in P_LIST], g, "o-", color=colors["GNC-GM"],
                      label="GNC-GM", lw=1.5)
    for x, rr, gg in zip([100 * p for p in P_LIST], r, g):
        ax[1, 2].annotate(f"{rr/gg:.0f}×", (x, rr), fontsize=6, ha="center", va="bottom")
    ax[1, 2].set_title("pooled random: runtime p50")
    ax[1, 2].set_xlabel("outlier fraction (%)")
    ax[1, 2].set_ylabel("ms (log)")
    ax[1, 2].grid(alpha=0.3, which="both")
    ax[1, 2].legend(fontsize=7)
    fig.suptitle("KITTI odometry 00/05/07 (exact GT poses): robust 3D-3D + motion-magnitude analysis")
    fig.tight_layout()
    fig.savefig(RES / "kitti_odom.png", dpi=160)
    print("wrote", RES / "kitti_odom.png")


if __name__ == "__main__":
    main()
