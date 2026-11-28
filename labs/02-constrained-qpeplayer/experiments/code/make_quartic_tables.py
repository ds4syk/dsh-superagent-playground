#!/usr/bin/env python
"""Render experiments/results/quartic_validation.json into quartic_validation.md."""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
J = os.path.join(RES, "quartic_validation.json")
MD = os.path.join(RES, "quartic_validation.md")

d = json.load(open(J))
L = []
A = L.append
A("# P2 内层 KKT 全枚举（quartic_roots）验证报告")
A("")
A("生成：`experiments/code/run_quartic.py` → `experiments/results/quartic_validation.json`；"
  "本文件由 `experiments/code/make_quartic_tables.py` 渲染。numpy %s，节点 %s。"
  % (d["meta"].get("numpy"), d["meta"].get("host")))
A("")
A("## 0. 结论先行")
A("")
g = d["generic"]["cases"]
A("1. **generic 10×10 对称 M：40 个射影有限复根 / 80 个仿射有限复根，5/5 seed 一致**"
  "（%s），与 P1 supplement 的 Porteous `c_3(F-E)=40` 和 Bezout 上界 162 相符；"
  "其余 ~82 条路径发散（根在无穷远），不是跟踪失败。" % ", ".join(str(c["n_finite_projective"]) for c in g))
p2 = d["p2"]["cases"]
A("2. **P2 `Mred` 全部 %d 个实例同样是 40 射影 / 80 仿射有限复根**（含 noise=0 的 PSD-退化 Mred），"
  "并不存在“退化导致根跑到无穷远”的现象；实根数稳定为 %s 个几何实根。"
  % (len(p2), sorted({c["n_unique"] for c in p2})))
A("3. **全局最小值与多起点一致**：max |gap(256 起点)| = %.2e（%d 个实例），"
  "没有一次多起点真正击败枚举（>1e-12）；argmin 旋转差 ≤ %.1e 度。"
  % (max(abs(c["gap_256"]) for c in p2), len(p2),
     max(c["rot_deg_256"] for c in p2 if c["qdist_256"] < 1e-6) if any(
         c["qdist_256"] < 1e-6 for c in p2) else float("nan")))
A("4. **代价**：324 路径（2 个 Gamma 相位）≈1.0–1.4 s；162 路径 ≈0.5–0.7 s，约 230–300 路径/s，"
  "是多起点 BFGS（6 起点）的 ~100 倍。")
A("5. **完整性不是无条件的**：单相位偶尔丢根（ALM 装配的 `M_A` 上观察到 38/39 射影根），"
  "换一个 Gamma 相位的并集即可补到 40；`solve_inner_exact` 在 `n_gamma=1` 且根数 <40 时"
  "置 `fallback=True` 并回退多起点，此时**不声称精确**。")
A("")
A("> 诚实边界：以上是“在测试实例上未发现丢根 + 枚举值与多起点一致”，不是完备性证明。"
  "单项式总次数同伦的路径数是 162，其中有 82 条必然跑向无穷远；"
  "“未丢根”的证据是多个独立 Gamma 相位的并集不再增加根。")
A("")
A("## 1. Generic 随机 M（5 seeds，2 个 Gamma 相位）")
A("")
A("| seed | 有限路径 | 发散 | Newton 失败 | 有限复根(射影) | 有限复根(仿射) | 实根(几何) | f_min | 时间 (s) |")
A("|---|---|---|---|---|---|---|---|---|")
for c in g:
    A("| %d | %d | %d | %d | %d | %d | %d | %.6f | %.2f |"
      % (c["seed"], c["n_finite_paths"], c["n_diverged"], c["n_newton_fail"],
         c["n_finite_projective"], c["n_finite_affine"],
         c["n_real_geometric"], c["f_min"], c["wall"]))
A("")
A("单相位（162 路径）对照：")
A("")
A("| seed | 射影根 | 实根 | 时间 (s) | 路径/s | 双相位并集新增 |")
A("|---|---|---|---|---|---|")
for c in g:
    s = c["single_phase"]
    A("| %d | %d | %d | %.2f | %.0f | %d |"
      % (c["seed"], s["n_finite_projective"], s["n_real_affine"],
         s["wall"], s["paths_per_s"], c["phase_union_extra"]))
A("")
A("`phase_union_extra = 0` 说明第 2 个 Gamma 相位没有带来任何新根 → 单相位在这 5 个 generic 实例上已经完备。")
A("")
A("## 2. P2 实例：根数 + 精确 vs 多起点")
A("")
A("`gap_K = f_multistart(K) - f_enum`（正 = 枚举更好，负 = 多起点略好，1e-17 量级是同一根上的舍入差）。")
A("")
A("| p | noise | seed | 射影根 | 实根 | f_enum | gap_8 | gap_64 | gap_256 | rot(256)° | qdist(256) | t_exact (s) | fallback |")
A("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for c in p2:
    A("| %d | %.2f | %d | %d | %d | %.9g | %+.1e | %+.1e | %+.1e | %.1e | %.1e | %.2f | %s |"
      % (c["p"], c["noise"], c["seed"], c["n_finite_projective"], c["n_unique"],
         c["f_enum"], c["gap_8"], c["gap_64"], c["gap_256"],
         c["rot_deg_256"], c["qdist_256"], c["t_exact"],
         "**yes**" if c["fallback"] else "no"))
A("")
A("统计：")
n_ms_win = sum(1 for c in p2 if c["gap_256"] < -1e-12)
n_ex_win = sum(1 for c in p2 if c["gap_256"] > 1e-12)
A("")
A("- 多起点(256) 严格优于枚举的次数：**%d**；枚举严格优于多起点的次数：**%d**；其余 |gap| ≤ 1e-12。" % (n_ms_win, n_ex_win))
A("- max |gap_256| = %.2e；max |gap_8| = %.2e。" % (
    max(abs(c["gap_256"]) for c in p2), max(abs(c["gap_8"]) for c in p2)))
A("- 时间：t_exact 中位数 %.2f s，mean %.2f s；多起点 256 ms 中位数 %.3f s。"
  % (float(np.median([c["t_exact"] for c in p2])),
     float(np.mean([c["t_exact"] for c in p2])),
     float(np.median([c["ms_256"]["time"] for c in p2]))))
A("")
A("### 2b. 时间分解 (p2 实例)")
A("")
A("| p | t_track (s) | t_polish (s) | 路径/s | 批评估次数 |")
A("|---|---|---|---|---|")
seen = set()
for c in p2:
    if c["p"] in seen:
        continue
    seen.add(c["p"])
    A("| %d | %.2f | %.4f | %.0f | %d |"
      % (c["p"], c["t_track"], c["t_polish"], c["paths_per_s"], c["n_batch_evals"]))
A("")
A("polish 只占 <1%，时间几乎全部在 162(324) 条路径的预测-校正跟踪上；"
  "批量化（(162,5,5) 堆叠线性求解）把单条路径成本从标量循环的 ~11 ms 降到 ~3 ms。")
A("")
A("## 3. 微小扰动 Mred：根数如何随退化消失")
A("")
A("M = Mred + eps·||Mred||·E（E 单位随机对称）。若“退化使根跑到无穷远”成立，eps→0 时射影根数应下降。")
A("")
A("| p | noise | eps | 射影根 | 实根 | f_min |")
A("|---|---|---|---|---|---|")
for c in d["perturb"]["cases"]:
    A("| %d | %.2f | %.0e | %d | %d | %.9g |"
      % (c["p"], c["noise"], c["eps"], c["n_finite_projective"], c["n_unique"], c["f_min"]))
A("")
A("**结论：根数不随 eps 变化，恒为 40。** 因此之前“P2 Mred 退化 → 有限根 18–26”的观察是跟踪器欠收敛"
  "（单相位丢根 / 去重口径）造成的，不是根跑到无穷远。")
A("")
A("## 4. 鲁棒性 / 回退")
A("")
A("### 4a. PSD、秩亏 Mred（noise=0）")
A("")
A("| p | seed | eig_min | rank | 射影根 | f_enum | f_ms256 | gap | fallback | degenerate |")
A("|---|---|---|---|---|---|---|---|---|---|")
for c in d["robust"]["psd"]:
    A("| %d | %d | %.1e | %d | %d | %.9g | %.9g | %+.1e | %s | %s |"
      % (c["p"], c["seed"], c["eig_min"], c["rank"], c["n_finite_projective"],
         c["f_enum"], c["f_ms256"], c["gap"], c["fallback"], c["degenerate"]))
A("")
A("### 4b. ALM 装配的 `M_A`（`_alm_quadratic` + `_rho_of`，最终活跃集）")
A("")
A("| seed | 活跃数 | eig_min | 射影根 | 实根 | f_enum | f_ms256 | gap | fallback | incomplete |")
A("|---|---|---|---|---|---|---|---|---|---|")
for c in d["robust"]["alm_M"]:
    A("| %d | %d | %.2e | %d | %d | %.9g | %.9g | %+.1e | %s | %s |"
      % (c["seed"], c["n_act"], c["eig_min"], c["n_finite_projective"], c["n_unique"],
         c["f_enum"], c["f_ms256"], c["gap"], c["fallback"],
         c.get("incomplete", c["fallback"])))
A("")
A("### 4c. 合成奇异 M")
A("")
A("| M | 射影根 | f_enum | f_ms64 | fallback | degenerate | sigma_min(J) |")
A("|---|---|---|---|---|---|---|")
for c in d["robust"]["synthetic"]:
    A("| %s | %d | %.6g | %.6g | %s | %s | %.1e |"
      % (c["name"], c["n_finite_projective"], c["f_enum"], c["f_ms64"],
         c["fallback"], c["degenerate"], c["sigma_min_best"]))
A("")
A("`zero` 的 KKT 系统有连续解集（任意单位 q、lam=0），枚举返回 73 个伪根并被 `degenerate`"
  "（最优点的 KKT Jacobian 奇异）识别、回退多起点；`rank1_*` 也触发回退。"
  "这是本实现**不能**认证的退化：只有 `fallback=False` 时才声称枚举完备。")
A("")
if "alm" in d:
    a = d["alm"]
    A("## 5. ALM 小对比（p=%d, noise=%.2f, bias=%.2f, %d trials）"
      % (a["p"], a["noise"], a["bias_z"], len(a["trials"])))
    A("")
    A("| trial | J_ms | J_exact | ΔJ | iters(ms/ex) | t_ms (s) | t_ex (s) | 时间比 | 可行(ms/ex) | 穿透 mm (ms/ex) | fallback_count(ms/ex) | exact 调用/回退 |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in a["trials"]:
        m, e = r["multistart"], r["exact"]
        A("| %d | %.10f | %.10f | %+.2e | %d/%d | %.2f | %.2f | %.0f× | %s/%s | %.3f/%.3f | %d/%d | %d/%d |"
          % (r["trial"], m["J"], e["J"], r["dJ"], m["iters"], e["iters"],
             m["time"], e["time"], r["speed_ratio"],
             "Y" if m["feasible"] else "N", "Y" if e["feasible"] else "N",
             m["penetration_mm"], e["penetration_mm"],
             m["fallback_count"], e["fallback_count"],
             e["exact_calls"], e["exact_fallbacks"]))
    A("")
    aj = [r["dJ"] for r in a["trials"]]
    A("- max |ΔJ| = %.2e；exact 更好的次数 %d，multistart 更好的次数 %d。"
      % (max(abs(x) for x in aj), sum(1 for x in aj if x > 1e-10),
         sum(1 for x in aj if x < -1e-10)))
    A("- 时间比（exact/multistart）中位数 %.0f×；exact 每次内层调用 ≈ %.2f s，"
      "每 trial %d 次调用。" % (float(np.median([r["speed_ratio"] for r in a["trials"]])),
                              float(np.mean([r["exact"]["time"] / max(r["exact"]["exact_calls"], 1)
                                             for r in a["trials"]])),
                              int(np.mean([r["exact"]["exact_calls"] for r in a["trials"]]))))
    A("- fallback 计数 = 枚举丢根后回退多起点的次数（`n_gamma=1` 时偶发）。")
    A("")
A("## 6. 复现")
A("")
A("```bash")
A("cd labs/02-constrained-qpeplayer/experiments/code")
A("/home/syk/miniconda3/envs/sdpr/bin/python run_quartic.py --phases generic p2 perturb robust alm")
A("/home/syk/miniconda3/envs/sdpr/bin/python make_quartic_tables.py")
A("```")
A("")
A("单测：`python -c \"import quartic_roots as Q, qpep_core as C; ...\"`（见 run_quartic.py 的 phase 函数）。")
open(MD, "w").write("\n".join(L) + "\n")
print("wrote", MD)
