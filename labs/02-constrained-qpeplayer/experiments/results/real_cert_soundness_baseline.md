# 真实 BOP 证书：两侧 gap 与严格性基线

> 从现有 `real/*/results/raw.json` 重算，不改数据。`J = J_ALM`，`Shor` 为 14×14 松弛值；
> 理论上有 `Shor <= J* <= J_ALM`，所以 `J_ALM - Shor >= 0`。**出现负值说明 SDP 数值解违背原始约束**
> （SCS 容差），此时 `J - Shor <= 1e-6` 的一侧判定不再构成证书。

| dataset | n | base 一侧 cert | base 两侧 `|J-Shor|<=1e-6` | base gap 范围 [min, med, max], #neg | RLT 一侧 cert | RLT 两侧 | RLT gap 范围, #neg |
|---|---|---|---|---|---|---|---|
| linemod | 18 | 9/18 = 50% | 7/18 = 39% | [-1.09e-05, 2.48e-06, 1.69e-04], 4 neg | 9/18 = 50% | 5/18 = 28% | [-3.68e-04, 1.59e-06, 1.49e-01], 5 neg |
| lmo | 18 | 4/18 = 22% | 1/18 = 6% | [-1.99e-04, 2.57e-05, 1.49e-03], 4 neg | 7/18 = 39% | 0/18 = 0% | [-4.28e-04, 1.52e-05, 6.19e-02], 7 neg |
| ycbv | 12 | 10/12 = 83% | 9/12 = 75% | [-6.95e-06, 6.79e-12, 1.24e-05], 6 neg | missing | missing | missing |

## 读数

- 一侧判定的 50%/22%/83% **不是严格证书**：LM/LM-O 有约 1/4–1/3 的行 `J-Shor<0`，最大负 gap 约 −4e-4，远大于 1e-6 阈值。
- 两侧 `|J-Shor|<=1e-6` 后：LM base 7/18、LM-O 1/18、YCB 9/12；+RLT 5/18、0/18。也就是说现有真实表的高认证率主要来自「下界高于可行目标值」的数值假象。
- 修正路线（R1 子代理）：用 SDP 对偶可行点 + PSD 修复得到**严格**下界 `bound_dual <= J*`；此时只需 `J_ALM - bound_dual <= 1e-6`，负 gap 不可能出现。真实数据必须用它重跑后才能写「certified」。
- 在此之前，论文/README 必须把真实 cert 率标成 upper bound（`paper/draft.md` §5.5(iv) 已如此处理）。

Source: `results/real_cert_soundness_baseline.json`（本文件由主 worker 汇总）。
