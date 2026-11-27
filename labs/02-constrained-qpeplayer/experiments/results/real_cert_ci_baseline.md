# 真实数据证书率的基线置信区间（Wilson 95%）

> 目的：检查现有真实 BOP 表的样本量是否足以支撑 RLT 结论。数据来自各 `real/*/results/raw.json` 的 ALM-QPEP 行；
> 每行是一个 GT 投影实例（帧×图像×物体×bias），不是独立帧，CI 偏乐观，仅作内部一致性检查。

| dataset | n (ALM rows) | cert base | 95% CI | cert +RLT | 95% CI |
|---|---|---|---|---|---|
| linemod | 18 | 9/18 = 50% | [29, 71] | 9/18 = 50% | [29, 71] |
| lmo | 18 | 4/18 = 22% | [9, 45] | 7/18 = 39% | [20, 61] |
| ycbv | 12 | 10/12 = 83% | [55, 95] | missing（run 早于 RLT） | n/a |

## 读数

- LM：base 与 RLT 都是 9/18，CI 完全重合；RLT 在 LM 上**没有可测增益**。
- LM-O：22% → 39%，但 95% CI [9,45] 与 [20,61] 大幅重叠；n=18 时该提升**不显著**。要支撑「RLT 在 LM-O 上有用」需要更大样本（目标 n≥40，最好 ≥80）。
- YCB：83%（10/12），CI [55,95]，且 raw.json 缺 `cert_rlt` —— 需要重跑。
- 因此真实数据主表的下一步优先级：YCB 补 RLT + 三数据集扩样本；论文里不能把 LM-O 的 22%→39% 写成显著提升。

Source: `results/real_cert_ci_baseline.json`（本文件由主 worker 从 raw.json 汇总；不修改真实数据）。
