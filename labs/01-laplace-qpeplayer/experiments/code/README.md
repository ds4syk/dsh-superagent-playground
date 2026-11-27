# `run_e2e_fit.py` — 端到端 toy 训练 / 梯度方差 / 吞吐

> **一句话结论**：默认全量配置（20 seeds × 120 steps）在 CPU 单核上 **11.1 min** 跑完（exit 0），产出 `results/e2e_fit.{json,md,png}`；`mix` 目标仍不可表示（0%），`star` 正侧 36% 成功、负侧 0%，MC 相对梯度 std 3.2e-2–6.9e-2，mode enumeration 占单样本 fwd+bwd 的 90–94%。

## 1. 测什么

1. **Fit through the layer**：学习标量 $\theta\in[-0.2,0.2]$，使 layer 输出位姿匹配
   - `mix`：$\theta=0$ 时两全局分支位姿的平均（有限 $\tau$ 下不可表示）；
   - `star`：$\theta^\ast=0.10$ 处全局分支的位姿（可表示，正对照）。
   方法：hard argmin / 多峰 Laplace（解析梯度）/ mode-level MC（n=200）。
2. **Gradient variance**：$\theta=0,0.01$ 处 `dpose/dtheta`（$R^{12}$）跨 20 seeds 的 std；hard 在 $\theta=0$ 是精确 tie。
3. **Throughput**：单样本 fwd+bwd（20 reps，BLAS 固定 1 线程）、enumeration 占比、B=1/8/64 的 loop batch（**非向量化**）。
4. **Landscape** 与 **clip sweep**（`mix` 目标的步长/条件数权衡）。
5. 6-panel 图 `results/e2e_fit.png`。

## 2. 精确命令（2026-11-28 全量运行，未缩减）

```bash
# clean cwd；脚本用自身路径定位 results/，与 cwd 无关
cd /tmp/e2e_clean.xJUkMA
timeout 1800 nice -n 10 /home/syk/miniconda3/envs/WM/bin/python -u \
  /mnt/storage1/syk/.superagent-work/wt-final-p1-e2e2/labs/01-laplace-qpeplayer/experiments/code/run_e2e_fit.py
```

- 默认参数：`--seeds 20 --steps 120 --control-seeds 8`；方差 20 seeds、throughput 20 reps、batch B=1/8/64。
- 实测 wall **664.2 s**（`/usr/bin/time` elapsed **11:06.30**，user 640.6 s，单核 96%，峰值 RSS 104 460 kB），exit 0。
- 运行日志：`results/logs/20261128-0100_e2e_fit_full.log`。

## 3. 环境 / CPU

| 项 | 值 |
|-|-|
| host | `ls`（单节点 Slurm 集群登录节点） |
| CPU | Intel Xeon w7-3445，1 socket × 20 core × 2 HT = **40 logical**（`nproc`=40） |
| Python | `/home/syk/miniconda3/envs/WM/bin/python` 3.10.22 |
| numpy / BLAS | numpy 2.2.6，scipy-openblas 0.3.29（脚本内把 OMP/OPENBLAS/MKL/NUMEXPR/VECLIB 全钉为 1 线程） |
| git | `2ca1ec3`（dirty worktree，未 commit） |
| 运行方式 | 登录节点空闲（load ≈6/40）时直接跑单核；未用 `srun`。按仓库 AGENTS.md 计算任务原则上应走 Slurm，此处仅 1 核 11 min，父任务明确允许空闲时直跑 |

## 4. 输出

| 文件 | 大小 | 内容 |
|-|-|-|
| `../results/e2e_fit.json` | 730 kB | 全部 per-seed 曲线、summary、variance、throughput、landscape、clip control、meta |
| `../results/e2e_fit.md` | 10 kB | 人读报告（表 + 解读 + caveats） |
| `../results/e2e_fit.png` | 290 kB | 6 面板图 A–F |

## 5. 全量关键数字（与 JSON 一致）

**Fit**（median final loss / success <1e-4）：

| target | hard | laplace | mc |
|-|-|-|-|
| `mix` | 2.064e+00 / 0% | 1.697e+00 / 0% | 1.663e+00 / 0% |
| `star` | 5.962e-04 / 25% | 5.962e-04 / 25% | 5.962e-04 / 25% |

`star` 按初值侧：正侧 **36%**（5/14，中位 63 steps [26,66]），负侧 **0%**（6/6 卡在 8.202）。三方法在 `star` 上 theta 轨迹逐步完全相同。

**Gradient variance**（std frob / rel std）：

| theta | hard | laplace | mc |
|-|-|-|-|
| 0 | 1.331e+01 / 5.729e-01（tie，导数不存在；两支相对差 1.87） | 7.69e-13 / 1.5e-15 | 3.463e+01 / 6.916e-02 |
| 0.01 | 3.92e-14 / 8.9e-16 | 4.15e-14 / 9.4e-16 | 1.423e+00 / 3.239e-02 |

**Throughput**（单样本 fwd+bwd，median，ms）：$\theta=0$：hard 57.14 / Laplace 58.76 / MC 56.44；$\theta=0.1$：12.93 / 13.49 / 12.98。cold `find_modes(128)` 单独 52.90 ms / 10.78 ms，**enumeration 占 90–94%**。Loop batch 每样本：B=1 18.17 ms → B=8 15.34 ms → B=64 30.53 ms，**无加速**（`find_modes` 是逐样本的，不是跨 instance 向量化）。

**Clip control**（Laplace on `mix`，median final loss）：clip 1.0 → 1.875；0.1 → 2.204；0.02 → 2.947（都 0% <1e-4）。

Landscape：$|\mathrm{dpose}|$ 0.22..3531，fold 在 $|\theta|=1.30\times10^{-3}$，曲率跨度 ~2.5e8。

## 6. 已知问题（未改现有 `.py`）

1. **`write_md()` 摘要硬编码“150 步”**（约第 715 行），与默认 `STEPS=120` 不一致——脚本 bug。本轮交付的 `e2e_fit.md` 已按 `meta["steps"]=120` 手工校正，并加了 §7 校正说明；重跑脚本会再次写出 150。
2. **`write_md()` 中 `star` 的定性段是 smoke 阶段残留**（“远处初值需要远超 N 步”“bit-identical”）。全量 20-seed 数据显示正侧 36% 成功、`theta0=+0.017` 72 步达标；三方法只有 theta 轨迹完全相同，loss 差 ≤1.5e-6。交付 md 的 §7 已注明；建议修复生成器时把这段改为由 summary 生成。
3. **Throughput 只有单样本 / loop batch**：没有跨 instance 的向量化 batch，B=64 反而更慢；引用时不要说成 batched throughput。
4. **smoke 配置记录不一致**：`results/_smoke_e2e.*` 的 meta 是 4 seeds / **15 steps** / 43.7 s，而 rounds 日志与论文 N3 行写“4 seeds / 40 steps / 78 s”。本轮全量结果已取代 smoke，论文 §5.7/N3 需按 `e2e_fit.md` 更新。
5. 计时是共享登录节点上的单次测量，绝对值随负载漂移；用枚举占比和相对比值，不把 ms 当硬常数。

## 7. 复现注意

- `--smoke` 会覆盖 `results/_smoke_e2e.*`；只想试跑时加 `--seeds 1 --steps 5` 也会写 `e2e_fit.*`（覆盖全量结果），建议先备份或用独立 worktree。
- 全量配置实测约 11 min；缩减配置（`--seeds 10 --steps 80`）本轮**未使用、未实测**，`--smoke` 只是小样本自检。
