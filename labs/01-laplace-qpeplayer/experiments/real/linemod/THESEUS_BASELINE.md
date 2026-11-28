# Theseus LM 基线（P1 真实双模代价，BOP LINEMOD）

> **结论先行**
> 1. **对账通过**：Theseus 目标 = `run_two_mode.py` 的 numpy 代价 `C(R,t)=Σ‖P_i^⊥(R p_i+t)‖²`，105 个位姿（15 帧 × 7）最大绝对差 **1.1e-13**；解析 Jacobian vs 中心差分 **7e-11–1.4e-10**。
> 2. **GT 初始化 = 局部收敛检查**：15/15 帧回到 P1 枚举出的模式（14 帧落在 k0，中位旋转误差 **2.06°**，与 `two_mode.md` 的 hard-argmin 一列逐帧相同）。
> 3. **冷启动基本失败**：identity 初始化 1/15 命中 GT；随机初始化 100/300（33.3%）；8 起点多启动 11/40（27.5%）。失败解要么是 P1 的 flip 模式，要么是**面朝后的近对映伪极小**（t≈−t_gt、全部点 z<0），后者代价只比物理模式高 1.08–2.27×，且在 **2/15 帧比物理全局极小更低**（0.939×、0.969×）。
> 4. **结论**：这是「不枚举就没有全局性」的诚实负结果；Theseus 不返回后验，不能当 Laplace 的替代估计器，只作为 *no-enumeration* 参照。

## 1. 做法：复用 P3 的 Theseus 基线，不重造

- 复用 `labs/03-robust-qpeplayer/experiments/baselines/theseus_cost.py` 的几何约定与工具：`se3_from_Rt`、`skew_batch`、`errors_from_pose`，并沿用它的有限差分自检结构（右扰动 `T ← T @ Exp(δ)`，`δ=(v,ω)`，平移在前）。
- 复用 `labs/03-robust-qpeplayer/experiments/baselines/run_theseus.py` 的求解协议：`th.LevenbergMarquardt` + `th.CholeskyDenseSolver`、float64 CPU、GT/随机两种初始化、20 trials、自检后再测。
- 新代价类 `BearingResidual`：`r_i = P_i^⊥(R p_i + t)`，`P_i^⊥ = I − u_i u_i^T`；与 P3 点对点 cost 的唯一区别是常数左因子 `P^⊥`。
  - 求解用**单块 3N 维残差**（batch 1）；测量数据以常量 tensor 保存、**不**注册为 aux Variable（Theseus 的 objective batch = 所有注册变量的最大 batch，aux 为 N 会让 objective 变成 batch N）。
  - P3 的**逐点 cost**（每个对应一个 3 维 cost）作为交叉校验：前 3 帧 21 个位姿、以及 000010/15 与 000010/90 的求解结果，位姿差 0°、0 m。
- CHOLMOD 缺失不影响：变量只有 6 维，分块稠密 Cholesky 足够。

## 2. 精确命令

```bash
cd /mnt/storage1/syk/.superagent-work/wt-p1-theseus/labs/01-laplace-qpeplayer/experiments/real/linemod
# 全量：15 帧 × (GT + identity + 20 随机) + 5 帧 × 8 多启动 = 370 次 LM，约 9 s（登录节点 CPU，无需 srun）
/mnt/storage1/syk/venvs/theseus/bin/python run_theseus_baseline.py
# 冒烟：数据 + 对账 + 单次 GT 求解
/mnt/storage1/syk/venvs/theseus/bin/python run_theseus_baseline.py --selftest
```

环境：`/mnt/storage1/syk/venvs/theseus`（theseus 0.2.3、torch 2.11、numpy 2.2.6、scipy 1.15）。数据：`/mnt/storage1/syk/data/bop/lm`。

## 3. 主要结果（完整表见 `results/theseus_baseline.md`）

| 方法 | 返回什么 | 中位旋转误差 | 命中 GT（<10°） |
|-|-|-|-|
| hard argmin（P1，**引用** `two_mode.md`） | 物理模式中的全局极小 | 2.06° | 14/15 |
| Laplace 均值（P1，**引用**） | 物理模式上的后验均值 | 58.28° | 0/15 |
| Theseus LM，GT 初始化 | 一个局部解 | 2.06° | 15/15 |
| Theseus LM，identity 初始化 | 一个局部解 | 178.82° | 1/15 |
| Theseus LM，随机初始化（20/帧） | 一个局部解 | 123.91° | 100/300 |
| Theseus LM，8 起点多启动 | best-of-8 | 4.95° | 11/40 |

- 随机初始化结果分解（300 次）：命中 GT 100、flip 57、其他 143（其中面朝后伪极小 134，物理但非 GT 的 9）。
- 多启动 5 帧：`000010/15` 3/8、`000010/72` 0/8、`000010/60` 2/8、`000011/20` 3/8、`000011/49` 3/8。
- 全部 370 次 LM 均 `status=1`（收敛），最大迭代 21（上限 100）。

## 4. 与 P1 数字的对比（引用，不重算）

`two_mode.md`：15 帧全部 K=2；hard argmin 中位旋转误差 2.06°、14/15 命中；Laplace 均值中位 58.28°、0/15 命中；hard argmin 在 `000010/72` 选错（44.46°）。本次 Theseus 的 GT 初始化逐帧复现 hard-argmin 一列（`000010/72` 例外：它落在 3.84° 的 k1 模式，即 GT 所在的盆地，而全局 argmin k0 在 44.46°）。

## 5. 诚实边界（负结果照报）

- Theseus 是**局部**优化器：一次调用给一个解，不枚举模式、不知道 P1 的 z>0 物理性过滤、不返回后验 ⇒ 不能与 Laplace 均值比“估计精度”，只能作为“不枚举的参照”。
- 无 z>0 约束 ⇒ 代价有面朝后的伪极小（`t≈−t_gt`，全部点 z<0，旋转≈180°），代价与物理模式近简并；在 `000010/66`、`000010/72` 上甚至**比物理全局极小更低**。因此“在 Theseus 目标上跑”与“P1 的代价”并不完全等价，任何 baseline 对比必须声明这一点。
- `000010/72`：8 次重启 5 次落到 44.5° 的全局 argmin 模式、0 次回到 GT —— 当 GT 本身不是全局极小时，增加局部重启没有用；只有“模式集合 + 物理/先验判据”能解决。
- 不声称：第三方 pipeline 精度、ADD(-S) AUC、检测器前端；本基线只针对 P1 的 cost。

## 6. 文件

- 新增：`run_theseus_baseline.py`、`results/theseus_baseline.json`、`results/theseus_baseline.md`、本文件。
- 复用（未修改）：P3 `theseus_cost.py` / `run_theseus.py`；P1 `run_two_mode.py` 的 `load_gt`/`real_correspondences`/`plane_ransac`/`reduce_cost`/`physical_modes`（仅用于数据构造与模式 label）。
- 未修改任何既有 `.py`，未 commit / push。
