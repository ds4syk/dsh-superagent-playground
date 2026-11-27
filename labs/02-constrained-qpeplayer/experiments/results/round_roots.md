# P2-C 轮次记录：各向同性 3D-3D + 单平面约束的根谱系（8 轮）

> 分支 `opt/p2c-roots`（worktree `wt-p2c-roots`，base `17d7ffc`），2026-11-27 晚。
> 定理 **P2-1b** 为用户提供证明（Feishu `RhzDwI1RKinQsekHDhUcjOVenHL`，2026-11-27）；本文件只记录本仓库的**独立数值测量**，不重述/不替换定理。
> 范围：`code/isotropic.py`、`code/isotropic_roots.py`、新增 `code/iso_spectrum.py`、`code/iso_parity.py`、`results/isotropic_*`、本文件。未改 `quartic_roots.py`、`theory/`、`verify/`、`deliverables/`。
> 运行环境：`/home/syk/miniconda3/envs/sdpr/bin/python`（numpy 1.26.3 / scipy 1.13.1），**单线程 BLAS**（`OMP_NUM_THREADS=1`）以保证 census 可复现。

## 结论（TL;DR）

1. **标准形谱系稳定**：`make_canonical_instance(rank_Sigma=0/1/2/3)`、seed 0–4、`n_gamma=2` →
   **4/12/16/16 复射影根（8/24/32/32 仿射）**，`n_lost=0`。rank 3 在 PSD 规范（$\lambda_{\min}(\Sigma)=0$）下等价于 rank 2。
2. **ALM 组装落在标准形族内**（开放点关闭）：isotropic Horn + 同法向平面约束的 `alm_reduced` 矩阵，其四次多项式到 15 维族
   $F_{\mathbf n}=\{(q^Tq)(q^TKq)\}\oplus\{w^T\Sigma w\}$ 的相对残差为 **1e-15 量级**（k=0..4，μ=0 与 μ>0 都是）；
   PSD 规范下 `rank(Σ_A)=0/1/2/2/2`，与定理的 4/12/16/16/16 一致。
   之前实测“k=2 只给 12”**不是** ALM→Σ_A 映射不对，而是**census 策略**：已提交的 `quartic_roots._run_homotopy`
   只对 status∈{1,2,4} 做 endgame polish，跟踪期判 diverged（status 3）的路径永不回捞，丢掉了 4 个位于
   `||q||≈29–34` 的真正解；对全部路径 polish 的 corrected census 读到 16。
3. **两法向**：1+1 布局（每条法向一条活跃约束）→ corrected **32/64**（定理 32）；2+1 → 36；2+2 → 40。
   即定理的“双法向 32”对应**每条法向一条活跃约束**。
4. **Type I 不坍缩**：anisotropic Horn / PnP 组装到 $F_{\mathbf n}$ 的残差 7e-2 / 5e-1（数量级拒绝），
   committed 与 corrected 均 **40/80**。canonical-family 残差因此是一个可用的结构判别器。
5. **16×16 饱和作用矩阵未实现为可发货模板**（诚实负结果）：符号 Gröbner/Rabinowitsch 参考未在本轮落地，
   `isotropic_roots.py` 旧 docstring 中“已建好 `results/isotropic_gb_reference.json`”的说法已撤回（该文件不存在）。
   可用替代：canonical 成员判定 + corrected census + 64 起点交叉验证的 reduced homotopy。

## 轮次表

| Round | 假设 | 改动 | 命令 | 指标 before → after | 判定 | 产物 |
|---|---|---|---|---|---|---|
| S1 | 已提交的 isotropic 装配/API 自检通过，k=0 与 Davenport 一致 | 无（只跑） | `python -c "import isotropic as ISO; ISO.validate()"`；`solve_isotropic_plane` / `davenport_solve` / `count_roots` smoke | `validate()` 9/9 case：$\max|J_{\rm form}-J_{\rm direct}|=8.5e-14$、$\max|g_{\rm form}-g|=5.6e-16$；horn k=0 → 4 根；Davenport $f$ 与同伦差 **3e-16** | keep | 本文件 |
| S2 | 标准形 4/12/16 谱系在 seed 0–4 稳定 | 新增 `code/iso_spectrum.py`（canonical 20 例，ng=2） | `OMP_NUM_THREADS=1 python code/iso_spectrum.py` | rank=0/1/2/3（5 seeds）→ committed **4/12/16/16**，corrected 同，`n_lost=0`；canonical rank2 census 0.61 s/例 | keep | `results/isotropic_spectrum.{json,md}` |
| S3 | ALM 组装落在标准形族，且“12 而非 16”是 census 伪影 | 在 `isotropic.py` 新增 `poly_coeffs`/`canonical_fit`/`classify_canonical`；在 `isotropic_roots.py` 新增 `_census`/`count_roots(norm_cap)`（polish 全部路径）；iso_spectrum 增加 committed vs corrected 两列与 F-残差 | 同上 + `count_roots(M,n_gamma=2,norm_cap=1e5)` | horn k=0..4：F-残差 **9e-16–7e-15**，rank(Σ_A)=**0/1/2/2/2**；committed **4/12/12/14/16**（seed0）vs corrected **4/12/16/16/16**；ALM μ>0（μ=0.570,ρ=250,iters=12）→ 16/16，残差 8.8e-16 | keep（核心结果） | 同上 |
| S4 | 双法向 → 32；各向异性/PnP → 40 | iso_spectrum 增加 two-normal（1+1/2+1/2+2）与 Type I（aniso/pnp, k=3） | 同上 | 1+1：committed **20/32/26**（seed0/1/2）→ corrected **32/32/32**；2+1 → 36；2+2 → 40；aniso **40/40**（残差 7.4–8.9e-2）、pnp **40/40**（残差 5.3–6.0e-1） | keep | 同上 |
| S5 | 可发货的 16×16 饱和作用矩阵（Rabinowitsch + Macaulay/SVD） | 评估实现；本轮**不落地**（时间盒）；撤回旧 docstring 的虚假 JSON 引用，写清“符号参考未建” | — | 状态：**blocked/partial**（用户证明给出 16×16；本仓库无数值模板）；替代：canonical 判定 + corrected census + 64-start 交叉验证 | blocked（诚实记录） | `isotropic_roots.py` docstring；本文件 §16-root status |
| S6 | 结构感知 dispatcher 与已提交 40 根求解器同解 | 新增 `isotropic_roots.classify_structure` / `solve_inner_auto`；新增 `code/iso_parity.py`（10 canonical-r2 + 5 ALM-k2） | `python code/iso_parity.py` | $\max|\Delta f|=$ **1.3e-15**，$\max\|q_1\pm q_2\|=$ **3.7e-10**（15/15 同解）；canonical-r2 census 16/16；ALM committed 12/14/16 vs reduced 16/16/16；committed fallback 5/15 vs reduced 0/15 | keep | `results/isotropic_parity.json`、`code/iso_parity.py` |
| S7 | 修正 census 的成本可忽略；reduced solver 与 committed 同量级 | 计时（iso_spectrum + iso_parity） | 同上 | census：committed 0.949 s → corrected **0.956 s**（×1.01，20+15 例均值）；canonical-r2 0.606→0.655 s；solver：reduced 0.910 s vs committed 0.680 s（×1.34，含 64 起点 BFGS 交叉验证；ALM 1.20 vs 1.15 s） | keep | 同上 |
| S8 | 谱系 + census 差异 + parity 可写成一份可复核产物 | 写 `results/isotropic_spectrum.{json,md}`、`results/isotropic_parity.json`、本文件；`iso_spectrum.py`/`iso_parity.py` 可重跑 | `python code/iso_spectrum.py && python code/iso_parity.py` | 全部数字与本文件一致；单线程 BLAS 下可复现 | keep | 上述三份 + 本文件 |

## 谱系表（关键交付）

### (a) 标准形（权威 16 根模板测试用例）

| rank(Σ)（PSD 规范） | seed 范围 | 复射影根 | 仿射根 | `n_lost` | 时间（s/例） |
|---|---|---|---|---|---|
| 0 | 0–4 | **4** | 8 | 0 | 0.98 |
| 1 | 0–4 | **12** | 24 | 0 | 0.73 |
| 2 | 0–4 | **16** | 32 | 0 | 0.61 |
| 3（≡2） | 0–4 | **16** | 32 | 0 | 0.61 |

$J(q)=(q^Tq)(q^TKq)+w(q)^T\Sigma w(q)$，$w=R^Tn$。`make_canonical_instance` 的 Mred 对 $F_n$ 残差 1.2–2.6e-15。

### (b) ALM 组装（isotropic Horn + 同法向 k 条活跃约束）

| k | F-残差 | rank(Σ_A) | 定理预测 | committed census | corrected census |
|---|---|---|---|---|---|
| 0 | 1.4–4.3e-15 | 0 | 4 | **4/8** | **4/8** |
| 1 | 1.2e-15–7.5e-15 | 1 | 12 | **12/24** | **12/24** |
| 2 | 1.4e-15–7.4e-15 | 2 | 16 | 12/14/16（seed 0–2） | **16/32** |
| 3 | 9.2e-16–7.1e-15 | 2 | 16 | 14/16/16 | **16/32** |
| 4 | 1.1e-15–7.3e-15 | 2 | 16 | 16/16/14 | **16/32** |

`committed` = `quartic_roots.critical_points(n_gamma=2)`（`||q||<1e3`、status 3 不 polish）；
`corrected` = `isotropic_roots.count_roots(n_gamma=2, norm_cap=1e5)`（全部路径 polish）。
μ>0 的真实 ALM 组装（seed0,k=2, μ=[0.570,0], ρ=250）与 μ=0 都是 16/16、F-残差 8.8e-16 → **μ 不改变谱系**（只移动 K）。

### (c) 两法向 / Type I

| 构造 | F-残差 | committed | corrected | 备注 |
|---|---|---|---|---|
| 两法向 1+1（k=2） | 1.1–1.3e-15 | 20/32/26 | **32/64** | 定理 32 对应“每法向一条约束” |
| 两法向 2+1（k=3） | 1.1–1.3e-15 | 24/32/28 | **36/72** | rank(Σ1,Σ2)=(2,1) |
| 两法向 2+2（k=4） | 1.1–1.5e-15 | 38/32/28 | **40/80** | rank=(2,2)，回到通用 40 |
| anisotropic Horn + 3 平面 | 7.4–8.9e-2 | 40 | 40 | 不在 $F_n$（类型 I） |
| PnP + 3 平面 | 5.3–6.0e-1 | 40 | 40 | 不在 $F_n$（类型 I） |

## 16 根饱和模板：状态与替代

* **定理**（用户证明）：k≥2 时有 17 元 Gröbner 阶梯 / **16×16 饱和作用矩阵**；Rabinowitsch 辅助变量
  $u_{\rm sat}(1+q_1^2+q_2^2+q_3^2)-1=0$ 饱和掉 Hopf 直纹线 $\mathcal L_{\bf n}=\{q^Tq=0,w=0\}$。
* **本仓库状态：未实现为可发货的数值模板（blocked/partial）**。原因与边界：
  - 未在本轮投入 Macaulay/SVD 数值商环实现（时间盒）；符号 Gröbner 参考也未落盘；
  - 旧 `isotropic_roots.py` docstring 声称存在 `results/isotropic_gb_reference.json`，**该文件不存在，声明已撤回**；
  - 对任意数值实例，可用的替代是 **canonical 成员判定 + corrected census + 64-start 交叉验证**：
    `solve_inner_auto(M, n=normal)` 会识别 $F_n$、按 rank(Σ_A) 预测 4/12/16，并返回 `census_consistent`。
* **可用性结论**：对**全局实数最小**（真正要用的量），40 根/16 根模板的差别不影响——实数根都在单位球上，
  committed 与 corrected 枚举给出同解（$\max|\Delta f|=1.3e-15$）。16 根谱系目前是**复根普查/结构判别**层面的结论，
  不是已发货的快速求解器。

## 命令

```bash
cd labs/02-constrained-qpeplayer
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/syk/miniconda3/envs/sdpr/bin/python

# S1 自检
$PY -c "import sys; sys.path.insert(0,'experiments/code'); import isotropic; isotropic.validate()"

# S2–S4 谱系（约 85 s，写 results/isotropic_spectrum.{json,md}）
$PY experiments/code/iso_spectrum.py

# S6/S7 parity（10 canonical-r2 + 5 ALM-k2，约 30 s）
$PY experiments/code/iso_parity.py
```

## 注意事项（caveats）

1. **census 依赖策略**：本文件同时给 committed 与 corrected 两列，没有把 committed 的 12 直接改写成 16。
   corrected 的 4 个额外复根位于复二次曲面 $q^Tq=1$ 的大分量区（seed0 k=2：`||q||≈29–34`，其余 ≤13），
   条件数较差；多线程 BLAS 下个别次运行会把它推到 `||q||≈8e3` 或丢失 → 本文件数字用单线程 BLAS 复现。
2. **census 策略的修复点在 `quartic_roots.py`**（`_run_homotopy` 的 `finite` 掩码与 status 3 不 polish），
   该文件属 eng 线、不在本 worktree 的修改范围；本轮以 `isotropic_roots._census` 旁路实现，未改 `quartic_roots.py`。
   建议 eng 线评估：把 endgame polish 扩到 status 3（`||q||<1e4` 内）并把 `xnorm<1e3` 提到 1e4–1e5。
3. **`n_lost`**：corrected census 的 k=2 seed0 有 2 条路径 Newton 失败（`n_lost=2`），但冗余路径仍找齐 16 根；
   其余 ALM 行 `n_lost=0`。canonical 20 例全部 `n_lost=0`。
4. 定理 P2-1b 的 4/12/16/32/40、16×16 作用矩阵、15/12/17 元阶梯均为**用户证明**的代数数字；
   本仓库只独立复现了其中可数值检验的谱系读数，未复跑 $\mathbb F_{32003}$ Gröbner 计算。
5. `canonical_fit` 是**多项式空间**（35 维四次单项式）上的拟合，不是矩阵空间的拟合：
   矩阵差可以落在 $\ker(M\mapsto \text{polynomial})$ 里而多项式相同；判定用 35 维残差（1e-15 量级 = 在族内，
   1e-1 量级 = 在族外）。PSD 规范 $\Sigma\mapsto\Sigma-\lambda_{\min}I,\ K\mapsto K+\lambda_{\min}|n|^2I$
   是定理 (2.8) 的取法，与矩阵零空间无冲突。
