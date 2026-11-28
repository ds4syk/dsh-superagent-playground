# 官方 SDPRLayers 前向基线（P1 10×10 QPEP）

> **一句话结论：发布的 SDPRLayers `SDPRLayer` 在 CPU（torch 2.5.1+cpu + diffcp/SCS）上**跑通**；10×10 QPEP SDP 上 tight 实例的前向值与我们的 Veronese SDP 差 ≤ 5.4e-12，位姿与 QPEP 硬解一致；但它的位姿出口被 `ER>1e5` 判据卡住，θ=0 退化实例直接 `xs=None`。**

## 0. 口径、来源、环境

- **发布版来源**：`utiasASRL/sdprlayer` clone 在 `/mnt/storage1/syk/third_party/sdprlayer`，commit **cb3aed2**（origin `https://github.com/utiasASRL/sdprlayer.git`）；README 未给单独 tag，该 commit 即 clone 的 upstream HEAD。
- **环境里的 `sdprlayers`**：`sdprlayers 0.0.1` 是 **editable 安装**，指向本地 fork `/home/syk/labs/SDPR/code/sdprlayer`，HEAD **62d8ac6**（= `cb3aed2` + 8 个兼容性 commit：把 mosek/theseus/sparseqr 变成可选、给 `SDPPoseEstimator` 加 SCS 后端、求解失败阈值开关）。这些**不是**发布版内容，是让发布版代码在无 MOSEK/无 CUDA 环境能跑的最小补丁；后文会把两者都跑一遍互相验证。
- **发布版能否直接 import**：`PYTHONPATH=<tree>/src python -c 'import sdprlayers'` → **失败**，`ModuleNotFoundError: No module named 'mosek'`。原因：发布版 `__init__.py` 在模块级 eager import `SDPEssMatEst`（→ `cert_tools` → `import mosek`）等可选层。完整 traceback 与本次临时树路径见 JSON `provenance.pristine_package_import`。
- **绕行方式（未改动发布版任何一行）**：`git archive cb3aed2` 导出干净树，用 `importlib` 直接按文件路径加载 `src/sdprlayers/layers/sdprlayer.py`（该模块没有包内 import）；它 module 级 `import sparseqr` 是编译版 SuiteSparse 绑定，本机未安装且在该代码路径上从未调用，因此注入一个 no-op 模块。文件 sha256 = `eb8b4cee93daf3d4a5e5179eba1c051d44739f06836dca0805c9a5fb8d5d2fdd`，`sparseqr_stubbed=True`。
- **求解器**：SCS（经 diffcp），论文设置 `eps=1e-9, max_iters=2e5`；`use_dual=True`；`diff_qcqp=False`（与 `SDPPoseEstimator` 默认一致，只做前向，不触发 KKT 反向）。
- **没有安装/修改任何包**：`pip install -e` 未执行；env 里已有的 `sdprlayers 0.0.1` editable 安装指向 fork（上一条），发布版只用临时导出树 + 按文件加载，系统环境零改动。
- **import 方式（可复制）**：

```python
# (a) 发布版文件（cb3aed2 干净树），绕过包的 eager __init__：
import importlib.util, sys, types
sys.modules.setdefault('sparseqr', types.ModuleType('sparseqr'))  # 本机缺该可选编译依赖
spec = importlib.util.spec_from_file_location('rel', '<tree>/src/sdprlayers/layers/sdprlayer.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
SDPRLayer = m.SDPRLayer          # 或从 <tree>/src 设 PYTHONPATH 后 from sdprlayers... import
# (b) 环境已安装的 fork@62d8ac6（含 SCS backend，SDPPoseEstimator 用这个）：
from sdprlayers.layers.sdprlayer import SDPRLayer
from sdprlayers.layers.pose_est import SDPPoseEstimator  # solver='SCS'
```

- **只做前向**：不涉及可微性/反向，不涉及 GPU。
- **环境**：`/home/syk/miniconda3/envs/sdpr/bin/python`（torch 2.5.1+cpu, numpy 1.26.3, cvxpy 1.5.4, scs 3.3.1, diffcp 1.0.23, `torch.cuda.is_available()=False`）。
- **命令**：

```bash
cd labs/01-laplace-qpeplayer/experiments/baselines
/home/syk/miniconda3/envs/sdpr/bin/python run_official_sdpr.py
```

运行时长 3.3 s。

## 1. 先验证 wrapper：4×4 随机 SDP（官方层 vs CVXPY+SCS）

| 量 | 官方 `SDPRLayer` | CVXPY+SCS | 差 |
|-|-|-|-|
| 最优值 | -17.613559140 | -17.613559140 | +2.3e-11 |
| `X_00` | 0.999999999983 | 1.000000000000 | -1.7e-11 |
| `||X-X_ref||_F/||X_ref||_F` | 4.0e-12 | — | — |
| rank / ER | 1 / 5.1e+15 | — | |
| 求解 ms | 318.1 | 11.7 | |

（官方层首次调用含 diffcp 一次性初始化，~0.3 s；后续 10×10 调用为 6–8 ms。）

官方层返回的 `Xs` 被解释为原 SDP 的 `X`（对偶约束的乘子），`xs = X[:,0]/X[0,0]` 是 rank-1 时的 lift。值/矩阵与独立 CVXPY 解在 `4e-12` 相对误差内一致 → wrapper 语义正确。

## 2. 发布版公开 API：`SDPPoseEstimator`（13×13，玩具 3D-3D 位姿）

20 个无噪声 3D 对应点，GT `T_trg_src` 由 se3 参数 `(0.15,-0.2,0.1,0.3,-0.1,0.2)` 生成；`T_s_v = I`；`solver="SCS"`。误差 `se3_log(T_est^{-1} T_gt)`。

| 实例 | se3 误差 max | ms | 备注 |
|-|-|-|-|
| exact | 4.0e-10 | 13.0 | ok |
| noisy | 4.3e-03 | 12.0 | ok |

说明：这一步用的是 fork@62d8ac6 的 `SDPPoseEstimator`（发布版该类硬编码 MOSEK，无 MOSEK 无法实例化）；兼容补丁只替换 backend 选择，不改进位姿提取。

## 3. P1 10×10 QPEP 旋转 SDP：官方层 vs 我们 vs QPEP 硬解

`Q10 = 0.25 U M U^T`，21 条 SO(3) 约束，`X_00=1`，SCS `eps=1e-9`。“官方”= 发布版 `sdprlayer.py@cb3aed2`（干净树、文件路径直载）；同一 Q/A 再用 fork@62d8ac6 安装版跑一遍，用于验证兼容补丁不改数值。

| θ | J_min | 我们 f10 | 官方 f | 官方−我们 | 官方−J_min | rank(官方) | ER(官方) | 官方 tight | 官方 xs=None | ms(官方) | ms(我们) |
|-|-|-|-|-|-|-|-|-|-|-|-|
| 0.00 | -0.482900108 | -0.482900108 | -0.482900108 | +3.7e-10 | +4.1e-11 | 2 | 1.0e+00 | False | True | 7 | 13 |
| 0.01 | -0.520090355 | -0.520090355 | -0.520090355 | +6.7e-14 | +6.7e-14 | 1 | 5.7e+14 | True | False | 12 | 21 |
| 0.10 | -1.037912209 | -1.037912209 | -1.037912209 | +5.4e-12 | +4.1e-12 | 1 | 2.1e+15 | True | False | 6 | 19 |

交叉验证（发布版 vs env fork）：最大 `|Δf| = 0.0e+00`（相对 0.0e+00），rank/tight 判定一致：True。

位姿（`col`=官方 `recovery_map` 第一列，`eig`=主特征向量，`xs`=官方返回值；非 tight 实例的 `col`/`xs` 不是可行 lift，标“—”不参与解释）：

| θ | 官方 tight | 官方 col vs 硬 (rot°, m) | 官方 eig vs 硬 | 官方 xs vs 硬 | 官方 col vs 我们 sdp10 col | 我们 sdp10 col vs 硬 |
|-|-|-|-|-|-|-|
| 0.00 | False | — | <1e-6° / 2.2e-10 m | — | — | — |
| 0.01 | True | <1e-6° / 6.0e-13 m | <1e-6° / 6.0e-13 m | <1e-6° / 6.0e-13 m | 1.71e-06° / 5.9e-13 m | <1e-6° / 3.8e-15 m |
| 0.10 | True | <1e-6° / 2.4e-10 m | <1e-6° / 2.4e-10 m | <1e-6° / 2.4e-10 m | <1e-6° / 2.4e-10 m | <1e-6° / 3.1e-14 m |

- θ=0 的 `X*` 是 rank-2（两个特征值几乎相等，`ER=1.00`），`col`/`xs` 不是可行 lift，对硬解的角度偏差（4.6°/6.6°，我们 `col` 也有 2.0°）**不作为位姿误差解释**；但主特征向量 `eig` 仍精确指向 QPEP 硬解（<1e-6°/2e-10 m）——即值对、方向对，只是双层特征值让 rank-1 提取不唯一。
- θ=0.01/0.1：官方 `col`（以及 `xs`）与 QPEP 硬解旋转差 <1e-6°、平移差 2.4e-10 m；与我们 10×10 `col` 的差 1.71e-06°。

## 4. 结论与对 P1 的影响

1. **官方前向在 CPU 上可跑，且 10×10 值与我们一致**：发布版 `SDPRLayer`（cb3aed2，文件路径直载）在 tight 实例上与我们的 CVXPY+SCS 值差 ≤ 5.4e-12；θ=0 的 rank-2 实例差 3.7e-10（两边各自的 SCS 误差已是这个量级）。4×4 玩具上 `f` 差 2.3e-11、`||ΔX||_F/||X||_F = 4.0e-12`。→ 10×10 也可以直接喂官方层，不必只依赖 13×13。
2. **但官方层的位姿出口由 `ER>1e5` 把关**：2/3 个 θ 通过；θ=0 的 rank-2 退化实例`xs=None`（`SDPPoseEstimator` 会 `RuntimeError`）。值仍然算得对，只是**不给位姿**——与 `sdpr_p1.json` 里 13×13 上游臂的结论一致，不是 10×10 特有。
3. **tight 实例上 10×10 官方位姿 ≈ QPEP 硬解**：θ=0.01/0.1 的 `col`/`xs` 对硬解旋转差 <1e-6°、平移差 ≤ 2.4e-10 m，且与我们 10×10 `col` 一致到 1.71e-06°。因此官方层可作为 P1 前向的**独立实现校验**，而不是替代 QPEP（退化处它没有输出）。
4. **发布版无法直接 `import sdprlayers`**（MOSEK/theseus/sparseqr 可选依赖被 eager import）。这是环境/打包问题，不是数学问题；复现官方数字必须用本文的“干净树 + 文件路径加载 + sparseqr stub”方式，或装齐 MOSEK。

### 复现命令与产物

```bash
git -C /mnt/storage1/syk/third_party/sdprlayer rev-parse HEAD          # cb3aed2
git -C /home/syk/labs/SDPR/code/sdprlayer rev-parse HEAD          # 62d8ac6
cd labs/01-laplace-qpeplayer/experiments/baselines
/home/syk/miniconda3/envs/sdpr/bin/python run_official_sdpr.py
```

- 机器可读结果：`experiments/results/official_sdpr.json`
- 本文件：`experiments/results/official_sdpr.md`

