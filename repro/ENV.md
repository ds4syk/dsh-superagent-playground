# 复现环境记录（`sdpr` conda env，Python 3.11）
> **可移植性说明**：下文的 `/home/syk/...` 是原始实验机的路径；在别的机器上请把
> `P` 改为自己的 conda 解释器，把 `REQ` 改为 `code/sdprlayer`（上游仓库，本仓库不内嵌）。
> 版本 pin 与错误修复记录仍适用。



> 本文件记录 SDPRLayers faithful 复现所用的**最终**环境、安装命令、遇到的错误与修复。
> 从零重建：`bash repro/upstream/install_env.sh`（本文件与脚本内容一致；
> `conda create -n sdpr python=3.11 -y` 后执行脚本即可）。

## 主机 / 约束

| 项目 | 值 |
|---|---|
| CPU | Intel Xeon w7-3445，40 核 |
| 内存 | 125.5 GB |
| GPU | 3× RTX 5090（sm_120）——**本复现全程 CPU**，torch 用 CPU wheel |
| 解释器 | `/home/syk/miniconda3/envs/sdpr/bin/python`（3.11.17） |
| 线程限制 | `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=8`，`MKL_THREADING_LAYER=GNU`，`torch.set_num_threads(8)`；分片跑时每进程 2–3 线程 |
| 网络 | GitHub / PyPI / conda 可访问；无 sudo（不能 apt install） |
| Mosek | **无 license，未安装**（论文 Table III/IV 的 1e-12 精度因此无法复现；§VI-B4 本身用 SCS） |

## 安装命令（与 `repro/upstream/install_env.sh` 一致）

```bash
P=/home/syk/miniconda3/envs/sdpr/bin/python
REQ=/home/syk/labs/SDPR/code/sdprlayer

# 1) 科学计算栈（numpy 必须 1.26.3，见"错误与修复"）
$P -m pip install "numpy==1.26.3" "scipy==1.13.1" "matplotlib==3.8.4" pytest tqdm \
    "scikit-image==0.22.0" "scikit-learn==1.4.2" "pandas==2.2.2" \
    "opencv-python-headless==4.10.0.84" kornia asrl-pylgmath

# 2) CPU-only torch（与上游 requirements 的 2.5.1 对齐）
$P -m pip install "torch==2.5.1+cpu" "torchvision==0.20.1+cpu" \
    --index-url https://download.pytorch.org/whl/cpu

# 3) torch/torchvision 可能把 numpy 顶到 2.x，强制回到 1.26.3
$P -m pip install --no-cache-dir --force-reinstall --no-deps "numpy==1.26.3" "scipy==1.13.1"

# 4) 锥规划求解器 + diffcp 编译依赖
$P -m pip install pybind11 scs ecos clarabel threadpoolctl cffi "cvxpy==1.5.4"

# 5) 本地包（依赖顺序；--no-build-isolation 让 diffcp 用已装 numpy/pybind11）
$P -m pip install -e $REQ/src/submodules/poly_matrix --no-deps
$P -m pip install -e $REQ/src/submodules/cert_tools --no-deps          # 仅 essmat/fundmat 用
$P -m pip install -e $REQ/src/submodules/diffcp --no-build-isolation --no-deps   # 编 C++ 扩展
$P -m pip install -e $REQ/src/submodules/cvxpylayers --no-deps
$P -m pip install -e $REQ --no-deps --no-build-isolation

# 6) OPTIONAL：Theseus 基线（Table V 用）。见"Theseus 依赖"
$P -m pip install --no-deps torchlie torchkin semantic-version pyyaml
$P -m pip install --no-deps --no-build-isolation "theseus-ai==0.2.1"
# 运行 Theseus 相关脚本时 PYTHONPATH 需包含 repro/upstream/stubs（sksparse stub）

# 7) 可选（本机失败，见下）：sparseqr（需要 SuiteSparseQR，未安装；patch 0001 已让它 optional）
```

## 最终包版本（`pip list` 摘要）

| 包 | 版本 | 备注 |
|---|---|---|
| numpy | 1.26.3 | 上游 pin；diffcp C++ 扩展按此编译 |
| scipy | 1.13.1 | |
| torch / torchvision | 2.5.1+cpu / 0.20.1+cpu | CPU；`cuda_available=False` |
| cvxpy | 1.5.4 | ≥1.6 强制 numpy≥2，会破坏 diffcp |
| scs | 3.3.1 | SDP 后端，论文 §VI-B4 用 eps=1e-9 |
| ecos / clarabel / osqp / highspy | 2.0.14 / 0.11.1 / 1.1.3 / 1.15.1 | diffcp/cvxpy 依赖 |
| diffcp | 1.0.23（源码编译；`holmesco/diffcp@c09ee314`） | C++ 扩展 `_diffcp` 编译成功 |
| cvxpylayers | 0.1.7（`holmesco/cvxpylayers@728b8f20`，editable） | |
| poly_matrix | 0.2（`utiasASRL/poly_matrix@671d6b77`，editable） | |
| cert_tools | 0.0.2（`utiasASRL/certifiable-tools@afcd203b`，editable） | 仅 test_essmat/test_fundmat 需要 |
| sdprlayers | 0.0.1（本仓库，editable） | |
| theseus-ai | 0.2.1 | optional；真实 Theseus GN 基线 |
| torchlie / torchkin / urdf-parser-py / pyyaml / semantic-version | 0.1.0 / 0.1.1 / 0.0.4 / 6.0.3 / 2.10.0 | theseus 的纯 Python 依赖 |
| scikit-image / scikit-learn / pandas / matplotlib | 0.22.0 / 1.4.2 / 2.2.2 / 3.8.4 | |
| kornia / asrl-pylgmath / opencv-python-headless | 0.9.0 / 1.0.3 / 4.10.0.84 | `opencv 5.x 要求 numpy>=2`，故降到 4.10 |

未安装：**mosek**（无 license）、**sparseqr**（缺 SuiteSparseQR 头文件）、**scikit-sparse/sksparse**（见下）。

## 遇到的错误与修复

| # | 错误 | 根因 | 修复 |
|---|---|---|---|
| 1 | `git submodule update` 挂在 SSH 凭证；4 个子模块目录为空 | `.gitmodules` 用 `git@github.com:`，且 `git submodule--helper clone` **不认 repo 级 `url.insteadOf`** | 全局 `git config --global url."https://github.com/".insteadOf "git@github.com:"`；并逐个 `git config submodule.<path>.url https://...`；`rm -rf src/submodules/* .git/modules/src/submodules/*` 后重新 `--init --recursive` |
| 2 | `diffcp` 编译失败 / `ModuleNotFoundError: ecos` | `pip install sparseqr` 缺 `SuiteSparseQR_C.h` 导致整批 pip 事务回滚（ecos/scs/clarabel/cvxpy 全没装）；diffcp 又用 `--no-deps` 装了 | 把 solver 依赖拆成单独一条 `pip install`；`sparseqr` 单独 try（patch 0001 已使其 optional） |
| 3 | `import sdprlayers` → `No module named 'mosek'` | `cert_tools/__init__` 模块级 `import mosek`（essential_est/fundmat 用到） | patch 0001：`sdprlayers/__init__.py` 对这些层 lazy import |
| 4 | `import sdprlayers` → `No module named 'theseus'` | `stereo_tuner.py` 模块级 `import theseus` | patch 0001：移到 `build_theseus_layer()` 内 lazy import |
| 5 | `import sdprlayers` → `No module named 'sparseqr'` | `sdprlayer.py` 顶层 import（模块内从未使用） | patch 0001：optional import |
| 6 | `pytest` 直接报 `unrecognized arguments: --cov=hello` | `setup.cfg` 的 `addopts = --cov=hello`，但无 pytest-cov、无 hello 包 | patch 0002：清空 addopts |
| 7 | numpy 被 pip/conda 悄悄升到 2.x，`scipy`/`diffcp` 报 numpy ABI 错误 | torch 装 numpy2、conda-forge `scikit-sparse` 要求 numpy>2 | 最终强制 `--force-reinstall --no-deps numpy==1.26.3 scipy==1.13.1`；**不再用 conda 装任何东西** |
| 8 | `scikit-sparse` 编译失败：`SuiteSparseQR_C.h / cholmod.h not found`；conda 版又要求 numpy>2 | 机器学习机无 sudo、未装 SuiteSparse dev 包 | 放弃 sksparse：`repro/upstream/stubs/sksparse` 提供 **import-only stub**（CHOLMOD 真的被调用时抛错）。Theseus 默认线性求解器是 `CholeskyDenseSolver`，实际不触碰 CHOLMOD |
| 9 | theseus `sdist` 安装报 `Theseus installation requires torch` | PEP517 build isolation 里没有 torch | `pip install theseus-ai==0.2.1 --no-deps --no-build-isolation` |
| 10 | `import theseus` → `No module named 'torchlie' / 'urdf_parser_py' / 'yaml'` | 这些是 theseus 的运行时依赖（--no-deps 跳过） | 补装 `torchlie torchkin semantic-version urdf-parser-py pyyaml`（全部 --no-deps，不碰 numpy） |
| 11 | theseus 前向 `RuntimeError: size of tensor a (20) must match b (3)` | theseus 0.2.1 的 `torch.where((B,1),(B,*,*))` 依赖旧版 torch 隐式广播；torch≥2.4 不再允许 | `repro/upstream/theseus_compat.py` 重写 `_merge_infos`（只把 cond reshape 成 `(B,*1…)`） |
| 12 | `test_tune_optimum` 报 `Differential KKT system residual is high` | 差分 KKT 矩阵结构奇异（rank 1 SDP 解使 cond≈1e17），`np.linalg.solve` 数值不稳定 | patch 0003：cond>1e12 时用最小范数最小二乘 |
| 13 | SDPR-IS 在 stereo 问题 outer iter 0/1 报同一 KKT 断言 | SCS eps=1e-9 的对偶乘子/证书与数值证书不一致（LSQR 残差 ≫ 1e-5） | patch 0003：残差超限时按一阶 KKT 重算乘子与证书（=SDPR-CIFT）再解 |
| 14 | `SDPPoseEstimator.forward() got an unexpected keyword argument 'solver_args'` | `tune_stereo_params_sdpr` 传 `solver_args=None`，而 forward 不接受；该路径上游无测试覆盖 | patch 0003：forward 增加 `solver_args` 并合并到 SCS 参数 |

## 环境自检命令

```bash
P=/home/syk/miniconda3/envs/sdpr/bin/python
cd /home/syk/labs/SDPR/code/sdprlayer
PYTHONPATH=$PWD $P -c "import numpy,scipy,torch,diffcp,cvxpylayers,poly_matrix; print('ok')"
PYTHONPATH=$PWD $P -c "import sdprlayers; from sdprlayers import SDPRLayer, SDPPoseEstimator; print('sdprlayers ok')"
# Theseus（含 sksparse stub）
PYTHONPATH=/home/syk/labs/SDPR/repro/upstream/stubs $P -c "import theseus; print(theseus.__version__)"
# pytest（patch 0002 后可直接跑）
PYTHONPATH=$PWD $P -m pytest _test/test_sdpr_poly6.py _test/test_sdpr_poly4.py -q
```
