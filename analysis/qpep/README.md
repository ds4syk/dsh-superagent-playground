# analysis/qpep：QPEP 两种模式与可微性实验

对应 `kb/qpep-layer/` 与 `deliverables/QPEP_可微优化层_2026-11.docx`。两组独立实验：

## 1. 结构实验（点到面 QPEP，纯 Python）

`qpep_ptop.py`：

- 点到面配准（论文 Eq. 9）的 $L(q)$、$\nabla L$、$\nabla^2L$ 解析式（已用复步长校验到 1e-16）；
- full KKT 系统 $(∇L-λq=0,\ \|q\|^2=1)$ 的全部实根（向量化 LM 多起点枚举）；
- 四个 3-of-4 约化系统（$\lambda=0$）的全部实根；
- full KKT 与分支系统的隐式微分。

运行（需要 numpy/scipy；本机用 `~/miniconda3/envs/sdpr-clean/bin/python`）：

```bash
python run_ptop.py        # -> results/ptop_results.json（根、λ、分支 recall、重叠）
python run_grads.py       # -> results/ptop_grads.json（full vs branch vs FD 导数）
python make_figures.py    # -> figures/*.png + docs/visuals/2026-11-11-qpep-diff-layer.png
```

## 2. SDP 规模标定

```bash
# sdpr env（含 cvxpy/scs）
python run_sdp_scaling.py # -> results/sdp_scaling.json
```

随机 Shor 型 SDP（PSD 目标、可行点构造右端项），SCS `eps=1e-9`，量级对照用。

## 3. LibQPEP 实测（C++）

`libqpep_bench/`：

- `qpep_pnp_bench.cpp`：用 LibQPEP 官方 `data/pnp_data-*.txt`，跑 full（596 模板）/ 近似（249 模板，行求和与四排列），记录分段计时与位姿误差；输出 CSV。
- `bench_lu.cpp`：596³ 与 249³ 模板的纯 LU+RHS+eig 代价（Eigen `PartialPivLU`）。
- `pnp_data-*-1_approx.csv`：30 trials × 4 个点数的结果。

复现步骤：

```bash
git clone --depth 1 https://github.com/zarathustr/LibQPEP.git /tmp/libqpep
# 本机无 LAPACK/OpenCV/CMake：用最小 CMakeLists 只编核心（去 csdp/CovEstimation/OpenCV demo），
# 依赖 Eigen 3.4 + OpenBLAS，-O3（不要 -ffast-math，否则 596 模板段错误）。
# 然后把 qpep_pnp_bench.cpp 放进 /tmp/libqpep，编译链接 libLibQPEPCore.so：
#   g++ -O3 -std=c++11 -Iinclude -Ifsolve_wrapper -I<eigen> qpep_pnp_bench.cpp \
#       -Lbuild -lLibQPEPCore -o build/qpep_pnp_bench
# 运行（工作目录在 build/）：
#   LD_LIBRARY_PATH=.:<conda>/lib ./qpep_pnp_bench --trials 30 --data ../data/pnp_data-50pt-1.txt --noise 1.0 --seed 1
```

注意：`pnp_WQD` 只写 $W$ 的稀疏项，调用前必须 `setZero()`（上游 demo 没有，会得到 NaN）。
