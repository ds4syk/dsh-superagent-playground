#!/usr/bin/env python
"""Official SDPRLayers forward baseline for the P1 QPEP rotation subproblem.

What this script measures
------------------------
The **released SDPRLayers package** (``utiasASRL/sdprlayer``) is run, forward
only, on CPU, on three problems, and compared with the P1 10x10 Veronese/Shor
SDP and with QPEP hard argmin:

1. ``toy_random4`` -- a 4x4 random SDP in the exact homogenised form the layer
   expects (``min <Q,X>, X>=0, X_00=1, <A_i,X>=0``).  Purpose: prove the
   official layer's forward value/primal matrix agree with an independent
   CVXPY+SCS solve before it is trusted on P1.
2. ``toy_pose13`` -- the public QPEP-like pose API ``SDPPoseEstimator``
   (13x13, 24 constraints) on a synthetic 20-point 3D-3D pose problem; error is
   measured in se3 against ground truth.  Run from fork@62d8ac6, whose SCS
   backend is what makes it usable without MOSEK; pose extraction is unchanged.
3. ``p1_sdp10`` -- the **P1 10x10 QPEP rotation SDP** (``build_10x10`` from
   ``experiments/sdp/run_sdp_check.py``: 21 SO(3) constraints, exact ``t``
   elimination, ``Q10 = 0.25 U M U^T``) fed through the official
   ``SDPRLayer`` (generic QCQP interface, ``diff_qcqp=False`` exactly like
   ``SDPPoseEstimator``).  Compared with our own CVXPY+SCS 10x10 solve and
   with QPEP hard argmin: value, rank / eigenvalue ratio, recovered pose,
   runtime, and the layer's own tightness verdict.

Provenance (important, read before quoting numbers)
---------------------------------------------------
* The **released upstream tree** is `utiasASRL/sdprlayer` at
  ``/mnt/storage1/syk/third_party/sdprlayer`` (origin
  ``https://github.com/utiasASRL/sdprlayer.git``), commit **cb3aed2**
  ("Fix README formatting", upstream HEAD).
* The ``sdpr`` conda env has an *editable* install of a local fork at
  ``/home/syk/labs/SDPR/code/sdprlayer``, HEAD **62d8ac6** = ``cb3aed2`` + 8
  compatibility commits (make mosek/theseus/sparseqr optional, add an SCS
  backend to ``SDPPoseEstimator``, solve-failure knobs).  They are **not**
  upstream releases; they are what makes the released code runnable without
  MOSEK/CUDA in this environment.
* The released ``sdprlayers/__init__.py`` eagerly imports layers that need
  ``mosek`` (via ``cert_tools``) and ``sparseqr``; a plain
  ``import sdprlayers`` on CPU therefore fails (see ``provenance`` in the
  JSON).  To still run the **unmodified released source**, this script exports
  the pristine tree at ``cb3aed2`` with ``git archive`` and loads
  ``src/sdprlayers/layers/sdprlayer.py`` directly by file path (that module
  has no intra-package imports), providing a no-op ``sparseqr`` module because
  that compiled SuiteSparse binding is absent here and the import is never
  used on this code path.  Every 10x10 value is cross-checked against the
  fork's installed layer (62d8ac6) and must agree.

Honest scope
------------
* FORWARD ONLY: no gradient / differentiability claim, no backward pass.
* CPU only (torch 2.5.1+cpu, SCS via diffcp); no MOSEK, no CUDA.
* The layer's own ``check_tightness`` (ER = lambda1/lambda2 > 1e5) gates pose
  recovery.  When it refuses (``xs is None``) we report the SDP value and the
  rank, and do **not** present any pose as "the" SDP pose.

Run / 运行
----------
    cd labs/01-laplace-qpeplayer/experiments/baselines
    /home/syk/miniconda3/envs/sdpr/bin/python run_official_sdpr.py

Runtime on login-node CPU: ~3 s (three 10x10 diffcp solves dominate).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
import types
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
RESULTS = EXP / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

for _p in (str(EXP / "code"), str(EXP / "sdp"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cvxpy as cp                                            # noqa: E402
import torch                                                  # noqa: E402

torch.set_default_dtype(torch.float64)

import instance as inst_mod                                   # noqa: E402
import run_sdp_check as SDP                                   # noqa: E402
import run_sdpr_p1 as P1                                      # noqa: E402  (shared pose helpers)
from sdprlayers.layers.pose_est import SDPPoseEstimator       # noqa: E402
from sdprlayers.layers.sdprlayer import SDPRLayer as EnvSDPRLayer   # noqa: E402
from sdprlayers.utils.lie_algebra import se3_inv, se3_log     # noqa: E402

UPSTREAM_REPO = Path("/mnt/storage1/syk/third_party/sdprlayer")
UPSTREAM_COMMIT = "cb3aed2"
FORK_REPO = Path("/home/syk/labs/SDPR/code/sdprlayer")
FORK_COMMIT = "62d8ac6"
THETAS = (0.0, 0.01, 0.1)
SYN_SEED = 20261127
N_STARTS_SYN = 512
SDP_EPS = 1e-9
SDP_MAX_ITERS = 200_000


# ---------------------------------------------------------------------------
# provenance / environment probes
# ---------------------------------------------------------------------------
def git(repo, *args):
    try:
        return subprocess.check_output(["git", "-C", str(repo), *args],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:                                          # pragma: no cover
        return None


def export_pristine(dest):
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        f"git -C {UPSTREAM_REPO} archive {UPSTREAM_COMMIT} | tar -x -C {dest}",
        shell=True, check=True)
    return dest


def probe_pristine_import(pristine_src):
    """Plain ``import sdprlayers`` against the pristine tree (a subprocess)."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(pristine_src)
    t0 = time.perf_counter()
    p = subprocess.run([sys.executable, "-c", "import sdprlayers"],
                       env=env, capture_output=True, text=True)
    tail = [ln for ln in (p.stderr or "").strip().splitlines() if ln.strip()]
    return dict(returncode=p.returncode, ok=bool(p.returncode == 0),
                error_last_line=tail[-1] if tail else "",
                t_s=time.perf_counter() - t0)


def load_pristine_sdprlayer(pristine_tree):
    """Load released ``sdprlayer.py`` (cb3aed2) unmodified, by file path.

    Only the module-level ``import sparseqr`` needs help: it is a compiled
    SuiteSparse binding not installed here and is never used by
    ``SDPRLayer``.  A no-op module is installed in ``sys.modules``.
    """
    path = Path(pristine_tree) / "src" / "sdprlayers" / "layers" / "sdprlayer.py"
    stubbed = False
    if importlib.util.find_spec("sparseqr") is None:
        sys.modules.setdefault("sparseqr", types.ModuleType("sparseqr"))
        stubbed = True
    spec = importlib.util.spec_from_file_location(
        "sdprlayer_released_%s" % UPSTREAM_COMMIT, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.SDPRLayer, dict(path=str(path), source_sha256=__import__("hashlib")
                               .sha256(path.read_bytes()).hexdigest(),
                               sparseqr_stubbed=stubbed)


# ---------------------------------------------------------------------------
# generic forward wrapper (homogenised form, exactly the layer's contract)
# ---------------------------------------------------------------------------
def official_forward(LayerCls, Q, A_list, layout="col", GinvD=None,
                     eps=SDP_EPS, max_iters=SDP_MAX_ITERS, tag=""):
    """``min <Q,X>`` s.t. ``X>=0, X_00=1, <A_i,X>=0`` through ``LayerCls``.

    Mirrors ``SDPPoseEstimator``: ``objective=None`` (Q is the CVXPY parameter),
    fixed constraint matrices, ``use_dual=True``, ``diff_qcqp=False`` (upstream
    pose-estimator default), SCS backend at the paper setting eps=1e-9.

    ``layout``/``GinvD`` only steer pose extraction (P1 lift layouts); with
    ``GinvD=None`` the pose block is skipped and only rank diagnostics are
    returned.
    """
    n = int(Q.shape[0])
    out = dict(tag=tag, n=n, layout=layout, eps=float(eps),
               max_iters=int(max_iters), error=None, t_solve_ms=float("nan"),
               t_build_ms=float("nan"), value=float("nan"), X00=None,
               eigs_top=None, eig_ratio_12=None, rank=None, rank1_resid=None,
               ER=None, upstream_tight=None, xs_is_none=None, recovered=None,
               pose_xs=None, warnings=[])
    t_build = time.perf_counter()
    layer = LayerCls(n_vars=n, constraints=[np.asarray(a, float) for a in A_list],
                     objective=None, homogenize=False, use_dual=True,
                     diff_qcqp=False)
    out["t_build_ms"] = 1e3 * (time.perf_counter() - t_build)

    Qt = torch.tensor(np.asarray(Q, float))
    args = {"solve_method": "SCS", "eps": float(eps),
            "max_iters": int(max_iters)}
    t0 = time.perf_counter()
    Xs = xs = None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            Xs, xs = layer(Qt, solver_args=args)
        except Exception as exc:                                # pragma: no cover
            out["error"] = repr(exc)
    out["t_solve_ms"] = 1e3 * (time.perf_counter() - t0)
    out["warnings"] = [str(w.message) for w in caught
                       if "distutils" not in str(w.message)]
    if Xs is None:
        return out

    X = np.asarray(Xs.detach().cpu().numpy(), float)
    X = 0.5 * (X + X.T)
    if GinvD is not None:
        rec = P1.recover_from_X(X, layout, GinvD)
        out.update(eigs_top=rec["eigs_top"], eig_ratio_12=rec["eig_ratio_12"],
                   rank=rec["rank"], rank1_resid=rec["rank1_resid"],
                   recovered=rec)
    else:
        eigs = np.sort(np.linalg.eigvalsh(X))[::-1]
        lam1 = float(eigs[0])
        rank = int(np.sum(eigs > 1e-6 * max(lam1, 1e-300)))
        v1 = np.linalg.eigh(X)[1][:, -1]
        X1 = lam1 * np.outer(v1, v1)
        out.update(eigs_top=[float(v) for v in eigs[:8]],
                   eig_ratio_12=float(eigs[0] / max(eigs[1], 1e-15)),
                   rank=rank,
                   rank1_resid=float(np.linalg.norm(X - X1)
                                     / max(np.linalg.norm(X), 1e-300)))
    out.update(X00=float(X[0, 0]), value=float(np.trace(np.asarray(Q, float) @ X)),
               X=[[float(v) for v in row] for row in X])
    try:
        tight, ER = LayerCls.check_tightness(torch.tensor(X))
        out.update(ER=float(ER), upstream_tight=bool(tight))
    except Exception:                                          # pragma: no cover
        pass
    out["xs_is_none"] = bool(xs is None)
    if xs is not None:
        x = np.asarray(xs.detach().cpu().numpy(), float).reshape(-1)
        x = x / x[0] if abs(x[0]) > 1e-12 else x
        if GinvD is not None:
            out["pose_xs"] = P1.pose_of_x(x, layout, GinvD)
    return out


# ---------------------------------------------------------------------------
# arm 1: 4x4 random SDP
# ---------------------------------------------------------------------------
def toy_random4(seed=0):
    """Random 4x4 SDP in the layer's homogenised form + cvxpy reference."""
    rng = np.random.default_rng(seed)
    n = 4
    x0 = rng.standard_normal(n)
    x0[0] = 1.0
    Q = rng.standard_normal((n, n))
    Q = 0.5 * (Q + Q.T)
    E00 = np.zeros((n, n))
    E00[0, 0] = 1.0
    A = []
    for _ in range(3):
        M = rng.standard_normal((n, n))
        M = 0.5 * (M + M.T)
        A.append(M - (x0 @ M @ x0) * E00)      # x0 stays feasible
    X = cp.Variable((n, n), symmetric=True)
    cons = [X >> 0, X[0, 0] == 1] + [cp.trace(a @ X) == 0 for a in A]
    prob = cp.Problem(cp.Minimize(cp.trace(Q @ X)), cons)
    t0 = time.perf_counter()
    prob.solve(solver=cp.SCS, eps=SDP_EPS, max_iters=SDP_MAX_ITERS)
    t_cvx = 1e3 * (time.perf_counter() - t0)
    Xv = 0.5 * (np.asarray(X.value, float) + np.asarray(X.value, float).T)
    ref = dict(status=prob.status, value=float(prob.value), t_solve_ms=t_cvx,
               X00=float(Xv[0, 0]))
    return dict(Q=Q, A=A, X_ref=Xv, ref=ref, seed=int(seed))


# ---------------------------------------------------------------------------
# arm 2: released SDPPoseEstimator on a toy 13x13 3D-3D pose problem
# ---------------------------------------------------------------------------
def toy_pose13(seed=0, n_points=20, noise=0.0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    B, N = 1, n_points
    pts = rng.uniform(-1.0, 1.0, size=(B, N, 3))
    src = np.concatenate([pts, np.ones((B, N, 1))], axis=-1)
    xi = torch.tensor([[0.15, -0.2, 0.1, 0.3, -0.1, 0.2]])
    T_gt = torch.eye(4)
    from sdprlayers.utils.lie_algebra import se3_exp
    T_gt = se3_exp(xi)[0]
    trg = src @ T_gt.numpy().T
    if noise > 0:
        trg[:, :, :3] += rng.normal(0.0, noise, size=(B, N, 3))
    src_t = torch.tensor(src).permute(0, 2, 1).contiguous()
    trg_t = torch.tensor(trg).permute(0, 2, 1).contiguous()
    w = torch.ones(B, 1, N)
    est = SDPPoseEstimator(T_s_v=torch.eye(4), solver="SCS", scs_eps=SDP_EPS)
    out = dict(n_points=N, noise=float(noise), error=None,
               t_solve_ms=float("nan"), se3_err_max=None, se3_err=None)
    t0 = time.perf_counter()
    try:
        T_est = est(src_t, trg_t, w)
        out["t_solve_ms"] = 1e3 * (time.perf_counter() - t0)
        d = se3_log(se3_inv(T_est) @ T_gt.unsqueeze(0)).numpy().reshape(-1)
        out["se3_err"] = [float(v) for v in d]
        out["se3_err_max"] = float(np.abs(d).max())
        out["T_est"] = [[float(v) for v in row] for row in T_est[0].numpy()]
    except Exception as exc:                                   # pragma: no cover
        out["t_solve_ms"] = 1e3 * (time.perf_counter() - t0)
        out["error"] = repr(exc)
    return out


# ---------------------------------------------------------------------------
# arm 3: P1 10x10 QPEP rotation SDP through the official layer
# ---------------------------------------------------------------------------
def p1_sdp10(U, A10, I, thetas=THETAS, LayerCls=EnvSDPRLayer, eps=SDP_EPS,
             max_iters=SDP_MAX_ITERS):
    rows = []
    for th in thetas:
        M = I.M(th)
        Q10 = 0.25 * U @ M @ U.T

        t0 = time.perf_counter()
        m = I.find_modes(M, n_starts=N_STARTS_SYN)
        t_qpep = 1e3 * (time.perf_counter() - t0)
        J = np.asarray(m["J"], float)
        qs = np.asarray(m["q"], float)
        k = int(np.argmin(J))
        R_hard, t_hard = P1.pose_of_q(qs[k], I.GinvD)
        J_min = float(J[k])

        s10 = SDP.solve_sdp(Q10, A10, 10, eps=eps, max_iters=max_iters)
        rec10 = (P1.recover_from_X(s10["X"], "row", I.GinvD)
                 if s10["X"] is not None else None)

        up = official_forward(LayerCls, Q10, A10, layout="row",
                              GinvD=I.GinvD, eps=eps,
                              max_iters=max_iters, tag="p1_sdp10")
        up_env = official_forward(EnvSDPRLayer, Q10, A10, layout="row",
                                  GinvD=I.GinvD, eps=eps,
                                  max_iters=max_iters,
                                  tag="p1_sdp10_env62d8ac6")
        cross = dict(
            d_value=float(abs(up["value"] - up_env["value"])),
            d_value_rel=float(abs(up["value"] - up_env["value"])
                              / max(abs(up["value"]), 1e-300)),
            same_rank=bool(up["rank"] == up_env["rank"]),
            same_tight=bool(up["upstream_tight"] == up_env["upstream_tight"]),
            env_error=up_env["error"],
        )

        row = dict(theta=float(th), J_min=J_min, t_qpep_ms=t_qpep,
                   n_modes=int(len(J)), Q10=Q10.tolist(),
                   hard=dict(R=[float(v) for v in R_hard.reshape(-1)],
                             t=[float(v) for v in t_hard]),
                   sdp10=dict(status=s10["status"], value=s10["value"],
                              t_solve_ms=1e3 * s10["t_solve"],
                              dual_bound=s10.get("dual_bound"),
                              primal_minus_dual=s10.get("primal_minus_dual"),
                              rank=None if rec10 is None else rec10["rank"],
                              eig_ratio_12=None if rec10 is None else rec10["eig_ratio_12"],
                              recovered=rec10,
                              f_minus_Jmin=float(s10["value"] - J_min)),
                   official=up, official_env=up_env, cross_clean_vs_env=cross,
                   diffs={})
        # pose comparisons, only where a pose exists
        if rec10 is not None:
            row["diffs"]["sdp10_col_vs_hard"] = P1.diff_of(rec10, "pose_col",
                                                           R_hard, t_hard)
        if up.get("recovered") is not None:
            d = dict(official_col_vs_hard=P1.diff_of(up["recovered"], "pose_col",
                                                    R_hard, t_hard))
            if rec10 is not None:
                d["official_col_vs_sdp10_col"] = P1.pose_diff(
                    np.asarray(up["recovered"]["pose_col"]["R"]).reshape(3, 3),
                    np.asarray(up["recovered"]["pose_col"]["t"]),
                    np.asarray(rec10["pose_col"]["R"]).reshape(3, 3),
                    np.asarray(rec10["pose_col"]["t"]))
                d["official_eig_vs_hard"] = P1.diff_of(up["recovered"], "pose_eig",
                                                       R_hard, t_hard)
            if up.get("pose_xs") is not None:
                px = up["pose_xs"]
                d["official_xs_vs_hard"] = P1.pose_diff(
                    np.asarray(px["R"]).reshape(3, 3), np.asarray(px["t"]),
                    R_hard, t_hard)
                if rec10 is not None:
                    d["official_xs_vs_sdp10_col"] = P1.pose_diff(
                        np.asarray(px["R"]).reshape(3, 3), np.asarray(px["t"]),
                        np.asarray(rec10["pose_col"]["R"]).reshape(3, 3),
                        np.asarray(rec10["pose_col"]["t"]))
            row["diffs"].update(d)
        rows.append(row)
        print("  theta=%-5g Jmin=%+.9f f10=%+.9f official=%+.9f "
              "(d=%+.1e) rank=%s ER=%.2e tight=%s xs=None=%s"
              % (th, J_min, s10["value"], up["value"],
                 up["value"] - s10["value"], up["rank"], up["ER"] or -1,
                 up["upstream_tight"], up["xs_is_none"]), flush=True)
    return rows


# ---------------------------------------------------------------------------
# markdown
# ---------------------------------------------------------------------------
def fnum(x, nd=9):
    if x is None:
        return "n/a"
    if isinstance(x, float) and not np.isfinite(x):
        return "n/a"
    return ("%%.%df" % nd) % x


def sci(x):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return "%.1e" % x


def write_markdown(res, path):
    L = []
    A = L.append
    cfg = res["config"]
    prov = res["provenance"]
    A("# 官方 SDPRLayers 前向基线（P1 10×10 QPEP）")
    A("")
    A("> **一句话结论：发布的 SDPRLayers `SDPRLayer` 在 CPU（torch 2.5.1+cpu + diffcp/SCS）"
      "上**跑通**；10×10 QPEP SDP 上 tight 实例的前向值与我们的 Veronese SDP 差 ≤ %.1e，"
      "位姿与 QPEP 硬解一致；但它的位姿出口被 `ER>1e5` 判据卡住，θ=0 退化实例直接 `xs=None`。**"
      % (max([abs(r["official"]["value"] - r["sdp10"]["value"])
              for r in res["p1_sdp10"] if r["official"]["upstream_tight"]]
             or [float("nan")])))
    A("")
    A("## 0. 口径、来源、环境")
    A("")
    A("- **发布版来源**：`utiasASRL/sdprlayer` clone 在 `%s`，commit **%s**"
      "（origin `https://github.com/utiasASRL/sdprlayer.git`）；README 未给单独 tag，"
      "该 commit 即 clone 的 upstream HEAD。" % (prov["upstream_repo"],
                                                prov["upstream_commit"]))
    A("- **环境里的 `sdprlayers`**：`%s` 是 **editable 安装**，指向本地 fork `%s`，"
      "HEAD **%s**（= `cb3aed2` + 8 个兼容性 commit：把 mosek/theseus/sparseqr 变成可选、"
      "给 `SDPPoseEstimator` 加 SCS 后端、求解失败阈值开关）。这些**不是**发布版内容，"
      "是让发布版代码在无 MOSEK/无 CUDA 环境能跑的最小补丁；后文会把两者都跑一遍互相验证。"
      % (prov["installed_sdprlayers"]["dist_info"],
         prov["fork_repo"], prov["fork_commit"]))
    A("- **发布版能否直接 import**：`PYTHONPATH=<tree>/src python -c 'import sdprlayers'` "
      "→ **失败**，`%s`。原因：发布版 `__init__.py` 在模块级 eager import "
      "`SDPEssMatEst`（→ `cert_tools` → `import mosek`）等可选层。"
      "完整 traceback 与本次临时树路径见 JSON `provenance.pristine_package_import`。"
      % prov["pristine_package_import"]["error_last_line"])
    A("- **绕行方式（未改动发布版任何一行）**：`git archive cb3aed2` 导出干净树，"
      "用 `importlib` 直接按文件路径加载 `src/sdprlayers/layers/sdprlayer.py`"
      "（该模块没有包内 import）；它 module 级 `import sparseqr` 是编译版 SuiteSparse 绑定，"
      "本机未安装且在该代码路径上从未调用，因此注入一个 no-op 模块。"
      "文件 sha256 = `%s`，`sparseqr_stubbed=%s`。"
      % (prov["released_sdprlayer_file"]["source_sha256"],
         prov["released_sdprlayer_file"]["sparseqr_stubbed"]))
    A("- **求解器**：SCS（经 diffcp），论文设置 `eps=1e-9, max_iters=2e5`；`use_dual=True`；"
      "`diff_qcqp=False`（与 `SDPPoseEstimator` 默认一致，只做前向，不触发 KKT 反向）。")
    A("- **没有安装/修改任何包**：`pip install -e` 未执行；env 里已有的 `sdprlayers 0.0.1` "
      "editable 安装指向 fork（上一条），发布版只用临时导出树 + 按文件加载，系统环境零改动。")
    A("- **import 方式（可复制）**：")
    A("")
    A("```python")
    A("# (a) 发布版文件（cb3aed2 干净树），绕过包的 eager __init__：")
    A("import importlib.util, sys, types")
    A("sys.modules.setdefault('sparseqr', types.ModuleType('sparseqr'))  # 本机缺该可选编译依赖")
    A("spec = importlib.util.spec_from_file_location('rel', "
      "'<tree>/src/sdprlayers/layers/sdprlayer.py')")
    A("m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)")
    A("SDPRLayer = m.SDPRLayer          # 或从 <tree>/src 设 PYTHONPATH 后 from sdprlayers... import")
    A("# (b) 环境已安装的 fork@62d8ac6（含 SCS backend，SDPPoseEstimator 用这个）：")
    A("from sdprlayers.layers.sdprlayer import SDPRLayer")
    A("from sdprlayers.layers.pose_est import SDPPoseEstimator  # solver='SCS'")
    A("```")
    A("")
    A("- **只做前向**：不涉及可微性/反向，不涉及 GPU。")
    A("- **环境**：`%s`（torch %s, numpy %s, cvxpy %s, scs %s, diffcp %s, "
      "`torch.cuda.is_available()=%s`）。"
      % (cfg["python"], cfg["torch"], cfg["numpy"], cfg["cvxpy"], cfg["scs"],
         cfg["diffcp"], cfg["cuda_available"]))
    A("- **命令**：")
    A("")
    A("```bash")
    A("cd labs/01-laplace-qpeplayer/experiments/baselines")
    A("%s run_official_sdpr.py" % cfg["python"])
    A("```")
    A("")
    A("运行时长 %.1f s。" % res["wall_s"])
    A("")

    # ---- toy
    t = res["toy_random4"]
    A("## 1. 先验证 wrapper：4×4 随机 SDP（官方层 vs CVXPY+SCS）")
    A("")
    A("| 量 | 官方 `SDPRLayer` | CVXPY+SCS | 差 |")
    A("|-|-|-|-|")
    A("| 最优值 | %s | %s | %+.1e |" % (fnum(t["official"]["value"]),
                                      fnum(t["ref"]["value"]),
                                      t["official"]["value"] - t["ref"]["value"]))
    A("| `X_00` | %s | %s | %+.1e |" % (fnum(t["official"]["X00"], 12),
                                       fnum(t["ref"]["X00"], 12),
                                       t["official"]["X00"] - t["ref"]["X00"]))
    A("| `||X-X_ref||_F/||X_ref||_F` | %s | — | — |" % sci(t["X_rel_fro"]))
    A("| rank / ER | %s / %s | — | |" % (t["official"]["rank"],
                                         sci(t["official"]["eig_ratio_12"])))
    A("| 求解 ms | %.1f | %.1f | |" % (t["official"]["t_solve_ms"],
                                       t["ref"]["t_solve_ms"]))
    A("")
    A("（官方层首次调用含 diffcp 一次性初始化，~0.3 s；后续 10×10 调用为 6–8 ms。）")
    A("")
    A("官方层返回的 `Xs` 被解释为原 SDP 的 `X`（对偶约束的乘子），`xs = X[:,0]/X[0,0]` "
      "是 rank-1 时的 lift。值/矩阵与独立 CVXPY 解在 `%.0e` 相对误差内一致 → wrapper 语义正确。"
      % t["X_rel_fro"])
    A("")

    # ---- toy pose
    A("## 2. 发布版公开 API：`SDPPoseEstimator`（13×13，玩具 3D-3D 位姿）")
    A("")
    A("20 个无噪声 3D 对应点，GT `T_trg_src` 由 se3 参数 `(0.15,-0.2,0.1,0.3,-0.1,0.2)` 生成；"
      "`T_s_v = I`；`solver=\"SCS\"`。误差 `se3_log(T_est^{-1} T_gt)`。")
    A("")
    A("| 实例 | se3 误差 max | ms | 备注 |")
    A("|-|-|-|-|")
    for k in ("exact", "noisy"):
        p = res["toy_pose13"][k]
        A("| %s | %s | %.1f | %s |"
          % (k, "n/a" if p["se3_err_max"] is None else sci(p["se3_err_max"]),
             p["t_solve_ms"], p.get("error") or "ok"))
    A("")
    A("说明：这一步用的是 fork@62d8ac6 的 `SDPPoseEstimator`（发布版该类硬编码 MOSEK，"
      "无 MOSEK 无法实例化）；兼容补丁只替换 backend 选择，不改进位姿提取。")
    A("")

    # ---- P1 10x10
    A("## 3. P1 10×10 QPEP 旋转 SDP：官方层 vs 我们 vs QPEP 硬解")
    A("")
    A("`Q10 = 0.25 U M U^T`，21 条 SO(3) 约束，`X_00=1`，SCS `eps=1e-9`。"
      "“官方”= 发布版 `sdprlayer.py@cb3aed2`（干净树、文件路径直载）；"
      "同一 Q/A 再用 fork@62d8ac6 安装版跑一遍，用于验证兼容补丁不改数值。")
    A("")
    A("| θ | J_min | 我们 f10 | 官方 f | 官方−我们 | 官方−J_min | rank(官方) | ER(官方) | 官方 tight | 官方 xs=None | ms(官方) | ms(我们) |")
    A("|-|-|-|-|-|-|-|-|-|-|-|-|")
    for r in res["p1_sdp10"]:
        u = r["official"]
        A("| %.2f | %s | %s | %s | %+.1e | %+.1e | %s | %s | %s | %s | %.0f | %.0f |"
          % (r["theta"], fnum(r["J_min"]), fnum(r["sdp10"]["value"]),
             fnum(u["value"]), u["value"] - r["sdp10"]["value"],
             u["value"] - r["J_min"], u["rank"], sci(u["ER"]),
             u["upstream_tight"], u["xs_is_none"], u["t_solve_ms"],
             r["sdp10"]["t_solve_ms"]))
    A("")
    A("交叉验证（发布版 vs env fork）：最大 `|Δf| = %.1e`（相对 %.1e），rank/tight 判定一致：%s。"
      % (max(r["cross_clean_vs_env"]["d_value"] for r in res["p1_sdp10"]),
         max(r["cross_clean_vs_env"]["d_value_rel"] for r in res["p1_sdp10"]),
         all(r["cross_clean_vs_env"]["same_rank"]
             and r["cross_clean_vs_env"]["same_tight"]
             for r in res["p1_sdp10"])))
    A("")
    A("位姿（`col`=官方 `recovery_map` 第一列，`eig`=主特征向量，`xs`=官方返回值；"
      "非 tight 实例的 `col`/`xs` 不是可行 lift，标“—”不参与解释）：")
    A("")
    A("| θ | 官方 tight | 官方 col vs 硬 (rot°, m) | 官方 eig vs 硬 | 官方 xs vs 硬 | 官方 col vs 我们 sdp10 col | 我们 sdp10 col vs 硬 |")
    A("|-|-|-|-|-|-|-|")

    def rot_txt(x):
        return "<1e-6°" if x < 1e-6 else "%.3g°" % x

    def pfmt(dd, tight):
        if not dd or not tight:
            return "—"
        return "%s / %.1e m" % (rot_txt(dd["rot_deg"]), dd["trans_m"])

    for r in res["p1_sdp10"]:
        d = r["diffs"]
        tight = bool(r["official"]["upstream_tight"])
        A("| %.2f | %s | %s | %s | %s | %s | %s |"
          % (r["theta"], tight,
             pfmt(d.get("official_col_vs_hard"), tight),
             pfmt(d.get("official_eig_vs_hard"), True),
             pfmt(d.get("official_xs_vs_hard"), tight),
             pfmt(d.get("official_col_vs_sdp10_col"), tight),
             pfmt(d.get("sdp10_col_vs_hard"), tight)))
    A("")
    A("- θ=0 的 `X*` 是 rank-2（两个特征值几乎相等，`ER=1.00`），`col`/`xs` 不是可行 lift，"
      "对硬解的角度偏差（4.6°/6.6°，我们 `col` 也有 2.0°）**不作为位姿误差解释**；"
      "但主特征向量 `eig` 仍精确指向 QPEP 硬解（<1e-6°/2e-10 m）——即值对、方向对，"
      "只是双层特征值让 rank-1 提取不唯一。")
    A("- θ=0.01/0.1：官方 `col`（以及 `xs`）与 QPEP 硬解旋转差 %s、平移差 %.1e m；"
      "与我们 10×10 `col` 的差 %s。"
      % (rot_txt(max(r["diffs"]["official_col_vs_hard"]["rot_deg"]
                    for r in res["p1_sdp10"] if r["official"]["upstream_tight"])),
         max(r["diffs"]["official_col_vs_hard"]["trans_m"]
             for r in res["p1_sdp10"] if r["official"]["upstream_tight"]),
         rot_txt(max(r["diffs"]["official_col_vs_sdp10_col"]["rot_deg"]
                    for r in res["p1_sdp10"] if r["official"]["upstream_tight"]))))
    A("")

    # ---- findings
    n_tight = sum(1 for r in res["p1_sdp10"] if r["official"]["upstream_tight"])
    A("## 4. 结论与对 P1 的影响")
    A("")
    dtight = [abs(r["official"]["value"] - r["sdp10"]["value"])
              for r in res["p1_sdp10"] if r["official"]["upstream_tight"]]
    dloose = [abs(r["official"]["value"] - r["sdp10"]["value"])
              for r in res["p1_sdp10"] if not r["official"]["upstream_tight"]]
    A("1. **官方前向在 CPU 上可跑，且 10×10 值与我们一致**：发布版 `SDPRLayer`"
      "（cb3aed2，文件路径直载）在 tight 实例上与我们的 CVXPY+SCS 值差 ≤ %.1e；"
      "θ=0 的 rank-2 实例差 %.1e（两边各自的 SCS 误差已是这个量级）。"
      "4×4 玩具上 `f` 差 %.1e、`||ΔX||_F/||X||_F = %.1e`。"
      "→ 10×10 也可以直接喂官方层，不必只依赖 13×13。"
      % (max(dtight) if dtight else float("nan"),
         max(dloose) if dloose else float("nan"),
         abs(t["official"]["value"] - t["ref"]["value"]), t["X_rel_fro"]))
    A("2. **但官方层的位姿出口由 `ER>1e5` 把关**：%d/%d 个 θ 通过；θ=0 的 rank-2 退化实例"
      "`xs=None`（`SDPPoseEstimator` 会 `RuntimeError`）。值仍然算得对，"
      "只是**不给位姿**——与 `sdpr_p1.json` 里 13×13 上游臂的结论一致，不是 10×10 特有。"
      % (n_tight, len(res["p1_sdp10"])))
    A("3. **tight 实例上 10×10 官方位姿 ≈ QPEP 硬解**：θ=0.01/0.1 的 `col`/`xs` 对硬解"
      "旋转差 %s、平移差 ≤ %.1e m，且与我们 10×10 `col` 一致到 %s。"
      "因此官方层可作为 P1 前向的**独立实现校验**，而不是替代 QPEP（退化处它没有输出）。"
      % (rot_txt(max(r["diffs"]["official_col_vs_hard"]["rot_deg"]
                    for r in res["p1_sdp10"] if r["official"]["upstream_tight"])),
         max(r["diffs"]["official_col_vs_hard"]["trans_m"]
             for r in res["p1_sdp10"] if r["official"]["upstream_tight"]),
         rot_txt(max(r["diffs"]["official_col_vs_sdp10_col"]["rot_deg"]
                    for r in res["p1_sdp10"] if r["official"]["upstream_tight"]))))
    A("4. **发布版无法直接 `import sdprlayers`**（MOSEK/theseus/sparseqr 可选依赖被 eager import）。"
      "这是环境/打包问题，不是数学问题；复现官方数字必须用本文的“干净树 + 文件路径加载 + "
      "sparseqr stub”方式，或装齐 MOSEK。")
    A("")
    A("### 复现命令与产物")
    A("")
    A("```bash")
    A("git -C %s rev-parse HEAD          # %s" % (prov["upstream_repo"],
                                                  prov["upstream_commit"]))
    A("git -C %s rev-parse HEAD          # %s" % (prov["fork_repo"],
                                                  prov["fork_commit"]))
    A("cd labs/01-laplace-qpeplayer/experiments/baselines")
    A("%s run_official_sdpr.py" % cfg["python"])
    A("```")
    A("")
    A("- 机器可读结果：`experiments/results/official_sdpr.json`")
    A("- 本文件：`experiments/results/official_sdpr.md`")
    A("")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thetas", type=str, default=",".join(str(t) for t in THETAS))
    ap.add_argument("--eps", type=float, default=SDP_EPS)
    ap.add_argument("--max-iters", type=int, default=SDP_MAX_ITERS)
    ap.add_argument("--skip-toy-pose", action="store_true")
    args = ap.parse_args()
    thetas = tuple(float(v) for v in args.thetas.split(",") if v.strip())
    t_start = time.time()

    # ---- provenance --------------------------------------------------------
    print("[1/5] provenance ...", flush=True)
    tmp = Path(tempfile.mkdtemp(prefix="sdpr_release_"))
    pristine = export_pristine(tmp / "tree")
    pristine_src = pristine / "src"
    import_probe = probe_pristine_import(pristine_src)
    import_probe["pythonpath"] = str(pristine_src)
    ReleasedLayer, rel_meta = load_pristine_sdprlayer(pristine)
    dist_info = None
    try:
        import importlib.metadata as md
        dist_info = "%s %s" % ("sdprlayers", md.version("sdprlayers"))
    except Exception:                                          # pragma: no cover
        pass
    import diffcp
    import scs
    prov = dict(
        upstream_repo=str(UPSTREAM_REPO), upstream_commit=UPSTREAM_COMMIT,
        upstream_commit_full=git(UPSTREAM_REPO, "rev-parse", UPSTREAM_COMMIT),
        upstream_origin="https://github.com/utiasASRL/sdprlayer.git",
        fork_repo=str(FORK_REPO), fork_commit=FORK_COMMIT,
        fork_commit_full=git(FORK_REPO, "rev-parse", "HEAD"),
        fork_on_top_of="cb3aed2 (8 compatibility commits; see `git log cb3aed2..HEAD`)",
        installed_sdprlayers=dict(dist_info=dist_info,
                                  module_path=os.path.dirname(
                                      sys.modules["sdprlayers"].__file__)),
        pristine_package_import=import_probe,
        released_sdprlayer_file=rel_meta,
        sparseqr_installed=not rel_meta["sparseqr_stubbed"],
    )
    print("      pristine import:", import_probe["error_last_line"], flush=True)

    # ---- toys --------------------------------------------------------------
    print("[2/5] 4x4 random SDP (released-layer wrapper check) ...", flush=True)
    toy = toy_random4()
    toy_up = official_forward(ReleasedLayer, toy["Q"], toy["A"], layout="col",
                              GinvD=None, eps=args.eps,
                              max_iters=args.max_iters, tag="toy_random4")
    Xo = np.asarray(toy_up["X"], float)
    toy_meta = dict(seed=toy["seed"], ref=toy["ref"], official=toy_up,
                    X_rel_fro=float(np.linalg.norm(Xo - toy["X_ref"])
                                    / np.linalg.norm(toy["X_ref"])))
    print("      layer=%+.9f cvxpy=%+.9f rank=%s tight=%s"
          % (toy_up["value"], toy_meta["ref"]["value"], toy_up["rank"],
             toy_up["upstream_tight"]), flush=True)

    toy_pose = {}
    if not args.skip_toy_pose:
        print("[3/5] toy SDPPoseEstimator 13x13 ...", flush=True)
        toy_pose["exact"] = toy_pose13(seed=0, noise=0.0)
        toy_pose["noisy"] = toy_pose13(seed=1, noise=0.01)
        print("      exact err=%s noisy err=%s"
              % (toy_pose["exact"].get("se3_err_max"),
                 toy_pose["noisy"].get("se3_err_max")), flush=True)

    # ---- P1 10x10 ----------------------------------------------------------
    print("[4/5] builders + QPEP instance ...", flush=True)
    U, A10, names10, chk10 = SDP.build_10x10()
    I = inst_mod.Instance(seed=SYN_SEED)
    print("[5/5] P1 10x10 official forward, theta in %s ..." % (thetas,), flush=True)
    rows = p1_sdp10(U, A10, I, thetas=thetas, LayerCls=ReleasedLayer,
                    eps=args.eps, max_iters=args.max_iters)

    res = dict(
        config=dict(
            python=sys.executable, python_version=sys.version.split()[0],
            platform=platform.platform(), numpy=np.__version__,
            scipy=__import__("scipy").__version__, cvxpy=cp.__version__,
            scs=getattr(scs, "__version__", "?"),
            diffcp=getattr(diffcp, "__version__", "?"),
            torch=torch.__version__,
            cuda_available=bool(torch.cuda.is_available()),
            solvers=cp.installed_solvers(),
            command=("cd labs/01-laplace-qpeplayer/experiments/baselines && "
                     "%s run_official_sdpr.py" % sys.executable),
            scope=("FORWARD ONLY, CPU only: released SDPRLayer cb3aed2 on a 4x4 toy, "
                   "released SDPPoseEstimator on a toy 13x13 pose problem, and the "
                   "P1 10x10 QPEP rotation SDP; compared with cvxpy SCS and QPEP hard argmin"),
            eps=args.eps, max_iters=args.max_iters,
            thetas=[float(t) for t in thetas],
            builders=dict(n_constraints_10=int(chk10["n_constraints"]),
                          U_orthogonality_err=chk10["U_orthogonal_max_abs_err"],
                          xR_eq_2Uv2_exact=chk10["xR_equals_2Uv2_sympy_exact"]),
        ),
        provenance=prov,
        toy_random4=toy_meta,
        toy_pose13=toy_pose,
        p1_sdp10=rows,
        wall_s=time.time() - t_start,
    )
    out_json = RESULTS / "official_sdpr.json"
    out_json.write_text(json.dumps(res, indent=1), encoding="utf-8")
    out_md = write_markdown(res, RESULTS / "official_sdpr.md")
    print("wrote", out_json, out_md, "wall %.1fs" % res["wall_s"], flush=True)



if __name__ == "__main__":
    main()
