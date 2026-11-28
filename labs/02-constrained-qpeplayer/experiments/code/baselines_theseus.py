#!/usr/bin/env python
"""B2: Theseus 库软惩罚基线（官方库，LM/GN + quadratic/Huber penalty）。

目的
----
论文 §5.7 之前写的是「Theseus 未运行（缺 CHOLMOD）」。本轮把 `import theseus`
修好（隔离的 scikit-sparse/CHOLMOD 覆盖层，见 results/round_dqp_final.md B1），
然后用**官方 Theseus 库**在 `run_main.py` 的同一批合成实例上跑软惩罚基线：

    min_{R,t}  J(R,t) + λ Σ_j h(g_j(R,t)),   g_j = A_j v2(q) + B_j t <= 0

- `quad` : h(g) = [g]_+²（二次惩罚；Theseus 官方 API 里没有硬约束）
- `huber`: h(g) = 0.5 g² (g<=δ), δ(g-0.5δ) (g>δ)，δ=1e-3（与 g 的物理单位一致：
  pen/depth/fov 行是米，tilt 行是无量纲二次型）

实现要点
--------
- 姿态用 `theseus.SE3`（tensor 为 3×4 [R|t]），目标函数用
  `theseus.AutoDiffCostFunction`（Theseus 自己 autograd 出 Jacobian，
  内部把 ∂e/∂tensor 投影到 SE3 切空间；我们用 `pose.retract` 的有限差分
  验证过一致性，见 B1 记录）。
- bearing 残差：J = Σ_i (R p_i + t)^T (I - u_i u_i^T)(R p_i + t) = Σ_i ||L_i^T(R p_i + t)||²，
  L_i 是 P_i 的两个单位特征向量（3×2）。Theseus 的 cost = ½Σ||e||²，
  因此 e_i = √2 L_i^T(R p_i + t) 时 cost 恰为 J。
- 约束残差：e_j = √(2λ) [g_j]_+（quad）；Huber 版 e_j = √(2λ h(g_j))。
  于是 ½||e||² = λ h(g_j)，与上式一致。
- v2(q) 直接由旋转矩阵 R 精确表出（q_i² 与 q_i q_j 都是 R 的线性函数），
  不需要从 R 反解四元数，避免了 sqrt 的符号/奇异问题。
- 同一实例、同一热启动（`qpep_core.solve_qp_global` 的 48 起点无约束 QPEP 解）、
  同一指标与证书口径（`certificate.shor_bound(enforce_ineq=True)`）。

运行（CHOLMOD 覆盖层；不修改 sdpr 环境本身）
--------------------------------------------
    THSP=/mnt/storage1/syk/miniconda3/envs/thsparse
    PYTHONPATH=/mnt/storage1/syk/theseus_deps/pylib \
    LD_LIBRARY_PATH=$THSP/lib \
    nice -n 10 /home/syk/miniconda3/envs/sdpr/bin/python baselines_theseus.py

环境变量：QPE_TH_P（默认 "8,16,32"）、QPE_TH_TRIALS（30）、
QPE_TH_LAMS（"1e2,1e3,1e4,1e5"）、QPE_TH_KINDS（"quad,huber"）、
QPE_TH_MAX_ITER（200）、QPE_TH_HUBER_DELTA（1e-3）。
"""
import json
import os
import sys
import time
import traceback
import warnings

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qpep_core as C
from certificate import shor_bound

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
os.makedirs(RES, exist_ok=True)

import torch  # noqa: E402
import theseus as th  # noqa: E402

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

SQRT2 = float(np.sqrt(2.0))

P_LIST = [int(v) for v in os.environ.get("QPE_TH_P", "8,16,32").split(",") if v]
N_TRIALS = int(os.environ.get("QPE_TH_TRIALS", "30"))
LAMS = [float(v) for v in os.environ.get("QPE_TH_LAMS", "1,10,1e2,1e3,1e4,1e5").split(",") if v]
KINDS = [v for v in os.environ.get("QPE_TH_KINDS", "quad,huber").split(",") if v]
MAX_ITER = int(os.environ.get("QPE_TH_MAX_ITER", "200"))
HUBER_DELTA = float(os.environ.get("QPE_TH_HUBER_DELTA", "1e-3"))
NOISE, BIAS_Z = 0.01, 0.08      # 与 run_main.py 主表一致
CERT_RULE = ("feasible(viol<=1e-6) & J - Shor <= 1e-6 & SDP tight(lam2/lam1<=1e-6)")


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------------------
# 1. 把实例转成 Theseus 图（常量预先转好）
# ---------------------------------------------------------------------------

def _bearing_basis(U):
    """P_i = I - u u^T (u 单位向量) -> L_i (3x2)，满足 ||L_i^T x||^2 = x^T P_i x。"""
    Ls = []
    for u in U:
        u = np.asarray(u, float)
        u = u / np.linalg.norm(u)
        a = np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        e1 = np.cross(u, a)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(u, e1)
        Ls.append(np.stack([e1, e2], axis=1))
    return np.array(Ls)  # (N,3,2)


def _v2_from_R_tensor(R):
    """R: (B,3,3) torch -> v2(q): (B,10)，精确且对 R 可微。"""
    R00, R01, R02 = R[:, 0, 0], R[:, 0, 1], R[:, 0, 2]
    R10, R11, R12 = R[:, 1, 0], R[:, 1, 1], R[:, 1, 2]
    R20, R21, R22 = R[:, 2, 0], R[:, 2, 1], R[:, 2, 2]
    q0sq = 0.25 * (1.0 + R00 + R11 + R22)
    q1sq = 0.25 * (1.0 + R00 - R11 - R22)
    q2sq = 0.25 * (1.0 - R00 + R11 - R22)
    q3sq = 0.25 * (1.0 - R00 - R11 + R22)
    m01 = 0.25 * (R21 - R12)
    m02 = 0.25 * (R02 - R20)
    m03 = 0.25 * (R10 - R01)
    m12 = 0.25 * (R01 + R10)
    m13 = 0.25 * (R02 + R20)
    m23 = 0.25 * (R12 + R21)
    return torch.stack([q0sq, q1sq, q2sq, q3sq,
                        SQRT2 * m01, SQRT2 * m02, SQRT2 * m03,
                        SQRT2 * m12, SQRT2 * m13, SQRT2 * m23], dim=1)


def _pose_tensor(q, t):
    """q(w,x,y,z), t(3) -> (1,3,4) [R|t]。"""
    R = C.R_of(np.asarray(q, float))
    T = np.concatenate([R, np.asarray(t, float)[:, None]], axis=1)
    return T[None]


class _Problem:
    """一个实例 + 一个 (λ,kind) 的 Theseus 目标。"""

    def __init__(self, ins, lam, kind):
        self.ins = ins
        self.lam = float(lam)
        self.kind = kind
        A = torch.tensor(np.asarray(ins["A"], float))
        B = torch.tensor(np.asarray(ins["B"], float))
        P = torch.tensor(np.asarray(ins["P"], float))[None]          # (1,N,3)
        L = torch.tensor(_bearing_basis(ins["U"]))                  # (N,3,2)
        self._A, self._B, self._P = A, B, P
        self._L = L
        self._n_pts = P.shape[1]
        self._p = A.shape[0]
        delta = torch.full((1, self._p), HUBER_DELTA)

        def bearing_err(optim_vars, aux_vars):
            se3 = optim_vars[0]
            R = se3.tensor[:, :, :3]
            t = se3.tensor[:, :, 3]
            x = torch.einsum("bij,bnj->bni", R, P) + t[:, None, :]   # (1,N,3)
            r = torch.einsum("nij,bnj->bni", L.transpose(1, 2), x)  # (1,N,2)
            return SQRT2 * r.reshape(1, -1)

        def pen_err(optim_vars, aux_vars):
            se3 = optim_vars[0]
            R = se3.tensor[:, :, :3]
            t = se3.tensor[:, :, 3]
            v = _v2_from_R_tensor(R)                                 # (1,10)
            g = v @ A.T + t @ B.T                                    # (1,p)
            gp = torch.clamp(g, min=0.0)
            if kind == "quad":
                return np.sqrt(2.0 * self.lam) * gp
            if kind == "huber":
                h = torch.where(gp <= HUBER_DELTA,
                                0.5 * gp * gp,
                                HUBER_DELTA * (gp - 0.5 * HUBER_DELTA))
                return torch.sqrt(2.0 * self.lam * h + 1e-24)
            raise ValueError(kind)

        T0 = np.concatenate([np.asarray(ins["R_gt"], float),
                             np.asarray(ins["t_gt"], float)[:, None]], axis=1)[None]
        self.pose = th.SE3(tensor=torch.tensor(T0), disable_checks=True)
        self.bearing_cost = th.AutoDiffCostFunction(
            [self.pose], bearing_err, 2 * self._n_pts,
            aux_vars=[], autograd_mode="loop_batch")
        self.pen_cost = th.AutoDiffCostFunction(
            [self.pose], pen_err, self._p,
            aux_vars=[], autograd_mode="loop_batch")
        self.obj = th.Objective()
        self.obj.add(self.bearing_cost)
        self.obj.add(self.pen_cost)

    def set_pose(self, q, t):
        self.pose.update(torch.tensor(_pose_tensor(q, t)))
        self.obj.update()   # reconcile batch size after pose/tensor update

    def get_pose(self):
        T = self.pose.tensor.detach().numpy()[0]
        R, t = T[:, :3], T[:, 3]
        q = C.q_from_R(R)
        return q, t

    def cost_parts(self):
        """返回 (bearing_cost, penalty_cost, total)；与 C.obj_value + λΣ[g]_+² 对齐。"""
        with torch.no_grad():
            eb = self.bearing_cost.error()
            ep = self.pen_cost.error()
            cb = float(0.5 * (eb ** 2).sum())
            cp = float(0.5 * (ep ** 2).sum())
        return cb, cp, cb + cp


def _new_pose(prob, q, t):
    prob.set_pose(q, t)
    # 每次求解前重建 optimizer，避免 Theseus 的 objective 版本/状态复用问题
    opt = th.LevenbergMarquardt(
        prob.obj, th.CholeskyDenseSolver,
        max_iterations=MAX_ITER,
        abs_err_tolerance=1e-12, rel_err_tolerance=1e-12,
        track_err_history=True)
    return opt


def theseus_solve(prob, q0, t0):
    prob.set_pose(q0, t0)
    opt = _new_pose(prob, q0, t0)
    t_start = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # adaptive_damping=True 是 Theseus LM 自带的可选线搜索式步长控制：
        # 默认 adaptive_damping=False 时 LM 不做步长拒绝，在本问题（J 的 GN
        # Hessian 量级 ~1e-4、惩罚项量级 ~λ）上会大幅过冲并在两点间振荡。
        info = opt.optimize(track_err_history=True, adaptive_damping=True)
    dt = time.perf_counter() - t_start
    q, t = prob.get_pose()
    st = np.asarray(info.status).ravel()
    status_name = getattr(st[0], "name", str(st[0])) if st.size else "?"
    conv = int(np.asarray(info.converged_iter, dtype=float).max()) if info.converged_iter is not None else -1
    hist = (np.asarray(info.err_history.detach())[0]
            if info.err_history is not None else None)
    n_hist = int(np.isfinite(hist).sum()) if hist is not None else 0
    iters = conv if conv > 0 else n_hist
    le = info.last_err.detach() if hasattr(info.last_err, "detach") else info.last_err
    le = np.asarray(le, dtype=float)
    last_err = float(le.max()) if le.size else float("nan")
    return dict(q=q, t=t, time_ms=1e3 * dt, iters=iters, n_hist=n_hist,
                status_name=status_name, last_err=last_err)


# ---------------------------------------------------------------------------
# 2. 指标 / trial
# ---------------------------------------------------------------------------

def metrics(ins, q, t, J=None):
    R = C.R_of(q)
    g = ins["A"] @ C.v2(q) + ins["B"] @ t
    return dict(
        J=float(C.obj_value(q, t, ins["M0"], ins["D0"], ins["G0"]) if J is None else J),
        viol=float(max(0.0, g.max())),
        feasible=bool(max(0.0, g.max()) <= 1e-6),
        pen_mm=C.penetration_mm(R, t, ins["V_dense"]),
        rot_deg=C.rot_deg(q, ins["q_gt"]),
        trans_mm=C.trans_mm(t, ins["t_gt"]),
        depth_min_mm=float(1000.0 * ((R @ ins["P"].T).T[:, 2] + np.asarray(t)[2]).min()),
        q=[float(x) for x in q], t=[float(x) for x in t],
    )


def _pen_h(gp, kind):
    """与 _Problem.pen_err 对应的逐行惩罚 h(g)，用于核对总目标值。"""
    gp = np.asarray(gp, float)
    if kind == "quad":
        return gp ** 2
    if kind == "huber":
        return np.where(gp <= HUBER_DELTA, 0.5 * gp ** 2,
                        HUBER_DELTA * (gp - 0.5 * HUBER_DELTA))
    raise ValueError(kind)


def run_trial(ins, lams=LAMS, kinds=KINDS):
    rec = dict(seed=ins["seed"], p=ins["p"], noise=ins["noise"], bias_z=ins["bias_z"],
               actypes=ins["actypes"])
    M0, D0, G0, A, B = ins["M0"], ins["D0"], ins["G0"], ins["A"], ins["B"]
    seed = ins["seed"]

    # 与 run_main.py 完全相同的热启动：无约束 QPEP（48 起点）
    t0 = time.perf_counter()
    Ju, qu, tu = C.solve_qp_global(M0, D0, G0, n_starts=48,
                                   rng=np.random.default_rng(1000 + seed))
    dt_u = time.perf_counter() - t0
    rec["unc"] = metrics(ins, qu, tu, J=Ju)
    rec["unc"]["time_ms"] = 1e3 * dt_u

    # ALM 参照（与 run_main.py 同配置；用于 sound 检查与配对比较）
    t0 = time.perf_counter()
    ok = C.alm_qpep(M0, D0, G0, A, B, qu, tu, rho0=10.0, n_starts=6,
                    rng=np.random.default_rng(4000 + seed))
    dt_a = time.perf_counter() - t0
    rec["alm"] = metrics(ins, ok["q"], ok["t"], J=ok["J"])
    rec["alm"].update(time_ms=1e3 * dt_a, iters=ok["iters"], ok=ok["ok"])

    # 证书（实例级，与方法/λ 无关）
    t0 = time.perf_counter()
    sd = shor_bound(M0, D0, G0, A, B, enforce_ineq=True)
    dt_sdp = time.perf_counter() - t0
    rec["sdp"] = dict(value=sd["value"], ratio=sd["ratio"], lam1=sd["lam1"],
                      status=sd["status"], time_ms=1e3 * dt_sdp,
                      tight=bool(sd["ratio"] <= 1e-6),
                      sound_vs_alm=bool(sd["value"] <= rec["alm"]["J"] + 1e-7))

    for name in ("unc", "alm"):
        m = rec[name]
        m["cert_gap"] = float(m["J"] - sd["value"])
        m["certified"] = bool(m["feasible"] and (m["J"] - sd["value"] <= 1e-6)
                              and rec["sdp"]["tight"])

    rec["runs"] = {}
    for kind in kinds:
        for lam in lams:
            key = "%s_%g" % (kind, lam)
            try:
                prob = _Problem(ins, lam, kind)
                out = theseus_solve(prob, qu, tu)
                cb, cp, ct = prob.cost_parts()
                m = metrics(ins, out["q"], out["t"])
                m.update(time_ms=out["time_ms"], iters=out["iters"],
                         n_hist=out["n_hist"], status=out["status_name"],
                         last_err=out["last_err"],
                         lam=lam, kind=kind,
                         cost_bearing=cb, cost_pen=cp, cost_total=ct,
                         cost_expected=float(m["J"] + lam *
                                             np.sum(_pen_h(np.maximum(
                                                 A @ C.v2(out["q"]) + B @ out["t"], 0.0), kind))))
                m["cert_gap"] = float(m["J"] - sd["value"])
                m["certified"] = bool(m["feasible"] and (m["J"] - sd["value"] <= 1e-6)
                                      and rec["sdp"]["tight"])
                rec["runs"][key] = m
            except Exception:
                rec["runs"][key] = dict(error=traceback.format_exc(), lam=lam, kind=kind)
    return rec


# ---------------------------------------------------------------------------
# 3. 汇总 / markdown
# ---------------------------------------------------------------------------

def med_iqr(xs):
    xs = np.asarray([x for x in xs if np.isfinite(x)], float)
    if xs.size == 0:
        return dict(median=float("nan"), q1=float("nan"), q3=float("nan"), n=0)
    q1, q50, q3 = np.percentile(xs, [25, 50, 75])
    return dict(median=float(q50), q1=float(q1), q3=float(q3), n=int(xs.size))


def summarize(recs):
    out = dict(n=len([r for r in recs if "error" not in r]), runs={}, unc={})
    unc = [r["unc"] for r in recs if "error" not in r]
    if unc:
        out["unc"] = dict(
            pen_mm=med_iqr([m["pen_mm"] for m in unc]),
            pen_rate=float(np.mean([m["pen_mm"] > 1.0 for m in unc])),
            rot_deg=med_iqr([m["rot_deg"] for m in unc]),
            trans_mm=med_iqr([m["trans_mm"] for m in unc]),
            feasible=float(np.mean([m["feasible"] for m in unc])),
            certified=float(np.mean([m.get("certified", False) for m in unc])),
            time_ms=med_iqr([m["time_ms"] for m in unc]))
    alm = [r["alm"] for r in recs if "error" not in r and "alm" in r]
    if alm:
        out["alm"] = dict(
            pen_mm=med_iqr([m["pen_mm"] for m in alm]),
            rot_deg=med_iqr([m["rot_deg"] for m in alm]),
            trans_mm=med_iqr([m["trans_mm"] for m in alm]),
            feasible=float(np.mean([m["feasible"] for m in alm])),
            certified=float(np.mean([m.get("certified", False) for m in alm])),
            time_ms=med_iqr([m["time_ms"] for m in alm]))
    keys = sorted({k for r in recs if "error" not in r for k in r.get("runs", {})},
                  key=lambda s: (s.split("_")[0], float(s.split("_")[1])))
    for key in keys:
        ms = [r["runs"][key] for r in recs if "error" not in r and key in r.get("runs", {})
              and "error" not in r["runs"][key]]
        if not ms:
            out["runs"][key] = dict(n=0)
            continue
        out["runs"][key] = dict(
            n=len(ms),
            lam=ms[0]["lam"], kind=ms[0]["kind"],
            pen_mm=med_iqr([m["pen_mm"] for m in ms]),
            pen_mean=float(np.mean([m["pen_mm"] for m in ms])),
            pen_rate=float(np.mean([m["pen_mm"] > 1.0 for m in ms])),
            rot_deg=med_iqr([m["rot_deg"] for m in ms]),
            trans_mm=med_iqr([m["trans_mm"] for m in ms]),
            viol_max=float(np.max([m["viol"] for m in ms])),
            viol_med=float(np.median([m["viol"] for m in ms])),
            feasible=float(np.mean([m["feasible"] for m in ms])),
            certified=float(np.mean([m["certified"] for m in ms])),
            J_med=float(np.median([m["J"] for m in ms])),
            cert_gap_med=float(np.median([m["cert_gap"] for m in ms])),
            time_ms=med_iqr([m["time_ms"] for m in ms]),
            iters=med_iqr([float(m["iters"]) for m in ms]),
            cost_abs_gap_max=float(np.max([abs(m["cost_total"] - m["cost_expected"])
                                           for m in ms])),
            n_error=len([r for r in recs if key in r.get("runs", {})
                         and "error" in r["runs"][key]]),
        )
    sdp = [r["sdp"] for r in recs if "error" not in r and "sdp" in r]
    if sdp:
        out["sdp"] = dict(tight=float(np.mean([s["tight"] for s in sdp])),
                          sound=float(np.mean([s["sound_vs_alm"] for s in sdp])),
                          time_ms=med_iqr([s["time_ms"] for s in sdp]),
                          ratio_max=float(np.max([s["ratio"] for s in sdp])))
    return out


def run_cell(p, n_trials, tag):
    recs = []
    t_cell = time.perf_counter()
    for trial in range(n_trials):
        try:
            ins = C.make_instance(trial, p, noise=NOISE, bias_z=BIAS_Z)
            recs.append(run_trial(ins))
        except Exception:
            log("  !! trial %d FAILED %s\n%s" % (trial, tag, traceback.format_exc()))
            recs.append(dict(seed=trial, p=p, error=traceback.format_exc()))
        if (trial + 1) % 10 == 0:
            good = [r for r in recs if "error" not in r]
            log("  [%s] %d/%d (%.1fs)" % (tag, trial + 1, n_trials, time.perf_counter() - t_cell))
    summ = summarize(recs)
    log("  ---- %s summary (n=%d) ----" % (tag, summ["n"]))
    u = summ.get("unc", {})
    if u:
        log("   unc   pen=%7.2f mm rot=%5.2f trans=%6.2f feas=%3.0f%% time=%6.1f ms"
            % (u["pen_mm"]["median"], u["rot_deg"]["median"], u["trans_mm"]["median"],
               100 * u["feasible"], u["time_ms"]["median"]))
    for key, ss in summ["runs"].items():
        if ss.get("n", 0) == 0:
            continue
        log("   %-10s pen=%7.3f mm feas=%3.0f%% cert=%3.0f%% viol_med=%.2e iters=%4.0f time=%6.1f ms costgap=%.1e"
            % (key, ss["pen_mm"]["median"], 100 * ss["feasible"], 100 * ss["certified"],
               ss["viol_med"], ss["iters"]["median"], ss["time_ms"]["median"],
               ss["cost_abs_gap_max"]))
    return dict(tag=tag, p=p, n=summ["n"], summary=summ, recs=recs)


def fmt_iqr(m):
    return "%.3f [%.3f, %.3f]" % (m["median"], m["q1"], m["q3"])


def write_md(raw):
    cfg = raw["config"]
    L = []
    A = L.append
    A("# B2 baseline：Theseus 官方库的软惩罚层（F17–F22）")
    A("")
    A("> 由 `experiments/code/baselines_theseus.py` 自动生成；**不要手改**。逐 trial 原始数据在 "
      "`baseline_theseus.json`。")
    A("")
    A("库：`theseus-ai %s`（sdpr 环境自带）+ `scikit-sparse %s`（隔离覆盖层，未改动 sdpr 本身）。"
      % (raw["config"].get("theseus_version", "?"), raw["config"].get("sksparse_version", "?")))
    A("求解器：Theseus `LevenbergMarquardt` + `CholeskyDenseSolver`（单姿态 6 维，稠密足够；"
      "CHOLMOD 已可用但本问题不需要稀疏路径）。")
    A("")
    A("层定义：`min J(R,t) + λ Σ_j h(g_j)`，`g_j = A_j v2(q)+B_j t <= 0`；"
      "`quad`: h=[g]_+²；`huber`: δ=%g。Σ 为约束行。" % cfg["huber_delta"])
    A("热启动与主表/ALM/SCP/dQP 相同（无约束 QPEP 48 起点）；λ∈%s；p∈%s；n=%d trials/p。"
      % (cfg["lams"], cfg["p"], cfg["n_trials"]))
    A("证书与主表同口径（`certificate.shor_bound(enforce_neq=True)`，"
      "certified = 可行且 J−Shor≤1e-6 且 SDP tight）。")
    A("")
    A("## 表 1. 穿透-精度 Pareto（每一格：pen mm 中位 [IQR] / rot deg 中位 / trans mm 中位 / 可行率 / 认证率 / 耗时 ms 中位）")
    A("")
    heads = []
    for key in raw["config"]["cells"]:
        pass
    cells = cfg["cells"]
    # 组装宽表：每个 p 一块
    for p in cfg["p"]:
        c = cells.get("p%d" % p)
        if not c:
            continue
        A("### p = %d（n=%d）" % (p, c["n"]))
        A("")
        A("| method | pen mm (median [IQR]) | rot deg | trans mm | feasible | cert | ms | iters |")
        A("|---|---|---|---|---|---|---|---|")
        u = c["summary"].get("unc")
        if u:
            A("| Unconstrained QPEP | %s | %.2f | %.2f | %.0f%% | %.0f%% | %.1f [%.1f, %.1f] | — |"
              % (fmt_iqr(u["pen_mm"]), u["rot_deg"]["median"], u["trans_mm"]["median"],
                 100 * u["feasible"], 100 * u["certified"], u["time_ms"]["median"],
                 u["time_ms"]["q1"], u["time_ms"]["q3"]))
        a = c["summary"].get("alm")
        if a:
            A("| ALM-QPEP (reference) | %s | %.2f | %.2f | %.0f%% | %.0f%% | %.1f [%.1f, %.1f] | — |"
              % (fmt_iqr(a["pen_mm"]), a["rot_deg"]["median"], a["trans_mm"]["median"],
                 100 * a["feasible"], 100 * a["certified"], a["time_ms"]["median"],
                 a["time_ms"]["q1"], a["time_ms"]["q3"]))
        for key, ss in c["summary"]["runs"].items():
            if ss.get("n", 0) == 0:
                A("| %s | —— | —— | —— | —— | —— | —— | —— |" % key)
                continue
            A("| Theseus %s | %s | %.2f | %.2f | %.0f%% | %.0f%% | %.1f [%.1f, %.1f] | %.0f |"
              % (key, fmt_iqr(ss["pen_mm"]), ss["rot_deg"]["median"], ss["trans_mm"]["median"],
                 100 * ss["feasible"], 100 * ss["certified"], ss["time_ms"]["median"],
                 ss["time_ms"]["q1"], ss["time_ms"]["q3"], ss["iters"]["median"]))
        A("")
        A("最小穿透/最大穿透（λ 维度的两端）与可行率：")
        A("")
        A("| kind | λ | pen mm median | pen rate (>1mm) | viol median | feasible | cert |")
        A("|---|---|---|---|---|---|---|")
        for key, ss in c["summary"]["runs"].items():
            if ss.get("n", 0) == 0:
                continue
            A("| %s | %g | %.3f | %.0f%% | %.2e | %.0f%% | %.0f%% |"
              % (ss["kind"], ss["lam"], ss["pen_mm"]["median"], 100 * ss["pen_rate"],
                 ss["viol_med"], 100 * ss["feasible"], 100 * ss["certified"]))
        A("")
        sdp = c["summary"].get("sdp", {})
        if sdp:
            A("证书 SDP：tight %.0f%%、sound %.0f%%、cost %.1f ms、max λ2/λ1 %.1e。"
              % (100 * sdp["tight"], 100 * sdp["sound"], sdp["time_ms"]["median"], sdp["ratio_max"]))
            A("")
    A("## 表 2. 收敛/迭代/成本一致性")
    A("")
    A("`cost gap` = |Theseus 目标值 − (J+λΣ[g]_+²)| 的逐 trial 最大值；应 <1e-6（只验证装配与权重口径）。")
    A("")
    A("| p | kind | λ | iters median [IQR] | cost gap max |")
    A("|---|---|---|---|---|")
    for p in cfg["p"]:
        c = cells.get("p%d" % p)
        if not c:
            continue
        for key, ss in c["summary"]["runs"].items():
            if ss.get("n", 0) == 0:
                continue
            A("| %d | %s | %g | %.0f [%.0f, %.0f] | %.1e |"
              % (p, ss["kind"], ss["lam"], ss["iters"]["median"], ss["iters"]["q1"],
                 ss["iters"]["q3"], ss["cost_abs_gap_max"]))
    A("")
    A("## 结论（数字全部来自本 JSON）")
    A("")
    # 动态写几条
    for p in cfg["p"]:
        c = cells.get("p%d" % p)
        if not c:
            continue
        u = c["summary"].get("unc", {})
        runs = c["summary"]["runs"]
        q = [r for k, r in runs.items() if r.get("kind") == "quad"]
        h = [r for k, r in runs.items() if r.get("kind") == "huber"]
        def line(rs, tag):
            if not rs:
                return ""
            mn = min(rs, key=lambda r: r["pen_mm"]["median"])
            mx = max(rs, key=lambda r: r["pen_mm"]["median"])
            return ("%s: λ=%g 时 pen=%.3f mm（最小），λ=%g 时 pen=%.3f mm（最大）；"
                    "可行率 %.0f%%→%.0f%%" %
                    (tag, mn["lam"], mn["pen_mm"]["median"], mx["lam"],
                     mx["pen_mm"]["median"], 100 * mx["feasible"], 100 * mn["feasible"]))
        A("- **p=%d**：Unconstrained pen %.2f mm、feasible %.0f%%。%s；%s。"
          % (p, u.get("pen_mm", {}).get("median", float("nan")),
             100 * u.get("feasible", float("nan")), line(q, "quad"), line(h, "Huber")))
    A("")
    A("## Caveats")
    A("")
    A("- 这是**软惩罚**基线：任何有限 λ 都只把穿透压到 O(λ^{-1})，不是硬约束；"
      "certified 要求 viol≤1e-6，因此 λ 扫描的多数格子认证率为 0——这正是与本文硬约束层的差异。")
    A("- Theseus 只做前向（LM）；本表不涉及隐式微分/反向。官方库的 `TheseusLayer` 反向未测。")
    A("- 单热启动（无约束 QPEP 解）；我们自己的 soft baseline 用 14 起点 BFGS。"
      "两者初始化相同（同一个 qu,tu），但 restart 数不同，比较时请注意。")
    A("")
    with open(os.path.join(RES, "baseline_theseus.md"), "w") as f:
        f.write("\n".join(L) + "\n")


def main():
    t_start = time.perf_counter()
    log("=" * 100)
    log("Theseus soft-penalty baseline | p=%s n=%d lams=%s kinds=%s max_iter=%d"
        % (P_LIST, N_TRIALS, LAMS, KINDS, MAX_ITER))
    import sksparse
    log("theseus=%s | sksparse=%s | torch=%s" % (getattr(th, "__version__", "?"),
                                                 getattr(sksparse, "__version__", "?"),
                                                 torch.__version__))
    raw = dict(config=dict(
        p=P_LIST, n_trials=N_TRIALS, lams=LAMS, kinds=KINDS, max_iter=MAX_ITER,
        huber_delta=HUBER_DELTA, noise=NOISE, bias_z=BIAS_Z,
        theseus_version=getattr(th, "__version__", "unknown"),
        sksparse_version=getattr(sksparse, "__version__", "unknown"),
        torch_version=torch.__version__,
        solver="LevenbergMarquardt + CholeskyDenseSolver",
        warm_start="unconstrained QPEP (48 starts), identical to run_main/SCP/ALM",
        cost_convention="Theseus objective = 0.5*sum(e^2); bearing e=sqrt(2)L^T(Rp+t); pen e=sqrt(2*lam*h(g))",
        cert_rule=CERT_RULE, cells={}))
    for p in P_LIST:
        log("\n--- Theseus p=%d (noise=%.3f bias=%.3f, n=%d) ---" % (p, NOISE, BIAS_Z, N_TRIALS))
        raw["config"]["cells"]["p%d" % p] = run_cell(p, N_TRIALS, "p%d" % p)
        with open(os.path.join(RES, "baseline_theseus.json"), "w") as f:
            json.dump(raw, f, indent=1, default=float)
        log("  [saved] baseline_theseus.json (%.1fs)" % (time.perf_counter() - t_start))
    raw["elapsed_s"] = time.perf_counter() - t_start
    with open(os.path.join(RES, "baseline_theseus.json"), "w") as f:
        json.dump(raw, f, indent=1, default=float)
    write_md(raw)
    log("\n完成：%.1fs" % raw["elapsed_s"])


if __name__ == "__main__":
    main()
