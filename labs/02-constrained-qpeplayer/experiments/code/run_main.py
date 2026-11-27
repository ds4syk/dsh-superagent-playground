#!/usr/bin/env python
"""Constrained-QPEPLayer 主实验：合成桌面 PnP 的 method × p / method × 噪声。

方法（同一实例、同一目标、同一约束）：
  unc   无约束 QPEP（多起点数值全局替代，48 起点）
  soft  软二次惩罚（λ=1e3 主表；λ∈{1,10,100,1e3,1e4} 另跑 Pareto）
  proj  事后投影（无约束解 → 最近可行点，SLSQP；失败回退只投影 t）
  alm   ALM-QPEP（冻结活跃集 + 定理 2.1 的 10×10/3×10/3×3 装配 + 一致性检查）
指标：穿透深度/穿透率、rot/trans、约束违反/可行率、runtime、14×14 Shor tight/ certified。

每组 50 trials；seed = trial，所以不同 p / 不同噪声档共享同一批场景（配对比较）。
所有原始数据落 results/raw_main.json / raw_pareto.json；下游 make_results.py 只读 JSON。

运行：srun --partition=cpu --time=00:30:00 --ntasks=1 \
      /home/syk/miniconda3/envs/sdpr/bin/python run_main.py
"""
import json
import os
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qpep_core as C
from certificate import shor_bound

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
os.makedirs(RES, exist_ok=True)

N_TRIALS = int(os.environ.get("QPE_TRIALS", "50"))
LAM_MAIN = 1e3
LAM_PARETO = [1.0, 10.0, 100.0, 1e3, 1e4]

MAIN_P = [int(v) for v in os.environ.get("QPE_MAIN_P", "8,16,32,64").split(",") if v]
NOISE_CELLS = [
    dict(tag="n000_b000", noise=0.0, bias_z=0.0),
    dict(tag="n005_b040", noise=0.005, bias_z=0.04),
    dict(tag="n010_b080", noise=0.010, bias_z=0.08),
    dict(tag="n020_b120", noise=0.020, bias_z=0.12),
]
if os.environ.get("QPE_NOISE", "1") == "0":
    NOISE_CELLS = []
DO_PARETO = os.environ.get("QPE_PARETO", "1") == "1"


def log(*a):
    print(*a, flush=True)


def metrics(ins, q, t, A, B, M0, D0, G0, J=None):
    R = C.R_of(q)
    g = A @ C.v2(q) + B @ t
    return dict(
        J=float(C.obj_value(q, t, M0, D0, G0) if J is None else J),
        viol=float(max(0.0, g.max())),
        feasible=bool(max(0.0, g.max()) <= 1e-6),
        pen_mm=C.penetration_mm(R, t, ins["V_dense"]),
        rot_deg=C.rot_deg(q, ins["q_gt"]),
        trans_mm=C.trans_mm(t, ins["t_gt"]),
        q=[float(x) for x in q], t=[float(x) for x in t],
        depth_min_mm=float(1000.0 * ((R @ ins["P"].T).T[:, 2] + np.asarray(t)[2]).min()),
    )


def run_trial(ins, want_sdp=True):
    """跑一个实例的四种方法 + 证书；返回 dict（含每个方法/证书的记录）。"""
    rec = dict(seed=ins["seed"], p=ins["p"], noise=ins["noise"], bias_z=ins["bias_z"],
               actypes=ins["actypes"])
    M0, D0, G0, A, B = ins["M0"], ins["D0"], ins["G0"], ins["A"], ins["B"]

    # --- 无约束 QPEP ---
    t0 = time.perf_counter()
    Ju, qu, tu = C.solve_qp_global(M0, D0, G0, n_starts=48, rng=np.random.default_rng(1000 + ins["seed"]))
    dt_u = time.perf_counter() - t0
    rec["unc"] = metrics(ins, qu, tu, A, B, M0, D0, G0, J=Ju)
    rec["unc"]["time_ms"] = 1e3 * dt_u

    # --- 软惩罚 ---
    t0 = time.perf_counter()
    sp = C.soft_penalty(M0, D0, G0, A, B, LAM_MAIN, qu, tu, n_starts=14,
                        rng=np.random.default_rng(2000 + ins["seed"]))
    dt_s = time.perf_counter() - t0
    rec["soft"] = metrics(ins, sp["q"], sp["t"], A, B, M0, D0, G0, J=sp["J"])
    rec["soft"]["time_ms"] = 1e3 * dt_s
    rec["soft"]["lam"] = LAM_MAIN

    # --- 事后投影 ---
    t0 = time.perf_counter()
    pr = C.project_feasible(M0, D0, G0, A, B, qu, tu, n_starts=4,
                            rng=np.random.default_rng(3000 + ins["seed"]))
    dt_p = time.perf_counter() - t0
    rec["proj"] = metrics(ins, pr["q"], pr["t"], A, B, M0, D0, G0)
    rec["proj"]["time_ms"] = 1e3 * dt_p
    rec["proj"]["variant"] = pr.get("method")

    # --- ALM-QPEP ---
    t0 = time.perf_counter()
    ok = C.alm_qpep(M0, D0, G0, A, B, qu, tu, rho0=10.0, n_starts=6,
                    rng=np.random.default_rng(4000 + ins["seed"]))
    dt_a = time.perf_counter() - t0
    rec["alm"] = metrics(ins, ok["q"], ok["t"], A, B, M0, D0, G0, J=ok["J"])
    rec["alm"]["time_ms"] = 1e3 * dt_a
    rec["alm"].update(iters=ok["iters"], rho=ok["rho"], fallback=ok["fallback_count"],
                      mu_max=float(ok["mu"].max()), mu_min=float(ok["mu"].min()),
                      n_act=int(np.sum(ok["mu"] > 1e-9)), ok=ok["ok"])
    # 活跃集 Jacobian 秩（LICQ 诊断）
    act = np.where(ok["mu"] > 1e-9)[0]
    if len(act):
        rank, _ = C.active_jacobian_rank(ok["q"], ok["t"], A, B, act)
    else:
        rank = 0
    rec["alm"]["act_rank"] = int(rank)

    # --- 证书（与实例有关，与方法是哪一个无关）---
    if want_sdp:
        t0 = time.perf_counter()
        sd = shor_bound(M0, D0, G0, A, B, enforce_ineq=True)
        dt_sdp = time.perf_counter() - t0
        rec["sdp"] = dict(value=sd["value"], ratio=sd["ratio"], lam1=sd["lam1"],
                          status=sd["status"], time_ms=1e3 * dt_sdp,
                          tight=bool(sd["ratio"] <= 1e-6),
                          sound=bool(sd["value"] <= ok["J"] + 1e-7))
        # 每个方法的 certified = 可行 且 (J - Shor) <= 1e-6
        for name in ("unc", "soft", "proj", "alm"):
            m = rec[name]
            m["cert_gap"] = float(m["J"] - sd["value"])
            m["certified"] = bool(m["feasible"] and (m["J"] - sd["value"] <= 1e-6)
                                  and rec["sdp"]["tight"])
    return rec


def summarize(cell_recs):
    out = {}
    methods = ("unc", "soft", "proj", "alm")
    for name in methods:
        ms = [r[name] for r in cell_recs]
        out[name] = dict(
            pen_mean=float(np.mean([m["pen_mm"] for m in ms])),
            pen_std=float(np.std([m["pen_mm"] for m in ms])),
            pen_rate=float(np.mean([m["pen_mm"] > 1.0 for m in ms])),
            rot_mean=float(np.mean([m["rot_deg"] for m in ms])),
            rot_std=float(np.std([m["rot_deg"] for m in ms])),
            trans_mean=float(np.mean([m["trans_mm"] for m in ms])),
            trans_std=float(np.std([m["trans_mm"] for m in ms])),
            viol_max=float(np.max([m["viol"] for m in ms])),
            feasible=float(np.mean([m["feasible"] for m in ms])),
            J_mean=float(np.mean([m["J"] for m in ms])),
            time_ms_mean=float(np.mean([m["time_ms"] for m in ms])),
            time_ms_med=float(np.median([m["time_ms"] for m in ms])),
            certified=float(np.mean([m.get("certified", False) for m in ms])),
            cert_gap_med=float(np.median([m.get("cert_gap", np.nan) for m in ms]))
            if "cert_gap" in ms[0] else float("nan"),
        )
    sdp_vals = [r.get("sdp") for r in cell_recs if r.get("sdp")]
    if sdp_vals:
        out["sdp"] = dict(
            tight=float(np.mean([s["tight"] for s in sdp_vals])),
            sound=float(np.mean([s["sound"] for s in sdp_vals])),
            ratio_max=float(np.max([s["ratio"] for s in sdp_vals])),
            value_mean=float(np.mean([s["value"] for s in sdp_vals])),
            time_ms_mean=float(np.mean([s["time_ms"] for s in sdp_vals])),
        )
    out["n"] = len(cell_recs)
    return out


def run_cell(tag, p, noise, bias_z, n_trials=N_TRIALS, want_sdp=True):
    recs = []
    t_cell = time.perf_counter()
    for trial in range(n_trials):
        ins = C.make_instance(trial, p, noise=noise, bias_z=bias_z)
        try:
            rec = run_trial(ins, want_sdp=want_sdp)
        except Exception:
            log("  !! trial %d FAILED for cell %s" % (trial, tag))
            traceback.print_exc()
            rec = dict(seed=trial, p=p, noise=noise, bias_z=bias_z, error=traceback.format_exc())
        recs.append(rec)
        if (trial + 1) % 10 == 0:
            log("  [%s] %d/%d  (%.1fs)" % (tag, trial + 1, n_trials, time.perf_counter() - t_cell))
    summ = summarize(recs)
    log("  ---- %s summary (p=%d noise=%.3f bias=%.3f) ----" % (tag, p, noise, bias_z))
    for name in ("unc", "soft", "proj", "alm"):
        s = summ[name]
        log("   %-5s pen=%7.2f±%6.2f mm (rate %4.0f%%) rot=%5.2f±%4.2f deg trans=%6.2f±%6.2f mm "
            "feas=%3.0f%% cert=%3.0f%% time=%6.1f ms"
            % (name, s["pen_mean"], s["pen_std"], 100 * s["pen_rate"], s["rot_mean"], s["rot_std"],
               s["trans_mean"], s["trans_std"], 100 * s["feasible"], 100 * s["certified"], s["time_ms_mean"]))
    if "sdp" in summ:
        log("   sdp  tight=%3.0f%% sound=%3.0f%% max λ2/λ1=%.2e  time=%.1f ms"
            % (100 * summ["sdp"]["tight"], 100 * summ["sdp"]["sound"],
               summ["sdp"]["ratio_max"], summ["sdp"]["time_ms_mean"]))
    # 诊断：异常 trial
    bad = [r for r in recs if "error" in r]
    if bad:
        log("  !! %d 个 trial 报错（见 JSON）" % len(bad))
    low_alm = [r for r in recs if "error" not in r and not r["alm"]["feasible"]]
    if low_alm:
        log("  !! %d 个 trial ALM 不可行：viol=%s" % (len(low_alm), [round(r["alm"]["viol"], 8) for r in low_alm]))
    bad_cert = [r for r in recs if r.get("sdp") and r["sdp"]["tight"] and r["alm"].get("cert_gap", 0) > 1e-6]
    if bad_cert:
        log("  !! %d 个 trial 证书 gap>1e-6" % len(bad_cert))
    return dict(tag=tag, p=p, noise=noise, bias_z=bias_z, n=n_trials,
                summary=summ, recs=recs)


def main():
    log("=" * 100)
    log("Constrained-QPEPLayer main experiment  |  N_TRIALS=%d  LAM_MAIN=%g" % (N_TRIALS, LAM_MAIN))
    log("p ∈ %s ; noise cells %s" % (MAIN_P, [c["tag"] for c in NOISE_CELLS]))
    t_start = time.perf_counter()

    raw = dict(config=dict(n_trials=N_TRIALS, lam_main=LAM_MAIN, lam_pareto=LAM_PARETO,
                           main_p=MAIN_P, noise_cells=NOISE_CELLS,
                           split=C.P_SPLIT, box=[C.BOX_HX, C.BOX_HY, C.BOX_HZ],
                           H_TAB=C.H_TAB, zmin=C.ZMIN, tau=C.TAU,
                           alpha_deg=float(np.degrees(C.ALPHA)), N_pts=C.N_PTS),
               main={}, noise={}, pareto={})

    # ---- 主表：method × p（固定 noise=0.01, bias=0.08）----
    for p in MAIN_P:
        cell = run_cell("main_p%d" % p, p, 0.01, 0.08)
        raw["main"]["p%d" % p] = cell
        with open(os.path.join(RES, "raw_main.json"), "w") as f:
            json.dump(raw, f, indent=1, default=float)
        log("  [saved] results/raw_main.json  (%.1fs elapsed)" % (time.perf_counter() - t_start))

    # ---- 噪声/偏差分组：p=16 ----
    for c in NOISE_CELLS:
        cell = run_cell("noise_%s" % c["tag"], 16, c["noise"], c["bias_z"])
        raw["noise"][c["tag"]] = cell
        with open(os.path.join(RES, "raw_main.json"), "w") as f:
            json.dump(raw, f, indent=1, default=float)
        log("  [saved] results/raw_main.json  (%.1fs elapsed)" % (time.perf_counter() - t_start))

    # ---- 软惩罚 Pareto：p=16, λ 扫描 ----
    if not DO_PARETO:
        return
    log("\n=== 软惩罚 Pareto（p=16, noise=0.01, bias=0.08, %d trials × %d λ） ===" % (N_TRIALS, len(LAM_PARETO)))
    pareto = []
    t_par = time.perf_counter()
    for trial in range(N_TRIALS):
        ins = C.make_instance(trial, 16, noise=0.01, bias_z=0.08)
        M0, D0, G0, A, B = ins["M0"], ins["D0"], ins["G0"], ins["A"], ins["B"]
        Ju, qu, tu = C.solve_qp_global(M0, D0, G0, n_starts=48, rng=np.random.default_rng(1000 + trial))
        for lam in LAM_PARETO:
            sp = C.soft_penalty(M0, D0, G0, A, B, lam, qu, tu, n_starts=14,
                                rng=np.random.default_rng(5000 + trial))
            pareto.append(dict(seed=trial, lam=lam,
                               pen_mm=C.penetration_mm(C.R_of(sp["q"]), sp["t"], ins["V_dense"]),
                               rot_deg=C.rot_deg(sp["q"], ins["q_gt"]),
                               trans_mm=C.trans_mm(sp["t"], ins["t_gt"]),
                               viol=C.max_viol(A, B, sp["q"], sp["t"]), J=sp["J"]))
        if (trial + 1) % 10 == 0:
            log("  pareto %d/%d (%.1fs)" % (trial + 1, N_TRIALS, time.perf_counter() - t_par))
    raw["pareto"] = pareto
    with open(os.path.join(RES, "raw_pareto.json"), "w") as f:
        json.dump(pareto, f, indent=1, default=float)
    log("  [saved] results/raw_pareto.json")

    with open(os.path.join(RES, "raw_main.json"), "w") as f:
        json.dump(raw, f, indent=1, default=float)
    log("\n总耗时 %.1f s ; 输出 results/raw_main.json / results/raw_pareto.json"
        % (time.perf_counter() - t_start))


if __name__ == "__main__":
    main()
