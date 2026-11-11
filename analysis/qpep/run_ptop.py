"""实验 1：QPEP 结构（point-to-plane）
- full KKT 全部实根 vs 4 组 3-of-4 约化系统的全部实根；
- lambda 大小、分支召回、分支重叠、条件数；
- full KKT 隐式微分 vs 分支隐式微分 vs 有限差分。

输出 analysis/qpep/results/ptop_results.json
"""

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from qpep_ptop import (  # noqa: E402
    canonical,
    enumerate_branch,
    enumerate_full,
    implicit_branch,
    implicit_full,
    jac_branch,
    jac_full,
    loss_grad_hess,
    quat_to_R,
    resid_branch,
    resid_full,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)


def make_instance(seed, n_pts=12, noise=0.01, out_frac=0.0):
    rng = np.random.default_rng(seed)
    u = rng.uniform(-1, 1, size=(n_pts, 3))
    n = rng.normal(size=(n_pts, 3))
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    q0 = rng.normal(size=4)
    q0 /= np.linalg.norm(q0)
    R0 = quat_to_R(q0)
    t0 = rng.uniform(-1, 1, size=3) + np.array([0.0, 0.0, 2.5])
    b = u @ R0.T + t0
    # outlier：把若干 b 拉偏
    if out_frac > 0:
        m = int(round(out_frac * n_pts))
        idx = rng.choice(n_pts, size=m, replace=False)
        b[idx] += rng.normal(scale=1.0, size=(m, 3))
    b = b + noise * rng.normal(size=(n_pts, 3))
    return dict(u=u, b=b, n=n), (R0, t0, q0)


def theta_deriv_full(z, data, h=1e-8):
    """dF/dtheta by complex step, theta = b[0,0]."""
    d = {k: v.copy() for k, v in data.items()}
    d["b"] = d["b"].astype(complex)
    d["b"][0, 0] += 1j * h
    return np.imag(resid_full(z, d)) / h


def theta_deriv_branch(q, data, k, h=1e-8):
    d = {k: v.copy() for k, v in data.items()}
    d["b"] = d["b"].astype(complex)
    d["b"][0, 0] += 1j * h
    return np.imag(resid_branch(q, d, k)) / h


def solve_full_from(q, lam, data):
    from qpep_ptop import _lm_batch

    z0 = np.concatenate([q, [lam]])[None, :]
    z, nr = _lm_batch(resid_full, jac_full, z0, data, maxit=100, tol=1e-14)
    return z[0], nr[0]


def solve_branch_from(q, data, k):
    from qpep_ptop import _lm_batch

    z, nr = _lm_batch(lambda x, d: resid_branch(x, d, k),
                      lambda x, d: jac_branch(x, d, k), q[None, :], data, maxit=100, tol=1e-14)
    return z[0], nr[0]


def main():
    n_starts_full = int(os.environ.get("NS_FULL", 300))
    n_starts_br = int(os.environ.get("NS_BR", 200))
    noise_levels = [0.0, 0.002, 0.005, 0.01, 0.02, 0.05]
    seeds = [0, 1, 2, 3]
    n_pts = 12

    rows = []
    t0 = time.time()
    for noise in noise_levels:
        for seed in seeds:
            data, gt = make_instance(1000 * seed + int(1e5 * noise), n_pts=n_pts, noise=noise)
            R0, tt0, q0true = gt

            roots, losses = enumerate_full(data, n_starts=n_starts_full, seed=seed)
            lam = np.array([np.dot(r, loss_grad_hess(r, data)[1]) for r in roots])
            # 全局最优（目标值最小）
            imin = int(np.argmin(losses))
            qstar = roots[imin]
            Lstar = float(losses[imin])
            lamstar = float(lam[imin])

            # 4 个约化分支
            br = {}
            for k in range(4):
                rk = enumerate_branch(data, k, n_starts=n_starts_br, seed=seed + 7 * k)
                Lk = np.array([loss_grad_hess(x, data)[0] for x in rk]) if len(rk) else np.zeros(0)
                br[k] = dict(roots=rk, losses=Lk)

            # 每个分支最优点、与全局最优的距离、dropped residual
            branch_info = {}
            for k in range(4):
                rk, Lk = br[k]["roots"], br[k]["losses"]
                if len(rk) == 0:
                    branch_info[k] = None
                    continue
                j = int(np.argmin(Lk))
                qk = rk[j]
                dist = float(np.linalg.norm(canonical(qk) - canonical(qstar)))
                # dropped residual at the FULL optimum: g_k(q*) = lam*q*_k
                g = loss_grad_hess(qstar, data)[1]
                branch_info[k] = dict(
                    n_roots=int(len(rk)),
                    best_loss=float(Lk[j]),
                    loss_gap=float(Lk[j] - Lstar),
                    dist_to_global=dist,
                    dropped_resid_at_global=float(g[k]),
                    pred_lam_qk=float(lamstar * qstar[k]),
                    contains_global=bool(np.min(np.linalg.norm(
                        np.array([canonical(x) - canonical(qstar) for x in rk]), axis=1)) < 1e-6)
                    if len(rk) else False,
                )

            # 分支间重叠：同一根出现在几个分支（以 1e-6 判定）
            all_roots = []
            for k in range(4):
                for x in br[k]["roots"]:
                    all_roots.append((k, canonical(x)))
            overlap_pairs = 0
            overlap_global = 0
            for i in range(len(all_roots)):
                for j2 in range(i + 1, len(all_roots)):
                    if all_roots[i][0] == all_roots[j2][0]:
                        continue
                    if np.linalg.norm(all_roots[i][1] - all_roots[j2][1]) < 1e-6:
                        overlap_pairs += 1
            # 全局最优被几个分支同时包含
            for k in range(4):
                if branch_info[k] and branch_info[k]["contains_global"]:
                    overlap_global += 1

            rows.append(dict(
                noise=noise, seed=seed, n_pts=n_pts,
                n_roots_full=int(len(roots)),
                losses_all=[float(x) for x in np.sort(losses)],
                lams_all=[float(x) for x in lam[np.argsort(losses)]],
                loss_star=Lstar, lam_star=lamstar,
                q_star=qstar.tolist(),
                q_absmin_idx=int(np.argmin(np.abs(qstar))),
                q_absmin=float(np.min(np.abs(qstar))),
                branch=branch_info,
                overlap_pairs=int(overlap_pairs),
                n_branches_containing_global=int(overlap_global),
                n_roots_branches=[int(len(br[k]["roots"])) for k in range(4)],
            ))
        print(f"noise={noise} done, elapsed={time.time()-t0:.1f}s", flush=True)

    # ---- 微分实验：在若干实例上比较 full vs branch vs FD ----
    grad_rows = []
    for noise in [0.0, 0.01, 0.05]:
        for seed in [0, 1, 2]:
            data, gt = make_instance(1000 * seed + int(1e5 * noise), n_pts=n_pts, noise=noise)
            roots, losses = enumerate_full(data, n_starts=n_starts_full, seed=seed)
            imin = int(np.argmin(losses))
            qstar = roots[imin]
            lamstar = float(np.dot(qstar, loss_grad_hess(qstar, data)[1]))
            zstar = np.concatenate([qstar, [lamstar]])

            h = 1e-6
            res = dict(noise=noise, seed=seed, lam=lamstar, loss=float(losses[imin]))

            # full: implicit vs FD
            dq_imp, dlam_imp, M = implicit_full(qstar, lamstar, data, theta_deriv_full)
            data_p = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in data.items()}
            data_m = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in data.items()}
            data_p["b"] = data_p["b"].copy(); data_p["b"][0, 0] += h
            data_m["b"] = data_m["b"].copy(); data_m["b"][0, 0] -= h
            zp, okp = solve_full_from(qstar, lamstar, data_p)
            zm, okm = solve_full_from(qstar, lamstar, data_m)
            dq_fd = (zp[:4] / np.linalg.norm(zp[:4]) - zm[:4] / np.linalg.norm(zm[:4])) / (2 * h)
            res["full_impl_vs_fd"] = float(np.linalg.norm(dq_imp - dq_fd) / max(1e-12, np.linalg.norm(dq_fd)))
            res["full_cond"] = float(np.linalg.cond(M))
            res["dq_full"] = dq_imp.tolist()

            # branches
            binfo = {}
            for k in range(4):
                # 分支根：从 q* 出发求解该分支系统
                qk, nrk = solve_branch_from(qstar, data, k)
                if nrk > 1e-9:
                    binfo[k] = None
                    continue
                qk = qk / np.linalg.norm(qk)
                try:
                    dqk_imp, Jk = implicit_branch(qk, data, k, theta_deriv_branch)
                except np.linalg.LinAlgError:
                    binfo[k] = dict(singular=True)
                    continue
                # FD of this branch root
                qp, np_ok = solve_branch_from(qk, data_p, k)
                qm, nm_ok = solve_branch_from(qk, data_m, k)
                dqk_fd = (qp / np.linalg.norm(qp) - qm / np.linalg.norm(qm)) / (2 * h)
                binfo[k] = dict(
                    cond=float(np.linalg.cond(Jk)),
                    impl_vs_fd=float(np.linalg.norm(dqk_imp - dqk_fd) / max(1e-12, np.linalg.norm(dqk_fd))),
                    diff_vs_full=float(np.linalg.norm(dqk_imp - dq_imp) / max(1e-12, np.linalg.norm(dq_imp))),
                    dq=dqk_imp.tolist(),
                    dist_root_to_global=float(np.linalg.norm(canonical(qk) - canonical(qstar))),
                )
            res["branch"] = binfo
            grad_rows.append(res)

    with open(os.path.join(OUT, "ptop_results.json"), "w") as f:
        json.dump(dict(roots=rows, grads=grad_rows,
                       config=dict(n_pts=n_pts, n_starts_full=n_starts_full,
                                   n_starts_branch=n_starts_br,
                                   noise_levels=noise_levels, seeds=seeds)),
                  f, indent=1)
    print("saved", os.path.join(OUT, "ptop_results.json"))


if __name__ == "__main__":
    main()
