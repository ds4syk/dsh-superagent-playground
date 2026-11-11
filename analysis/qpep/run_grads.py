"""实验 1b：可微性对照（full KKT vs 3-of-4 约化分支 vs 有限差分）。

θ = b[0,0]（第一个目标点的 x 坐标）。对每个实例：
- full KKT：隐式微分 dq/dθ = -(∂F/∂z)^{-1} ∂F/∂θ；与解映射的中心差分比较；
- 每个约化分支：先求该分支在 q* 附近的根 q_k，再对其做隐式微分与差分；
- 输出分支导数与 full 导数的相对差、条件数、根到全局最优的距离。

结果写 results/ptop_grads.json
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from qpep_ptop import (  # noqa: E402
    _lm_batch,
    canonical,
    enumerate_full,
    jac_branch,
    jac_full,
    loss_grad_hess,
    resid_branch,
    resid_full,
)
from run_ptop import make_instance, solve_branch_from, solve_full_from, theta_deriv_branch, theta_deriv_full  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")


def main():
    rows = []
    for noise in [0.0, 0.002, 0.01, 0.05]:
        for seed in [0, 1]:
            data, gt = make_instance(1000 * seed + int(1e5 * noise), n_pts=12, noise=noise)
            roots, losses = enumerate_full(data, n_starts=120, seed=seed)
            imin = int(np.argmin(losses))
            qstar = roots[imin]
            lam = float(np.dot(qstar, loss_grad_hess(qstar, data)[1]))
            zstar = np.concatenate([qstar, [lam]])

            h = 1e-6
            rec = dict(noise=noise, seed=seed, lam=lam, loss=float(losses[imin]),
                       n_roots=int(len(roots)))

            # ---- full ----
            N = theta_deriv_full(zstar, data)
            dq_full = -np.linalg.solve(jac_full(zstar, data), N)[:4]
            data_p = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in data.items()}
            data_m = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in data.items()}
            data_p["b"] = data_p["b"].copy(); data_p["b"][0, 0] += h
            data_m["b"] = data_m["b"].copy(); data_m["b"][0, 0] -= h
            zp, _ = solve_full_from(qstar, lam, data_p)
            zm, _ = solve_full_from(qstar, lam, data_m)
            qp = zp[:4] / np.linalg.norm(zp[:4])
            qm = zm[:4] / np.linalg.norm(zm[:4])
            dq_fd = (qp - qm) / (2 * h)
            rec["full_impl_vs_fd"] = float(np.linalg.norm(dq_full - dq_fd) / max(1e-12, np.linalg.norm(dq_fd)))
            rec["full_cond"] = float(np.linalg.cond(jac_full(zstar, data)))
            rec["dq_full"] = dq_full.tolist()

            # ---- branches ----
            binfo = {}
            for k in range(4):
                qk, nrk = solve_branch_from(qstar, data, k)
                if nrk > 1e-9:
                    binfo[k] = dict(converged=False)
                    continue
                qk = qk / np.linalg.norm(qk)
                Jk = jac_branch(qk, data, k)
                Nk = theta_deriv_branch(qk, data, k)
                dqk = -np.linalg.solve(Jk, Nk)
                qkp, _ = solve_branch_from(qk, data_p, k)
                qkm, _ = solve_branch_from(qk, data_m, k)
                qkp = qkp / np.linalg.norm(qkp)
                qkm = qkm / np.linalg.norm(qkm)
                dqk_fd = (qkp - qkm) / (2 * h)
                g = loss_grad_hess(qstar, data)[1]
                binfo[k] = dict(
                    converged=True,
                    cond=float(np.linalg.cond(Jk)),
                    impl_vs_fd=float(np.linalg.norm(dqk - dqk_fd) / max(1e-12, np.linalg.norm(dqk_fd))),
                    diff_vs_full=float(np.linalg.norm(dqk - dq_full) / max(1e-12, np.linalg.norm(dq_full))),
                    dist_root_to_global=float(np.linalg.norm(canonical(qk) - canonical(qstar))),
                    dropped_resid=float(g[k]),
                    lam_qk=float(lam * qstar[k]),
                    dq=dqk.tolist(),
                )
            rec["branch"] = binfo
            rows.append(rec)
            print(json.dumps({k: (round(v, 6) if isinstance(v, float) else v)
                              for k, v in rec.items() if k not in ("branch", "dq_full")}, ensure_ascii=False),
                  flush=True)
            for k, b in binfo.items():
                if b.get("converged"):
                    print(f"   branch {k}: cond={b['cond']:.3g} impl_vs_fd={b['impl_vs_fd']:.2e} "
                          f"diff_vs_full={b['diff_vs_full']:.3g} dist={b['dist_root_to_global']:.2e}", flush=True)

    json.dump(rows, open(os.path.join(OUT, "ptop_grads.json"), "w"), indent=1)
    print("saved", os.path.join(OUT, "ptop_grads.json"))


if __name__ == "__main__":
    main()
