"""Finite-difference validation of the SDPR-IS gradient for a given run.

Used to check that the (possibly relaxed `kkt_tol` / non-tight recovery)
gradients used for the failed-run fix still match the true derivative of the
outer (P7) loss w.r.t. the stereo baseline b.

Settings come from the same env vars as retry_runs.py:
  SDPR_KKT_TOL, SDPR_SCS_EPS, SDPR_USE_DUAL, SDPR_ER_MIN,
  SDPR_SCS_MAX_ITERS, SDPR_SCS_SCALE
Usage: python fd_check.py --run 8 --b 0.243
"""
import argparse
import json
import os
import pickle

import numpy as np
import torch

import repro_common as rc

rc.setup_env(threads=int(os.environ.get("SLURM_CPUS_PER_TASK", "2")))
sc = rc.load_stereo_cal()  # noqa: E402  (sets float64)
st = rc.load_stereo_tuner()  # noqa: E402


def build_cam(b, grad):
    g = sc.cam_gt
    return st.StereoCamera(
        f_u=g.f_u, f_v=g.f_v, c_u=g.c_u, c_v=g.c_v,
        b=torch.tensor(float(b), requires_grad=grad),
        sigma_u=g.sigma_u, sigma_v=g.sigma_v,
    )


def outer_loss(prob, b, grad=False):
    r_p0s_gt, C_p0s_gt, r_ls, pixel_meass = prob
    cam = build_cam(b, grad)
    rs_l_inT, inv_cov_weights = cam.inverse(pixel_meass)
    weights = torch.ones(rs_l_inT.shape[0], 1, rs_l_inT.shape[2])
    est = st.SDPPoseEstimator(T_s_v=torch.eye(4), diff_qcqp=True, solver="SCS")
    T, _ = est(
        keypoints_3D_src=r_ls, keypoints_3D_trg=rs_l_inT,
        weights=weights, inv_cov_weights=inv_cov_weights,
        solver_args=None, return_loss=True,
    )
    C = T[:, :3, :3]
    r = C.mT.bmm(-T[:, :3, [3]])
    loss = st.get_outer_loss(r, C, r_p0s_gt, C_p0s_gt)
    return (loss, cam.b) if grad else (loss, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=int, required=True)
    ap.add_argument("--b", type=float, default=0.243)
    ap.add_argument("--tag", default=os.environ.get("FD_TAG", "fd"))
    args = ap.parse_args()
    out = os.path.join(rc.RESULTS, "fd", args.tag)
    os.makedirs(out, exist_ok=True)
    info = pickle.load(open(os.path.join(rc.RESULTS, "final",
                                         "stereo_tune_prob_data.pkl"), "rb"))
    prob = info[args.run]
    cfg = {k: os.environ.get(k) for k in
           ("SDPR_KKT_TOL", "SDPR_SCS_EPS", "SDPR_USE_DUAL", "SDPR_ER_MIN",
            "SDPR_SCS_MAX_ITERS", "SDPR_SCS_SCALE")}
    res = dict(run=args.run, b=args.b, config=cfg)
    try:
        L, b_t = outer_loss(prob, args.b, grad=True)
        L.backward()
        g_an = float(b_t.grad)
        res.update(ok=True, loss=float(L), grad_analytic=g_an)
        for h in (1e-3, 1e-4, 1e-5):
            Lp = float(outer_loss(prob, args.b + h)[0])
            Lm = float(outer_loss(prob, args.b - h)[0])
            fd = (Lp - Lm) / (2 * h)
            res[f"fd_{h:.0e}"] = fd
            res[f"rel_err_{h:.0e}"] = abs(g_an - fd) / max(abs(fd), 1e-12)
    except Exception as e:  # noqa: BLE001
        res.update(ok=False, error=f"{type(e).__name__}: {e}")
    json.dump(res, open(os.path.join(out, f"run_{args.run:03d}.json"), "w"), indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
