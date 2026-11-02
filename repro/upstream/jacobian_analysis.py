"""Jacobian verification for the stereo-localization layer (Tables III/IV analog).

The paper validates the implicit-differentiation machinery by comparing the
Jacobian of the solution (C, t) with respect to the measurements against a
reference Jacobian (SVD closed form for scalar weights, unrolled Theseus for
matrix weights), using the infinity-norm relative error

    dJ = || J_est - J_true ||_inf / || J_true ||_inf          (paper eq. 29)

We have no Mosek license (the paper's Tables III/IV used Mosek at 1e-12), so the
reference here is a high-accuracy centred finite difference of the *same*
forward solver.  With SCS at eps=1e-9/1e-10 the forward solution noise limits the
finite-difference Jacobian to roughly 1e-4..1e-6 relative accuracy; this floor is
reported explicitly (paper's Mosek-based numbers are 1e-5..1e-6).

Two experiments:
  A) dLoss/db -- the scalar gradient the Sec. VI-B4 outer loop actually uses.
     analytic (autograd through the layer) vs centred FD, per method.
  B) d(C,t)/d(m_k) -- 3x3 Jacobian w.r.t. one feature's (u,v,d), infinity-norm
     relative difference vs centred FD, aggregated over poses (Table IV analog,
     matrix-weighted / inv-cov weights).

Run: python jacobian_analysis.py --n-poses 5 --eps 1e-9
"""

import argparse
import json
import os

import numpy as np
import torch

import repro_common as rc


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-poses", type=int, default=5)
    p.add_argument("--eps", type=float, default=1e-9)
    p.add_argument("--h-pixel", type=float, default=1e-4)
    p.add_argument("--h-b", type=float, default=1e-6)
    p.add_argument("--feature", type=int, default=0)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--skip-theseus", action="store_true")
    p.add_argument("--skip-b", action="store_true", help="skip d(C,t)/d(pixel) experiment")
    return p.parse_args()


METHODS = ["sdpr-is", "sdpr-sdp", "sdpr-cift", "theseus_gt"]


# ---------------------------------------------------------------------------
# forward + loss
# ---------------------------------------------------------------------------
def make_cam(cal, b):
    """StereoCamera whose baseline is a (possibly differentiable) tensor."""
    st = cal.st
    return st.StereoCamera(
        f_u=torch.tensor(cal.cam_gt.f_u),
        f_v=torch.tensor(cal.cam_gt.f_v),
        c_u=torch.tensor(cal.cam_gt.c_u),
        c_v=torch.tensor(cal.cam_gt.c_v),
        b=b,
        sigma_u=cal.cam_gt.sigma_u,
        sigma_v=cal.cam_gt.sigma_v,
    )


def outer_loss(r_est, C_est, r_gt, C_gt):
    """stereo_tuner.get_outer_loss (P7)."""
    if r_gt.ndim == 3:
        r_gt = r_gt.squeeze(-1)
    loss = torch.norm(r_est - r_gt) ** 2
    loss = loss + torch.norm(C_est.transpose(-2, -1) @ C_gt - torch.eye(3), p="fro") ** 2
    return loss.sum()


def theseus_layer(cal, prob_data, abs_tol=1e-10, rel_tol=1e-6):
    st = cal.st
    r_p0s, C_p0s, r_ls, pixel_meass = prob_data
    return st.build_theseus_layer(
        N_map=r_ls.shape[-1],
        N_batch=r_p0s.shape[0],
        opt_kwargs_in={
            "abs_err_tolerance": abs_tol,
            "rel_err_tolerance": rel_tol,
            "max_iterations": 500,
            "step_size": 0.2,
        },
    )


def pose_forward(cal, prob_data, method, b, pixel_meass, eps):
    """Return (C_est, r_est) for one method; everything differentiable wrt b/pixels."""
    from sdprlayers import SDPPoseEstimator

    r_p0s, C_p0s, r_ls, _ = prob_data
    cam = make_cam(cal, b)
    rs_l_inT, inv_cov = cam.inverse(pixel_meass)
    weights = torch.ones(rs_l_inT.shape[0], 1, rs_l_inT.shape[2])

    if method == "theseus_gt":
        layer = theseus_layer(cal, prob_data)
        inputs = {
            "C_p0s": C_p0s.clone(),
            "r_p0s": r_p0s.clone().squeeze(-1),
            "r_ls": r_ls,
            "meas": rs_l_inT,
            "weights": inv_cov,
        }
        out, _ = layer.forward(
            inputs,
            optimizer_kwargs={"track_best_solution": True, "backward_mode": "implicit"},
        )
        return out["C_p0s"], out["r_p0s"]
    diff_qcqp = method != "sdpr-sdp"
    est = SDPPoseEstimator(
        T_s_v=torch.eye(4),
        diff_qcqp=diff_qcqp,
        compute_multipliers=(method == "sdpr-cift"),
        solver="SCS",
        scs_eps=eps,
    )
    T, _ = est(
        keypoints_3D_src=r_ls,
        keypoints_3D_trg=rs_l_inT,
        weights=weights,
        inv_cov_weights=inv_cov,
        return_loss=True,
    )
    C_est = T[:, :3, :3]
    r_est = -C_est.mT.bmm(T[:, :3, [3]]).squeeze(-1)
    return C_est, r_est


def loss_at(cal, prob_data, method, b_val, pixel_meass, eps, grad=False):
    b = torch.tensor(float(b_val), requires_grad=grad)
    C_est, r_est = pose_forward(cal, prob_data, method, b, pixel_meass, eps)
    r_gt, C_gt = prob_data[0], prob_data[1]
    loss = outer_loss(r_est, C_est, r_gt, C_gt)
    return loss, b


# ---------------------------------------------------------------------------
# experiment A: dLoss/db
# ---------------------------------------------------------------------------
def exp_A(cal, prob_datas, methods, eps, h_b):
    out = {}
    for method in methods:
        rel, pairs = [], []
        for i, pd in enumerate(prob_datas):
            pix = pd[3]
            try:
                loss, b = loss_at(cal, pd, method, cal.cam_gt.b, pix, eps, grad=True)
                loss.backward()
                g_an = float(b.grad.item())
                with torch.no_grad():
                    lp, _ = loss_at(cal, pd, method, cal.cam_gt.b + h_b, pix, eps)
                    lm, _ = loss_at(cal, pd, method, cal.cam_gt.b - h_b, pix, eps)
                g_fd = float((lp - lm) / (2 * h_b))
            except Exception as exc:
                print(f"[A] {method} pose {i} failed: {type(exc).__name__}: {exc}")
                continue
            r = abs(g_an - g_fd) / abs(g_fd) if g_fd != 0 else np.nan
            rel.append(r)
            pairs.append(dict(analytic=g_an, fd=g_fd, rel=r))
            print(f"[A] {method} pose {i}: analytic={g_an:+.6e} fd={g_fd:+.6e} rel={r:.3e}")
        if rel:
            out[method] = dict(
                rel_err_mean=float(np.nanmean(rel)),
                rel_err_std=float(np.nanstd(rel)),
                rel_err_max=float(np.nanmax(rel)),
                pairs=pairs,
            )
    return out


# ---------------------------------------------------------------------------
# experiment B: d(C,t)/d(u,v,d of one feature)
# ---------------------------------------------------------------------------
def pose_vecs(C, r):
    """Flat [t(3), vec(C)(9)]; translation t = -C r (world->camera convention)."""
    t = -torch.einsum("bij,bj->bi", C, r)
    return torch.cat([t[0], C[0].reshape(-1)])


def exp_B(cal, prob_datas, methods, eps, h_pix, feature):
    out = {}
    for method in methods:
        errs = []
        for i, pd in enumerate(prob_datas):
            pix0 = np.asarray(pd[3], dtype=float)
            try:
                pix_t = torch.tensor(pix0, dtype=torch.float64, requires_grad=True)
                C_est, r_est = pose_forward(
                    cal, pd, method, torch.tensor(cal.cam_gt.b), pix_t, eps
                )
                phi = pose_vecs(C_est, r_est)
                # keep only the 3 perturbed scalars as inputs
                idx = [(0, 0, feature), (0, 1, feature), (0, 2, feature)]
                theta = [pix_t[j, k, feature] for j, k, _f in idx]
                J_est = np.zeros((phi.numel(), 3))
                for row in range(phi.numel()):
                    grads = torch.autograd.grad(
                        phi[row], theta, retain_graph=True, allow_unused=True
                    )
                    for j in range(3):
                        J_est[row, j] = 0.0 if grads[j] is None else float(grads[j])
            except Exception as exc:
                print(f"[B] {method} pose {i} analytic failed: {type(exc).__name__}: {exc}")
                continue

            def f_theta(th):
                pix = pix0.copy()
                for j, (a, b_, c) in enumerate(idx):
                    pix[a, b_, c] = th[j]
                with torch.no_grad():
                    C_e, r_e = pose_forward(
                        cal, pd, method, torch.tensor(cal.cam_gt.b),
                        torch.tensor(pix, dtype=torch.float64), eps,
                    )
                return pose_vecs(C_e, r_e).numpy()

            theta0 = np.array([pix0[a, b_, c] for a, b_, c in idx])
            J_fd = np.zeros_like(J_est)
            for j in range(3):
                tp, tm = theta0.copy(), theta0.copy()
                tp[j] += h_pix
                tm[j] -= h_pix
                J_fd[:, j] = (f_theta(tp) - f_theta(tm)) / (2 * h_pix)
            num = np.linalg.norm(J_est - J_fd, ord=np.inf)
            den = np.linalg.norm(J_fd, ord=np.inf)
            e = float(num / den) if den > 0 else np.nan
            errs.append(e)
            print(f"[B] {method} pose {i}: dJ_inf={e:.3e}")
        if errs:
            out[method] = dict(
                jac_diff_mean=float(np.nanmean(errs)),
                jac_diff_std=float(np.nanstd(errs)),
                jac_diff_max=float(np.nanmax(errs)),
                per_pose=errs,
            )
    return out


def main():
    args = parse_args()
    rc.setup_env(args.threads)
    cal = rc.load_stereo_cal()
    torch.set_default_dtype(torch.float64)

    methods = [m for m in METHODS if m != "theseus_gt"]
    if not args.skip_theseus:
        try:
            import theseus  # noqa: F401

            methods.append("theseus_gt")
        except Exception as exc:
            print(f"[warn] theseus unavailable -> skipping Theseus-GT: {exc}")

    prob_datas = [
        cal.get_cal_data(
            radius=3,
            board_dims=[1.0, 1.0],
            N_squares=[8, 8],
            N_batch=1,
            setup="cone",
            plot=False,
        )
        for _ in range(args.n_poses)
    ]

    out = dict(
        meta=dict(
            n_poses=args.n_poses,
            eps=args.eps,
            h_pixel=args.h_pixel,
            h_b=args.h_b,
            feature=args.feature,
            methods=methods,
            host=rc.host_info(),
        )
    )
    out["A_dloss_db"] = exp_A(cal, prob_datas, methods, args.eps, args.h_b)
    if not args.skip_b:
        out["B_dpose_dpixel"] = exp_B(
            cal, prob_datas, methods, args.eps, args.h_pixel, args.feature
        )
    rc.write_json(os.path.join(rc.RESULTS, "jacobian.json"), out)
    print(
        json.dumps(
            {k: v for k, v in out.items() if k != "meta"}, indent=2, default=str
        )
    )


if __name__ == "__main__":
    main()
