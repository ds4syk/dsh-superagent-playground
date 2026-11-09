"""Tightness / certificate-matrix analysis for the stereo-localization SDP (P6).

For every instance of the calibration problem (N_batch poses, 64 features) we
solve the Shor relaxation both in primal form (X >= 0, trace(A0 X)=1) and in
dual form (H = Q - sum_i y_i A_i >= 0, returned by the default SDPRLayer), and
record
  * the two largest eigenvalues of the solution -> r = lambda_1 / lambda_2
    (the tightness ratio; upstream calls this ER and requires ER > 1e5),
  * numerical rank / nullity of X and H (rank-1 => nullity 12 for 13x13),
  * eigmin(H) (must be >= -tol for a valid certificate),
  * SDP size (scalar variable count, cone dimensions) and per-solve timings.

Run: python tightness_analysis.py --n-batch 20 --n-instances 10
"""

import argparse
import json
import os
import time

import numpy as np
import torch

import repro_common as rc


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-batch", type=int, default=20)
    p.add_argument("--n-instances", type=int, default=10)
    p.add_argument("--eps", type=float, default=1e-9, help="SCS tolerance")
    p.add_argument("--threads", type=int, default=8)
    return p.parse_args()


def build_cam(cal):
    st = cal.st
    cam = st.StereoCamera(
        f_u=torch.tensor(cal.cam_gt.f_u),
        f_v=torch.tensor(cal.cam_gt.f_v),
        c_u=torch.tensor(cal.cam_gt.c_u),
        c_v=torch.tensor(cal.cam_gt.c_v),
        b=torch.tensor(cal.cam_gt.b),
        sigma_u=cal.cam_gt.sigma_u,
        sigma_v=cal.cam_gt.sigma_v,
    )
    return cam


def spectra(mats):
    out = []
    for X in mats:
        X = X.detach().cpu().numpy()
        eigs = np.linalg.eigvalsh((X + X.T) / 2)
        order = np.sort(np.abs(eigs))[::-1]
        er = float(order[0] / order[1]) if order[1] > 0 else np.inf
        tol = max(order[0], 1e-300) * 1e-8
        rank = int(np.sum(np.abs(eigs) > tol))
        out.append(
            dict(
                eig_ratio=er,
                eigmin=float(eigs.min()),
                eigmax=float(eigs.max()),
                rank=rank,
                nullity=int(X.shape[0] - rank),
                tight_1e5=bool(er > 1e5),
                tight_1e8=bool(er > 1e8),
                eigvals=[float(x) for x in eigs[::-1]],
            )
        )
    return out


def main():
    args = parse_args()
    rc.setup_env(args.threads)
    cal = rc.load_stereo_cal()
    from sdprlayers import SDPPoseEstimator

    cam = build_cam(cal)
    results = {"instances": [], "meta": {}, "timing": {}}
    solve_times, back_sdp_times, back_is_times = [], [], []
    size_info = None

    for inst in range(args.n_instances):
        r_p0s, C_p0s, r_ls, pixel_meass = cal.get_cal_data(
            radius=3,
            board_dims=[1.0, 1.0],
            N_squares=[8, 8],
            N_batch=args.n_batch,
            setup="cone",
            plot=False,
        )
        rs_l_inT, inv_cov = cam.inverse(pixel_meass)
        weights = torch.ones(rs_l_inT.shape[0], 1, rs_l_inT.shape[2])

        # --- dual form: returned matrix is the certificate H = Q - sum y_i A_i
        est_dual = SDPPoseEstimator(
            T_s_v=torch.eye(4), use_dual=True, solver="SCS", scs_eps=args.eps
        )
        Qs, scales, offsets = est_dual.get_obj_matrix_vec(
            r_ls, rs_l_inT, weights, inv_cov
        )
        # --- primal form: returned matrix is X (the SDP primal variable)
        est_primal = SDPPoseEstimator(
            T_s_v=torch.eye(4), use_dual=False, solver="SCS", scs_eps=args.eps
        )
        if size_info is None:
            comp = est_dual.sdprlayer
            size_info = dict(
                n_vars_x=13,
                X_dim=[13, 13],
                n_scalar_entries=13 * 14 // 2,
                n_constraint_matrices=len(comp.constr_list) + 1,
                cone_dims={k: list(v) if isinstance(v, (list, tuple)) else v
                           for k, v in comp.cone_dims.items()},
            )

        t0 = time.perf_counter()
        with torch.no_grad():
            Hs, _ = est_dual.sdprlayer(
                Qs, solver_args={"solve_method": "SCS", "eps": args.eps}
            )
        solve_times.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        with torch.no_grad():
            Xs, _ = est_primal.sdprlayer(
                Qs, solver_args={"solve_method": "SCS", "eps": args.eps}
            )
        solve_times.append(time.perf_counter() - t0)

        H_sp = spectra(Hs)
        X_sp = spectra(Xs)

        # --- backward timings for the two SDPR flavours
        q = Qs.clone().requires_grad_(True)
        est_sdpdiff = SDPPoseEstimator(
            T_s_v=torch.eye(4), diff_qcqp=False, solver="SCS", scs_eps=args.eps
        )
        t0 = time.perf_counter()
        H1, _ = est_sdpdiff.sdprlayer(
            q, solver_args={"solve_method": "SCS", "eps": args.eps}
        )
        t_fwd = time.perf_counter() - t0
        w = torch.randn_like(H1)
        t0 = time.perf_counter()
        (H1 * w).sum().backward()
        back_sdp_times.append(time.perf_counter() - t0)

        q = Qs.clone().requires_grad_(True)
        est_is = SDPPoseEstimator(
            T_s_v=torch.eye(4), diff_qcqp=True, solver="SCS", scs_eps=args.eps
        )
        H2, _ = est_is.sdprlayer(
            q, solver_args={"solve_method": "SCS", "eps": args.eps}
        )
        w = torch.randn_like(H2)
        t0 = time.perf_counter()
        (H2 * w).sum().backward()
        back_is_times.append(time.perf_counter() - t0)

        results["instances"].append(
            dict(
                instance=inst,
                H=H_sp,
                X=X_sp,
                forward_solve_time_s=t_fwd,
            )
        )
        print(
            f"[tightness] inst {inst}: min ER(X)={min(s['eig_ratio'] for s in X_sp):.3e} "
            f"min ER(H)={min(s['eig_ratio'] for s in H_sp):.3e} "
            f"all tight(1e5)={all(s['tight_1e5'] for s in X_sp) and all(s['tight_1e5'] for s in H_sp)}"
        )

    # aggregate
    allX = [s for i in results["instances"] for s in i["X"]]
    allH = [s for i in results["instances"] for s in i["H"]]
    results["meta"] = dict(
        n_batch=args.n_batch,
        n_instances=args.n_instances,
        scs_eps=args.eps,
        sdp_size=size_info,
        host=rc.host_info(),
    )
    results["summary"] = dict(
        n_solves=len(allX),
        X_eig_ratio_median=float(np.median([s["eig_ratio"] for s in allX])),
        X_eig_ratio_min=float(np.min([s["eig_ratio"] for s in allX])),
        X_rank1_frac_1e5=float(np.mean([s["tight_1e5"] for s in allX])),
        X_rank1_frac_1e8=float(np.mean([s["tight_1e8"] for s in allX])),
        X_eigmin_max=float(np.max([s["eigmin"] for s in allX])),
        X_rank_values=sorted({s["rank"] for s in allX}),
        H_eig_ratio_median=float(np.median([s["eig_ratio"] for s in allH])),
        H_eig_ratio_min=float(np.min([s["eig_ratio"] for s in allH])),
        H_rank1_frac_1e5=float(np.mean([s["tight_1e5"] for s in allH])),
        H_rank1_frac_1e8=float(np.mean([s["tight_1e8"] for s in allH])),
        H_eigmin_min=float(np.min([s["eigmin"] for s in allH])),
        H_eigmin_max=float(np.max([s["eigmin"] for s in allH])),
        H_rank_values=sorted({s["rank"] for s in allH}),
        H_nullity_values=sorted({s["nullity"] for s in allH}),
    )
    results["timing"] = dict(
        solve_time_mean_s=float(np.mean(solve_times)),
        solve_time_median_s=float(np.median(solve_times)),
        n_solves_timed=len(solve_times),
        backprop_sdp_mean_s=float(np.mean(back_sdp_times)),
        backprop_is_mean_s=float(np.mean(back_is_times)),
    )
    rc.write_json(os.path.join(rc.RESULTS, "tightness.json"), results)
    # one-row-per-solve CSV
    import pandas as pd

    rows = []
    for inst in results["instances"]:
        for kind in ("X", "H"):
            for b, s in enumerate(inst[kind]):
                rows.append(
                    dict(
                        instance=inst["instance"],
                        matrix=kind,
                        batch=b,
                        eig_ratio=s["eig_ratio"],
                        eigmin=s["eigmin"],
                        eigmax=s["eigmax"],
                        rank=s["rank"],
                        nullity=s["nullity"],
                        tight_1e5=s["tight_1e5"],
                        tight_1e8=s["tight_1e8"],
                    )
                )
    pd.DataFrame(rows).to_csv(os.path.join(rc.RESULTS, "tightness.csv"), index=False)
    print(json.dumps(results["summary"], indent=2))
    print(json.dumps(results["timing"], indent=2))


if __name__ == "__main__":
    main()
