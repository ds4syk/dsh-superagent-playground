#!/usr/bin/env python3
"""External baseline: Theseus LM/GN on the P1 real two-mode bearing cost.

Object of study
---------------
The exact cost built by ``run_two_mode.py`` (PART A) on BOP LINEMOD eggbox
(obj 10) / glue (obj 11), for the same 15 frames:

    C(R, t) = sum_i || P_i^perp (R p_i + t) ||^2 ,
    P_i^perp = I - u_i u_i^T ,   ||u_i|| = 1 (real unit bearings),

where ``p_i`` is the real mesh vertex and ``u_i`` the real depth-pixel bearing
(``run_two_mode.real_correspondences``), and only the RANSAC-dominant planar
face inliers enter the sum (``run_two_mode.plane_randac/reduce_cost``).

Why this file exists
--------------------
P1's two-mode layer computes the exact global modes of the *reduced* cost
(translation eliminated analytically), reports hard argmin and the Laplace
posterior over the physical modes.  The natural external baseline is a local
optimiser on the *same* cost with the translation kept as a free SE(3)
variable: Theseus Levenberg-Marquardt with an analytic Jacobian.  It tests
whether local optimisation reaches the global (true) mode without
enumeration, and what it does when two modes are nearly tied.

Reuse (not reinvention)
-----------------------
The Theseus pattern is the P3 track's working baseline
``labs/03-robust-qpeplayer/experiments/baselines/theseus_cost.py`` and
``.../run_theseus.py``: a custom ``CostFunction`` with an analytic Jacobian,
``th.LevenbergMarquardt`` + ``th.CholeskyDenseSolver`` in float64, GT/random
initialisation and 20 trials.  We import ``se3_from_Rt``, ``skew_batch`` and
``errors_from_pose`` from ``theseus_cost`` and copy its finite-difference
self-check structure (the Jacobian convention ``T <- T @ Exp(delta)``,
``delta = (v, omega)``, is verified numerically below, not assumed).

Two objective constructions are built from the same class:
  * ``vector``   -- one residual block of dimension 3N (fast; used for the grid);
  * ``perpoint`` -- N cost functions of dimension 3, one per correspondence
    (the literal P3 ``build_gm_objective`` pattern; used as a cross-check).
They are the same function, and the report records the maximum pose
difference between them.

Local optimiser, not a posterior
-------------------------------
Theseus returns one pose per initialisation.  It does not enumerate modes, does
not know about the physical-mode filter (all transformed points in front of the
camera), and returns no posterior.  Where it converges to a compromise pose or
to a non-physical (behind-camera) solution, that is the result and is reported.

Usage
-----
  /mnt/storage1/syk/venvs/theseus/bin/python run_theseus_baseline.py --selftest
  /mnt/storage1/syk/venvs/theseus/bin/python run_theseus_baseline.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
P1CODE = HERE.parents[1] / "code"
P3BASELINES = HERE.parents[3] / "03-robust-qpeplayer" / "experiments" / "baselines"
for _p in (str(HERE), str(P1CODE), str(P3BASELINES)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import theseus as th  # noqa: E402
from theseus.core import CostFunction, ScaleCostWeight  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
from scipy.spatial.transform import Rotation as Rot  # noqa: E402

import qpep_core as core  # noqa: E402
import instance as inst_mod  # noqa: E402
import run_two_mode as tm  # noqa: E402
from run_symmetry import ROOT, load_model  # noqa: E402

# P3 re-use: geometry conventions + pose extraction, not re-derived here.
from theseus_cost import errors_from_pose, se3_from_Rt, skew_batch  # noqa: E402

torch.set_default_dtype(torch.float64)

MAX_ITERS = 100
ABS_TOL = 1e-14
REL_TOL = 1e-10
N_RANDOM = int(os.environ.get("THESEUS_N_RANDOM", "20"))
N_MULTISTART = int(os.environ.get("THESEUS_N_MULTISTART", "8"))
Z_FRONT = 0.05            # same "in front of the camera" threshold as run_two_mode
GT_TOL_DEG = 10.0         # success = within 10 deg of GT (task protocol)
SEED_ROOT = 20261127
MULTISTART_FRAMES = [("000010", "15"), ("000010", "72"), ("000010", "60"),
                     ("000011", "20"), ("000011", "49")]
OUT = HERE / "results"


# --------------------------------------------------------------------- cost
class BearingResidual(CostFunction):
    """r_i = P_i^perp (R p_i + t) for a block of n correspondences.

    * ``n = N`` (one 3N-dim residual block, batch 1) -> ``vector`` construction;
    * ``n = 1`` per correspondence, one cost function each -> ``perpoint``.

    Analytic Jacobian (Theseus convention ``T <- T @ Exp(delta)``,
    ``delta = (v, omega)``, translation first, verified by finite differences)::

        d r_i / d v     =  P_i^perp R
        d r_i / d omega = -P_i^perp R [p_i]_x

    ``[p]_x`` comes from the P3 ``skew_batch``; the only change w.r.t. the P3
    point-to-point cost is the constant left factor ``P_i^perp``, which is the
    bearing (point-to-line) projection of ``run_two_mode``'s cost.
    """

    def __init__(self, pose: th.SE3, P: torch.Tensor, U: torch.Tensor,
                 name: str = "bearing"):
        super().__init__(ScaleCostWeight(1.0), name=name)
        self.pose = pose
        self.register_optim_var("pose")
        # Constant measurement data are plain tensors, deliberately NOT registered
        # as aux Variables: Theseus' objective batch size is the max over *all*
        # registered variables, so an aux variable of batch N would make the
        # objective batched (N) while the pose is batch 1.  With a single 3N-dim
        # residual block the objective batch must stay 1.  (The per-point
        # construction of ``build_objective`` is the P3 aux-variable pattern.)
        self.P = P.clone()                   # (n,3) model points
        self.U = U.clone()                   # (n,3) unit bearings
        self.n = int(P.shape[0])

    def dim(self) -> int:
        return 3 * self.n

    def _points_cam(self) -> torch.Tensor:
        R = self.pose.rotation().tensor[0]        # (3,3)
        t = self.pose.translation().tensor[0]     # (3,)
        return self.P @ R.T + t                   # (n,3)

    def _project(self, v: torch.Tensor) -> torch.Tensor:
        # P^perp v = v - (u.v) u, per point.  U is unit to 1e-16 (asserted below).
        u = self.U
        return v - (u * v).sum(-1, keepdim=True) * u

    def error(self) -> torch.Tensor:
        return self._project(self._points_cam()).reshape(1, 3 * self.n)

    def _copy_impl(self, new_name=None):
        return BearingResidual(self.pose.copy(), self.P.clone(), self.U.clone(),
                               name=new_name or self.name)

    def jacobians(self):
        R = self.pose.rotation().tensor[0]
        u = self.U
        Pperp = torch.eye(3, dtype=R.dtype, device=R.device) - \
            u.unsqueeze(-1) * u.unsqueeze(-2)                      # (n,3,3)
        S = skew_batch(self.P)                                     # (n,3,3)
        J = torch.zeros(self.n, 3, 6, dtype=R.dtype, device=R.device)
        J[..., :3] = torch.matmul(Pperp, R.unsqueeze(0).expand(self.n, -1, -1))
        J[..., 3:] = -torch.matmul(Pperp, torch.matmul(
            R.unsqueeze(0).expand(self.n, -1, -1), S))
        return [J.reshape(1, 3 * self.n, 6)], self.error()


def build_objective(P, U, R0, t0, impl="vector"):
    """Theseus objective for C(R,t) with one shared SE(3) variable."""
    pose = th.SE3(tensor=se3_from_Rt(R0, t0))
    obj = th.Objective(dtype=torch.float64)
    P_t = torch.as_tensor(np.ascontiguousarray(P))
    U_t = torch.as_tensor(np.ascontiguousarray(U))
    if impl == "vector":
        obj.add(BearingResidual(pose, P_t, U_t, name="bearing"))
    elif impl == "perpoint":
        # literal P3 build_gm_objective pattern: one cost per correspondence
        for i in range(P.shape[0]):
            obj.add(BearingResidual(pose, P_t[i:i + 1], U_t[i:i + 1],
                                    name=f"bearing_{i:04d}"))
    else:
        raise ValueError(impl)
    obj.update()
    return obj, pose


def numpy_cost(P, U, R, t):
    """Reference cost, the same expression as run_two_mode's residual sum."""
    v = P @ R.T + t
    r = v - (U * v).sum(1, keepdims=True) * U
    return float((r * r).sum())


def solve_lm(obj, pose, R0, t0, max_iters=MAX_ITERS):
    """Reset the pose variable to (R0,t0), then run Theseus LM on ``obj``."""
    pose.update(se3_from_Rt(R0, t0))
    obj.update()
    solver = th.LevenbergMarquardt(
        obj, linear_solver_cls=th.CholeskyDenseSolver,
        max_iterations=max_iters, vectorize=False,
        abs_err_tolerance=ABS_TOL, rel_err_tolerance=REL_TOL,
    )
    t0c = time.perf_counter()
    info = solver.optimize()
    dt = time.perf_counter() - t0c
    R, t = errors_from_pose(pose)
    iters = int(np.asarray(info.converged_iter).reshape(-1)[0])
    raw_status = np.asarray(info.status).reshape(-1)[0]
    status = int(getattr(raw_status, "value", raw_status))
    last_err = float(np.asarray(info.last_err).reshape(-1)[0])
    return R, t, dt, iters, status, last_err


# --------------------------------------------------------------------- data
def load_frame(fr, im, oid, tree, verts, I):
    """Real correspondences + dominant plane + P1's physical modes (labels only)."""
    K, R, t, _ = tm.load_gt(fr, im, 0)
    data = tm.real_correspondences(fr, im, 0, K, R, t, tree, verts)
    if data is None:
        return None
    P, U, dist = data
    pl = tm.plane_ransac(P)
    if pl is None:
        return None
    inl, nrm, off = pl
    if inl.sum() < tm.MIN_PLANE or (~inl).sum() < tm.MIN_OTHER:
        return None
    Pi, Ui = P[inl], U[inl]
    M, Gi = tm.reduce_cost(Pi, Ui)
    pm = tm.physical_modes(I, M, Gi, Pi)
    frame = dict(frame=fr, img=im, obj=oid, R=R, t=t, P=Pi, U=Ui,
                 n_plane=int(inl.sum()), n_other=int((~inl).sum()),
                 u_norm_dev=float(np.abs(np.linalg.norm(Ui, axis=1) - 1).max()))
    if pm is None:
        frame.update(modes=[], k0=None, k1=None, flip_deg=None, dJ=None,
                     k0_cost=None, k1_cost=None,
                     K_phys=0, K_all=0)
        return frame
    modes = []
    for k in pm["keep"]:
        q = pm["m"]["q"][k]
        modes.append(dict(k=int(k), R=core.Rmat(q), t=Gi @ core.v2(q),
                          J=float(pm["m"]["J"][k])))
    k0 = pm["k0"]
    others = [k for k in pm["keep"] if k != k0]
    k1 = min(others, key=lambda k: core.rot_angle(core.Rmat(pm["m"]["q"][k]), R)) \
        if others else None
    for md in modes:
        md["label"] = "k0" if md["k"] == k0 else ("k1" if md["k"] == k1 else f"k{md['k']}")
    frame.update(modes=modes, k0=k0, k1=k1,
                 k0_cost=float(pm["m"]["J"][k0]),
                 k1_cost=float(pm["m"]["J"][k1]) if k1 is not None else None,
                 flip_deg=float(np.degrees(core.rot_angle(
                     core.Rmat(pm["m"]["q"][k1]), R))) if k1 is not None else None,
                 dJ=float(pm["m"]["J"][k1] - pm["m"]["J"][k0]) if k1 is not None else None,
                 K_phys=int(len(pm["keep"])), K_all=int(len(pm["m"]["q"])))
    return frame


def all_frames(max_frames=8):
    """Exactly the 15 frames of run_two_mode.part_a, in the same order.

    P1 keeps a frame only when the physical-mode filter leaves at least two
    modes (``run_two_mode.part_a`` skips ``len(pm['keep']) < 2``); the skipped
    frame(s) are returned separately for the report.
    """
    models = {oid: load_model(oid) for oid in tm.FRAMES.values()}
    trees = {oid: cKDTree(v) for oid, v in models.items()}
    I = inst_mod.Instance(seed=SEED_ROOT)
    frames, skipped = [], []
    for fr, oid in tm.FRAMES.items():
        gt_all = json.load(open(ROOT / "test" / fr / "scene_gt.json"))
        imgs = [im for im in gt_all if gt_all[im][0]["obj_id"] == oid][:max_frames]
        for im in imgs:
            f = load_frame(fr, im, oid, trees[oid], models[oid], I)
            if f is None:
                skipped.append(dict(frame=fr, img=im, reason="no correspondences/plane"))
            elif f["K_phys"] < 2:
                skipped.append(dict(frame=fr, img=im, K_phys=f["K_phys"],
                                    reason="fewer than 2 physical modes (P1 rule)"))
            else:
                frames.append(f)
    return frames, skipped


# ------------------------------------------------------------------ outcome
def classify(frame, R, t):
    """Pose errors, physicality and nearest enumerated P1 mode for one solution."""
    v = frame["P"] @ R.T + t
    z = v[:, 2]
    rot_err = float(np.degrees(core.rot_angle(R, frame["R"])))
    trans_err = float(np.linalg.norm(t - frame["t"]))
    near = None
    if frame["modes"]:
        ds = [(float(np.degrees(core.rot_angle(R, m["R"]))), m["label"]) for m in frame["modes"]]
        d_near, near = min(ds)
    else:
        d_near = None
    if rot_err < GT_TOL_DEG:
        cat = "gt"
    elif near == "k1" and d_near is not None and d_near < GT_TOL_DEG:
        cat = "flip"
    else:
        cat = "other"
    return dict(rot_err_deg=rot_err, trans_err=trans_err,
                frac_front=float(np.mean(z > Z_FRONT)), min_z=float(z.min()),
                nearest_mode=near, ang_to_nearest_deg=d_near, outcome=cat)


def run_case(obj, pose, frame, init, R0, t0, trial=None, seed=None):
    init_cost = numpy_cost(frame["P"], frame["U"], R0, t0)
    R, t, dt, iters, status, last_err = solve_lm(obj, pose, R0, t0)
    final_cost = numpy_cost(frame["P"], frame["U"], R, t)
    o = classify(frame, R, t)
    row = dict(frame=frame["frame"], img=frame["img"], obj=frame["obj"], init=init,
               trial=trial, seed=seed,
               init_rot_err_deg=float(np.degrees(core.rot_angle(R0, frame["R"]))),
               init_trans_err=float(np.linalg.norm(t0 - frame["t"])),
               init_cost=init_cost, final_cost=final_cost,
               final_cost_theseus=2.0 * last_err,   # theseus metric is C/2
               cost_ratio=final_cost / max(init_cost, 1e-300),
               cost_over_k0=final_cost / frame["k0_cost"],
               iters=iters, status=status, time_s=dt, **o)
    return row, R, t


# -------------------------------------------------------------- verification
def fd_check(P, U, R, t, h=1e-6):
    """Central-difference check of the analytic 3N x 6 Jacobian (P3 pattern)."""
    obj, pose = build_objective(P, U, R, t, impl="vector")
    cost = obj.cost_functions["bearing"]
    J = cost.jacobians()[0][0].detach().numpy()[0]       # (3N,6), batch axis 0
    base = th.SE3(tensor=se3_from_Rt(R, t))
    Jfd = np.zeros_like(J)
    for k in range(6):
        d = np.zeros((1, 6))
        d[0, k] = h
        pose.update(base.retract(torch.as_tensor(d)).tensor)
        ep = cost.error().detach().numpy().ravel()
        d[0, k] = -h
        pose.update(base.retract(torch.as_tensor(d)).tensor)
        em = cost.error().detach().numpy().ravel()
        pose.update(se3_from_Rt(R, t))
        Jfd[:, k] = (ep - em) / (2 * h)
    return float(np.linalg.norm(J - Jfd) / max(np.linalg.norm(Jfd), 1e-12))


def verify(frames, n_rand=5, seed=7701, n_perpoint_frames=3):
    """Theseus objective == numpy cost at many poses, + analytic Jacobian check.

    The single-3N-residual construction is compared on *all* frames; the slower
    per-point construction (the P3 pattern) on the first ``n_perpoint_frames``.
    """
    rng = np.random.default_rng(seed)
    rows, jac = [], []
    for fi, frame in enumerate(frames):
        poses = [("gt", frame["R"], frame["t"]),
                 ("identity", np.eye(3), np.zeros(3))]
        for j in range(n_rand):
            poses.append((f"rand{j}", Rot.random(random_state=rng).as_matrix(),
                          rng.uniform(-1.0, 1.0, 3)))
        for name, R, t in poses:
            c_np = numpy_cost(frame["P"], frame["U"], R, t)
            obj, _ = build_objective(frame["P"], frame["U"], R, t, impl="vector")
            # theseus' error metric is 0.5 * sum ||e||^2 (error_squared_norm_fn)
            c_vec = 2.0 * float(obj.error_metric())
            c_pp = None
            if fi < n_perpoint_frames:
                objpp, _ = build_objective(frame["P"], frame["U"], R, t, impl="perpoint")
                c_pp = 2.0 * float(objpp.error_metric())
            rows.append(dict(frame=frame["frame"], img=frame["img"], pose=name,
                             cost_numpy=c_np, cost_vector=c_vec, cost_perpoint=c_pp,
                             abs_diff_vector=abs(c_vec - c_np),
                             abs_diff_perpoint=None if c_pp is None else abs(c_pp - c_np),
                             rel_diff_vector=abs(c_vec - c_np) / max(c_np, 1e-300)))
        jac.append(dict(frame=frame["frame"], img=frame["img"],
                        jac_rel_err=fd_check(frame["P"], frame["U"],
                                             frame["R"], frame["t"])))
    return rows, jac


# ------------------------------------------------------------------- report
def summary(rows):
    def med(x):
        return float(np.median(x)) if len(x) else float("nan")

    if not rows:
        return {}
    rot = [r["rot_err_deg"] for r in rows]
    tr = [r["trans_err"] for r in rows]
    return dict(
        n=len(rows), n_gt=sum(r["outcome"] == "gt" for r in rows),
        n_flip=sum(r["outcome"] == "flip" for r in rows),
        n_other=sum(r["outcome"] == "other" for r in rows),
        n_nonphysical=sum(r["frac_front"] < 0.99 for r in rows),
        rot_err_deg=dict(median=med(rot), mean=float(np.mean(rot)),
                         p90=float(np.percentile(rot, 90)), max=float(np.max(rot))),
        trans_err=dict(median=med(tr), max=float(np.max(tr))),
        final_cost=dict(median=med([r["final_cost"] for r in rows])),
        iters=dict(median=med([r["iters"] for r in rows]),
                   max=int(np.max([r["iters"] for r in rows]))),
        n_max_iters=int(sum(r["iters"] >= MAX_ITERS for r in rows)),
        n_fail_status=int(sum(r["status"] < 0 for r in rows)),
        time_s=dict(median=med([r["time_s"] for r in rows]),
                    mean=float(np.mean([r["time_s"] for r in rows]))),
    )


def fmt(x, nd=3):
    return f"{x:.{nd}g}" if isinstance(x, float) else str(x)


def write_report(path_md, path_json, payload, frames):
    L = []
    A = L.append
    A("# Theseus LM baseline on the P1 real two-mode bearing cost (BOP LINEMOD)")
    A("")
    proto = payload["protocol"]
    A(f"Cost: `C(R,t) = sum_i ||P_i^perp (R p_i + t)||^2`, `P_i^perp = I - u_i u_i^T`; "
      f"same 15 frames, same real depth-pixel bearings and nearest-mesh-vertex model "
      f"points, same RANSAC planar-face inliers as `run_two_mode.py` PART A.")
    A("")
    A(f"Theseus {proto['theseus_version']}, torch {proto['torch_version']}, "
      f"float64 CPU, `LevenbergMarquardt` + `CholeskyDenseSolver`, "
      f"max {proto['max_iters']} iters, abs/rel tol {proto['abs_tol']:.0e}/{proto['rel_tol']:.0e}. "
      f"Objective: {proto['cost_impl']}; {proto['n_trials_random']} random inits per frame "
      f"(rotation uniform, t ~ U(-1,1)^3), plus GT and identity inits. "
      f"Timings are login-node CPU, objective construction excluded from solve time.")
    A("")

    # 1 verification
    v = payload["verification"]
    A("## 1. Objective equality and analytic Jacobian")
    A("")
    A("Theseus objective value vs the numpy reference at GT, identity and "
      f"{proto['n_verify_random']} random poses per frame "
      f"({len(v['cost_rows'])} poses x 2 objective constructions):")
    A("")
    A("| construction | max abs cost diff | max rel diff |")
    A("|-|-|-|")
    A(f"| single 3N residual (used below) | {v['max_abs_diff_vector']:.3e} | "
      f"{v['max_rel_diff_vector']:.3e} |")
    A(f"| one cost per correspondence (P3 pattern) | {v['max_abs_diff_perpoint']:.3e} | "
      f"{v['max_rel_diff_perpoint']:.3e} |")
    A("")
    A("Central-difference check of the analytic 3N x 6 Jacobian "
      "(right perturbation `T <- T @ Exp(delta)`, `delta=(v,omega)`):")
    A("")
    A("| frame/img | jac rel err |")
    A("|-|-|")
    for r in v["jac"]:
        A(f"| {r['frame']}/{r['img']} | {r['jac_rel_err']:.2e} |")
    A("")
    A(f"Unit-bearing deviation `max||u_i||-1| = {v['u_norm_dev_max']:.2e}` "
      f"(so `I - u u^T` is P1's `P^perp` to machine precision).")
    A("")

    # 2 labels
    A("## 2. Mode labels used for the GT-vs-flip classification (from `two_mode.json`)")
    A("")
    A("The mode poses (`k0` = P1's global argmin among physical modes, `k1` = the "
      "other physical mode nearest the GT rotation) are re-derived with the same "
      "deterministic construction only to *label* convergence outcomes; the "
      "hard-argmin / Laplace numbers are cited in section 6, not recomputed.")
    A("")
    A("| frame/img | n_plane | K_phys | Delta J | flip deg (recomputed) | flip deg (two_mode.md) |")
    A("|-|-|-|-|-|-|")
    for r in payload["labels"]:
        A(f"| {r['frame']}/{r['img']} | {r['n_plane']} | {r['K_phys']} | "
          f"{r['dJ']:.2e} | {r['flip_deg']:.2f} | {r['flip_deg_ref']:.2f} |")
    A("")

    # 3 GT init
    A("## 3. Initialisation at GT (local convergence check)")
    A("")
    A("| frame/img | rot err (deg) | trans err (m) | final cost | iters | status | nearest P1 mode | frac z>0.05 |")
    A("|-|-|-|-|-|-|-|-|")
    for r in payload["main"]["gt"]:
        A(f"| {r['frame']}/{r['img']} | {r['rot_err_deg']:.2f} | {r['trans_err']:.4f} | "
          f"{r['final_cost']:.3e} | {r['iters']} | {r['status']} | {r['nearest_mode']} | "
          f"{r['frac_front']:.3f} |")
    s = payload["main_summary"]["gt"]
    A("")
    A(f"- median rot err {s['rot_err_deg']['median']:.2f} deg, max {s['rot_err_deg']['max']:.2f} deg; "
      f"median trans err {s['trans_err']['median']:.4f} m; "
      f"median iters {s['iters']['median']:.0f}; "
      f"gt-rate {s['n_gt']}/{s['n']}.")
    A("")

    # 4 identity
    A("## 4. Initialisation at identity (t=0, R=I)")
    A("")
    A("| frame/img | rot err (deg) | trans err (m) | final cost | iters | outcome | nearest P1 mode | frac z>0.05 |")
    A("|-|-|-|-|-|-|-|-|")
    for r in payload["main"]["identity"]:
        A(f"| {r['frame']}/{r['img']} | {r['rot_err_deg']:.2f} | {r['trans_err']:.4f} | "
          f"{r['final_cost']:.3e} | {r['iters']} | {r['outcome']} | {r['nearest_mode']} | "
          f"{r['frac_front']:.3f} |")
    s = payload["main_summary"]["identity"]
    A("")
    A(f"- gt-rate {s['n_gt']}/{s['n']}, flip {s['n_flip']}, other {s['n_other']}, "
      f"non-physical {s['n_nonphysical']}; median rot err "
      f"{s['rot_err_deg']['median']:.2f} deg.")
    A("")

    # 5 random
    A("## 5. Random initialisation (20 trials per frame)")
    A("")
    A("| frame/img | GT hit | flip | other | non-physical | median rot err (deg) | min rot err (deg) |")
    A("|-|-|-|-|-|-|-|")
    for r in payload["main"]["random_per_frame"]:
        A(f"| {r['frame']}/{r['img']} | {r['n_gt']}/{r['n']} | {r['n_flip']} | "
          f"{r['n_other']} | {r['n_nonphysical']} | {r['rot_med']:.2f} | {r['rot_min']:.2f} |")
    s = payload["main_summary"]["random"]
    A("")
    A(f"- all {s['n']} trials: gt-rate {s['n_gt']}/{s['n']} "
      f"({100.0*s['n_gt']/s['n']:.1f}%), flip {s['n_flip']}, other {s['n_other']}, "
      f"non-physical {s['n_nonphysical']} (frac z>0.05 < 0.99); median rot err "
      f"{s['rot_err_deg']['median']:.2f} deg; median time {s['time_s']['median']*1e3:.0f} ms.")
    A("")

    # 6 multistart
    m = payload["multistart"]
    A("## 6. Random-init multistart on representative frames (8 inits each)")
    A("")
    A("| frame/img | GT hits | flip | other | non-physical | rot errs (deg) |")
    A("|-|-|-|-|-|-|")
    for r in m["per_frame"]:
        A(f"| {r['frame']}/{r['img']} | {r['n_gt']}/{r['n']} | {r['n_flip']} | "
          f"{r['n_other']} | {r['n_nonphysical']} | "
          f"{', '.join(f'{x:.1f}' for x in r['rot_errs'])} |")
    A("")
    A(f"- aggregate multistart success rate (rot err < {GT_TOL_DEG:.0f} deg): "
      f"{m['n_gt']}/{m['n']} = {100.0*m['n_gt']/m['n']:.1f}%; "
      f"non-physical solutions {m['n_nonphysical']}/{m['n']}.")
    A("")

    # 7 comparison
    c = payload["comparison"]
    A("## 7. Comparison with P1's hard-argmin / Laplace numbers (cited, not recomputed)")
    A("")
    A(f"From `{c['source']}` (same 15 frames, same cost): median hard-argmin rotation "
      f"error {c['hard_rot_err_median_deg']:.2f} deg, median Laplace posterior-mean "
      f"rotation error {c['lap_rot_err_median_deg']:.2f} deg, hard argmin >10 deg in "
      f"{c['hard_fail_gt10']}/{c['frames']} frames, K=2 physical modes "
      f"{c['k2_frames']}/{c['frames']}. (Medians recomputed from `two_mode.json` "
      f"agree to {c['median_check_max_abs_diff']:.2e} deg.)")
    A("")
    A("| method | what it returns | median rot err (deg) | gt-rate |")
    A("|-|-|-|-|")
    A(f"| hard argmin (P1, cited) | global minimum among physical modes | "
      f"{c['hard_rot_err_median_deg']:.2f} | {c['hard_success']}/{c['frames']} |")
    A(f"| Laplace mean (P1, cited) | posterior mean over physical modes | "
      f"{c['lap_rot_err_median_deg']:.2f} | {c['lap_success']}/{c['frames']} |")
    A(f"| Theseus LM from GT (this work) | one local solution | "
      f"{payload['main_summary']['gt']['rot_err_deg']['median']:.2f} | "
      f"{payload['main_summary']['gt']['n_gt']}/{payload['main_summary']['gt']['n']} |")
    A(f"| Theseus LM, identity init (this work) | one local solution | "
      f"{payload['main_summary']['identity']['rot_err_deg']['median']:.2f} | "
      f"{payload['main_summary']['identity']['n_gt']}/{payload['main_summary']['identity']['n']} |")
    A(f"| Theseus LM, random init (this work) | one local solution | "
      f"{payload['main_summary']['random']['rot_err_deg']['median']:.2f} | "
      f"{payload['main_summary']['random']['n_gt']}/{payload['main_summary']['random']['n']} |")
    A(f"| Theseus LM multistart, 8 inits (this work) | best-of-8 local solutions | "
      f"{m['best_rot_err_median_deg']:.2f} | {m['n_gt']}/{m['n']} |")
    A("")
    A(c["framing"])
    A("")

    # 8 caveats
    A("## 8. Caveats")
    A("")
    for line in payload["caveats"]:
        A(f"- {line}")
    A("")
    Path(path_md).write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true",
                    help="3 frames, verification + one GT solve, no grid")
    ap.add_argument("--max-frames", type=int, default=8)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    print("[data] building the 15 real frames (same as run_two_mode PART A)", flush=True)
    frames, skipped_frames = all_frames(args.max_frames)
    print(f"[data] {len(frames)} frames (skipped {len(skipped_frames)})", flush=True)

    # cross-check the frame set and flip angles with the existing P1 artifact
    ref = {f"{r['frame']}/{r['img']}": r
           for r in json.load(open(OUT / "two_mode.json"))["part_a"]}
    assert set(ref) == {f"{f['frame']}/{f['img']}" for f in frames}, \
        "frame set differs from results/two_mode.json"
    labels = []
    for f in frames:
        r = ref[f"{f['frame']}/{f['img']}"]
        labels.append(dict(frame=f["frame"], img=f["img"], obj=f["obj"],
                           n_plane=f["n_plane"], K_phys=f["K_phys"], K_all=f["K_all"],
                           dJ=f["dJ"], flip_deg=f["flip_deg"],
                           k0_cost=f["k0_cost"], k1_cost=f["k1_cost"],
                           flip_deg_ref=float(r["flip_deg"]),
                           flip_deg_diff=abs(f["flip_deg"] - float(r["flip_deg"]))))
    print("[data] flip-angle vs two_mode.json max abs diff: "
          f"{max(l['flip_deg_diff'] for l in labels):.2e} deg", flush=True)

    verify_frames = frames[:3]
    print("[verify] objective equality + FD Jacobian on 3 frames", flush=True)
    cost_rows, jac = verify(verify_frames)

    protocol = dict(
        theseus_version=th.__version__, torch_version=torch.__version__,
        numpy_version=np.__version__, dtype="float64", device="CPU",
        solver="LevenbergMarquardt + CholeskyDenseSolver, vectorize=False",
        max_iters=MAX_ITERS, abs_tol=ABS_TOL, rel_tol=REL_TOL,
        cost_impl="single 3N-dim residual block (cross-checked against per-point costs)",
        n_points=tm.N_POINTS, plane_ransac_band_mm=tm.PLANE_THR * 1000,
        mode_enumeration_starts=tm.N_STARTS,
        n_trials_random=N_RANDOM, n_verify_random=5,
        init_random="uniform random rotation (scipy Rotation.random), t ~ U(-1,1)^3",
        seed="main random inits SeedSequence([20261127,111,0,frame_code,trial]); "
             "multistart SeedSequence([20261127,111,1,frame_code,trial])",
        gt_tolerance_deg=GT_TOL_DEG, z_front_threshold=Z_FRONT,
        frames=[f"{f['frame']}/{f['img']}" for f in frames],
        skipped_frames=skipped_frames,
        command="/mnt/storage1/syk/venvs/theseus/bin/python "
                "labs/01-laplace-qpeplayer/experiments/real/linemod/run_theseus_baseline.py",
    )

    if args.selftest:
        f = frames[0]
        obj, pose = build_objective(f["P"], f["U"], f["R"], f["t"], impl="vector")
        t0 = time.perf_counter()
        row, R, t = run_case(obj, pose, f, "gt", f["R"], f["t"])
        print(f"[selftest] gt solve {time.perf_counter()-t0:.3f}s", json.dumps(row, default=float))
        objp, posep = build_objective(f["P"], f["U"], f["R"], f["t"], impl="perpoint")
        t0 = time.perf_counter()
        rowp, Rp, tp = run_case(objp, posep, f, "gt", f["R"], f["t"])
        print(f"[selftest] gt solve perpoint {time.perf_counter()-t0:.3f}s "
              f"rot_diff={np.degrees(core.rot_angle(R, Rp)):.2e} deg", json.dumps(rowp, default=float))
        return

    # ---------------------------------------------------------------- grid
    main_rows = dict(gt=[], identity=[], random=[])
    main_pose_diff = []
    for fi, f in enumerate(frames):
        obj, pose = build_objective(f["P"], f["U"], f["R"], f["t"], impl="vector")
        # (a) GT init
        row, Rg, tg = run_case(obj, pose, f, "gt", f["R"], f["t"])
        main_rows["gt"].append(row)
        # (b) identity init
        row, Ri, ti = run_case(obj, pose, f, "identity", np.eye(3), np.zeros(3))
        main_rows["identity"].append(row)
        # (c) N random inits, distinct seeds per frame
        code = int(f["frame"]) * 100 + int(f["img"])
        for trial in range(N_RANDOM):
            ss = np.random.SeedSequence([SEED_ROOT, 111, 0, code, trial])
            rng = np.random.default_rng(ss)
            R0 = Rot.random(random_state=rng).as_matrix()
            t0 = rng.uniform(-1.0, 1.0, 3)
            row, _, _ = run_case(obj, pose, f, "random", R0, t0, trial=trial,
                                 seed=int(ss.generate_state(1)[0]))
            main_rows["random"].append(row)
        # cross-check: same GT solve with the per-point construction, frame 0 and 7
        if fi in (0, 7):
            objp, posep = build_objective(f["P"], f["U"], f["R"], f["t"], impl="perpoint")
            _, Rp, tp = run_case(objp, posep, f, "gt", f["R"], f["t"])
            main_pose_diff.append(dict(
                frame=f["frame"], img=f["img"],
                rot_diff_deg=float(np.degrees(core.rot_angle(Rg, Rp))),
                trans_diff=float(np.linalg.norm(tg - tp))))
        print(f"[grid] {f['frame']}/{f['img']}: gt rot={main_rows['gt'][-1]['rot_err_deg']:.2f} "
              f"id rot={main_rows['identity'][-1]['rot_err_deg']:.2f} "
              f"rand gt={sum(r['outcome']=='gt' for r in main_rows['random'][-N_RANDOM:])}"
              f"/{N_RANDOM} ({time.time()-t_start:.0f}s)", flush=True)

    random_per_frame = []
    for f in frames:
        rows = [r for r in main_rows["random"]
                if r["frame"] == f["frame"] and r["img"] == f["img"]]
        random_per_frame.append(dict(
            frame=f["frame"], img=f["img"], n=len(rows),
            n_gt=sum(r["outcome"] == "gt" for r in rows),
            n_flip=sum(r["outcome"] == "flip" for r in rows),
            n_other=sum(r["outcome"] == "other" for r in rows),
            n_nonphysical=sum(r["frac_front"] < 0.99 for r in rows),
            rot_med=float(np.median([r["rot_err_deg"] for r in rows])),
            rot_min=float(np.min([r["rot_err_deg"] for r in rows]))))

    # ---------------------------------------------------------- multistart
    print("[multistart] 8 inits on 5 frames", flush=True)
    ms_rows = []
    for fr, im in MULTISTART_FRAMES:
        f = next(x for x in frames if x["frame"] == fr and x["img"] == im)
        obj, pose = build_objective(f["P"], f["U"], f["R"], f["t"], impl="vector")
        code = int(fr) * 100 + int(im)
        for trial in range(N_MULTISTART):
            ss = np.random.SeedSequence([SEED_ROOT, 111, 1, code, trial])
            rng = np.random.default_rng(ss)
            R0 = Rot.random(random_state=rng).as_matrix()
            t0 = rng.uniform(-1.0, 1.0, 3)
            row, _, _ = run_case(obj, pose, f, "multistart", R0, t0, trial=trial,
                                 seed=int(ss.generate_state(1)[0]))
            ms_rows.append(row)
    ms_per_frame = []
    for fr, im in MULTISTART_FRAMES:
        rows = [r for r in ms_rows if r["frame"] == fr and r["img"] == im]
        ms_per_frame.append(dict(
            frame=fr, img=im, n=len(rows),
            n_gt=sum(r["outcome"] == "gt" for r in rows),
            n_flip=sum(r["outcome"] == "flip" for r in rows),
            n_other=sum(r["outcome"] == "other" for r in rows),
            n_nonphysical=sum(r["frac_front"] < 0.99 for r in rows),
            rot_errs=[r["rot_err_deg"] for r in rows],
            best_rot_err=float(np.min([r["rot_err_deg"] for r in rows]))))
    multistart = dict(per_frame=ms_per_frame, rows=ms_rows,
                      n=len(ms_rows),
                      n_gt=sum(r["outcome"] == "gt" for r in ms_rows),
                      n_flip=sum(r["outcome"] == "flip" for r in ms_rows),
                      n_other=sum(r["outcome"] == "other" for r in ms_rows),
                      n_nonphysical=sum(r["frac_front"] < 0.99 for r in ms_rows),
                      best_rot_err_median_deg=float(np.median(
                          [p["best_rot_err"] for p in ms_per_frame])))

    # ---------------------------------------------------------- comparison
    ref_rows = list(ref.values())
    hard = [r["hard_rot_err"] for r in ref_rows]
    lap = [r["lap_rot_err"] for r in ref_rows]
    CITED = dict(hard_rot_err_median_deg=2.06, lap_rot_err_median_deg=58.28,
                 frames=15, k2_frames=15, hard_fail_gt10=1,
                 source="labs/01-laplace-qpeplayer/experiments/real/linemod/results/two_mode.md")
    chk = max(abs(float(np.median(hard)) - CITED["hard_rot_err_median_deg"]),
              abs(float(np.median(lap)) - CITED["lap_rot_err_median_deg"]))
    assert chk < 0.01, f"two_mode.json medians disagree with two_mode.md by {chk}"
    comparison = dict(
        **CITED,
        median_check_max_abs_diff=chk,
        hard_success=sum(1 for x in hard if x < GT_TOL_DEG),
        lap_success=sum(1 for x in lap if x < GT_TOL_DEG),
        framing=(
            "**Honest framing.** Theseus is a *local* optimiser: each call returns one "
            "pose and has no notion of the enumerated mode set or of the z>0 "
            "physicality filter that P1 applies. From GT it recovers the nearest "
            "minimum (local convergence, not a baseline win). From a cold/random init "
            "it recovers whichever basin the descent enters -- on this near-tied cost "
            "that is often the P1 flip mode or a non-physical (behind-camera) pose, "
            "which the layer's mode enumeration rejects by construction. Theseus also "
            "returns no posterior, so it is not comparable to the Laplace mean as an "
            "estimator; it is the natural *no-enumeration* reference point. A negative "
            "or mixed result is reported as-is: local optimisation does not replace "
            "mode enumeration on this cost. Two concrete observations: (i) on "
            "000010/72, the frame where P1's *global* argmin is 44 deg from GT, the "
            "GT-init solve lands on the 3.8 deg mode (P1's k1 candidate) -- a local "
            "minimum that is closer to GT than the global argmin, so the GT-init row "
            "is not simply 'hard argmin'; (ii) the identity-init solve usually ends at "
            "a rotation ~70-180 deg from GT, i.e. a non-physical (behind-camera) "
            "stationary point of the unconstrained cost."),
    )

    caveats = [
        "The Theseus problem is the *full* (R,t) cost; P1's modes are minima of the "
        "reduced cost after eliminating t analytically. For fixed R, t*(R) is the "
        "linear least-squares optimum, so the mode sets coincide; the residual is not "
        "zero because the nearest-mesh-vertex correspondence is a discretisation "
        "(it is the same residual P1 optimises).",
        "The cost has no z>0 constraint, so behind-camera poses are legitimate "
        "stationary points of the Theseus objective. Rows with `frac z>0.05 < 0.99` "
        "are non-physical and correspond to poses the P1 layer discards.",
        "Random inits are deliberately hard (uniform rotation, t ~ U(-1,1)^3, so most "
        "starts are far outside the object's 0.6-1 m depth range). Their success rate "
        "measures basin capture from cold start, not detector accuracy.",
        "'GT pose' here means the BOP ground truth; the cost's own global mode is "
        "1.5-8 deg from it on most frames because of the nearest-vertex "
        "discretisation, so a GT-init solve cannot be more accurate than that mode.",
        "Solve times exclude objective construction (the 3N-residual objective is "
        "built once per frame and the pose variable is reset for each init); they are "
        "single-threaded CPU numbers on the login node.",
        "This is an external baseline on P1's *cost*, not a claim about Theseus' "
        "performance on published pipelines, and it does not include a posterior.",
    ]

    payload = dict(
        protocol=protocol,
        verification=dict(
            cost_rows=cost_rows, jac=jac,
            max_abs_diff_vector=max(r["abs_diff_vector"] for r in cost_rows),
            max_rel_diff_vector=max(r["rel_diff_vector"] for r in cost_rows),
            max_abs_diff_perpoint=max(r["abs_diff_perpoint"] for r in cost_rows),
            max_rel_diff_perpoint=max(r["abs_diff_perpoint"] / max(r["cost_numpy"], 1e-300)
                                      for r in cost_rows),
            u_norm_dev_max=max(f["u_norm_dev"] for f in frames)),
        labels=labels,
        main=dict(gt=main_rows["gt"], identity=main_rows["identity"],
                  random=main_rows["random"], random_per_frame=random_per_frame,
                  impl_pose_diff=main_pose_diff),
        main_summary={k: summary(main_rows[k]) for k in ("gt", "identity", "random")},
        multistart=multistart,
        comparison=comparison,
        caveats=caveats,
        wall_time_s=time.time() - t_start,
    )
    tag = f"_{args.tag}" if args.tag else ""
    path_json = OUT / f"theseus_baseline{tag}.json"
    path_md = OUT / f"theseus_baseline{tag}.md"
    json.dump(payload, open(path_json, "w"), indent=2, default=float)
    write_report(path_md, path_json, payload, frames)
    print(f"[done] {time.time()-t_start:.1f}s -> {path_md}", flush=True)


if __name__ == "__main__":
    main()
