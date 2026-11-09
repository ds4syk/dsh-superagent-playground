"""Retry the SDPR-IS runs that failed in the full faithful Table V run.

The failures (14/50) are two kinds:
  1) `Failed to find adequate Lagrange multipliers` -- the first-order KKT
     residual of the SDP solution exceeds `kkt_tol` (SCS 1e-9 accuracy);
  2) the SDP solution is not rank-1 (`xs=None` path).

This script re-runs one failed instance per Slurm array task with settings
controlled by environment variables (so we can A/B test fixes without changing
code):
  SDPR_KKT_TOL   default 1e-5
  SDPR_SCS_EPS   default 1e-9
  SDPR_USE_DUAL  1/0 (dual vs primal DCP form)
Results -> repro/results_upstream/retry/<RETRY_TAG>/run_<idx>.json
"""
import glob
import json
import os
import pickle
import time

import repro_common as rc


def selected_indices():
    """Indices to retry: RETRY_RUNS env (comma list) overrides the failed set."""
    override = os.environ.get("RETRY_RUNS")
    if override:
        return [int(x) for x in override.split(",") if x.strip()]
    fails = []
    pat = os.path.join(rc.RESULTS, "final", "shards", "shard_*_sdpr-dq.json")
    for f in glob.glob(pat):
        for r in json.load(open(f))["records"]:
            if not r.get("ok"):
                fails.append(r["run"])
    return sorted(set(fails))


def main():
    tag = os.environ.get("RETRY_TAG", "retry")
    out = os.path.join(rc.RESULTS, "retry", tag)
    os.makedirs(out, exist_ok=True)
    task = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    fails = selected_indices()
    if task >= len(fails):
        print(f"[retry] task {task} >= n_failed {len(fails)}; nothing to do")
        return
    idx = fails[task]
    threads = int(os.environ.get("SLURM_CPUS_PER_TASK", "2"))
    rc.setup_env(threads=threads)
    sc = rc.load_stereo_cal()
    info = pickle.load(open(os.path.join(rc.RESULTS, "final",
                                         "stereo_tune_prob_data.pkl"), "rb"))
    pd = info[idx]
    cfg = dict(run=idx, kkt_tol=os.environ.get("SDPR_KKT_TOL"),
               scs_eps=os.environ.get("SDPR_SCS_EPS"),
               use_dual=os.environ.get("SDPR_USE_DUAL"), tag=tag)
    t0 = time.time()
    try:
        df = sc.tune_baseline(
            "sdpr-dq", opt_select="sgd", b_offs=0.003, N_batch=20, radius=3,
            prob_data=pd,
            term_crit={"max_iter": 150, "tol_grad_sq": 1e-5, "tol_loss": 1e-10},
        )
        res = dict(ok=True, err=float(df["params"].iloc[-1]) - 0.24,
                   iters=int(len(df)), loss=float(df["loss"].values[-1]),
                   wall=time.time() - t0)
    except Exception as e:  # noqa: BLE001 - we want to record any failure
        res = dict(ok=False, error=f"{type(e).__name__}: {e}",
                   wall=time.time() - t0)
    res.update(cfg)
    with open(os.path.join(out, f"run_{idx:03d}.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("[retry]", json.dumps(res))


if __name__ == "__main__":
    main()
