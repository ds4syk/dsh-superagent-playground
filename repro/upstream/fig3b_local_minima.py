"""Fig. 3b: fraction of random initialisations that converge to a local minimum.

Upstream does this in ``_scripts/stereo_cal.py::find_local_minima(N_inits=100)``
+ ``intialization_plots()``: 100 random camera poses on a 3 m ball are used as
Gauss-Newton initialisations for Problem (P6); the paper reports that roughly 60%
converge to poor local minima (Fig. 3b).

This driver calls the upstream functions unchanged (Theseus Gauss-Newton,
abs 1e-10 / rel 1e-8 / max 300, step_size 0.2) and additionally saves the
numbers + figures into repro/results_upstream.

Run: python fig3b_local_minima.py [--n-inits 100]
"""

import argparse
import json
import os
import pickle
import shutil

import numpy as np

import repro_common as rc


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-inits", type=int, default=100)
    p.add_argument("--local-threshold", type=float, default=5.0,
                   help="upstream: |loss - min loss| > 5 => local minimum")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--reuse", action="store_true", help="reuse existing local-min pkl/figures")
    return p.parse_args()


def main():
    args = parse_args()
    rc.setup_env(args.threads)
    import theseus_compat

    theseus_compat.install()
    cal = rc.load_stereo_cal()
    from sdprlayers.utils import plot_tools

    plot_tools.fig_dir = rc.FIGS
    os.makedirs(rc.FIGS, exist_ok=True)

    # upstream generators (writes _scripts/outputs/stereo_cal_local_min_init.pkl);
    # reuse the pkl if a previous invocation already produced it (the GN batch
    # search over 100 initialisations is expensive)
    out_pkl = os.path.join(rc.REQ, "_scripts", "outputs", "stereo_cal_local_min_init.pkl")
    if not (os.path.exists(out_pkl) and args.reuse):
        cal.find_local_minima(N_inits=args.n_inits, store_data=True)
    assert os.path.exists(out_pkl), f"missing {out_pkl}"
    if not args.reuse:
        cal.intialization_plots()  # saves stereo_cal_local_min_init.png / stereo_setup.png

    with open(out_pkl, "rb") as f:
        data = pickle.load(f)
    n_local = int(len(data["r_p0s_init_l"]))
    n_global = int(len(data["r_p0s_init_g"]))
    n_total = n_local + n_global
    frac = n_local / max(n_total, 1)
    result = dict(
        n_inits=args.n_inits,
        n_used=int(n_total),
        n_local_minima=n_local,
        n_global_minima=n_global,
        local_minima_fraction=frac,
        paper_fraction=0.60,
        local_threshold=args.local_threshold,
        host=rc.host_info(),
    )
    rc.write_json(os.path.join(rc.RESULTS, "fig3b_local_minima.json"), result)

    # copy the upstream figures into our results dir
    for name in ("stereo_cal_local_min_init.png", "stereo_setup.png"):
        src = os.path.join(rc.FIGS, name)
        dst = os.path.join(rc.FIGS, name.replace("stereo_cal_local_min_init", "fig3b_local_min_init"))
        if os.path.exists(src) and os.path.abspath(src) != os.path.abspath(dst):
            shutil.copyfile(src, dst)
    # also store the raw initialisation data (small) for later replotting
    with open(os.path.join(rc.RESULTS, "fig3b_data.pkl"), "wb") as f:
        pickle.dump(
            dict(
                losses=np.asarray(data.get("losses", [])),
                n_local=n_local,
                n_global=n_global,
            ),
            f,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
