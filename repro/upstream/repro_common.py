"""Shared helpers for the upstream SDPRLayers stereo-calibration reproduction.

Nothing here modifies upstream code: the idea is to reuse the *original*
``_scripts/stereo_cal.py`` and ``sdprlayers.utils.stereo_tuner`` entry points
(``get_cal_data``, ``tune_baseline``, ``get_statistics``, ...) while
  1. redirecting outputs into ``repro/results_upstream`` and
  2. instrumenting the SDP solves for tightness / corank / resource statistics.

Paths/limits follow notes/CONTEXT.md: CPU only, 8 threads.
"""

import importlib.util
import json
import os
import resource
import sys
import time

import matplotlib

matplotlib.use("Agg")  # headless

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = os.environ.get(
    "SDPR_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
)
REQ = os.path.join(ROOT, "code", "sdprlayer")
RESULTS = os.path.join(ROOT, "repro", "results_upstream")
LOGS = os.path.join(RESULTS, "logs")
FIGS = os.path.join(RESULTS, "figures")
STUBS = os.path.join(ROOT, "repro", "upstream", "stubs")

for _p in (STUBS, REQ):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def setup_env(threads=8):
    """CPU-only run: limit BLAS/OMP threads and torch threads."""
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(v, str(threads))
    os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
    torch.set_num_threads(threads)
    torch.set_default_dtype(torch.float64)


def load_stereo_cal():
    """Import the upstream ``_scripts/stereo_cal.py`` as a module."""
    path = os.path.join(REQ, "_scripts", "stereo_cal.py")
    spec = importlib.util.spec_from_file_location("stereo_cal", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_stereo_tuner():
    import sdprlayers.utils.stereo_tuner as st

    return st


# ---------------------------------------------------------------------------
# tightness telemetry: wrap SDPRLayer.check_tightness (used by every forward)
# ---------------------------------------------------------------------------
_REC = {
    "eig_ratios": [],  # lambda_1 / lambda_2 of the (primal or dual) solution
    "spe_tr": [],  # eigmin / eigmax of the certificate matrix
    "eigmin": [],
    "eigmax": [],
}


def reset_recorder():
    for k in _REC:
        _REC[k].clear()


def snapshot_recorder():
    return {k: list(v) for k, v in _REC.items()}


def _install_tightness_hook():
    from sdprlayers.layers.sdprlayer import SDPRLayer

    if getattr(SDPRLayer.check_tightness, "_dsh_hooked", False):
        return
    _st = SDPRLayer.__dict__.get("check_tightness")
    orig = getattr(_st, "__func__", _st)

    def check_tightness(X, ER_min=1e5):
        X = X.detach().cpu().numpy() if torch.is_tensor(X) else np.asarray(X)
        eigs = np.linalg.eigvalsh(X)
        abs_eigs = np.abs(eigs)
        order = np.sort(abs_eigs)
        er = float(order[-1] / order[-2]) if order[-2] > 0 else np.inf
        _REC["eig_ratios"].append(er)
        _REC["eigmin"].append(float(eigs[0]))
        _REC["eigmax"].append(float(eigs[-1]))
        _REC["spe_tr"].append(float(eigs[0] / eigs[-1]) if eigs[-1] != 0 else np.nan)
        return er > ER_min, er

    check_tightness._dsh_hooked = True
    SDPRLayer.check_tightness = staticmethod(check_tightness)


def install_hooks():
    """Install tightness instrumentation (idempotent)."""
    _install_tightness_hook()


# ---------------------------------------------------------------------------
# resource accounting
# ---------------------------------------------------------------------------
def rss_mb():
    """Peak resident set size of this process, in MiB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def cpu_model():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "unknown"


def total_mem_gb():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    return int(line.split()[1]) / 1024.0**2
    except OSError:
        pass
    return float("nan")


def host_info():
    return dict(
        cpu=cpu_model(),
        n_cores=os.cpu_count(),
        total_mem_gb=total_mem_gb(),
        torch=torch.__version__,
        numpy=np.__version__,
        cuda_available=torch.cuda.is_available(),
        omp_num_threads=os.environ.get("OMP_NUM_THREADS"),
        mkl_num_threads=os.environ.get("MKL_NUM_THREADS"),
        torch_num_threads=torch.get_num_threads(),
    )


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().numpy().tolist()
    return str(o)


def summarise_tightness(snap):
    er = np.asarray(snap["eig_ratios"], dtype=float)
    spe = np.asarray(snap["spe_tr"], dtype=float)
    if er.size == 0:
        return dict(n=0)
    return dict(
        n=int(er.size),
        eig_ratio_median=float(np.median(er)),
        eig_ratio_p05=float(np.percentile(er, 5)),
        eig_ratio_min=float(np.min(er)),
        rank1_frac_1e5=float(np.mean(er > 1e5)),
        rank1_frac_1e8=float(np.mean(er > 1e8)),
        rank1_frac_1e10=float(np.mean(er > 1e10)),
        spe_median=float(np.median(spe)),
        eigmin_max=float(np.max(snap["eigmin"])),
        eigmin_median=float(np.median(snap["eigmin"])),
    )
