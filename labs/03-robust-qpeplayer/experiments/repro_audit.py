#!/usr/bin/env python3
"""Reproducibility audit for the P3 experiment tree.

Checks (all read-only):
1. every result JSON parses and has its expected top-level keys;
2. raw JSONL files have the expected line counts / are non-empty;
3. key result files are hashed (SHA256) into REPRO_AUDIT.json;
4. no data caches or huge binaries were accidentally committed under the
   experiment tree (the data disk is the only place for point clouds);
5. the Python/NumPy/SciPy versions and git HEAD are recorded.

Run:
  /home/syk/miniconda3/envs/WM/bin/python repro_audit.py
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
P3 = HERE.parent

EXPECTED_JSON = {
    "synthetic/results/main_table_extended.json": ["rows", "budget", "scaling"],
    "synthetic/results/gradient_table.json": ["settings", "bins", "sign_flips", "hessian_check"],
    "synthetic/results/e2e_table.json": ["settings", "aggregate", "paired", "gradient_check"],
    "real/modelnet/results/main_table.json": None,
    "real/modelnet10/results/main_table.json": None,
    "real/modelnet10/results/meta.json": ["dataset", "split", "n_objects"],
    "real/threedmatch/results/threedmatch.json": ["config", "rows"],
    "real/threedmatch/results/descriptor.json": ["config", "rows"],
    "real/threedmatch/results/n_scaling.json": ["config", "rows"],
    "real/threedmatch/results/verify_gt_direction.json": ["scenes"],
    "real/threedmatch_extra/results/extra_scenes.json": ["config", "rows"],
    "real/threedmatch_extra/results/verify_gt_direction_extra.json": ["scenes"],
    "real/kitti/results/kitti.json": ["config", "rows"],
    "real/kitti/results/c_sensitivity.json": ["config", "rows"],
    "real/kitti/results/n_scaling.json": ["config", "rows"],
    "real/kitti2/results/kitti2.json": ["config", "rows"],
    "real/kitti2/results/verify_kitti2.json": ["rows"],
    "e2e_torch/results/e2e_torch.json": ["settings", "gradcheck", "protocols"],
    "e2e_torch/results/gradcheck.json": ["cases", "summary"],
    "inner/results/inner_solver.json": ["meta", "correctness", "runtime", "verdict"],
    "systems/results/batch_gnc.json": ["config", "cpu", "gpu"],
    "ablations/kernels/results/kernel_ablation.json": ["config", "rows"],
    "ablations/gnc_schedule/results/gnc_schedule.json": ["config", "rows"],
    "ablations/anisotropic/results/anisotropic.json": ["meta", "rows", "correctness"],
    "ablations/breakdown/results/breakdown.json": ["config", "rows"],
    "ablations/noise/results/noise.json": ["config", "rows"],
    "ablations/ransac_lo/results/ransac_lo.json": ["config", "rows"],
    # phase-3 additions (2026-11-27)
    "real/kitti_stereo/results/kitti_stereo_main.json": ["settings", "pairs", "records", "summary"],
    "real/kitti_odom/results/verify_kitti_odom.json": ["summary", "pairs"],
    "real/inthedark/results/inthedark_dense.json": ["dataset", "protocol", "pairs"],
    "real/inthedark/results/inthedark_sparse.json": ["dataset", "protocol", "pairs"],
    "real/inthedark/results/inthedark_run12_dense.json": ["dataset", "protocol", "pairs"],
    "real/inthedark/results/verify_run_000002.json": ["checks", "pairs"],
    "real/inthedark/results/verify_run_000012.json": ["checks", "pairs"],
    "../results/baselines.json": ["protocol", "methods", "cells"],
    "real/kitti_odom/results/raw_drive_motion.json": ["2011_09_26_drive_0001_sync", "2011_09_26_drive_0002_sync"],
    "real/threedmatch_full/results/manifest.json": ["data_root", "url_template", "scenes"],
    "real/threedmatch_full/results/verify_gt_direction_full.json": ["tau", "n_sample", "scenes"],
    "real/threedmatch_full/results/pairs_redkitchen.json": ["scene", "selected", "n_all"],
    "real/threedmatch_full/results/pairs_sun3d-home_at.json": ["scene", "selected", "n_all"],
    "real/threedmatch_full/results/pairs_sun3d-home_md.json": ["scene", "selected", "n_all"],
    "real/threedmatch_full/results/pairs_sun3d-hotel_uc.json": ["scene", "selected", "n_all"],
    "qpep_gpu/results/qpep_gpu.json": ["meta", "rows", "errors"],
    "real/kitti_odom/results/kitti_odom.json": ["config", "per_sequence", "pooled"],
    "real/threedmatch_full/results/full_scenes.json": ["config", "per_scene", "pooled"],
    "real/threedmatch_full/results/overlap_bins.json": ["config", "tables", "advantage"],
    "real/threedmatch_full/results/overlap_dependence.json": ["config", "cells", "interaction"],
    "results/phase3_rounds.json": ["meta", "rounds"],
}
EXPECTED_JSONL = {
    "synthetic/results/main_table_extended_raw.jsonl": 14500,
    "real/threedmatch/results/threedmatch_raw.jsonl": 4800,
    "real/threedmatch/results/descriptor_raw.jsonl": 600,
    "real/threedmatch/results/n_scaling_raw.jsonl": 1280,
    "real/threedmatch_extra/results/extra_scenes_raw.jsonl": 4800,
    "real/kitti/results/kitti_raw.jsonl": 2400,
    "real/kitti/results/c_sensitivity_raw.jsonl": 720,
    "systems/results/batch_gnc_raw.jsonl": 800,
    "ablations/anisotropic/results/anisotropic_raw.jsonl": 300,
    "ablations/ransac_lo/results/ransac_lo_raw.jsonl": 4800,
    "real/kitti2/results/kitti2_raw.jsonl": 2400,
    "real/threedmatch_full/results/full_scenes_raw.jsonl": 19200,
    "real/threedmatch_full/results/overlap_bins_raw.jsonl": 13800,
}
MAX_BINARY_BYTES = 20 * 1024 * 1024


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head():
    try:
        return subprocess.check_output(
            ["git", "-C", str(P3), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def main():
    report = dict(
        python=sys.version.split()[0],
        numpy=None,
        scipy=None,
        platform=platform.platform(),
        git_head=git_head(),
        cwd=os.getcwd(),
        checks={},
        hashes={},
        large_binaries=[],
        ok=True,
    )
    try:
        import numpy
        report["numpy"] = numpy.__version__
    except Exception:
        pass
    try:
        import scipy
        report["scipy"] = scipy.__version__
    except Exception:
        pass

    for rel, keys in EXPECTED_JSON.items():
        path = HERE / rel
        if not path.exists():
            report["checks"][rel] = "MISSING"
            report["ok"] = False
            continue
        try:
            d = json.loads(path.read_text())
        except Exception as e:
            report["checks"][rel] = f"JSON ERROR: {e}"
            report["ok"] = False
            continue
        if keys is None:
            ok = isinstance(d, list) and len(d) > 0
        else:
            ok = isinstance(d, dict) and all(k in d for k in keys)
        report["checks"][rel] = "ok" if ok else f"BAD SCHEMA (keys={list(d)[:6] if isinstance(d, dict) else type(d).__name__})"
        if not ok:
            report["ok"] = False
        report["hashes"][rel] = sha256(path)

    for rel, n_expected in EXPECTED_JSONL.items():
        path = HERE / rel
        if not path.exists():
            report["checks"][rel] = "MISSING"
            report["ok"] = False
            continue
        n = sum(1 for _ in open(path))
        ok = n > 0 and (n_expected is None or n == n_expected)
        report["checks"][rel] = f"ok ({n} lines)" if ok else f"BAD LINE COUNT {n} (expected {n_expected})"
        if not ok:
            report["ok"] = False
        report["hashes"][rel] = sha256(path)

    for p in HERE.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".npy", ".npz", ".ply", ".bin", ".zip"}:
            report["large_binaries"].append(str(p.relative_to(HERE)))
            report["ok"] = False
        if p.is_file() and p.stat().st_size > MAX_BINARY_BYTES:
            report["large_binaries"].append(f"{p.relative_to(HERE)} ({p.stat().st_size} bytes)")
            report["ok"] = False

    (HERE / "REPRO_AUDIT.json").write_text(json.dumps(report, indent=2))
    lines = ["# P3 reproducibility audit", "",
             f"- Python {report['python']} / NumPy {report['numpy']} / SciPy {report['scipy']}",
             f"- git HEAD `{report['git_head']}`",
             f"- cwd `{report['cwd']}`",
             f"- overall: **{'PASS' if report['ok'] else 'FAIL'}**", "",
             "## Checks", "", "| file | status |", "|---|---|"]
    for k, v in report["checks"].items():
        lines.append(f"| `{k}` | {v} |")
    lines += ["", "## Large/cache files under experiments", ""]
    lines += [f"- `{x}`" for x in report["large_binaries"]] or ["- none"]
    lines += ["", "## SHA256 (key results)", ""]
    for k, v in report["hashes"].items():
        lines.append(f"- `{k}`: `{v}`")
    (HERE / "REPRO_AUDIT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:12]))
    print("wrote REPRO_AUDIT.json / REPRO_AUDIT.md")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
