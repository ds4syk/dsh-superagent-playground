#!/usr/bin/env python3
"""Reproducibility audit for the P3 experiment tree.

Checks (all read-only):
1. every result JSON parses and has its expected top-level keys;
2. raw JSONL files have the expected line counts / are non-empty;
3. key result files are hashed (SHA256) into REPRO_AUDIT.json;
4. no data caches or huge binaries were accidentally committed under the
   experiment tree (the data disk is the only place for point clouds);
5. the Python/NumPy/SciPy versions and git HEAD are recorded;
6. exact-text checks for the blocked-GPU record and other audit evidence.

Large re-generatable artifacts are gitignored by design (see the repo
`.gitignore`: ``threedmatch_full/results/*raw.jsonl``, ``full_scenes.json``,
``overlap_scan.json``, ``kitti_odom/results/*raw.jsonl``).  In a fresh
worktree those files are absent.  The default mode treats an absent-but-
gitignored file as "absent-ignored" and does **not** fail, because the
committed summary (``full_scenes.md``) is the cited artifact.  Pass
``--strict`` to require them anyway (e.g. on the machine that produced them).

Run:
  /home/syk/miniconda3/envs/WM/bin/python repro_audit.py
  /home/syk/miniconda3/envs/WM/bin/python repro_audit.py --strict
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
    "real/kitti2/results/kitti2_compare.json": ["runs", "rows", "checks"],
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
    "real/inthedark/results/night17_dense_prefix.json": ["dataset", "protocol", "pairs"],
    "real/inthedark/results/night18_dense_prefix.json": ["dataset", "protocol", "pairs"],
    "real/inthedark/results/night_dense_run_000017.json": ["dataset", "protocol", "pairs"],
    "real/inthedark/results/night_dense_run_000018.json": ["dataset", "protocol", "pairs"],
    "real/inthedark/results/night_sparse_run_000017.json": ["dataset", "protocol", "pairs"],
    "real/inthedark/results/night_summary.json": ["front_end", "configs", "methods"],
    "real/kitti_more/results/kitti_more.json": ["config", "per_sequence", "pooled_all11"],
    "real/threedmatch_low_overlap/results/low_overlap.json": ["config", "tables"],
    "real/theseus/results/theseus_real.json": ["protocol", "rows", "instance_identity_check"],
    "phase5/anisotropic_grid/results/summary.json": ["accepted", "rows", "config"],
    "results/p3_140_accepted_ledger.json": ["total_accepted", "new_rounds"],
    "results/p3_extra_validation_630.json": ["total_accepted", "breakdown"],
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
    # phase-4 additions (2026-11-28): paper-consistency round log and the
    # artifacts that were previously only cited from the README
    "results/final_rounds.json": ["meta", "rounds"],
    "analysis/wilson.json": None,
    "real/inthedark/results/inthedark_rounds.json": ["rounds"],
    "real/inthedark/results/convention_calib_r2.json": ["run", "rows", "summary"],
    "real/kitti_odom/results/rounds.json": ["rounds"],
    "real/threedmatch_full/results/rounds.json": ["rounds"],
    "baselines/results/theseus.json": ["protocol", "cells"],
    "baselines/results/sdpr_forward.json": ["protocol", "cells"],
    "baselines/results/sdpr_kabsch_equiv.json": ["n_pairs", "max_abs_rot_diff_deg"],
    "baselines/results/theseus_selfcheck.json": ["jac_rel_err", "cost_rel_err"],
    "baselines/results/reference_n20.json": ["protocol", "cells"],
}
EXPECTED_JSONL = {
    "synthetic/results/main_table_extended_raw.jsonl": 14500,
    "real/threedmatch/results/threedmatch_raw.jsonl": 4800,
    "real/threedmatch/results/descriptor_raw.jsonl": 600,
    "real/threedmatch/results/n_scaling_raw.jsonl": 1280,
    "real/threedmatch_extra/results/extra_scenes_raw.jsonl": 4800,
    "real/threedmatch_low_overlap/results/low_overlap_raw.jsonl": 9200,
    "phase5/anisotropic_grid/results/ledger.jsonl": 550,
    "results/p3_extra_validation_630.jsonl": 630,
    "real/kitti/results/kitti_raw.jsonl": 2400,
    "real/kitti/results/c_sensitivity_raw.jsonl": 720,
    "systems/results/batch_gnc_raw.jsonl": 800,
    "ablations/anisotropic/results/anisotropic_raw.jsonl": 300,
    "ablations/ransac_lo/results/ransac_lo_raw.jsonl": 4800,
    "real/kitti2/results/kitti2_raw.jsonl": 2400,
    "real/threedmatch_full/results/full_scenes_raw.jsonl": 19200,
    "real/threedmatch_full/results/overlap_bins_raw.jsonl": 13800,
}
# Exact-text checks for evidence that is not a parsed JSON table.
TEXT_CHECKS = {
    "qpep_gpu/results/logs/slurm_gpu_3184_state.txt": [
        "JobState=PENDING",
        "Reason=Resources",
        "StartTime=2026-11-28T21:23:07",
        "JobState=CANCELLED",
        "scancel 3184",
    ],
    "real/threedmatch_full/results/full_scenes.md": [
        "Pooled across 8 scenes",
        "GNC-GM",
    ],
    "real/inthedark/results/inthedark_main.md": [
        "Both measured runs are **daylight**",
    ],
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


def git_ignored(rel):
    """True if the path would be ignored by git (so absence is expected)."""
    try:
        r = subprocess.run(
            ["git", "-C", str(P3), "check-ignore", "-q", str(HERE / rel)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return r.returncode == 0
    except Exception:
        return False


def main():
    strict = "--strict" in sys.argv
    report = dict(
        python=sys.version.split()[0],
        numpy=None,
        scipy=None,
        platform=platform.platform(),
        git_head=git_head(),
        cwd=os.getcwd(),
        strict=strict,
        checks={},
        hashes={},
        ignored_absent=[],
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

    def handle_missing(rel):
        if not strict and git_ignored(rel):
            report["checks"][rel] = "absent-ignored (regeneratable; summary committed)"
            report["ignored_absent"].append(rel)
            return
        report["checks"][rel] = "MISSING"
        report["ok"] = False

    for rel, keys in EXPECTED_JSON.items():
        path = HERE / rel
        if not path.exists():
            handle_missing(rel)
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
            handle_missing(rel)
            continue
        n = sum(1 for _ in open(path))
        ok = n > 0 and (n_expected is None or n == n_expected)
        report["checks"][rel] = f"ok ({n} lines)" if ok else f"BAD LINE COUNT {n} (expected {n_expected})"
        if not ok:
            report["ok"] = False
        report["hashes"][rel] = sha256(path)

    for rel, needles in TEXT_CHECKS.items():
        path = HERE / rel
        if not path.exists():
            report["checks"][rel] = "MISSING"
            report["ok"] = False
            continue
        text = path.read_text(errors="replace")
        missing = [x for x in needles if x not in text]
        report["checks"][rel] = "ok (text)" if not missing else f"MISSING TEXT: {missing[:2]}"
        if missing:
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
             f"- mode: **{'strict' if strict else 'default (gitignored regen artifacts optional)'}**",
             f"- overall: **{'PASS' if report['ok'] else 'FAIL'}**", "",
             "## Checks", "", "| file | status |", "|---|---|"]
    for k, v in report["checks"].items():
        lines.append(f"| `{k}` | {v} |")
    lines += ["", "## Ignored/regeneratable artifacts absent in this checkout", ""]
    lines += [f"- `{x}`" for x in report["ignored_absent"]] or ["- none"]
    lines += ["", "These files are gitignored by design; the committed `summary` "
              "artifact (for example `real/threedmatch_full/results/full_scenes.md`) "
              "is what the paper cites. Re-run the corresponding `run_*.py` script "
              "on the data disk to regenerate them, or run this audit with "
              "`--strict` on that machine.", ""]
    lines += ["## Large/cache files under experiments", ""]
    lines += [f"- `{x}`" for x in report["large_binaries"]] or ["- none"]
    lines += ["", "## SHA256 (key results)", ""]
    for k, v in report["hashes"].items():
        lines.append(f"- `{k}`: `{v}`")
    (HERE / "REPRO_AUDIT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:14]))
    print("wrote REPRO_AUDIT.json / REPRO_AUDIT.md")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
