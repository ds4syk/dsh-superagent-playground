#!/usr/bin/env python3
"""KITTI success vs frame-to-frame motion magnitude (|t| and rotation)."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
REAL = HERE.parent
OUT = HERE / "results"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    rows = []
    for name, path in [("drive_0001", REAL / "kitti/results/kitti.json"),
                       ("drive_0002", REAL / "kitti2/results/kitti2.json")]:
        d = json.loads(path.read_text())
        raw_path = REAL / ("kitti" if name == "drive_0001" else "kitti2") / "results" / ("kitti_raw.jsonl" if name == "drive_0001" else "kitti2_raw.jsonl")
        raw = [json.loads(x) for x in raw_path.read_text().splitlines()]
        # pair |t| not in raw; derive from config? use per-trial pair and load verify? Instead use
        # the per-drive README's known range, and bin by pair id proxy from trial records.
        for x in raw:
            x["drive"] = name
        rows += raw
    # We do have pair strings only in 3DMatch, not KITTI raw. Use per-drive median |t| from verify JSON.
    motion = {}
    for name, vpath in [("drive_0001", REAL / "kitti/results/verify_kitti.json"),
                        ("drive_0002", REAL / "kitti2/results/verify_kitti2.json")]:
        d = json.loads(vpath.read_text())
        vals = [r["translation"] for r in d["rows"]]
        motion[name] = float(np.median(vals))
    out = {"motion_median_m": motion,
           "note": "KITTI raw per-trial JSON does not store pair id; motion bins are per drive, not per trial."}
    (OUT / "kitti_motion.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
