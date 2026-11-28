#!/usr/bin/env python3
"""Write ``results/manifest.json``: data provenance and scene sizes.

All numbers in the manifest are measured from the local data directories; no
hand-entered counts.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import full_common as F  # noqa: E402

OUT = HERE / "results"


def main():
    manifest = {"data_root": str(F.ROOT),
                "url_template": "http://3dvision.princeton.edu/projects/2016/3DMatch/"
                                "downloads/scene-fragments/{name}.zip",
                "scenes": {}}
    for scene in F.SCENES:
        scene_dir = F.ROOT / scene
        gt = F.ROOT / f"{scene}-evaluation" / "gt.log"
        pairs = F.L.parse_gt_log(gt)
        clouds = sorted(scene_dir.glob("cloud_bin_*.ply"))
        zp = F.ROOT / f"{scene}.zip"
        ze = F.ROOT / f"{scene}-evaluation.zip"
        manifest["scenes"][scene] = dict(
            short=F.SCENE_SHORT[scene],
            new=scene in F.NEW_SCENES,
            n_fragments=len(clouds),
            n_gt_pairs=len(pairs),
            gt_log=str(gt),
            zip_bytes=zp.stat().st_size if zp.exists() else None,
            eval_zip_bytes=ze.stat().st_size if ze.exists() else None,
            has_gt_info=(F.ROOT / f"{scene}-evaluation" / "gt.info").exists(),
        )
        print(scene, manifest["scenes"][scene])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("wrote", OUT / "manifest.json")


if __name__ == "__main__":
    main()
