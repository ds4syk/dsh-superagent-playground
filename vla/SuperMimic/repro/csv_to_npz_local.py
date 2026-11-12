"""Convert Unitree-convention motion CSVs to mjlab ``.npz`` clips, locally.

mjlab's ``mjlab.scripts.csv_to_npz`` always uploads the result to a W&B
registry. This reproduction needs the same conversion without a W&B account,
so this script reuses mjlab's ``MotionLoader`` and writes the npz files next to
the CSVs.

Example:
  uv run python repro/csv_to_npz_local.py \
    --input-glob 'motions/lafan1_csv/g1/*.csv' \
    --output-dir motions/lafan1_npz

The output fields are exactly ``MotionLoader.arrays()``:
  fps, joint_names, joint_pos, joint_vel, root_pos_w, root_quat_w,
  root_lin_vel_w, root_ang_vel_w
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np

from mjlab.scripts.csv_to_npz import MotionLoader


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--input-glob", required=True, help="glob of Unitree CSV files")
  parser.add_argument("--output-dir", required=True, help="directory for .npz clips")
  parser.add_argument("--input-fps", type=float, default=30.0)
  parser.add_argument("--output-fps", type=float, default=50.0)
  parser.add_argument("--device", default="cpu")
  args = parser.parse_args()

  files = sorted(glob.glob(os.path.expanduser(args.input_glob)))
  if not files:
    raise SystemExit(f"no CSV matched {args.input_glob}")
  os.makedirs(args.output_dir, exist_ok=True)
  print(f"[convert] {len(files)} CSV -> {args.output_dir}", flush=True)

  total_frames = 0
  for i, path in enumerate(files, 1):
    motion = MotionLoader(
      motion_file=os.path.expanduser(path),
      input_fps=args.input_fps,
      output_fps=args.output_fps,
      device=args.device,
    )
    arrays = motion.arrays()
    out = os.path.join(args.output_dir, os.path.splitext(os.path.basename(path))[0] + ".npz")
    np.savez(out, **arrays)
    total_frames += arrays["joint_pos"].shape[0]
    if i % 10 == 0 or i == len(files):
      print(f"[convert] {i}/{len(files)} clips, {total_frames} frames", flush=True)

  print(f"[convert] done: {len(files)} clips, {total_frames} frames "
        f"({total_frames / args.output_fps / 3600:.3f} h)", flush=True)


if __name__ == "__main__":
  main()
