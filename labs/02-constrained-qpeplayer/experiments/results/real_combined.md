# Real-data layer results: combined table

Three real datasets, one layer protocol: pick 3D structure from the sensor,
project/propagate it to a second view to get 3D-2D correspondences, add pixel
noise and (where needed) a systematic pixel bias / wrong matches, then solve the
same constrained PnP with the four methods of the paper.  Nothing here is a
detector or a VO system; only the *layer* is measured.

| dataset | 3D source | 2D source | constraints | n instances |
|---|---|---|---|---|
| BOP LM/LM-O/YCB-V | object PLY vertices | GT-pose projection + 1 px noise + 10/20/30 px downward bias | non-penetration (16 lowest vertices) | 90 / 72 / 84 |
| KITTI odom 00/05/07 | Velodyne points | GT-motion projection + bias + outliers | depth + 10 deg roll/pitch cone | 189 (3 seq x 21 pairs x 4 settings) |
| UTIAS run_000000 | stereo ORB triangulation | GT-motion projection + bias + outliers | depth + FoV frustum | 114 (38 pairs x 3 settings) |

## Headline numbers

| dataset | metric | unconstrained | constrained (ALM) |
|---|---|---|---|
| KITTI `clean` | rot RMSE deg | 0.06 | 0.06 |
| KITTI `clean` | depth-violation rate | 0.0% | 0.0% |
| KITTI `bias6` | rot RMSE deg | 0.49 | 0.49 |
| KITTI `bias6` | depth-violation rate | 0.0% | 0.0% |
| KITTI `bias6+out50` | rot RMSE deg | 110.38 | 16.96 |
| KITTI `bias6+out50` | depth-violation rate | 93.7% | 0.0% |
| UTIAS `clean` | rot RMSE deg | 0.16 | 0.16 |
| UTIAS `clean` | FoV-violation rate | 0.0% | 0.0% |
| UTIAS `bias3` | rot RMSE deg | 0.45 | 0.45 |
| UTIAS `bias3` | FoV-violation rate | 0.0% | 0.0% |
| UTIAS `bias20+out5` | rot RMSE deg | 57.57 | 10.87 |
| UTIAS `bias20+out5` | FoV-violation rate | 42.1% | 36.8% |
