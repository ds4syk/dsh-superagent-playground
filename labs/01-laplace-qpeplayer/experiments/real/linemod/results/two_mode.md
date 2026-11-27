# P1 real-data two-mode experiment (BOP LINEMOD eggbox/glue)

> **Correction (2026-11-27 phase-4).** The `Laplace rot err` column and the summary line below use the *unprojected* 9-vector mean, which is not a rotation; the valid projected chordal mean is in `pose_error.md` (median **44.24 deg** vs the unprojected 58.28 deg; bias direction unchanged).

tau=0.01, plane RANSAC band=2.5 mm, n_points=900, mode enumeration n_starts=384, MC n=400, timings on login-node CPU.

## A. Real bimodal bearing cost on the dominant planar face

Real depth pixels -> bearings; nearest real mesh vertex -> 3-D model point; RANSAC selects the dominant planar face. Modes behind the camera (z<=0) are rejected. The true pose is the global minimum; `w_wrong` is the Laplace weight of the flip mode.

| frame/img | obj | n_plane | n_phys | K_all | Delta J | flip deg | w_wrong | hard rot err (deg) | Laplace rot err (deg) | min det H | fwd ms |
|-|-|-|-|-|-|-|-|-|-|-|-|
| 000010/15 | 10 | 558 | 2 | 4 | 9.58e-04 | 69.3 | 0.483 | 1.47 | 46.56 | 2.33e+00 | 61.8 |
| 000010/18 | 10 | 474 | 2 | 4 | 1.25e-03 | 91.9 | 0.472 | 0.51 | 59.17 | 4.32e+00 | 50.3 |
| 000010/40 | 10 | 482 | 2 | 4 | 9.71e-04 | 69.1 | 0.482 | 2.20 | 46.41 | 1.21e+00 | 56.7 |
| 000010/60 | 10 | 452 | 2 | 4 | 1.69e-03 | 104.0 | 0.459 | 0.96 | 64.56 | 6.12e+00 | 47.8 |
| 000010/66 | 10 | 460 | 2 | 4 | 7.00e-04 | 88.4 | 0.488 | 2.29 | 58.28 | 2.58e+00 | 54.2 |
| 000010/72 | 10 | 501 | 2 | 4 | 1.25e-04 | 3.8 | 0.504 | 44.46 | 31.02 | 1.87e-01 | 149.0 |
| 000010/80 | 10 | 508 | 2 | 4 | 5.11e-04 | 54.7 | 0.501 | 1.08 | 37.98 | 5.14e-01 | 70.0 |
| 000010/90 | 10 | 474 | 2 | 4 | 4.13e-04 | 54.8 | 0.497 | 3.49 | 37.96 | 3.42e-01 | 73.6 |
| 000011/20 | 11 | 413 | 2 | 4 | 2.07e-03 | 69.1 | 0.608 | 8.14 | 52.80 | 6.09e-03 | 392.4 |
| 000011/24 | 11 | 423 | 2 | 4 | 1.65e-03 | 131.1 | 0.457 | 1.37 | 75.98 | 4.45e+00 | 50.8 |
| 000011/26 | 11 | 439 | 2 | 4 | 1.90e-03 | 126.9 | 0.450 | 2.06 | 73.75 | 4.14e+00 | 47.2 |
| 000011/32 | 11 | 474 | 2 | 4 | 2.10e-03 | 123.9 | 0.451 | 1.78 | 72.70 | 6.35e+00 | 53.8 |
| 000011/33 | 11 | 365 | 2 | 4 | 1.49e-03 | 149.6 | 0.460 | 1.62 | 81.79 | 4.77e+00 | 44.1 |
| 000011/40 | 11 | 428 | 2 | 4 | 2.56e-03 | 106.0 | 0.457 | 2.77 | 65.39 | 1.44e-01 | 113.2 |
| 000011/49 | 11 | 347 | 2 | 4 | 2.10e-03 | 83.7 | 0.528 | 4.95 | 58.11 | 1.25e-02 | 340.9 |

- frames: 15; physical modes K=2 in 15/15; median Delta J = 1.49e-03; median w_wrong = 0.482.
- hard argmin rotation error: median 2.06 deg; Laplace mean rotation error: median 58.28 deg.
- hard argmin fails (>10 deg) in 1/15 frames: the true pose is the global minimum only when the mesh/depth residual of the flip mode exceeds it. The Laplace mean is strongly biased toward the flip mode (median ~58 deg), i.e. on this real near-tie the posterior mean is **not** an accuracy win over hard argmin; it trades accuracy for smoothness.

## B. Controlled transverse crossing on real geometry (two labelled hypotheses)

J_theta = (1-theta) J_true-labelling + theta J_symmetry-labelling; each hypothesis eliminates its **own** translation, so the two wells are not forced to share a compromise t. theta=1/2 is an exact tie by the theta <-> 1-theta symmetry of the construction. Gradient is d(Laplace mean pose)/d(theta), checked against central FD of the layer output.

| theta | Delta J | w(A) | Laplace grad rel | MC grad rel | |dpose| | |FD| | det H (min) | ms |
|-|-|-|-|-|-|-|-|-|
| 0.3500 | +1.64e-05 | 0.50031 | 2.59e-10 | 9.51e-03 | 1.129e-02 | 1.129e-02 | 2.47e+00 | 79.0 |
| 0.4200 | +7.34e-06 | 0.50014 | 9.50e-10 | 5.35e-03 | 1.125e-02 | 1.125e-02 | 2.47e+00 | 64.6 |
| 0.4600 | +2.20e-06 | 0.50004 | 8.61e-10 | 8.94e-05 | 1.123e-02 | 1.123e-02 | 2.47e+00 | 56.9 |
| 0.4790 | -2.22e-07 | 0.50000 | 7.82e-10 | 1.68e-01 | 1.122e-02 | 1.122e-02 | 2.47e+00 | 57.9 |
| 0.4900 | -1.62e-06 | 0.49997 | 1.91e-10 | 2.04e-02 | 1.121e-02 | 1.121e-02 | 2.47e+00 | 60.9 |
| 0.5000 | -2.89e-06 | 0.49995 | 1.76e-09 | 5.09e-02 | 1.121e-02 | 1.121e-02 | 2.47e+00 | 55.2 |
| 0.5100 | -4.16e-06 | 0.49993 | 5.85e-10 | 4.93e-03 | 1.120e-02 | 1.120e-02 | 2.47e+00 | 56.7 |
| 0.5200 | -5.43e-06 | 0.49990 | 7.02e-10 | 2.05e-02 | 1.119e-02 | 1.119e-02 | 2.47e+00 | 72.8 |
| 0.5400 | -7.95e-06 | 0.49986 | 3.25e-10 | 3.59e-02 | 1.118e-02 | 1.118e-02 | 2.47e+00 | 33.9 |
| 0.5800 | -1.30e-05 | 0.49976 | 9.39e-10 | 2.59e-02 | 1.116e-02 | 1.116e-02 | 2.47e+00 | 34.0 |
| 0.6500 | -2.17e-05 | 0.49960 | 4.03e-10 | 1.10e-02 | 1.112e-02 | 1.112e-02 | 2.47e+00 | 34.3 |

Notes:
- Both crossings are **transverse** (Delta J changes sign): on the real frame the tie is at theta* ~ 0.475, not exactly 0.5, because the two labelled costs are not exactly symmetric after resampling; the two minima stay within ~2 deg of their own labelled poses.
- `core.dlogdetH_dtheta` (closed-form third-derivative contraction) was **buggy** before 2026-11-27: `qpep_core.dE` had wrong entries at rows 1-2 of d1/d2/d3, so the analytic d log det H disagreed with the validated `dlogdet_exact`/FD by O(1). After fixing `dE` it matches to 1e-16; the Part B gradient check exercises exactly that path.
- Hard argmin has a jump at theta=1/2; FD of the hard map is undefined there, so its gradient entry is omitted rather than faked.
- Part A is the honest real-data statement: a genuine two-mode cost exists, but the true pose is the global minimum, so hard argmin is correct and the Laplace posterior is biased.
