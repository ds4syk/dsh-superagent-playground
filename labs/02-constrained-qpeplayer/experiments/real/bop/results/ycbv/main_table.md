# BOP YCB-Video: constrained PnP on real object meshes / real table planes

Controlled layer experiment: the 2D-3D correspondences are 30 model vertices
projected at the GT pose, then perturbed by pixel noise + downward pixel bias.
This is NOT a learned detector; no end-to-end perception claim is made.

N=30, P_CON=16, noise=1.0px, bias=[10.0, 20.0, 30.0]px

frames=12, max_img=2, obj_ids=[4, 3, 1], plane_calib=always, seed=20261127

trial = (frame, image, object, bias) pose; the four methods share it. 95% CI = Wilson score interval.

CERTIFICATION CAVEAT: a sound lower bound needs Shor <= J, so the one-sided flag
(J - Shor <= 1e-6) also accepts Shor > J and is an UPPER BOUND on the sound rate.
The two-sided |J - Shor| <= 1e-6 count is reported next to it. SCS is not
dual-feasible, so neither number is yet a rigorous certificate; a dual-repair
pass is required before quoting a certified rate.

## Unconstrained

- trials: 84
- penetration median: 14.09 mm
- rot median: 1.08 deg
- trans median: 17.33 mm
- feasible: 6.0%
- certified, one-sided (UPPER BOUND): 6.0% (95% CI 2.6-13.2%, k=5/84)
- certified, two-sided (necessary condition): 3.6% (95% CI 1.2-10.0%, k=3/84)
- certified + RLT, one-sided (UPPER BOUND): 4.8% (95% CI 1.9-11.6%, k=4/84)
- certified + RLT, two-sided (necessary condition): 2.4% (95% CI 0.7-8.3%, k=2/84)
- gap J-Shor: min -1.809e-04, median -8.131e-06, max +6.413e-04, negative 80/84
- gap J-Shor(RLT): min -4.908e-02, median -2.279e-06, max +2.727e+00, negative 65/84
- RLT paired gain, one-sided: -1.2 pp (0->1: 0, 1->0: 1, exact McNemar p=1)
- RLT paired gain, two-sided: -1.2 pp (0->1: 1, 1->0: 2, exact McNemar p=1)
- solve runtime (4 solvers): 9.3 s (0.11 s/trial)

## Soft penalty

- trials: 84
- penetration median: 0.00 mm
- rot median: 1.40 deg
- trans median: 29.38 mm
- feasible: 90.5%
- certified, one-sided (UPPER BOUND): 21.4% (95% CI 14.0-31.3%, k=18/84)
- certified, two-sided (necessary condition): 7.1% (95% CI 3.3-14.7%, k=6/84)
- certified + RLT, one-sided (UPPER BOUND): 21.4% (95% CI 14.0-31.3%, k=18/84)
- certified + RLT, two-sided (necessary condition): 6.0% (95% CI 2.6-13.2%, k=5/84)
- gap J-Shor: min -1.333e-04, median +3.333e-05, max +3.771e-03, negative 15/84
- gap J-Shor(RLT): min -4.904e-02, median +8.697e-05, max +2.727e+00, negative 14/84
- RLT paired gain, one-sided: +0.0 pp (0->1: 9, 1->0: 9, exact McNemar p=1)
- RLT paired gain, two-sided: -1.2 pp (0->1: 3, 1->0: 4, exact McNemar p=1)
- solve runtime (4 solvers): 9.3 s (0.11 s/trial)

## Projection

- trials: 84
- penetration median: 0.00 mm
- rot median: 1.08 deg
- trans median: 8.93 mm
- feasible: 100.0%
- certified, one-sided (UPPER BOUND): 7.1% (95% CI 3.3-14.7%, k=6/84)
- certified, two-sided (necessary condition): 4.8% (95% CI 1.9-11.6%, k=4/84)
- certified + RLT, one-sided (UPPER BOUND): 8.3% (95% CI 4.1-16.2%, k=7/84)
- certified + RLT, two-sided (necessary condition): 2.4% (95% CI 0.7-8.3%, k=2/84)
- gap J-Shor: min -1.222e-04, median +3.369e-03, max +1.628e-02, negative 2/84
- gap J-Shor(RLT): min -4.535e-02, median +3.872e-03, max +2.732e+00, negative 5/84
- RLT paired gain, one-sided: +1.2 pp (0->1: 2, 1->0: 1, exact McNemar p=1)
- RLT paired gain, two-sided: -2.4 pp (0->1: 1, 1->0: 3, exact McNemar p=0.625)
- solve runtime (4 solvers): 9.3 s (0.11 s/trial)

## ALM-QPEP

- trials: 84
- penetration median: 0.00 mm
- rot median: 1.11 deg
- trans median: 25.18 mm
- feasible: 100.0%
- certified, one-sided (UPPER BOUND): 33.3% (95% CI 24.2-43.9%, k=28/84)
- certified, two-sided (necessary condition): 9.5% (95% CI 4.9-17.7%, k=8/84)
- certified + RLT, one-sided (UPPER BOUND): 23.8% (95% CI 16.0-33.9%, k=20/84)
- certified + RLT, two-sided (necessary condition): 6.0% (95% CI 2.6-13.2%, k=5/84)
- gap J-Shor: min -1.438e-04, median +1.755e-05, max +6.510e-04, negative 22/84
- gap J-Shor(RLT): min -4.905e-02, median +3.067e-05, max +2.727e+00, negative 15/84
- RLT paired gain, one-sided: -9.5 pp (0->1: 8, 1->0: 16, exact McNemar p=0.152)
- RLT paired gain, two-sided: -3.6 pp (0->1: 3, 1->0: 6, exact McNemar p=0.508)
- solve runtime (4 solvers): 9.3 s (0.11 s/trial)

Total wall clock: 176.7 s; n_poses=28; mean plane 0.03 s, solve 0.44 s, Shor 0.68 s, Shor+RLT 0.96 s per pose; wall/trial 2.10 s.

