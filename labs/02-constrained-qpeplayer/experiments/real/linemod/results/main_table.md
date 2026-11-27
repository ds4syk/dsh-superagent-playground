# BOP LINEMOD: constrained PnP on real object meshes / real table planes

Controlled layer experiment: the 2D-3D correspondences are 30 model vertices
projected at the GT pose, then perturbed by pixel noise + downward pixel bias.
This is NOT a learned detector; no end-to-end perception claim is made.

N=30, P_CON=16, noise=1.0px, bias=[10.0, 20.0, 30.0]px

frames=15, max_img=2, obj_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], plane_calib=always, seed=20261127

trial = (frame, image, object, bias) pose; the four methods share it. 95% CI = Wilson score interval.

CERTIFICATION CAVEAT: a sound lower bound needs Shor <= J, so the one-sided flag
(J - Shor <= 1e-6) also accepts Shor > J and is an UPPER BOUND on the sound rate.
The two-sided |J - Shor| <= 1e-6 count is reported next to it. SCS is not
dual-feasible, so neither number is yet a rigorous certificate; a dual-repair
pass is required before quoting a certified rate.

## Unconstrained

- trials: 90
- penetration median: 11.26 mm
- rot median: 2.08 deg
- trans median: 35.63 mm
- feasible: 13.3%
- certified, one-sided (UPPER BOUND): 13.3% (95% CI 7.8-21.9%, k=12/90)
- certified, two-sided (necessary condition): 8.9% (95% CI 4.6-16.6%, k=8/90)
- certified + RLT, one-sided (UPPER BOUND): 10.0% (95% CI 5.4-17.9%, k=9/90)
- certified + RLT, two-sided (necessary condition): 8.9% (95% CI 4.6-16.6%, k=8/90)
- gap J-Shor: min -3.725e-02, median -3.638e-06, max +1.268e-10, negative 84/90
- gap J-Shor(RLT): min -1.151e-03, median -5.208e-07, max +2.579e+00, negative 59/90
- RLT paired gain, one-sided: -3.3 pp (0->1: 0, 1->0: 3, exact McNemar p=0.25)
- RLT paired gain, two-sided: +0.0 pp (0->1: 3, 1->0: 3, exact McNemar p=1)
- solve runtime (4 solvers): 8.7 s (0.10 s/trial)

## Soft penalty

- trials: 90
- penetration median: 0.00 mm
- rot median: 2.13 deg
- trans median: 39.81 mm
- feasible: 88.9%
- certified, one-sided (UPPER BOUND): 26.7% (95% CI 18.6-36.6%, k=24/90)
- certified, two-sided (necessary condition): 16.7% (95% CI 10.4-25.7%, k=15/90)
- certified + RLT, one-sided (UPPER BOUND): 23.3% (95% CI 15.8-33.1%, k=21/90)
- certified + RLT, two-sided (necessary condition): 12.2% (95% CI 7.0-20.6%, k=11/90)
- gap J-Shor: min -3.718e-02, median +1.438e-05, max +1.809e-02, negative 11/90
- gap J-Shor(RLT): min -1.125e-03, median +7.337e-05, max +2.579e+00, negative 17/90
- RLT paired gain, one-sided: -3.3 pp (0->1: 8, 1->0: 11, exact McNemar p=0.648)
- RLT paired gain, two-sided: -4.4 pp (0->1: 3, 1->0: 7, exact McNemar p=0.344)
- solve runtime (4 solvers): 8.7 s (0.10 s/trial)

## Projection

- trials: 90
- penetration median: 0.00 mm
- rot median: 2.05 deg
- trans median: 30.80 mm
- feasible: 100.0%
- certified, one-sided (UPPER BOUND): 15.6% (95% CI 9.5-24.4%, k=14/90)
- certified, two-sided (necessary condition): 10.0% (95% CI 5.4-17.9%, k=9/90)
- certified + RLT, one-sided (UPPER BOUND): 15.6% (95% CI 9.5-24.4%, k=14/90)
- certified + RLT, two-sided (necessary condition): 10.0% (95% CI 5.4-17.9%, k=9/90)
- gap J-Shor: min -2.538e-02, median +6.545e-04, max +3.687e-02, negative 8/90
- gap J-Shor(RLT): min -2.213e-04, median +2.472e-03, max +2.579e+00, negative 11/90
- RLT paired gain, one-sided: +0.0 pp (0->1: 4, 1->0: 4, exact McNemar p=1)
- RLT paired gain, two-sided: +0.0 pp (0->1: 3, 1->0: 3, exact McNemar p=1)
- solve runtime (4 solvers): 8.7 s (0.10 s/trial)

## ALM-QPEP

- trials: 90
- penetration median: 0.00 mm
- rot median: 1.97 deg
- trans median: 37.66 mm
- feasible: 100.0%
- certified, one-sided (UPPER BOUND): 33.3% (95% CI 24.5-43.6%, k=30/90)
- certified, two-sided (necessary condition): 18.9% (95% CI 12.1-28.2%, k=17/90)
- certified + RLT, one-sided (UPPER BOUND): 31.1% (95% CI 22.5-41.3%, k=28/90)
- certified + RLT, two-sided (necessary condition): 14.4% (95% CI 8.6-23.2%, k=13/90)
- gap J-Shor: min -3.718e-02, median +7.948e-06, max +3.990e-04, negative 16/90
- gap J-Shor(RLT): min -1.125e-03, median +1.389e-05, max +2.579e+00, negative 21/90
- RLT paired gain, one-sided: -2.2 pp (0->1: 12, 1->0: 14, exact McNemar p=0.845)
- RLT paired gain, two-sided: -4.4 pp (0->1: 4, 1->0: 8, exact McNemar p=0.388)
- solve runtime (4 solvers): 8.7 s (0.10 s/trial)

Total wall clock: 186.7 s; n_poses=30; mean plane 0.05 s, solve 0.39 s, Shor 0.66 s, Shor+RLT 0.93 s per pose; wall/trial 2.07 s.

