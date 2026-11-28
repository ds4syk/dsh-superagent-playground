# BOP LM-O: constrained PnP on real object meshes / real table planes

Controlled layer experiment: the 2D-3D correspondences are 30 model vertices
projected at the GT pose, then perturbed by pixel noise + downward pixel bias.
This is NOT a learned detector; no end-to-end perception claim is made.

N=30, P_CON=16, noise=1.0px, bias=[10.0, 20.0, 30.0]px

frames=1, max_img=8, obj_ids=[8, 12, 5], plane_calib=always, seed=20261127

trial = (frame, image, object, bias) pose; the four methods share it. 95% CI = Wilson score interval.

CERTIFICATION CAVEAT: a sound lower bound needs Shor <= J, so the one-sided flag
(J - Shor <= 1e-6) also accepts Shor > J and is an UPPER BOUND on the sound rate.
The two-sided |J - Shor| <= 1e-6 count is reported next to it. SCS is not
dual-feasible, so neither number is yet a rigorous certificate; a dual-repair
pass is required before quoting a certified rate.

## Unconstrained

- trials: 72
- penetration median: 14.93 mm
- rot median: 2.07 deg
- trans median: 34.69 mm
- feasible: 6.9%
- certified, one-sided (UPPER BOUND): 6.9% (95% CI 3.0-15.2%, k=5/72)
- certified, two-sided (necessary condition): 6.9% (95% CI 3.0-15.2%, k=5/72)
- certified + RLT, one-sided (UPPER BOUND): 4.2% (95% CI 1.4-11.5%, k=3/72)
- certified + RLT, two-sided (necessary condition): 2.8% (95% CI 0.8-9.6%, k=2/72)
- gap J-Shor: min -2.005e-04, median -7.269e-06, max +1.840e-02, negative 68/72
- gap J-Shor(RLT): min -7.420e-02, median -6.797e-07, max +3.298e+00, negative 48/72
- RLT paired gain, one-sided: -2.8 pp (0->1: 0, 1->0: 2, exact McNemar p=0.5)
- RLT paired gain, two-sided: -4.2 pp (0->1: 0, 1->0: 3, exact McNemar p=0.25)
- solve runtime (4 solvers): 6.8 s (0.09 s/trial)

## Soft penalty

- trials: 72
- penetration median: 0.00 mm
- rot median: 1.90 deg
- trans median: 39.91 mm
- feasible: 91.7%
- certified, one-sided (UPPER BOUND): 26.4% (95% CI 17.6-37.6%, k=19/72)
- certified, two-sided (necessary condition): 9.7% (95% CI 4.8-18.7%, k=7/72)
- certified + RLT, one-sided (UPPER BOUND): 30.6% (95% CI 21.1-42.0%, k=22/72)
- certified + RLT, two-sided (necessary condition): 4.2% (95% CI 1.4-11.5%, k=3/72)
- gap J-Shor: min -1.990e-04, median +1.753e-05, max +1.840e-02, negative 15/72
- gap J-Shor(RLT): min -7.416e-02, median +1.806e-05, max +3.298e+00, negative 22/72
- RLT paired gain, one-sided: +4.2 pp (0->1: 16, 1->0: 13, exact McNemar p=0.711)
- RLT paired gain, two-sided: -5.6 pp (0->1: 1, 1->0: 5, exact McNemar p=0.219)
- solve runtime (4 solvers): 6.8 s (0.09 s/trial)

## Projection

- trials: 72
- penetration median: 0.00 mm
- rot median: 2.04 deg
- trans median: 31.29 mm
- feasible: 100.0%
- certified, one-sided (UPPER BOUND): 8.3% (95% CI 3.9-17.0%, k=6/72)
- certified, two-sided (necessary condition): 6.9% (95% CI 3.0-15.2%, k=5/72)
- certified + RLT, one-sided (UPPER BOUND): 11.1% (95% CI 5.7-20.4%, k=8/72)
- certified + RLT, two-sided (necessary condition): 2.8% (95% CI 0.8-9.6%, k=2/72)
- gap J-Shor: min -1.068e-04, median +1.238e-03, max +2.236e-02, negative 4/72
- gap J-Shor(RLT): min -7.217e-02, median +2.077e-03, max +3.298e+00, negative 7/72
- RLT paired gain, one-sided: +2.8 pp (0->1: 5, 1->0: 3, exact McNemar p=0.727)
- RLT paired gain, two-sided: -4.2 pp (0->1: 0, 1->0: 3, exact McNemar p=0.25)
- solve runtime (4 solvers): 6.8 s (0.09 s/trial)

## ALM-QPEP

- trials: 72
- penetration median: 0.00 mm
- rot median: 1.99 deg
- trans median: 38.14 mm
- feasible: 100.0%
- certified, one-sided (UPPER BOUND): 30.6% (95% CI 21.1-42.0%, k=22/72)
- certified, two-sided (necessary condition): 12.5% (95% CI 6.7-22.1%, k=9/72)
- certified + RLT, one-sided (UPPER BOUND): 33.3% (95% CI 23.5-44.8%, k=24/72)
- certified + RLT, two-sided (necessary condition): 4.2% (95% CI 1.4-11.5%, k=3/72)
- gap J-Shor: min -1.995e-04, median +1.531e-05, max +1.840e-02, negative 17/72
- gap J-Shor(RLT): min -7.417e-02, median +1.403e-05, max +3.298e+00, negative 22/72
- RLT paired gain, one-sided: +2.8 pp (0->1: 18, 1->0: 16, exact McNemar p=0.864)
- RLT paired gain, two-sided: -8.3 pp (0->1: 1, 1->0: 7, exact McNemar p=0.0703)
- solve runtime (4 solvers): 6.8 s (0.09 s/trial)

Total wall clock: 154.4 s; n_poses=24; mean plane 0.09 s, solve 0.38 s, Shor 0.68 s, Shor+RLT 1.03 s per pose; wall/trial 2.14 s.

