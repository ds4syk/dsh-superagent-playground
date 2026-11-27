# BOP lmo: constrained PnP on real object meshes / real table planes

N=30, P_CON=16, noise=1.0px, bias=[10.0, 20.0, 30.0]px

## Unconstrained

- trials: 18
- penetration median: 23.08 mm
- rot median: 2.16 deg
- trans median: 35.14 mm
- feasible: 0%
- certified: 0%
- certified + RLT: 0%

## Soft penalty

- trials: 18
- penetration median: 0.00 mm
- rot median: 1.88 deg
- trans median: 48.91 mm
- feasible: 83%
- certified: 17%
- certified + RLT: 33%

## Projection

- trials: 18
- penetration median: 0.00 mm
- rot median: 2.10 deg
- trans median: 30.63 mm
- feasible: 100%
- certified: 6%
- certified + RLT: 0%

## ALM-QPEP

- trials: 18
- penetration median: 0.00 mm
- rot median: 1.92 deg
- trans median: 46.51 mm
- feasible: 100%
- certified: 22%
- certified + RLT: 39%

