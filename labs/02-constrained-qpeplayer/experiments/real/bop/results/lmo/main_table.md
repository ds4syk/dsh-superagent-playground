# BOP lmo: constrained PnP on real object meshes / real table planes

N=30, P_CON=16, noise=1.0px, bias=[10.0, 20.0, 30.0]px

## Unconstrained

- trials: 18
- penetration median: 23.06 mm
- rot median: 2.09 deg
- trans median: 34.51 mm
- feasible: 0%
- certified: 0%

## Soft penalty

- trials: 18
- penetration median: 0.00 mm
- rot median: 1.89 deg
- trans median: 63.07 mm
- feasible: 89%
- certified: 17%

## Projection

- trials: 18
- penetration median: 0.00 mm
- rot median: 2.06 deg
- trans median: 28.85 mm
- feasible: 100%
- certified: 6%

## ALM-QPEP

- trials: 18
- penetration median: 0.00 mm
- rot median: 1.87 deg
- trans median: 46.51 mm
- feasible: 100%
- certified: 22%

