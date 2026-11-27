# BOP ycbv: constrained PnP on real object meshes / real table planes

N=30, P_CON=16, noise=1.0px, bias=[10.0, 20.0, 30.0]px

## Unconstrained

- trials: 12
- penetration median: 0.00 mm
- rot median: 1.06 deg
- trans median: 17.07 mm
- feasible: 67%
- certified: 67%

## Soft penalty

- trials: 12
- penetration median: 0.00 mm
- rot median: 1.46 deg
- trans median: 20.00 mm
- feasible: 100%
- certified: 67%

## Projection

- trials: 12
- penetration median: 0.00 mm
- rot median: 1.05 deg
- trans median: 16.30 mm
- feasible: 100%
- certified: 75%

## ALM-QPEP

- trials: 12
- penetration median: 0.00 mm
- rot median: 1.33 deg
- trans median: 17.07 mm
- feasible: 100%
- certified: 83%

