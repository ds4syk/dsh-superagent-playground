# BOP LINEMOD: constrained PnP on real object meshes / real table planes

N=30, P_CON=16, noise=1.0px, bias=[10.0, 20.0, 30.0]px

## Unconstrained

- trials: 18
- penetration median: 4.76 mm
- rot median: 2.16 deg
- trans median: 36.82 mm
- feasible: 39%
- certified: 33%

## Soft penalty

- trials: 18
- penetration median: 0.00 mm
- rot median: 2.16 deg
- trans median: 38.55 mm
- feasible: 94%
- certified: 50%

## Projection

- trials: 18
- penetration median: 0.00 mm
- rot median: 2.03 deg
- trans median: 33.73 mm
- feasible: 100%
- certified: 33%

## ALM-QPEP

- trials: 18
- penetration median: 0.00 mm
- rot median: 2.11 deg
- trans median: 37.69 mm
- feasible: 100%
- certified: 50%

