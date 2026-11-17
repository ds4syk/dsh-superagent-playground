#!/bin/bash
# Reproduce every number quoted in report.md / notes/  (runtime: a few minutes).
PY=~/miniconda3/bin/python
cd "$(dirname "$0")"
echo "=========== (1) continued fraction: partial quotients, quality, empirical mu ==========="
$PY cf_pi.py 2>&1 | tail -7
echo "=========== (2) high precision CF: max a_n, tau_n range ==========="
$PY cf_pi2.py 2>&1 | sed -n '1,4p;12,20p'
echo "=========== (3) Picard-Fuchs / period -> pi / monodromy / CM / period algebra ==========="
$PY legendre_pf2.py 2>&1 | sed -n '1,12p;24,40p'
echo "=========== (4) Legendre barrier + quadratic (d=2) route ==========="
$PY barriers2.py 2>&1 | sed -n '1,6p;20,34p'
echo "=========== (5) ZZ saddle: all saddles, decay rho, quadrature slopes ==========="
$PY salikhov_saddle.py 2>&1
