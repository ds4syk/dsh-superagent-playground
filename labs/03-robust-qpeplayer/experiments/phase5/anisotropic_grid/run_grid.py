#!/usr/bin/env python3
"""Phase-5 anisotropic-covariance grid: 550 accepted solver units.

5 weight/noise regimes x 5 outlier rates x 11 seeds x 2 solvers = 550.
Each unit is one completed full-W GNC/IRLS solve; failures are recorded as
rejected and do not count toward the accepted-round ledger.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
EXP=HERE.parents[1]              # .../experiments
sys.path.insert(0,str(EXP/'synthetic'))
sys.path.insert(0,str(EXP/'ablations'/'anisotropic'))
import common as cm
import run_anisotropic as A

OUT=HERE/'results'; OUT.mkdir(parents=True,exist_ok=True)
BASE=20261128; N=100; SIGMA=cm.SIGMA
REGIMES=['identity','diag_matched','full_matched','diag_mismatch','full_mismatch']
P_LIST=[0.0,0.2,0.4,0.6,0.7]
SEEDS=list(range(11))
SOLVERS=['GNC-W','IRLS-W']
KIND={'identity':'identity','diag_matched':'diag','full_matched':'full',
      'diag_mismatch':'diag','full_mismatch':'full'}
MATCHED={'identity':True,'diag_matched':True,'full_matched':True,
         'diag_mismatch':False,'full_mismatch':False}

def make(regime,p,seed):
    rid=REGIMES.index(regime)
    rng=np.random.default_rng([BASE,rid,int(round(100*p)),seed])
    x,_,Rgt,tgt=cm._clean_instance(rng,N)
    kind=KIND[regime]
    W,S=A.make_weights(rng,kind,N)
    if MATCHED[regime]:
        z=SIGMA*rng.normal(size=(N,3))
        noise=np.linalg.solve(S,z[...,None])[...,0]
    else:
        noise=SIGMA*rng.normal(size=(N,3))
    y=(Rgt@x.T).T+tgt+noise
    nout=int(round(p*N))
    if nout:
        idx=rng.choice(N,nout,replace=False)
        y[idx]=rng.uniform(-2,2,size=(nout,3))
    return x,y,Rgt,tgt,W,S

def main():
    recs=[]
    for regime in REGIMES:
        for p in P_LIST:
            for seed in SEEDS:
                x,y,Rgt,tgt,W,S=make(regime,p,seed)
                XSKEW=A.skew_batch(x)
                for solver in SOLVERS:
                    t0=time.perf_counter()
                    try:
                        if solver=='GNC-W':
                            R,t,info=A.solve_gnc_w(x,y,S,XSKEW=XSKEW)
                        else:
                            R,t,info=A.solve_irls_w(x,y,S,XSKEW=XSKEW)
                        dt=1000*(time.perf_counter()-t0)
                        re,te=cm.pose_errors(R,t,Rgt,tgt)
                        rec=dict(regime=regime,matched=MATCHED[regime],p=p,seed=seed,
                                 solver=solver,accepted=True,success=cm.success(re,te),
                                 rot=re,trans=te,time_ms=dt,phi=float(info.get('phi',float('nan'))),
                                 cond_median=float(np.median(np.linalg.cond(W))))
                    except Exception as e:
                        rec=dict(regime=regime,matched=MATCHED[regime],p=p,seed=seed,
                                 solver=solver,accepted=False,error=repr(e))
                    recs.append(rec)
    (OUT/'ledger.jsonl').write_text('\n'.join(json.dumps(r) for r in recs)+'\n')
    acc=[r for r in recs if r['accepted']]
    (OUT/'summary.json').write_text(json.dumps(dict(
        total=len(recs),accepted=len(acc),rejected=len(recs)-len(acc),
        config=dict(regimes=REGIMES,p=P_LIST,seeds=SEEDS,solvers=SOLVERS,n=N,sigma=SIGMA),
        rows=acc),indent=2))
    L=['# Phase-5 anisotropic covariance grid','',
       f'{len(acc)}/{len(recs)} solver units accepted; N={N}, sigma={SIGMA}.','',
       '| regime | p | solver | n | success | rot med | trans med | p50 ms |',
       '|---|---|---|---|---|---|---|---|']
    for regime in REGIMES:
        for p in P_LIST:
            for solver in SOLVERS:
                rr=[r for r in acc if r['regime']==regime and r['p']==p and r['solver']==solver]
                if not rr: continue
                L.append(f"| {regime} | {p:.1f} | {solver} | {len(rr)} | {100*np.mean([r['success'] for r in rr]):.0f}% | "
                         f"{np.median([r['rot'] for r in rr]):.3f} | {np.median([r['trans'] for r in rr]):.4f} | "
                         f"{np.median([r['time_ms'] for r in rr]):.1f} |")
    L+=['','## Reading','',
        '* Matched regimes use noise covariance W_i^{-1}; mismatched regimes use isotropic noise with the same W_i, so accuracy loss is misspecification, not solver error.',
        '* Each row aggregates 11 seeds; each seed is one accepted ledger unit.']
    (OUT/'summary.md').write_text('\n'.join(L)+'\n')
    print(f'accepted {len(acc)}/{len(recs)}; wrote {OUT}')

if __name__=='__main__': main()
