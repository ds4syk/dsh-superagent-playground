import json, os
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES=os.path.join(ROOT,'diffqpep','results'); FIG=os.path.join(ROOT,'paper','figures')
# FCD schematic
fig,ax=plt.subplots(figsize=(7.2,2.2)); ax.set_xlim(0,10); ax.set_ylim(0,2.4); ax.axis('off')
boxes=[(0.1,0.8,1.5,0.8,'QPEP\nmeasurements'),
       (1.9,0.8,1.9,0.8,'lift: $X$, dual $H$\nface $F$, $\\kappa=\\mathrm{corank}H$'),
       (4.1,0.8,2.0,0.8,'certificates\n$m$ (tie), $\\nu$ (fold)'),
       (6.4,0.8,1.6,0.8,'three-state\nAPI'),
       (8.3,0.8,1.5,0.8,'pose + $d\\theta$\nor set/cov.')]
for x,y,w,h,t in boxes:
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.04,rounding_size=0.08',fc='#eef4fb',ec='#2b5d8a',lw=1.0))
    ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=7)
for i in range(len(boxes)-1):
    ax.add_patch(FancyArrowPatch((boxes[i][0]+boxes[i][2]+0.03,1.2),(boxes[i+1][0]-0.03,1.2),arrowstyle='->',lw=1,color='#333'))
ax.text(7.2,0.35,'CERTIFIED: $\\kappa=1,m>0,\\nu>0$ $\\to$ pose, KKT derivative\nAMBIGUOUS: $\\kappa\\geq2$ $\\to$ branch set, Jacobian set, mixture covariance\nVALUE-ONLY: non-physical atom $\\to$ bound only',ha='center',fontsize=6.8)
plt.tight_layout(pad=0.1); plt.savefig(os.path.join(FIG,'fig_fcd.png'),dpi=300,bbox_inches='tight'); plt.close()
# face diagnostics
d=json.load(open(os.path.join(RES,'correct.json')))
noises=[0.0,1e-3,1e-2]
rank=[np.median([r['rform12']['rank'] for r in d if r['noise']==n]) for n in noises]
orth=[np.median([r['rform12']['orth'] for r in d if r['noise']==n]) for n in noises]
det=[np.median([abs(r['rform12']['det']) for r in d if r['noise']==n]) for n in noises]
kdim=[np.median([r['qform']['dim_ker_W'] for r in d if r['noise']==n]) for n in noises]
rm=[np.median([r['qform']['rank_M'] for r in d if r['noise']==n]) for n in noises]
fig,axs=plt.subplots(1,2,figsize=(7.2,2.6))
ax=axs[0]; x=np.arange(3); w=0.26
ax.bar(x-w,rank,w,label='rank',color='#2b5d8a')
ax.bar(x,orth,w,label='$\\|R^TR-I\\|$',color='#7aa6c2')
ax.bar(x+w,det,w,label='$|\\det R|$',color='#d08c60')
ax.set_xticks(x); ax.set_xticklabels(['0','1e-3','1e-2']); ax.set_yscale('log'); ax.set_ylim(1e-12,3)
ax.set_xlabel('noise'); ax.legend(fontsize=7); ax.set_title('(a) R-form 12x12: $x^\\star=\\pm x_{gt}$',fontsize=8)
ax=axs[1]
ax.bar(x-w/2,kdim,w,label='dim ker $W$',color='#2b5d8a')
ax.bar(x+w/2,rm,w,label='rank $M$',color='#7aa6c2')
ax.set_xticks(x); ax.set_xticklabels(['0','1e-3','1e-2']); ax.set_xlabel('noise'); ax.set_ylabel('dimension')
ax.legend(fontsize=7); ax.set_title('(b) q-form: face dimension, not gap',fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(FIG,'fig_face.png'),dpi=300); plt.close()
print('done')
