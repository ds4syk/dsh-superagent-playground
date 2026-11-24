import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES=os.path.join(ROOT,'diffqpep','results'); OUT=os.path.join(ROOT,'docs','visuals','2026-11-24-diff-qpep.png')
plt.rcParams.update({'font.size':9.5,'axes.titlesize':10.5,'axes.labelsize':9.5,
                     'font.sans-serif':['Noto Sans CJK JP','DejaVu Sans'],'axes.unicode_minus':False})
fig=plt.figure(figsize=(12.8,7.0))
gs=fig.add_gridspec(2,3,height_ratios=[1.0,0.9],hspace=0.45,wspace=0.28)
fig.suptitle('原子位姿层：可微性属于最优面，不属于最优值',fontsize=16.5,y=0.98,weight='bold')
fig.text(0.5,0.925,'QPEP 的 R-form 在一般 PnP 上值紧且解紧；真正的困难是最优面的多原子结构。审计：99 vs 56.5 是 R→q 转换 bug；修复后 cue 基线 99.0/99.5，atomic 99.0——不是 SOTA。',ha='center',fontsize=10)
# a: correct diagnosis text
ax=fig.add_subplot(gs[0,0]); ax.axis('off')
ax.add_patch(FancyBboxPatch((0.01,0.03),0.98,0.94,boxstyle='round,pad=0.02,rounding_size=0.04',fc='#f7fafd',ec='#2b5d8a',lw=1,transform=ax.transAxes))
ax.set_title('(a) 正确诊断：R-form 不是不 tight',fontsize=9.5)
ax.text(0.05,0.84,'R-form 12×12（一般 PnP）',transform=ax.transAxes,fontsize=9,weight='bold')
ax.text(0.07,0.70,'rank 1, det ±1, orth 1e−11, x*=±x_gt',transform=ax.transAxes,fontsize=8.3)
ax.text(0.05,0.53,'13×13 last-row',transform=ax.transAxes,fontsize=9,weight='bold')
ax.text(0.07,0.39,'R=0 是恢复伪影，不是 tightness',transform=ax.transAxes,fontsize=8.3,color='#b03a2e')
ax.text(0.05,0.22,'q-form：dim ker W=6–7, rank M=6–7',transform=ax.transAxes,fontsize=8.3)
ax.text(0.05,0.08,'无噪声 gap=0（vacuous）；planar dim=10',transform=ax.transAxes,fontsize=8.3)
# b: planar double cover
ax=fig.add_subplot(gs[0,1])
ax.bar(['真值','flip','local'], [1e-15,1e-15,3.5e-17], color=['#2b5d8a','#7aa6c2','#d08c60'])
ax.set_yscale('log'); ax.set_ylim(1e-18,1e-13); ax.set_ylabel('QPEP cost')
ax.set_title('(b) planar PnP：两支精确同代价',fontsize=9)
ax.text(0.03,0.9,'pose error: 0 vs π',transform=ax.transAxes,fontsize=7.5)
# c: audit cue benchmark (not SOTA)
ax=fig.add_subplot(gs[0,2])
methods=['local','global','cue-init\nlocal','IPPE','atomic\n+cue']; vals=[40.5,51.0,99.0,99.5,99.0]
cols=['#7aa6c2','#9dc3d8','#d08c60','#b56576','#2b5d8a']
b=ax.bar(methods,vals,color=cols); ax.set_ylim(0,110); ax.set_ylabel('success (%)')
ax.set_title('(c) 审计后：cue-aware 基线也 99%，不是 SOTA',fontsize=8.6)
for bb,v in zip(b,vals): ax.text(bb.get_x()+bb.get_width()/2,v+2,'%.1f'%v,ha='center',fontsize=7.2)
# d: fold
ax=fig.add_subplot(gs[1,0])
d=json.load(open(os.path.join(RES,'fold.json')))
rows=d['rows']; th=np.array([r['theta'] for r in rows]); der=np.array([abs(r['deriv']) for r in rows])
ax.semilogy(th,der,color='#2b5d8a'); ax.axvline(d['fold_theta'],color='#b56576',ls=':')
ax.set_xlabel(r'$\theta$'); ax.set_ylabel(r'$|dx^\star/d\theta|$'); ax.set_title('(d) fold：ν→0 时导数按 ν$^{-1/2}$',fontsize=9)
# e: covariance/risk
ax=fig.add_subplot(gs[1,1])
u=json.load(open(os.path.join(RES,'exp1d.json')))['uncertainty']
ax.bar(['posterior','Laplace','mixture'], [u['var'],u['gmode_var'],u['mixture_var']], color=['#2b5d8a','#d08c60','#7aa6c2'])
ax.set_ylabel('variance'); ax.set_title('(e) 单模态 Laplace 低估 27×',fontsize=9)
# f: API text
ax=fig.add_subplot(gs[1,2]); ax.axis('off')
ax.add_patch(FancyBboxPatch((0.01,0.02),0.98,0.96,boxstyle='round,pad=0.02,rounding_size=0.04',fc='#eef4fb',ec='#2b5d8a',lw=1,transform=ax.transAxes))
ax.text(0.05,0.92,'三态合同',transform=ax.transAxes,fontsize=11,weight='bold',color='#1f4e79')
ax.text(0.05,0.74,'CERTIFIED: κ=1, m>0, ν>0\n→ pose + KKT 导数',transform=ax.transAxes,fontsize=8.3,va='top')
ax.text(0.05,0.52,'AMBIGUOUS: κ≥2\n→ 分支/原子集 + 混合协方差',transform=ax.transAxes,fontsize=8.3,va='top')
ax.text(0.05,0.30,'VALUE-ONLY: 非物理原子\n→ 只返回 bound，不发 pose/导数',transform=ax.transAxes,fontsize=8.3,va='top')
plt.tight_layout(rect=[0,0,1,0.90]); plt.savefig(OUT,dpi=300,bbox_inches='tight'); print('wrote',OUT,os.path.getsize(OUT))
