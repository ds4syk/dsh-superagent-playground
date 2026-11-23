import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, 'diffqpep', 'results')
OUT = os.path.join(ROOT, 'docs', 'visuals', '2026-11-24-diff-qpep.png')

plt.rcParams.update({'font.size': 9.5, 'axes.titlesize': 10.5, 'axes.labelsize': 9.5,
                     'font.sans-serif': ['Noto Sans CJK JP', 'DejaVu Sans'],
                     'axes.unicode_minus': False})
fig = plt.figure(figsize=(12.8, 7.0))
gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.9], hspace=0.45, wspace=0.28)

fig.suptitle('可微全局 QPEP：价值证书、分支切换与不确定性', fontsize=17, y=0.98, weight='bold')
fig.text(0.5, 0.925, '结论：真正的难点不是“求导”，而是“对哪个解求导”；全局解映射在切换点不连续，局部梯度可以为零而全局损失最大。',
         ha='center', fontsize=10.2, color='#333')

# panel 1: R-form failure + moment gap
ax = fig.add_subplot(gs[0, 0])
d3 = json.load(open(os.path.join(RES, 'exp3d.json')))
cert = d3['certificate']
noises = [0.0, 1e-3, 1e-2]
g2 = [np.median([r['moment_order2']['gap'] for r in cert if r['noise']==n]) for n in noises]
g3 = [np.median([r['moment_order3']['gap'] for r in cert if r['noise']==n]) for n in noises]
x = np.arange(3); w=0.36
ax.bar(x-w/2, g2, w, label='order 2', color='#9dc3d8')
ax.bar(x+w/2, g3, w, label='order 3', color='#2b5d8a')
ax.set_yscale('log'); ax.set_xticks(x); ax.set_xticklabels(['0','1e-3','1e-2'])
ax.set_xlabel('measurement noise'); ax.set_ylabel('certificate gap $U-L$')
ax.set_title('(a) 矩松弛 gap；R-form: 秩 2, 正交 1.73, det 0', fontsize=9.5)
ax.legend(fontsize=8, loc='lower right'); ax.grid(axis='y', alpha=0.3)


# panel 2: local vs global gradient
ax = fig.add_subplot(gs[0, 1])
d1 = json.load(open(os.path.join(RES, 'exp1d.json')))
loc, glob = d1['local'], d1['global_']
ax.plot([h['global_loss'] for h in loc], color='#7aa6c2', lw=2, label='local-branch gradient')
ax.plot([h['global_loss'] for h in glob], color='#2b5d8a', lw=2, label='global-branch gradient')
ax.set_yscale('symlog', linthresh=1e-3); ax.set_xlabel('gradient step'); ax.set_ylabel('global-branch loss')
ax.set_title('(b) 局部梯度污染：停在 4.04，全局梯度到 0')
ax.legend(fontsize=8); ax.grid(alpha=0.25)

# panel 3: switch / FD
ax = fig.add_subplot(gs[0, 2])
hs = np.array([r['h'] for r in d1['fd']]); fdv = np.array([abs(r['fd']) for r in d1['fd']])
ax.loglog(hs, fdv, 'o-', color='#b56576', label='central difference')
ax.loglog(hs, 1/hs, 'k--', lw=0.9, label='$1/h$')
ax.axhline(1/8, color='#2b5d8a', ls=':', label='branch derivative $1/8$')
ax.set_xlabel('step $h$'); ax.set_ylabel(r'$|\Delta x^\star/\Delta\theta|$')
ax.set_title('(c) 切换点：中心差分按 $1/h$ 发散')
ax.legend(fontsize=7.5); ax.grid(alpha=0.25, which='both')

# panel 4: uncertainty bars
ax = fig.add_subplot(gs[1, 0])
u = d1['uncertainty']
labels = ['posterior', 'Laplace\n(global mode)', 'Laplace\nmixture']
vals = [u['var'], u['gmode_var'], u['mixture_var']]
bars = ax.bar(labels, vals, color=['#2b5d8a', '#d08c60', '#7aa6c2'])
ax.set_ylabel('variance'); ax.set_title('(d) 单分支 Laplace 低估 27×')
for b, v in zip(bars, vals):
    ax.text(b.get_x()+b.get_width()/2, v*1.03, '%.3f' % v, ha='center', fontsize=8)
ax.set_ylim(0, max(vals)*1.18)

# panel 5: control cost
ax = fig.add_subplot(gs[1, 1])
acts = np.array(u['actions'])
ax.plot(acts, u['exp_mixture'], color='#2b5d8a', lw=2, label='mixture risk')
ax.plot(acts, u['exp_laplace'], color='#d08c60', lw=2, label='Laplace (global mode)')
ax.axvline(u['a_mixture'], color='#2b5d8a', ls=':')
ax.axvline(u['a_laplace'], color='#d08c60', ls=':')
ax.set_xlabel('action $a$'); ax.set_ylabel('expected cost')
ax.set_title('(e) 风险动作：真实代价 2.77 → 0.85')
ax.legend(fontsize=7.5); ax.grid(alpha=0.25)

# panel 6: takeaway text
ax = fig.add_subplot(gs[1, 2]); ax.axis('off')
ax.add_patch(FancyBboxPatch((0.01,0.02),0.98,0.96, boxstyle='round,pad=0.02,rounding_size=0.04',
                            fc='#eef4fb', ec='#2b5d8a', lw=1.0, transform=ax.transAxes))
ax.text(0.05, 0.90, '一句话结论', transform=ax.transAxes, fontsize=11, weight='bold', color='#1f4e79')
ax.text(0.05, 0.74, '• 值证书 ≠ 解证书：R-form 值紧但解无效；\n  四元数矩松弛只给价值 gap（秩 4–10）。', transform=ax.transAxes, fontsize=8.6, va='top')
ax.text(0.05, 0.49, '• 分支导数有限，但全局解映射跳跃；\n  局部梯度可为零而全局损失最大。', transform=ax.transAxes, fontsize=8.6, va='top')
ax.text(0.05, 0.26, '• 不确定性必须在分支层面混合，\n  否则控制器在几何歧义处过度自信。', transform=ax.transAxes, fontsize=8.6, va='top')

plt.tight_layout(rect=[0, 0, 1, 0.90])
plt.savefig(OUT, dpi=300, bbox_inches='tight')
print('wrote', OUT, os.path.getsize(OUT))
