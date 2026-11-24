import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, 'diffqpep', 'results')
FIG = os.path.join(ROOT, 'paper', 'figures')

# ---------- method schematic ----------
fig, ax = plt.subplots(figsize=(7.2, 1.9))
ax.set_xlim(0, 10); ax.set_ylim(0, 2.2); ax.axis('off')
boxes = [
    (0.1, 0.7, 1.5, 0.8, 'measurements\n$\\theta$'),
    (2.0, 0.7, 1.9, 0.8, 'quaternion\nQPEP cost $J(q;\\theta)$'),
    (4.3, 0.7, 2.1, 0.8, 'global solve\nmulti-start + moment'),
    (6.8, 0.7, 1.6, 0.8, 'certificate\n$U-L\\leq\\tau$?'),
    (8.8, 0.7, 1.1, 0.8, 'KKT\nderivative'),
]
for x, y, w, h, t in boxes:
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.04,rounding_size=0.08',
                                fc='#eef4fb', ec='#2b5d8a', lw=1.0))
    ax.text(x+w/2, y+h/2, t, ha='center', va='center', fontsize=7.2)
for i in range(len(boxes)-1):
    x0 = boxes[i][0]+boxes[i][2]; x1 = boxes[i+1][0]
    ax.add_patch(FancyArrowPatch((x0+0.04, 1.1), (x1-0.04, 1.1), arrowstyle='->', lw=1.0, color='#333'))
ax.text(5.35, 0.28, r'$q^\star,\ \Delta=J(q^\star)-L_{\rm mom},\ \Sigma,\ \kappa_{\rm sw}$', ha='center', fontsize=8)
ax.add_patch(FancyArrowPatch((7.6, 0.66), (7.6, 0.38), arrowstyle='->', lw=1.0, color='#333'))
plt.tight_layout(pad=0.1); plt.savefig(os.path.join(FIG, 'fig_method.png'), dpi=300, bbox_inches='tight'); plt.close()

# ---------- 3D certificate ----------
d = json.load(open(os.path.join(RES, 'exp3d.json')))
cert = d['certificate']
noises = [0.0, 1e-3, 1e-2]
g2 = [np.median([r['moment_order2']['gap'] for r in cert if r['noise']==n]) for n in noises]
g3 = [np.median([r['moment_order3']['gap'] for r in cert if r['noise']==n]) for n in noises]
fig, ax = plt.subplots(figsize=(5.2, 2.8))
x = np.arange(len(noises)); w = 0.36
ax.bar(x-w/2, g2, w, label='order 2', color='#9dc3d8')
ax.bar(x+w/2, g3, w, label='order 3', color='#2b5d8a')
ax.set_yscale('log'); ax.set_xticks(x); ax.set_xticklabels(['0', '1e-3', '1e-2'])
ax.set_xlabel('measurement noise'); ax.set_ylabel('certificate gap $U-L$')
ax.legend(fontsize=7.5, loc='upper right'); ax.grid(axis='y', alpha=0.3)
ax.set_ylim(1e-12, 1e-1)
ax.set_title('quaternion moment gap (R-form: rank 2, orth 1.73, det 0)', fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(FIG, 'fig_cert.png'), dpi=300); plt.close()

# ---------- 1D switch + FD ----------
d1 = json.load(open(os.path.join(RES, 'exp1d.json')))
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.7))
ax = axs[0]
th = np.array(d1['branch']['theta']); xg = np.array(d1['branch']['x_glob'])
ax.plot(th, xg, color='k', lw=1.6, label='global solution $q^\\star(\\theta)$')
ax.axvline(0, color='gray', ls=':', lw=0.8)
ax.set_xlabel(r'$\theta$'); ax.set_ylabel(r'$x^\star$'); ax.legend(fontsize=7)
ax.set_title('(a) global branch switch', fontsize=8)
ax = axs[1]
hs = np.array([r['h'] for r in d1['fd']]); fdv = np.array([abs(r['fd']) for r in d1['fd']])
ax.loglog(hs, fdv, 'o-', color='#b56576', label='central difference')
ax.loglog(hs, 1.0/hs, 'k--', lw=0.8, label='$1/h$')
ax.axhline(1/8, color='#2b5d8a', ls=':', lw=1, label='branch derivative $1/8$')
ax.set_xlabel('step $h$'); ax.set_ylabel('$|\\Delta x^\\star/\\Delta\\theta|$'); ax.legend(fontsize=6.5)
ax.set_title('(b) finite difference across the tie', fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(FIG, 'fig_switch.png'), dpi=300); plt.close()

# ---------- pollution + control ----------
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.7))
ax = axs[0]
loc = d1['local']; glob = d1['global_']
ax.plot([h['global_loss'] for h in loc], label='local-branch gradient', color='#7aa6c2')
ax.plot([h['global_loss'] for h in glob], label='global-branch gradient', color='#2b5d8a')
ax.set_yscale('symlog', linthresh=1e-3); ax.set_xlabel('step'); ax.set_ylabel('global-branch loss')
ax.legend(fontsize=7); ax.set_title('(a) gradient pollution', fontsize=8)
ax = axs[1]
u = d1['uncertainty']; acts = np.array(u['actions'])
ax.plot(acts, u['exp_mixture'], label='mixture risk', color='#2b5d8a')
ax.plot(acts, u['exp_laplace'], label='Laplace (global mode)', color='#d08c60')
ax.axvline(u['a_mixture'], color='#2b5d8a', ls=':')
ax.axvline(u['a_laplace'], color='#d08c60', ls=':')
ax.set_xlabel('action $a$'); ax.set_ylabel('expected control cost'); ax.legend(fontsize=7)
ax.set_title('(b) risk-sensitive action', fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(FIG, 'fig_pollution_control.png'), dpi=300); plt.close()
print('figures written')
