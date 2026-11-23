import sys, os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

OUT = os.path.join(os.path.dirname(__file__), '..', 'results')
FIG = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(OUT, exist_ok=True); os.makedirs(FIG, exist_ok=True)

def J(x, th): return (x*x - 1.0)**2 + th*x
def dJ(x, th): return 4*x*(x*x - 1.0) + th
def d2J(x): return 12*x*x - 4.0

def branches(th):
    roots = np.roots([4.0, 0.0, -4.0, th])
    out = []
    for r in roots:
        if abs(r.imag) < 1e-10:
            x = r.real
            h = d2J(x)
            if h > 0:
                out.append((x, J(x, th), h))
    out.sort(key=lambda z: z[0])
    return out

def global_min(th):
    b = branches(th)
    return min(b, key=lambda z: z[1])

# 1) branch costs and switch
th_grid = np.linspace(-0.4, 0.4, 161)
costs = [branches(t) for t in th_grid]
x_glob = np.array([min(b, key=lambda z: z[1])[0] for b in costs])
gap = np.array([abs(b[0][1]-b[1][1]) if len(b) == 2 else np.nan for b in costs])

# 2) finite difference across the tie
fd = []
for h in [1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 1e-4]:
    xp = global_min(h)[0]; xm = global_min(-h)[0]
    fd.append(dict(h=h, fd=float((xp-xm)/(2*h)), xp=float(xp), xm=float(xm)))
print('FD', fd)

# 3) local vs global gradient descent on the measurement parameter
target = 1.0
def loss(x): return (x - target)**2
def branch_deriv(x):
    return -1.0 / d2J(x)

def run(mode, steps=200, lr=0.05, th0=0.1):
    th = th0; x_local = 1.0; hist = []
    for k in range(steps):
        b = branches(th)
        if mode == 'local':
            x = min(b, key=lambda z: abs(z[0]-x_local))[0]
            x_local = x
        else:
            x = min(b, key=lambda z: z[1])[0]
        xg = min(b, key=lambda z: z[1])[0]
        g = 2*(x - target)*branch_deriv(x)
        hist.append(dict(th=float(th), x=float(x), branch_loss=float(loss(x)),
                         global_loss=float(loss(xg)), grad=float(g)))
        th = th - lr*g
    return hist

loc = run('local'); glob = run('global')
print('final local th %.3f x %.3f branch_loss %.3f global_loss %.3f' % (loc[-1]['th'], loc[-1]['x'], loc[-1]['branch_loss'], loc[-1]['global_loss']))
print('final global th %.3f x %.3f branch_loss %.3f global_loss %.3f' % (glob[-1]['th'], glob[-1]['x'], glob[-1]['branch_loss'], glob[-1]['global_loss']))

# 4) uncertainty and risk-sensitive action
sigma = 0.5; th0 = 0.1
xg = np.linspace(-2.2, 2.2, 4001)
logp = -J(xg, th0)/sigma**2
p = np.exp(logp - logp.max()); p /= np.trapz(p, xg)
mean = np.trapz(xg*p, xg); var = np.trapz((xg-mean)**2*p, xg)
b = branches(th0)
# Laplace per mode
lap = []
for x, j, h in b:
    lap.append(dict(x=float(x), j=float(j), h=float(h), var=float(sigma**2/h)))
ev = [np.exp(-l['j']/sigma**2)/np.sqrt(l['h']) for l in lap]
w = np.array(ev); w = w/w.sum()
mu_l = sum(wi*li['x'] for wi, li in zip(w, lap))
var_l = sum(wi*(li['var'] + (li['x']-mu_l)**2) for wi, li in zip(w, lap))
# global-mode Laplace only
gmode = min(lap, key=lambda l: l['j'])
# nonlinear control cost: quadratic inside a safe box, penalty outside
def ctrl_cost(z):
    return np.where(np.abs(z) <= 1.5, z**2, 5.0 + z**2)
acts = np.linspace(-2.0, 2.0, 401)
def expected_mixture(a):
    return np.trapz(ctrl_cost(xg+a)*p, xg)
def expected_laplace(a, m):
    xs = np.linspace(m['x']-4*np.sqrt(m['var']), m['x']+4*np.sqrt(m['var']), 2001)
    ps = np.exp(-(xs-m['x'])**2/(2*m['var'])); ps /= ps.sum()
    return np.sum(ctrl_cost(xs+a)*ps)
em = np.array([expected_mixture(a) for a in acts])
el = np.array([expected_laplace(a, gmode) for a in acts])
a_m = float(acts[np.argmin(em)]); a_l = float(acts[np.argmin(el)])
em_at_al = float(em[np.argmin(el)]); el_at_am = float(el[np.argmin(em)])
print('posterior mean %.3f var %.3f | laplace mixture mean %.3f var %.3f | gmode var %.3f' % (mean, var, mu_l, var_l, gmode['var']))
print('optimal action mixture %.3f cost %.3f | laplace %.3f cost %.3f | mixture cost of laplace action %.3f' % (a_m, em.min(), a_l, el.min(), em_at_al))

out = dict(branch=dict(theta=th_grid.tolist(), x_glob=x_glob.tolist(), gap=gap.tolist(),
                       branch_costs=[[float(z[0][1]), float(z[1][1])] if len(z)==2 else None for z in costs]),
           fd=fd, local=loc, global_=glob,
           uncertainty=dict(sigma=sigma, theta=th0, mean=float(mean), var=float(var),
                            laplace_w=[float(z) for z in w], laplace_modes=lap,
                            mixture_mean=float(mu_l), mixture_var=float(var_l),
                            gmode_var=float(gmode['var']),
                            actions=acts.tolist(), exp_mixture=em.tolist(), exp_laplace=el.tolist(),
                            a_mixture=a_m, a_laplace=a_l, em_at_al=em_at_al, el_at_am=el_at_am))
with open(os.path.join(OUT, 'exp1d.json'), 'w') as f:
    json.dump(out, f, indent=2)

# figures
plt.figure(figsize=(5.2, 3.2))
plt.plot(th_grid, [z[0][1] if len(z)==2 else np.nan for z in costs], label='branch 1')
plt.plot(th_grid, [z[1][1] if len(z)==2 else np.nan for z in costs], label='branch 2')
plt.plot(th_grid, [min(z, key=lambda t:t[1])[1] for z in costs], 'k--', lw=2, label='global value')
plt.xlabel(r'measurement parameter $\theta$'); plt.ylabel(r'$J(x;\theta)$')
plt.legend(fontsize=7); plt.tight_layout(); plt.savefig(os.path.join(FIG, 'fig1d_branches.png'), dpi=200); plt.close()

plt.figure(figsize=(5.2, 3.2))
steps = np.arange(len(loc))
plt.plot(steps, [h['global_loss'] for h in loc], label='local branch gradient')
plt.plot(steps, [h['global_loss'] for h in glob], label='global branch gradient')
plt.yscale('symlog', linthresh=1e-3); plt.xlabel('step'); plt.ylabel('global-branch loss')
plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(os.path.join(FIG, 'fig1d_pollution.png'), dpi=200); plt.close()

plt.figure(figsize=(5.2, 3.2))
plt.plot(xg, p, label='posterior')
for l in lap:
    plt.axvline(l['x'], color='gray', ls=':', lw=1)
plt.xlabel('pose x'); plt.ylabel('density'); plt.legend(fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(FIG, 'fig1d_posterior.png'), dpi=200); plt.close()

plt.figure(figsize=(5.2, 3.2))
plt.plot(acts, em, label='mixture risk')
plt.plot(acts, el, label='Laplace (global mode)')
plt.axvline(a_m, color='C0', ls=':'); plt.axvline(a_l, color='C1', ls=':')
plt.xlabel('action a'); plt.ylabel('expected control cost'); plt.legend(fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(FIG, 'fig1d_control.png'), dpi=200); plt.close()
print('saved')
