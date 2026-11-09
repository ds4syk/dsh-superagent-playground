"""SDPRLayers paper (Holmes, Duembgen, Barfoot, TRO 2025) -- Section VI-A
sixth-order polynomial bilevel example:

    y(x, theta) = sum_{i=0}^6 theta_i x^i ,   minimize over x

Homogenized QCQP: variable xbar = [x0, x1, x2, x3]^T with x0 = 1,
x2 = x1^2, x3 = x1^3  (see footnote 21 of the paper), so that
y = xbar^T Q_theta xbar with the 4x4 Q_theta of Table I / page 9.

Reference line numbers in paper/sdpr.txt:
  Q_theta                 : L706-L714
  A0, A1, A2, A3          : L717-L733
  constraints redunancy   : L736, L1278-L1340
  Thm.2 / (13)-(17)       : L475-L486
  N                       : L1192-L1199 (eq. 35)

Conventions (match the paper):
  * A0 is the homogenizing matrix: <A0, X> = 1            (x0^2 = 1)
  * A1, A2, A3 are the "m = 3" QCQP constraints.
  * g(x) = [x^T A1 x, x^T A2 x, x^T A3 x, x^T A0 x - 1]   (eq. 11, L387-L392)
    G = [A1 x, A2 x, A3 x, A0 x]^T                        (eq. 14, L433-L434)
  * H = Q + lam0 A0 + sum_i lam_i A_i                     (eq. 2 / eq. 3)
  * M = 2 [[H, G^T], [G, 0]]                              (eq. 13, L424-L427)
  * N as in eq. (35), L1192-L1199.
"""

import numpy as np
import cvxpy as cp

# ----------------------------------------------------------------------------
# problem data (Table I, L690-L692; Q at L706-L714; A_i at L717-L733)
# ----------------------------------------------------------------------------
THETA0 = np.array([10.0, 2.6334, -4.3443, 0.0, 0.8055, -0.1334, 0.0389])

A0 = np.zeros((4, 4)); A0[0, 0] = 1.0
A1 = np.zeros((4, 4)); A1[0, 2] = A1[2, 0] = 0.5; A1[1, 1] = -1.0
A2 = np.zeros((4, 4)); A2[0, 3] = A2[3, 0] = 1.0; A2[1, 2] = A2[2, 1] = -1.0
A3 = np.zeros((4, 4)); A3[1, 3] = A3[3, 1] = 0.5; A3[2, 2] = -1.0

CONS = [A1, A2, A3]          # the m = 3 parameter-free QCQP constraints


def Qmat(theta):
    """Q_theta of L706-L714 (exact rational coefficients)."""
    t = np.asarray(theta, dtype=float)
    Q = np.zeros((4, 4))
    Q[0, 0] = t[0]
    Q[0, 1] = Q[1, 0] = t[1] / 2
    Q[0, 2] = Q[2, 0] = t[2] / 3
    Q[1, 1] = t[2] / 3
    Q[0, 3] = Q[3, 0] = t[3] / 4
    Q[1, 2] = Q[2, 1] = t[3] / 4
    Q[1, 3] = Q[3, 1] = t[4] / 3
    Q[2, 2] = t[4] / 3
    Q[2, 3] = Q[3, 2] = t[5] / 2
    Q[3, 3] = t[6]
    return Q


def Q_basis():
    """B_j with Q_theta = sum_j theta_j B_j   (n^2 x 7, columns vec(B_j))."""
    V = np.zeros((16, 7))
    for j in range(7):
        e = np.zeros(7); e[j] = 1.0
        V[:, j] = Qmat(e).reshape(-1)
    return V


Q_BASIS = Q_basis()


def poly_coeffs(theta):
    return np.asarray(theta, dtype=float)


def poly_y(x, theta):
    """y(x, theta) = sum_i theta_i x^i  (eq. 26, L640)."""
    return sum(theta[i] * x ** i for i in range(7))


def poly_dy(x, theta):
    return sum(i * theta[i] * x ** (i - 1) for i in range(1, 7))


def poly_d2y(x, theta):
    return sum(i * (i - 1) * theta[i] * x ** (i - 2) for i in range(2, 7))


# ----------------------------------------------------------------------------
# forward pass: solve the Shor relaxation (P3) with cvxpy
# ----------------------------------------------------------------------------
def solve_sdp(theta, extra_cons=(), solver="SCS", solver_kw=None, warm_start=False, As=None, Q=None):
    """Solve min <Q,X> s.t. <A_i,X>=0, <A0,X>=1, X>=0.

    Returns dict with X, vector x (rank-1 factor), dual multipliers
    lam = [lam1, lam2, lam3, lam0] (paper ordering of g), H certificate,
    and diagnostics.

    CVXPY sign check: the dual y of <A,X> = b enters the Lagrangian as
    <Q,X> + sum y_k (<A_k,X> - b_k) - <Z,X>, hence
        Z = Q + y0 A0 + sum_i y_i A_i >= 0
    which is exactly the paper's certificate H (eq. 2/3), and y_k = lam_k.
    """
    solver_kw = dict(solver_kw or {})
    As = CONS if As is None else list(As)
    Q = Qmat(theta) if Q is None else np.asarray(Q, float)
    X = cp.Variable((4, 4), symmetric=True)
    cons = [cp.trace(C @ X) == 0 for C in As]
    cons.append(cp.trace(A0 @ X) == 1)
    cons.append(X >> 0)
    for C in extra_cons:
        cons.append(cp.trace(C @ X) == 0)
    prob = cp.Problem(cp.Minimize(cp.trace(Q @ X)), cons)
    prob.solve(solver=solver, warm_start=warm_start, **solver_kw)

    if prob.status not in ("optimal", "optimal_inaccurate") or X.value is None:
        return dict(status=prob.status, pstar=float(prob.value) if prob.value is not None
                    else float("nan"), X=None, x=None, eigX=None, lam=None, H=None,
                    Q=Q, Z_psd=None, rank1_ratio=float("nan"))
    Xv = np.array(X.value, dtype=float)
    w, V = np.linalg.eigh(0.5 * (Xv + Xv.T))
    order = np.argsort(w)[::-1]
    w, V = w[order], V[:, order]
    x = V[:, 0] * np.sqrt(max(w[0], 0.0))
    if x[0] < 0:
        x = -x
    duals = [np.asarray(c.dual_value, dtype=float).ravel()[0] for c in cons[:len(As) + 1]]
    # duals = [y1, y2, y3, y0]; PSD-constraint dual is cons[4]
    lam = np.array(duals)
    H = Q + lam[-1] * A0 + sum(lam[i] * As[i] for i in range(len(As)))
    out = dict(status=prob.status, pstar=float(prob.value), X=Xv, x=x,
               eigX=w, lam=lam, H=H, Q=Q,
               Z_psd=np.array(cons[4].dual_value, dtype=float),
               rank1_ratio=float(w[0] / w[1]) if w[1] > 0 else np.inf)
    return out


# ----------------------------------------------------------------------------
# certificate diagnostics (Q1)
# ----------------------------------------------------------------------------
def corank(M, rtol=1e-8):
    s = np.linalg.svd(M, compute_uv=False)
    return int(np.sum(s <= rtol * s[0]))


def certificate_report(sol, rtol=1e-8, As=None):
    As = CONS if As is None else list(As)
    H, x = sol["H"], sol["x"]
    ev = np.linalg.eigvalsh(0.5 * (H + H.T))
    Hn = np.linalg.norm(H, 2)
    x = x / np.linalg.norm(x)
    G = np.vstack([C @ x for C in As] + [A0 @ x])
    # SOSC: v^T H v > 0 on ker G
    if G.shape[0] > G.shape[1]:
        G = G[:G.shape[1]]
    _, sv, Vt = np.linalg.svd(G)
    r = int(np.sum(sv > rtol * max(sv[0], 1e-300)))
    Zker = Vt[r:].T                      # basis of ker G  (n x (n-r))
    sos = np.inf
    if Zker.shape[1] > 0:
        sos = float(np.min(np.linalg.eigvalsh(Zker.T @ H @ Zker)))
    return dict(
        eigH=ev,
        H_psd=bool(ev[0] >= -rtol * max(Hn, 1e-300)),
        corankH=corank(H, rtol=max(rtol, 1e-10)),
        eigH_rel=[float(v) / max(Hn, 1e-300) for v in ev],
        # is the rank decision numerically unambiguous?  (solver dual noise)
        corank_confident=("1" if ev[1] > 1e-6 * max(abs(ev[-1]), 1e-300)
                          else ("2+" if ev[1] < 1e-12 * max(abs(ev[-1]), 1e-300)
                                else "ambiguous")),
        Hx_res=float(np.linalg.norm(H @ x)),
        rankG=r,
        dim_kerG=int(4 - r),
        sosc_min=float(sos),
        sosc=bool(sos > 0),
        tightness_ratio=sol["rank1_ratio"],
    )


# ----------------------------------------------------------------------------
# Thm.2 Jacobian:  J = -P Mr^+ N_r     (eq. 16/17, L475-L486)
# ----------------------------------------------------------------------------
def jacobian_thm2(sol, sel=None, theta=None, use_all_lambda_in_N=True, As=None):
    """Compute d x / d nu  (nu = vec(Q), vec(A1), ..., vec(Am)), eq. (16).

    sel : indices (into the g-order [g1,g2,g3,g0]) of the r linearly
          independent constraint rows to keep.  None -> determined by SVD.
    """
    As = CONS if As is None else list(As)
    H = sol["H"]; x = sol["x"]; lam = sol["lam"]        # lam=[l1,...,lm,l0]
    m = len(As)                                          # 3
    G = np.vstack([C @ x for C in As] + [A0 @ x])        # (m+1) x n, eq.14
    n = G.shape[1]
    # pick maximal independent row subset
    if sel is None:
        sel = independent_rows(G)
    sel = list(sel)
    r = len(sel)
    Gr = G[sel]
    Mr = np.block([[2 * H, 2 * Gr.T],
                   [Gr, np.zeros((r, r))]])              # eq. (17)
    N = jacobian_N(x, lam, m, use_all_lambda_in_N=use_all_lambda_in_N)  # (n+m+1) x (m+1)n^2
    N_sel = keep_rows(N, n, sel, m)                      # reduced N_r
    P = np.hstack([np.eye(n), np.zeros((n, r))])         # n x (n+r)
    MrT = Mr.T
    # right pseudo-inverse Mr^T (Mr Mr^T)^{-1}
    Mr_pinv = MrT @ np.linalg.solve(Mr @ MrT, np.eye(n + r))
    J = -P @ Mr_pinv @ N_sel                              # n x (m+1)n^2
    return dict(J=J, Mr=Mr, Mr_pinv=Mr_pinv, N=N, N_sel=N_sel, G=G,
                Gr=Gr, sel=sel, r=r, P=P)


def jacobian_N(x, lam, m, use_all_lambda_in_N=True):
    """eq. (35), L1192-L1199.

    d(2Hx) = 2 (I x x^T) [ vec(dQ) + sum_i lam_i vec(dA_i) ]      (n rows)
    d(x^T A_i x) = vec(x x^T)^T vec(dA_i)                         (m rows)
    0 for the (parameter-free) homogenizing constraint             (1 row)
    """
    n = x.size
    I = np.eye(n)
    Ivec = np.kron(I, x.reshape(1, -1))                  # (I (x) x^T), eq. (31)
    lam_m = lam[:m] if use_all_lambda_in_N else np.zeros(m)
    top = np.hstack([2 * Ivec,
                     np.hstack([lam_m[i] * 2 * Ivec for i in range(m)])])
    vxx = np.outer(x, x).reshape(1, -1)                  # vec(xx^T)^T
    mid = np.hstack([np.zeros((m, n * n)),
                     np.kron(np.eye(m), vxx)])
    bot = np.zeros((1, (m + 1) * n * n))
    return np.vstack([top, mid, bot])


def keep_rows(N, n, sel, m):
    """Rows of N kept for the reduced system: stationarity rows (0..n-1) plus
    the rows of the selected constraints (g-order: 0..m-1 are A1..Am, m is A0)."""
    rows = list(range(n)) + [n + s for s in sel]
    return N[rows]


def independent_rows(G, rtol=1e-8):
    """Greedy maximal independent row subset, in original order."""
    sel = []
    for i in range(G.shape[0]):
        cand = sel + [i]
        r = np.linalg.matrix_rank(G[cand], tol=rtol)
        if r > len(sel):
            sel.append(i)
    return sel


def jacobian_theta(sol, theta, sel=None):
    """Chain rule to the 7 polynomial coefficients: J_theta = J_nu @ dnu/dtheta.
    Only the Q block depends on theta (constraints are parameter free)."""
    jac = jacobian_thm2(sol, sel=sel)
    Jtheta = jac["J"][:, :16] @ Q_BASIS                 # n x 7
    return Jtheta, jac


# ----------------------------------------------------------------------------
# ground-truth Jacobians
# ----------------------------------------------------------------------------
def exact_minimizer(theta):
    """Global minimizer of y(x,theta) via the roots of y'(x), machine precision.
    Returns (x1, y, y'') of the global minimum."""
    theta = np.asarray(theta, float)
    # y'(x) = theta1 + 2 theta2 x + ... + 6 theta6 x^5  -> ascending coeffs
    dcoef = np.array([i * theta[i] for i in range(1, 7)])   # c0 + c1 x + ...
    # strip leading zeros of the polynomial (highest degree first for np.roots)
    dc = dcoef.copy()
    while len(dc) > 1 and abs(dc[-1]) == 0:
        dc = dc[:-1]
    roots = np.roots(dc[::-1])
    cand = []
    for r in roots:
        if abs(r.imag) < 1e-9:
            xr = r.real
            if poly_d2y(xr, theta) > 0:       # local min
                cand.append(xr)
    if not cand:
        raise RuntimeError("no local min found")
    vals = [poly_y(xr, theta) for xr in cand]
    x1 = cand[int(np.argmin(vals))]
    return x1, poly_y(x1, theta), poly_d2y(x1, theta)


def exact_x(theta):
    x1 = exact_minimizer(theta)[0]
    return np.array([1.0, x1, x1 ** 2, x1 ** 3])


def exact_jacobian_theta(theta):
    """Analytic d x*/d theta for the sixth-order polynomial.
    x1 solves y'(x1)=0  =>  dx1/dtheta_j = -x1^{j-1}/y''(x1)  (j>=1),
    and x2=x1^2, x3=x1^3, x0=1."""
    x1, _, d2 = exact_minimizer(theta)
    dx1 = np.zeros(7)
    for j in range(1, 7):
        # d/dtheta_j of y'(x) = sum_k k theta_k x^{k-1} is  j * x^{j-1}
        dx1[j] = -j * x1 ** (j - 1) / d2
    J = np.zeros((4, 7))
    J[1] = dx1
    J[2] = 2 * x1 * dx1
    J[3] = 3 * x1 ** 2 * dx1
    return J


def fd_jacobian_theta(theta, h=1e-5, use_sdp=True, solver="SCS", solver_kw=None):
    """Central finite differences: perturb theta_j, re-solve (SDP or exact)."""
    theta = np.asarray(theta, float)
    J = np.zeros((4, 7))
    for j in range(7):
        e = np.zeros(7); e[j] = h
        if use_sdp:
            xp = solve_sdp(theta + e, solver=solver, solver_kw=solver_kw)["x"]
            xm = solve_sdp(theta - e, solver=solver, solver_kw=solver_kw)["x"]
            xp = fix_sign(xp); xm = fix_sign(xm)
        else:
            xp = exact_x(theta + e); xm = exact_x(theta - e)
        J[:, j] = (xp - xm) / (2 * h)
    return J


def fix_sign(x):
    return x if x[0] >= 0 else -x


def rel_jac_error(Jest, Jtrue):
    """eq. (29), L873-L877: ||Jest - Jtrue||_inf / ||Jtrue||_inf"""
    return float(np.linalg.norm(Jest - Jtrue, np.inf) / np.linalg.norm(Jtrue, np.inf))


# ----------------------------------------------------------------------------
# additional helpers used by the experiment scripts
# ----------------------------------------------------------------------------
def sosc_margin(H, x, As=None):
    """min eig of Zker^T H Zker, Zker = basis of ker G, G = gradients of the
    constraint equations at x (eq. 6 / Lemma 3).  Returns (margin, dim ker G)."""
    As = CONS if As is None else As
    G = np.vstack([C @ x for C in As] + [A0 @ x])
    u, s, Vt = np.linalg.svd(G)
    r = int(np.sum(s > 1e-7 * max(s[0], 1e-300)))
    Z = Vt[r:].T
    if Z.shape[1] == 0:
        return np.inf, 0, G
    return float(np.min(np.linalg.eigvalsh(Z.T @ H @ Z))), Z.shape[1], G


def cift_jacobian(sol, sel=None, theta=None, As=None):
    As = CONS if As is None else list(As)
    """SDPR-CIFT, Section V-B2 / eqs. (22)-(24): recompute the multipliers of a
    non-redundant subset from the primal stationarity, rebuild H_r, use the
    classic IFT.  Returns dict with J_theta and diagnostics."""
    x = sol["x"]; Q = sol["Q"]; n = x.size
    G = np.vstack([C @ x for C in As] + [A0 @ x])
    if sel is None:
        sel = independent_rows(G)
    sel = list(sel)
    r = len(sel)
    Gr = G[sel]
    lam_r = np.linalg.pinv(Gr.T) @ (-Q @ x)          # eq. (22)
    Hr = Q.copy()
    for k, s_idx in enumerate(sel):
        C = As[s_idx] if s_idx < len(As) else A0
        Hr = Hr + lam_r[k] * C                       # eq. (23)
    Mr = 2 * np.block([[Hr, Gr.T], [Gr, np.zeros((r, r))]])
    # N with the recomputed multipliers: lambda vector in g-order
    m = len(As)
    lam_g = np.zeros(m + 1)
    for k, s_idx in enumerate(sel):
        lam_g[s_idx] = lam_r[k]
    Nfull = jacobian_N(x, lam_g, m)
    Nsel = keep_rows(Nfull, n, sel, m)
    P = np.hstack([np.eye(n), np.zeros((n, r))])
    J = -P @ np.linalg.solve(Mr, Nsel)               # M y = P^T grad, J = -P M^{-1} N
    out = dict(J=J, Jtheta=J[:, :16] @ Q_BASIS, Mr=Mr, Hr=Hr, lam_r=lam_r,
               sel=sel, r=r, condMr=float(np.linalg.cond(Mr)),
               eigHr=np.linalg.eigvalsh(Hr), G=G, Gr=Gr)
    return out


def alternative_certificates(sol, n_grid=201, tmax=1.0, As=None):
    As = CONS if As is None else list(As)
    """Dual non-uniqueness: the dual optimal face around the solver's H.
    Directions v with G^T v = 0 keep H x = 0; explore H_t = H + t sum v_i A_i
    and keep the t with H_t >= 0.  Returns list of (t, lam_t, H_t, corankH_t)."""
    x = sol["x"]; H = sol["H"]; lam = sol["lam"]        # [l1,...,lm,l0]
    G = np.vstack([C @ x for C in As] + [A0 @ x])
    # v in ker(G^T):  G^T is n x 4, rank 3 -> 1-dim kernel
    u, s, Vt = np.linalg.svd(G.T)
    r = int(np.sum(s > 1e-8 * s[0]))
    V = Vt[r:].T                                        # 4 x (4-r)
    out = []
    for k in range(V.shape[1]):
        v = V[:, k]
        for t in np.linspace(-tmax, tmax, n_grid):
            lam_t = lam + t * v
            H_t = sol["Q"] + lam_t[-1] * A0 + sum(lam_t[i] * As[i] for i in range(len(As)))
            ev = np.linalg.eigvalsh(0.5 * (H_t + H_t.T))
            if ev[0] >= -1e-10 * max(1.0, abs(ev[-1])):
                out.append((float(t), lam_t, H_t, int(np.sum(ev < 1e-6 * max(1.0, ev[-1]))), ev))
    return out


def jacobian_from_H(sol, H, lam, sel=None, theta=None, As=None):
    As = CONS if As is None else list(As)
    """Thm.2 Jacobian for an arbitrary certificate H whose multipliers are lam
    (g-order [l1,l2,l3,l0]).  Used for the dual-non-uniqueness test."""
    x = sol["x"]; n = x.size; m = len(As)
    G = np.vstack([C @ x for C in As] + [A0 @ x])
    if sel is None:
        sel = independent_rows(G)
    sel = list(sel)
    r = len(sel)
    Gr = G[sel]
    Mr = 2 * np.block([[H, Gr.T], [Gr, np.zeros((r, r))]])
    Nfull = jacobian_N(x, lam, m)
    Nsel = keep_rows(Nfull, n, sel, m)
    P = np.hstack([np.eye(n), np.zeros((n, r))])
    Mrp = Mr.T @ np.linalg.solve(Mr @ Mr.T, np.eye(n + r))
    J = -P @ Mrp @ Nsel
    return J, J[:, :16] @ Q_BASIS, dict(Mr=Mr, condMr=float(np.linalg.cond(Mr)),
                                       sel=sel, r=r)
