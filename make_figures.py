"""
Figures and tables for Section 6 from the JSON results in results/.

  python make_figures.py <outdir>

Writes PDF figures and a file `numbers.json` with all numbers quoted in the
text, plus LaTeX table bodies (`tab_*.tex`).
"""
import json
import logging
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import matplotlib.patches

# Matplotlib subsets its bundled fonts with fontTools, which warns about their old
# header timestamps when writing PDFs. The warning is harmless; silence it.
logging.getLogger('fontTools').setLevel(logging.ERROR)
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'figs')
os.makedirs(OUT, exist_ok=True)

# ------------------------------------------------------------------ style
TEXTWIDTH_IN = 372.0 / 72.27          # sn-jnl \textwidth
SURFACE = '#ffffff'
INK = '#0b0b0b'
INK2 = '#52514e'
GRID = '#e4e3df'
C_PLUS = '#2a78d6'    # categorical slot 1: K^+ (multiple initial states)
C_ONE = '#eb6834'     # categorical slot 2: K^1 (one initial state)
C_LTL = '#1baf7a'     # aqua: LTL (random benchmark)
C_CTL = '#6250d6'     # violet: CTL (random benchmark)

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['cmr10', 'DejaVu Serif'],
    'mathtext.fontset': 'cm',
    'axes.formatter.use_mathtext': True,
    'font.size': 7.5,
    'axes.titlesize': 7.5,
    'axes.labelsize': 7.5,
    'xtick.labelsize': 6.5,
    'ytick.labelsize': 6.5,
    'legend.fontsize': 7,
    'axes.edgecolor': INK2,
    'axes.labelcolor': INK,
    'xtick.color': INK2,
    'ytick.color': INK2,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.major.size': 2.5,
    'ytick.major.size': 2.5,
    'lines.linewidth': 1.1,
    'axes.grid': True,
    'grid.color': GRID,
    'grid.linewidth': 0.5,
    'figure.facecolor': SURFACE,
    'axes.facecolor': SURFACE,
    'savefig.facecolor': SURFACE,
    'pdf.fonttype': 42,
})

EPS = 0.01


def load(name):
    with open(os.path.join(RES, name + '.json')) as fh:
        return json.load(fh)


def series(res, j, stat, nmax=None):
    ns, vals, ses = [], [], []
    for pt in res['points']:
        if nmax is not None and pt['n'] > nmax:
            continue
        ns.append(pt['n'])
        vals.append(pt[stat][j])
        ses.append(pt[stat + '_se'][j])
    return np.array(ns), np.array(vals), np.array(ses)


def deviation(res, j, model, nmax=None):
    """Estimated |P_n - lim| for formula j and model 'plus' / 'one'."""
    f = res['formulas'][j]
    if model == 'plus':
        stat = 'pplus' if f['cplus'] == 0.0 else 'oneminus_pplus'
        n, v, se = series(res, j, stat, nmax)
        return n, np.abs(v), se
    n, v, se = series(res, j, 'dev1', nmax)
    return n, np.abs(v), se


def n_eps(n, dev, eps=EPS):
    """Smallest grid n from which the estimated deviation stays <= eps."""
    ok = dev <= eps
    for i in range(len(n)):
        if ok[i:].all():
            return int(n[i])
    return None


def fit_rate(n, dev, se, nmin=None):
    """Least-squares slope of log(dev) on n, over points that are clearly
    resolved (dev > 3 se and dev > 0); returns (rate a, R^2, #points)."""
    m = (dev > 0) & (dev > 3 * se)
    if nmin is not None:
        m &= n >= nmin
    if m.sum() < 4:
        return None, None, int(m.sum())
    x, y = n[m], np.log(dev[m])
    A = np.vstack([x, np.ones_like(x)]).T
    coef, res_, _, _ = np.linalg.lstsq(A, y, rcond=None)
    yhat = A @ coef
    r2 = 1 - ((y - yhat) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-300)
    return -coef[0], r2, int(m.sum())



def upright_bb(texts):
    """Matplotlib's \\mathbb is slanted; use the Unicode character U+1D542 instead,
    so that the blackboard K is upright, as \\mathbb{K} in the paper."""
    for t in texts:
        t.set_text(t.get_text().replace(r'\mathbb{K}', '\U0001D542'))


def clean_axes(ax):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)


# ------------------------------------------------------------------ fixed formulas
def fig_fixed(res, logic, names, fname, ncols):
    idx = [j for j, f in enumerate(res['formulas']) if f['logic'] == logic]
    nf = len(idx)
    nrow_blocks = int(np.ceil(nf / ncols))
    height = 1.18 * 2 * nrow_blocks + 0.55
    fig, axes = plt.subplots(2 * nrow_blocks, ncols, figsize=(TEXTWIDTH_IN, height),
                             sharex=True, gridspec_kw={'height_ratios': [1.0, 0.85] * nrow_blocks})
    NMAX = 250
    for t, j in enumerate(idx):
        b, c = divmod(t, ncols)
        axp, axd = axes[2 * b, c], axes[2 * b + 1, c]
        f = res['formulas'][j]
        # probabilities (conditional estimator) and limits
        for model, stat, col, mk in (('plus', 'pplus', C_PLUS, 'o'), ('one', 'p1', C_ONE, 's')):
            n, v, se = series(res, j, stat, NMAX)
            axp.plot(n, v, color=col, lw=1.1, marker=mk, markersize=2.6, markevery=12,
                     markeredgewidth=0, zorder=3)
            lim = f['cplus'] if model == 'plus' else f['c1']
            axp.axhline(lim, color=col, lw=0.8, ls=(0, (3, 2)), zorder=2)
            nd, dv, dse = deviation(res, j, model, NMAX)
            m = dv > 0
            axd.semilogy(nd[m], dv[m], color=col, lw=1.0, marker=mk, markersize=2.4,
                         markevery=12, markeredgewidth=0)
        axp.set_ylim(-0.04, 1.04)
        axp.set_yticks([0, 0.5, 1])
        axd.set_ylim(1e-12, 1.5)
        axd.set_yticks([1e-12, 1e-8, 1e-4, 1])
        axd.axhline(EPS, color=INK2, lw=0.6, ls=':', zorder=1)
        axp.set_title(f'${names[t]}$ = ' + '$' + f['tex_short'] + '$', loc='left', pad=2.5,
                      fontsize=7)
        for ax in (axp, axd):
            clean_axes(ax)
            ax.set_xlim(0, NMAX)
            ax.set_xticks([0, 100, 200])
            if c != 0:
                ax.tick_params(labelleft=False)
        if c == 0:
            axp.set_ylabel('probability')
            axd.set_ylabel(r'$|P_n - \mathrm{limit}|$')
        if b == nrow_blocks - 1 or (b == nrow_blocks - 2 and t + ncols >= nf):
            axd.set_xlabel(r'number of states $n$')
    for t in range(nf, nrow_blocks * ncols):
        b, c = divmod(t, ncols)
        axes[2 * b, c].axis('off')
        axes[2 * b + 1, c].axis('off')
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=C_PLUS, marker='o', markersize=3, markeredgewidth=0,
                      label=r'multiple initial states, $\mathbb{K}^{+}_{n}$'),
               Line2D([], [], color=C_ONE, marker='s', markersize=3, markeredgewidth=0,
                      label=r'one initial state, $\mathbb{K}^{1}_{n}$'),
               Line2D([], [], color=INK2, ls=(0, (3, 2)), lw=0.8, label='predicted limit'),
               Line2D([], [], color=INK2, ls=':', lw=0.6, label=r'$\varepsilon = 0.01$')]
    leg = fig.legend(handles=handles, loc='upper center', ncol=4, frameon=False,
                     bbox_to_anchor=(0.5, 1.0), handlelength=2.2, columnspacing=1.4)
    upright_bb(leg.get_texts())
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.32 / height), h_pad=0.35, w_pad=0.5)
    fig.savefig(os.path.join(OUT, fname))
    plt.close(fig)


def short_tex(s):
    return s


# ------------------------------------------------------------------ random formulas
def fig_random(res, fname, numbers):
    F = res['formulas']
    pts = res['points']
    ns = np.array([p['n'] for p in pts])
    logics = np.array([f['logic'] for f in F])
    c1 = np.array([f['c1'] for f in F])
    cp = np.array([f['cplus'] for f in F])
    devp = np.zeros((len(F), len(ns)))
    dev1 = np.zeros((len(F), len(ns)))
    sep = np.zeros_like(devp)
    se1 = np.zeros_like(devp)
    for j in range(len(F)):
        _, devp[j], sep[j] = deviation(res, j, 'plus')
        _, dev1[j], se1[j] = deviation(res, j, 'one')

    fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH_IN, 2.25),
                             gridspec_kw={'width_ratios': [1.15, 1, 1]})
    # (a) distribution of limits for K^1
    ax = axes[0]
    vals = np.round(np.arange(0, 17) / 16, 4)
    width = 0.022
    for off, lg, col, hatch in ((-width / 2, 'LTL', C_LTL, None), (width / 2, 'CTL', C_CTL, None)):
        cnt = np.array([np.sum(np.isclose(c1[logics == lg], v)) for v in vals])
        frac = cnt / max((logics == lg).sum(), 1)
        ax.bar(vals + off, frac, width=width, color=col, label=lg, linewidth=0)
    upright_bb([ax.set_xlabel(r'limit $\lim_n P(\mathbb{K}^{1}_{n} \models \varphi)$')])
    ax.set_ylabel('fraction of formulae')
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.legend(frameon=False, loc='upper center')
    ax.set_title('(a) limits, one initial state', loc='left', fontsize=7, pad=2.5)
    clean_axes(ax)
    ax.grid(axis='x', visible=False)
    # (b), (c) quantiles of deviation
    for ax, dev, title in ((axes[1], devp, r'(b) $|P(\mathbb{K}^{+}_{n} \models \varphi) - \mathrm{limit}|$'),
                           (axes[2], dev1, r'(c) $|P(\mathbb{K}^{1}_{n} \models \varphi) - \mathrm{limit}|$')):
        for lg, col in (('LTL', C_LTL), ('CTL', C_CTL)):
            D = dev[logics == lg]
            med = np.quantile(D, 0.5, axis=0)
            q90 = np.quantile(D, 0.9, axis=0)
            mx = D.max(axis=0)
            for q, ls, lab in ((med, '-', 'median'), (q90, '--', '90%'), (mx, ':', 'max')):
                m = q > 0
                ax.semilogy(ns[m], q[m], color=col, ls=ls, lw=1.1 if ls == '-' else 0.9,
                            marker='o' if ls == '-' else None, markersize=2.4, markeredgewidth=0)
        ax.axhline(EPS, color=INK2, lw=0.6, ls=':')
        ax.set_ylim(1e-9, 1.5)
        ax.set_xlim(0, ns.max() * 1.02)
        ax.set_xlabel(r'number of states $n$')
        upright_bb([ax.set_title(title, loc='left', fontsize=7, pad=2.5)])
        clean_axes(ax)
    axes[2].tick_params(labelleft=False)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=C_LTL, label='LTL'), Line2D([], [], color=C_CTL, label='CTL'),
         Line2D([], [], color=INK2, ls='-', marker='o', markersize=2.4, label='median'),
         Line2D([], [], color=INK2, ls='--', lw=0.9, label='90% quantile'),
         Line2D([], [], color=INK2, ls=':', lw=0.9, label='maximum')]
    axes[0].get_legend().remove()
    h = [matplotlib.patches.Patch(color=C_LTL, label='LTL'), matplotlib.patches.Patch(color=C_CTL, label='CTL')] + h[2:]
    fig.legend(handles=h, frameon=False, loc='upper center', ncol=5, bbox_to_anchor=(0.5, 1.0),
               handlelength=2.2, columnspacing=1.4)
    fig.tight_layout(w_pad=0.6, rect=(0, 0, 1, 0.88))
    fig.savefig(os.path.join(OUT, fname))
    plt.close(fig)

    # numbers
    out = {}
    for lg in ('LTL', 'CTL'):
        sel = logics == lg
        out[lg] = {
            'count': int(sel.sum()),
            'cplus_one': int((cp[sel] == 1).sum()),
            'c1_zero': int(np.isclose(c1[sel], 0).sum()),
            'c1_one': int(np.isclose(c1[sel], 1).sum()),
            'c1_strict': int(((c1[sel] > 1e-9) & (c1[sel] < 1 - 1e-9)).sum()),
            'c1_distinct': int(len(set(np.round(c1[sel], 6)))),
        }
        for model, dev in (('plus', devp), ('one', dev1)):
            D = dev[sel]
            neps = [n_eps(ns, D[i]) for i in range(D.shape[0])]
            neps_arr = np.array([x if x is not None else np.inf for x in neps])
            out[lg][model] = {
                'frac_within_eps_by_n': {int(n): float((D[:, i] <= EPS).mean()) for i, n in enumerate(ns)},
                'max_dev_by_n': {int(n): float(D[:, i].max()) for i, n in enumerate(ns)},
                'median_dev_by_n': {int(n): float(np.median(D[:, i])) for i, n in enumerate(ns)},
                'n_eps_median': float(np.median(neps_arr)),
                'n_eps_q90': float(np.quantile(neps_arr, 0.9)),
                'n_eps_max': float(neps_arr.max()),
                'n_eps_none': int(np.isinf(neps_arr).sum()),
            }
            sizes = np.array([f['size'] for f in F])[sel]
            depths = np.array([f['depth'] for f in F])[sel]
            fin = np.isfinite(neps_arr)
            r_s = spearmanr(sizes[fin], neps_arr[fin]).correlation if fin.sum() > 3 else None
            r_d = spearmanr(depths[fin], neps_arr[fin]).correlation if fin.sum() > 3 else None
            out[lg][model]['spearman_size'] = None if r_s is None or np.isnan(r_s) else float(r_s)
            out[lg][model]['spearman_depth'] = None if r_d is None or np.isnan(r_d) else float(r_d)
        # agreement at the largest n: |estimate - limit| relative to SE
        last = len(ns) - 1
        zp = devp[sel, last] / np.maximum(sep[sel, last], 1e-300)
        z1 = dev1[sel, last] / np.maximum(se1[sel, last], 1e-300)
        out[lg]['largest_n'] = int(ns[last])
        out[lg]['max_dev_largest_n_plus'] = float(devp[sel, last].max())
        out[lg]['max_dev_largest_n_one'] = float(dev1[sel, last].max())
    out['aut_states_max'] = max(f['aut_states'] or 0 for f in F)
    out['aut_states_median'] = float(np.median([f['aut_states'] for f in F if f['logic'] == 'LTL']))
    out['sizes'] = [int(min(f['size'] for f in F)), int(max(f['size'] for f in F))]
    out['grid'] = [int(n) for n in ns]
    out['N'] = [int(p['N']) for p in pts]
    out['cpu_seconds'] = float(sum(p['cpu_seconds'] for p in pts))
    numbers['random'] = out


# ------------------------------------------------------------------ main
def main():
    numbers = {}
    PSI = [r'\psi_{%d}' % i for i in range(1, 9)]
    PHI = [r'\varphi_{%d}' % i for i in range(1, 11)]
    SHORT_LTL = [r'\mathsf{X}a_1', r'\mathsf{G}a_1', r'a_1\,\mathsf{U}\,a_2', r'\mathsf{F}(a_1\wedge\mathsf{F}a_2)',
                 r'\mathsf{G}(a_1\!\rightarrow\!\mathsf{F}a_2)', r'\mathsf{F}((a_1\wedge a_2)\,\mathsf{U}\,a_3)',
                 r'\mathsf{G}(\neg a_1\vee\mathsf{X}a_2)', r'\mathsf{F}(a_1\wedge\mathsf{X}(\ldots)\wedge\mathsf{F}a_4)']
    SHORT_CTL = [r'\mathsf{AF}a_1', r'\mathsf{EFEX}a_3', r'\mathsf{EF}(a_1\wedge\mathsf{EF}a_2)',
                 r'\mathsf{EG}(a_1\wedge(\ldots))', r'\mathsf{AG}(a_1\!\rightarrow\!\mathsf{AF}a_2)',
                 r'\mathsf{A}(\ldots\,\mathsf{U}\,a_3)', r'\mathsf{AG}(a_1\!\rightarrow\!\ldots)',
                 r'\mathsf{AG}(a_1\!\rightarrow\!\ldots)', r'\mathsf{AG}(\ldots)', r'\mathsf{AG}(\ldots)']
    if os.path.exists(os.path.join(RES, 'fixed_pR0.1.json')):
        res = load('fixed_pR0.1')
        L = [j for j, f in enumerate(res['formulas']) if f['logic'] == 'LTL']
        C = [j for j, f in enumerate(res['formulas']) if f['logic'] == 'CTL']
        for t, j in enumerate(L):
            res['formulas'][j]['tex_short'] = SHORT_LTL[t]
        for t, j in enumerate(C):
            res['formulas'][j]['tex_short'] = SHORT_CTL[t]
        fig_fixed(res, 'LTL', PSI, 'fig_ltl.pdf', 4)
        fig_fixed(res, 'CTL', PHI, 'fig_ctl.pdf', 5)
        timing = load('timing') if os.path.exists(os.path.join(RES, 'timing.json')) else None
        rows = []
        fx = {}
        for t, j in enumerate(L + C):
            f = res['formulas'][j]
            name = (PSI + PHI)[t]
            n_p, d_p, s_p = deviation(res, j, 'plus', 250)
            n_1, d_1, s_1 = deviation(res, j, 'one', 250)
            ne_p, ne_1 = n_eps(n_p, d_p), n_eps(n_1, d_1)
            ap, r2p, kp = fit_rate(n_p, d_p, s_p)
            a1, r21, k1 = fit_rate(n_1, d_1, s_1)
            # beyond 250
            nb_p, db_p, _ = deviation(res, j, 'plus')
            nb_1, db_1, _ = deviation(res, j, 'one')
            big = nb_p > 250
            tms = None if timing is None else 1000 * timing['250'][t]
            tms1000 = None if timing is None else 1000 * timing['1000'][t]
            pym = None if timing is None or 'pymc_250' not in timing else timing['pymc_250'][t]
            fx[name] = {'cplus': f['cplus'], 'c1': f['c1'], 'neps_plus': ne_p, 'neps_one': ne_1,
                        'rate_plus': ap, 'r2_plus': r2p, 'k_plus': kp,
                        'rate_one': a1, 'r2_one': r21, 'k_one': k1,
                        'maxdev_big_plus': float(db_p[big].max()) if big.any() else None,
                        'maxdev_big_one': float(db_1[big].max()) if big.any() else None,
                        'time_ms_250': tms, 'time_ms_1000': tms1000, 'pymc_s_250': pym,
                        'size': f['size'], 'depth': f['depth'], 'aut_states': f['aut_states']}
        numbers['fixed'] = fx
        # agreement of plain and conditional estimators: |raw - rb| <= 3 se(raw)
        tot = ok = 0
        worst = 0.0
        for pt in res['points']:
            N = pt['N']
            for j in range(len(res['formulas'])):
                for raw, rb in (('raw1', 'p1'), ('rawplus', 'pplus')):
                    r_, b_ = pt[raw][j], pt[rb][j]
                    se = max(pt[raw + '_se'][j], np.sqrt(max(b_ * (1 - b_), 0) / N), 1.0 / N)
                    z = abs(r_ - b_) / se
                    worst = max(worst, z)
                    tot += 1
                    ok += z <= 3
        numbers['raw_agreement'] = {'total': tot, 'within_3se': ok, 'worst_z': worst}
        numbers['fixed_meta'] = {'cpu_seconds': float(sum(p['cpu_seconds'] for p in res['points'])),
                                 'grid_max': max(p['n'] for p in res['points'])}
    if os.path.exists(os.path.join(RES, 'random_pR0.1.json')):
        fig_random(load('random_pR0.1'), 'fig_random.pdf', numbers)
    # sensitivity
    sens = {}
    for name in ('fixed_pR0.05', 'fixed_pR0.1', 'fixed_pR0.2'):
        if not os.path.exists(os.path.join(RES, name + '.json')):
            continue
        r = load(name)
        d = {}
        for j, f in enumerate(r['formulas']):
            for model in ('plus', 'one'):
                n, dv, _ = deviation(r, j, model, 512)
                d[f"{f['str']}|{model}"] = n_eps(n, dv)
        sens[name] = d
    numbers['sens'] = sens
    if os.path.exists(os.path.join(RES, 'sparse_c3.json')):
        r = load('sparse_c3')
        sp = {}
        for j, f in enumerate(r['formulas']):
            n, p1, s1 = series(r, j, 'p1')
            _, pp, spp = series(r, j, 'pplus')
            sp[f['str']] = {'n': n.tolist(), 'p1': p1.tolist(), 'p1_se': s1.tolist(),
                            'pplus': pp.tolist(), 'pplus_se': spp.tolist(),
                            'dense_c1': f['c1'], 'dense_cplus': f['cplus']}
        numbers['sparse'] = sp
    if os.path.exists(os.path.join(RES, 'validation.json')):
        numbers['validation'] = load('validation')
    with open(os.path.join(OUT, 'numbers.json'), 'w') as fh:
        json.dump(numbers, fh, indent=1, default=lambda o: None)


if __name__ == '__main__':
    main()
