"""LaTeX tables for Section 6 from figs/numbers.json (written by make_figures.py)."""
import json
import os
import sys

OUT = sys.argv[1]
with open(os.path.join(OUT, 'numbers.json')) as fh:
    NUM = json.load(fh)

LTL_TEX = [r'\X a_1', r'\G a_1', r'\U(a_1,a_2)', r'\F(a_1 \wedge \F a_2)', r'\G(a_1 \rightarrow \F a_2)',
           r'\F(\U(a_1 \wedge a_2, a_3))', r'\G(\neg a_1 \vee \X a_2)', r'\F(a_1 \wedge \X(a_2 \vee a_3) \wedge \F a_4)']
CTL_TEX = [r'\A\F a_1', r'\E\F\E\X a_3', r'\E\F(a_1 \wedge \E\F(a_2))', r'\E\G(a_1 \wedge (a_2 \vee a_3))',
           r'\A\G(a_1 \rightarrow \A\F a_2)', r'\A\U(a_1 \rightarrow a_2, a_3)',
           r'\A\G(a_1 \rightarrow \A\F(a_2 \wedge \E\X\E\U(a_3, a_4)))', r'\A\G(a_1 \rightarrow \A\F\E\X(\neg a_2))',
           r'\A\G(\E\F(a_1 \wedge \E\X a_2) \rightarrow \E\G(\neg a_3 \vee \A\U(a_4, \A\F a_5)))',
           None]
PHI10 = (r'\A\G((a_1 \rightarrow \A\F(a_2 \wedge \E\X(a_3 \vee \E\U(\neg a_3, a_4)))) \vee {}',
         r'\quad \E\G(\neg a_2 \wedge \E\U(a_1, a_4 \vee \A\F a_5)))')
NAMES = [r'\psi_{%d}' % i for i in range(1, 9)] + [r'\varphi_{%d}' % i for i in range(1, 11)]


def fmt_lim(x):
    if abs(x) < 1e-12:
        return '0'
    if abs(x - 1) < 1e-12:
        return '1'
    return f'{x:.4g}'


def fmt_n(x):
    return '--' if x is None else str(int(x))


def fmt_ms(x):
    if x is None:
        return '--'
    return f'{x:.1f}' if x < 100 else f'{x:.0f}'


def table_fixed():
    fx = NUM['fixed']
    lines = []
    for t, name in enumerate(NAMES):
        d = fx[name]
        tex = (LTL_TEX + CTL_TEX)[t]
        cells = [fmt_lim(d['cplus']), fmt_lim(d['c1']), fmt_n(d['neps_plus']), fmt_n(d['neps_one']),
                 fmt_ms(d['time_ms_250'])]
        if tex is None:
            lines.append(f'${name}$ & ${PHI10[0]}$ & ' + ' & '.join(cells) + r' \\')
            lines.append(f' & ${PHI10[1]}$ & & & & & \\\\')
        else:
            lines.append(f'${name}$ & ${tex}$ & ' + ' & '.join(cells) + r' \\')
        if t == 7:
            lines.append(r'\midrule')
    return '\n'.join(lines)


def table_sens():
    sens = NUM['sens']
    rows = []
    import numpy as np
    for name, label in (('fixed_pR0.05', '0.05'), ('fixed_pR0.1', '0.1'), ('fixed_pR0.2', '0.2')):
        if name not in sens:
            continue
        d = sens[name]
        cells = []
        for logic_idx, logic in ((range(0, 8), 'LTL'), (range(8, 18), 'CTL')):
            for model in ('plus', 'one'):
                keys = [k for k in d if k.endswith('|' + model)]
                # keep insertion order: formulas in order
                vals = [d[k] for k in keys]
                sel = [vals[i] for i in logic_idx]
                fin = [v for v in sel if v is not None]
                med = np.median(fin) if fin else None
                mx = max(fin) if len(fin) == len(sel) else None
                cells.append(f'{int(med)} / {int(mx)}' if med is not None and mx is not None else '--')
        rows.append(f'${label}$ & ' + ' & '.join(cells) + r' \\')
    return '\n'.join(rows)


def table_sparse():
    """Two blocks side by side: LTL (8 rows) | CTL (10 rows)."""
    sp = NUM['sparse']
    keys = list(sp.keys())
    def cells(t):
        d = sp[keys[t]]
        i = len(d['n']) - 1
        return (f"${NAMES[t]}$ & {fmt_lim(d['dense_cplus'])} & {d['pplus'][i]:.3f} & "
                f"{fmt_lim(d['dense_c1'])} & {d['p1'][i]:.3f}")
    lines = []
    for r in range(10):
        left = cells(r) if r < 8 else ' & & & & '
        right = cells(8 + r)
        lines.append(left + ' & ' + right + r' \\')
    return '\n'.join(lines)


with open(os.path.join(OUT, 'tab_fixed.tex'), 'w') as fh:
    fh.write(table_fixed())
if NUM.get('sens'):
    with open(os.path.join(OUT, 'tab_sens.tex'), 'w') as fh:
        fh.write(table_sens())
if NUM.get('sparse'):
    with open(os.path.join(OUT, 'tab_sparse.tex'), 'w') as fh:
        fh.write(table_sparse())
print('tables written')
