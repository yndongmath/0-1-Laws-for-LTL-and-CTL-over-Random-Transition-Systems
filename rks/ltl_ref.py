"""
Independent reference LTL model checker (used only for validation).

Uses the atom (Hintikka-set) construction of Lichtenstein and Pnueli instead
of the GPVW tableau used in ltl.py: product states are pairs (v, A) where A is
a truth assignment to the elementary subformulas (atomic propositions,
X-formulas and U-formulas) consistent with the label of v.  Exponential in the
formula, so only for small formulas.
"""
import itertools
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components


def _core(f):
    """Rewrite into the core syntax true/ap/not/and/X/U."""
    op = f[0]
    if op in ('true', 'ap'):
        return f
    if op == 'false':
        return ('not', ('true',))
    if op == 'not':
        return ('not', _core(f[1]))
    if op == 'and':
        return ('and', _core(f[1]), _core(f[2]))
    if op == 'or':
        return ('not', ('and', ('not', _core(f[1])), ('not', _core(f[2]))))
    if op == 'imp':
        return ('not', ('and', _core(f[1]), ('not', _core(f[2]))))
    if op == 'X':
        return ('X', _core(f[1]))
    if op == 'F':
        return ('U', ('true',), _core(f[1]))
    if op == 'G':
        return ('not', ('U', ('true',), ('not', _core(f[1]))))
    if op == 'U':
        return ('U', _core(f[1]), _core(f[2]))
    if op == 'R':
        return ('not', ('U', ('not', _core(f[1])), ('not', _core(f[2]))))
    raise ValueError(op)


def _subs(f, acc):
    if f not in acc:
        acc.append(f)
    for g in f[1:]:
        if isinstance(g, tuple):
            _subs(g, acc)
    return acc


def violating_ref(phi, A, Lab):
    f = _core(phi)
    subs = _subs(f, [])
    elem = [g for g in subs if g[0] in ('X', 'U')]
    n = A.shape[0]

    def evaluate(assign, lab):
        val = {}

        def ev(g):
            if g in val:
                return val[g]
            op = g[0]
            if op == 'true':
                r = True
            elif op == 'ap':
                r = bool(lab[g[1]])
            elif op == 'not':
                r = not ev(g[1])
            elif op == 'and':
                r = ev(g[1]) and ev(g[2])
            elif op == 'X':
                r = assign[g]
            elif op == 'U':
                r = assign[g]
            val[g] = r
            return r
        for g in subs:
            ev(g)
        return val

    # enumerate (vertex, assignment) states with local U-consistency
    states, vals = [], []
    for v in range(n):
        for bits in itertools.product([False, True], repeat=len(elem)):
            assign = dict(zip(elem, bits))
            val = evaluate(assign, Lab[v])
            ok = True
            for g in elem:
                if g[0] == 'U' and val[g[2]] and not val[g]:
                    ok = False          # b holds -> a U b holds
                if g[0] == 'U' and val[g] and not val[g[1]] and not val[g[2]]:
                    ok = False          # a U b holds -> a or b
            if ok:
                states.append((v, bits))
                vals.append(val)
    S = len(states)
    rows, cols = [], []
    for i, (v, _) in enumerate(states):
        for j, (w, _) in enumerate(states):
            if not A[v, w]:
                continue
            ok = True
            for g in elem:
                if g[0] == 'X' and vals[i][g] != vals[j][g[1]]:
                    ok = False
                    break
                if g[0] == 'U':
                    expected = vals[i][g[2]] or (vals[i][g[1]] and vals[j][g])
                    if vals[i][g] != expected:
                        ok = False
                        break
            if ok:
                rows.append(i)
                cols.append(j)
    if not rows:
        return np.zeros(n, dtype=bool)
    P = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(S, S))
    nc, lab = connected_components(P, directed=True, connection='strong')
    rows, cols = np.array(rows), np.array(cols)
    size = np.bincount(lab, minlength=nc)
    sl = np.zeros(nc, dtype=bool)
    sl[lab[rows[rows == cols]]] = True
    good = (size > 1) | sl
    for g in elem:
        if g[0] == 'U':
            F = np.array([(not vals[i][g]) or vals[i][g[2]] for i in range(S)])
            hit = np.zeros(nc, dtype=bool)
            hit[lab[F]] = True
            good &= hit
    goodn = good[lab]
    # backward closure
    reach = goodn.copy()
    while True:
        # nodes with an edge into `reach`
        new = reach | (P @ reach.astype(float) > 0)
        if (new == reach).all():
            break
        reach = new
    out = np.zeros(n, dtype=bool)
    for i, (v, _) in enumerate(states):
        if reach[i] and not vals[i][f]:
            out[v] = True
    return out
