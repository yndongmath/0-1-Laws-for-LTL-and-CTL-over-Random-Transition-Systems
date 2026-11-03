"""
LTL: explicit-state model checking via the automata-theoretic approach.

  1. negate phi and put it in negation normal form;
  2. translate to a generalised Buchi automaton with the tableau construction
     of Gerth, Peled, Vardi and Wolper (PSTV 1995);
  3. build the product with the Kripke structure (sparse matrices);
  4. a vertex v violates phi iff some initial product state (v, q0) can reach
     a non-trivial strongly connected component that meets every acceptance
     set.

`violating(aut, A, Lab)` returns, for *all* vertices at once, whether some
path from that vertex violates phi; hence V_phi = complement.  This is what
lets one sample of (R, L) serve both K^+ and K^1.

`limits(phi, ps)` computes the limit probabilities of Theorems 4.3 and 4.6 by
checking phi on the complete structure U_k: sigma -> phi is valid iff every
path of U_k from the vertex labelled sigma satisfies phi.
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components, breadth_first_order

from .sampler import universal_structure
from .ctl import minterm_probs


# ---------------------------------------------------------------- NNF

def nnf(f, neg=False):
    op = f[0]
    if op == 'true':
        return ('false',) if neg else ('true',)
    if op == 'false':
        return ('true',) if neg else ('false',)
    if op == 'ap':
        return ('nap', f[1]) if neg else f
    if op == 'not':
        return nnf(f[1], not neg)
    if op == 'imp':
        return nnf(('or', ('not', f[1]), f[2]), neg)
    if op == 'and':
        return (('or' if neg else 'and'), nnf(f[1], neg), nnf(f[2], neg))
    if op == 'or':
        return (('and' if neg else 'or'), nnf(f[1], neg), nnf(f[2], neg))
    if op == 'X':
        return ('X', nnf(f[1], neg))
    if op == 'F':      # F f = true U f ; !F f = G !f = false R !f
        return ('R', ('false',), nnf(f[1], True)) if neg else ('U', ('true',), nnf(f[1]))
    if op == 'G':      # G f = false R f ; !G f = F !f = true U !f
        return ('U', ('true',), nnf(f[1], True)) if neg else ('R', ('false',), nnf(f[1]))
    if op == 'U':
        return ('R', nnf(f[1], True), nnf(f[2], True)) if neg else ('U', nnf(f[1]), nnf(f[2]))
    if op == 'R':
        return ('U', nnf(f[1], True), nnf(f[2], True)) if neg else ('R', nnf(f[1]), nnf(f[2]))
    raise ValueError(f'not an LTL operator: {op}')


def _neg_lit(f):
    if f[0] == 'ap':
        return ('nap', f[1])
    if f[0] == 'nap':
        return ('ap', f[1])
    if f[0] == 'true':
        return ('false',)
    if f[0] == 'false':
        return ('true',)
    raise ValueError(f)


# -------------------------------------------------- GPVW tableau construction

class GBA:
    """Generalised Buchi automaton with state-based acceptance and literal
    labels on states (a run q0 q1 ... reads w0 w1 ... with w_i |= lits(q_i))."""

    def __init__(self, pos, neg, init, edges, acc):
        self.pos = pos        # list of lists of AP indices that must hold
        self.neg = neg        # list of lists of AP indices that must not hold
        self.init = init      # list of initial state ids
        self.edges = edges    # list of (q, q')
        self.acc = acc        # list of boolean arrays over states
        self.nstates = len(pos)


def translate(phi):
    """GBA for phi (phi is an LTL formula, not necessarily in NNF)."""
    f = nnf(phi)
    nodes = []   # dicts: incoming(set), old(frozenset), next(frozenset)
    counter = [0]

    def expand(incoming, new, old, nxt):
        while True:
            if not new:
                for nd in nodes:
                    if nd['old'] == old and nd['next'] == nxt:
                        nd['incoming'] |= incoming
                        return
                name = counter[0]
                counter[0] += 1
                nodes.append({'name': name, 'incoming': set(incoming),
                              'old': old, 'next': nxt})
                expand({name}, set(nxt), frozenset(), frozenset())
                return
            eta = new.pop()
            op = eta[0]
            if op in ('ap', 'nap', 'true', 'false'):
                if op == 'false' or _neg_lit(eta) in old:
                    return
                old = old | {eta}
                continue
            if op == 'and':
                new = new | ({eta[1], eta[2]} - old)
                old = old | {eta}
                continue
            if op == 'X':
                old = old | {eta}
                nxt = nxt | {eta[1]}
                continue
            if op in ('or', 'U', 'R'):
                if op == 'or':
                    new1, next1, new2 = {eta[1]}, set(), {eta[2]}
                elif op == 'U':
                    new1, next1, new2 = {eta[1]}, {eta}, {eta[2]}
                else:  # R
                    new1, next1, new2 = {eta[2]}, {eta}, {eta[1], eta[2]}
                old2 = old | {eta}
                expand(set(incoming), new | (new1 - old2), old2, nxt | next1)
                expand(set(incoming), new | (new2 - old2), old2, nxt)
                return
            raise ValueError(op)

    expand({'init'}, {f}, frozenset(), frozenset())

    # collect U-subformulas for acceptance
    us = set()

    def collect(g):
        if g[0] == 'U':
            us.add(g)
        if g[0] in ('and', 'or', 'U', 'R'):
            collect(g[1])
            collect(g[2])
        elif g[0] == 'X':
            collect(g[1])
    collect(f)

    idx = {nd['name']: i for i, nd in enumerate(nodes)}
    pos = [[l[1] for l in nd['old'] if l[0] == 'ap'] for nd in nodes]
    neg = [[l[1] for l in nd['old'] if l[0] == 'nap'] for nd in nodes]
    init = [i for i, nd in enumerate(nodes) if 'init' in nd['incoming']]
    edges = [(idx[src], i) for i, nd in enumerate(nodes)
             for src in nd['incoming'] if src != 'init']
    acc = [np.array([(u not in nd['old']) or (u[2] in nd['old']) for nd in nodes])
           for u in sorted(us, key=repr)]
    return _prune(GBA(pos, neg, init, edges, acc))


def _prune(aut):
    """Remove states that cannot reach an accepting cycle (sound: every
    accepting product run projects onto an accepting automaton run)."""
    Q = aut.nstates
    if Q == 0 or not aut.edges:
        return GBA([], [], [], [], [])
    r = np.array([e[0] for e in aut.edges])
    c = np.array([e[1] for e in aut.edges])
    M = sp.csr_matrix((np.ones(len(r), dtype=np.int8), (r, c)), shape=(Q, Q))
    nc, lab = connected_components(M, directed=True, connection='strong')
    size = np.bincount(lab, minlength=nc)
    sl = np.zeros(nc, dtype=bool)
    sl[lab[r[r == c]]] = True
    good = (size > 1) | sl
    for F in aut.acc:
        hit = np.zeros(nc, dtype=bool)
        hit[lab[F]] = True
        good &= hit
    useful = good[lab]
    while True:   # backward closure
        new = useful | (M @ useful.astype(np.int8) > 0)
        if (new == useful).all():
            break
        useful = new
    keep = np.flatnonzero(useful)
    if len(keep) == 0:
        return GBA([], [], [], [], [])
    re = {int(q): i for i, q in enumerate(keep)}
    return GBA([aut.pos[q] for q in keep], [aut.neg[q] for q in keep],
               [re[q] for q in aut.init if q in re],
               [(re[a], re[b]) for (a, b) in aut.edges if a in re and b in re],
               [F[keep] for F in aut.acc])


# ---------------------------------------------------------------- product

def violating(aut, A, Lab, src=None, dst=None):
    """Boolean vector: v has a path violating phi, where aut = translate(!phi)."""
    n = A.shape[0]
    Q = aut.nstates
    if Q == 0:
        return np.zeros(n, dtype=bool)
    if src is None:
        src, dst = np.nonzero(A)
    # which vertices are compatible with the literals of each automaton state
    valid = np.ones((Q, n), dtype=bool)
    for q in range(Q):
        for i in aut.pos[q]:
            valid[q] &= Lab[:, i]
        for i in aut.neg[q]:
            valid[q] &= ~Lab[:, i]
    if not aut.edges:
        return np.zeros(n, dtype=bool)
    ea = np.asarray(aut.edges, dtype=np.int64)
    VS = valid[:, src]
    VD = valid[:, dst]
    rows, cols = [], []
    step = max(1, int(2e7 // max(len(src), 1)))     # bound memory of the mask
    for s in range(0, len(ea), step):
        e = ea[s:s + step]
        ei, ej = np.nonzero(VS[e[:, 0]] & VD[e[:, 1]])
        rows.append(e[ei, 0] * n + src[ej])
        cols.append(e[ei, 1] * n + dst[ej])
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    N = Q * n
    if len(rows) == 0:
        return np.zeros(n, dtype=bool)
    P = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(N, N))
    ncomp, lab = connected_components(P, directed=True, connection='strong')
    # non-trivial components: size > 1 or a self-loop
    csize = np.bincount(lab, minlength=ncomp)
    selfloop = np.zeros(ncomp, dtype=bool)
    sl = rows == cols
    selfloop[lab[rows[sl]]] = True
    good = (csize > 1) | selfloop
    for F in aut.acc:
        Fn = np.repeat(F, n)  # product state (q, v) has index q*n+v
        hit = np.zeros(ncomp, dtype=bool)
        hit[lab[Fn]] = True
        good &= hit
    good_nodes = good[lab]
    # also require the node itself to be label-consistent
    good_nodes &= valid.reshape(-1)
    if not good_nodes.any():
        return np.zeros(n, dtype=bool)
    # backward reachability to good nodes: BFS from a virtual node N on the
    # transposed product graph, with edges N -> g for every good node g
    g = np.flatnonzero(good_nodes)
    r2 = np.concatenate([cols, np.full(len(g), N, dtype=np.int64)])
    c2 = np.concatenate([rows, g])
    PT_ext = sp.csr_matrix((np.ones(len(r2)), (r2, c2)), shape=(N + 1, N + 1))
    reach = breadth_first_order(PT_ext, N, directed=True, return_predecessors=False)
    can = np.zeros(N + 1, dtype=bool)
    can[reach] = True
    can = can[:N]
    out = np.zeros(n, dtype=bool)
    for q0 in aut.init:
        out |= can[q0 * n:(q0 + 1) * n] & valid[q0]
    return out


def sat(phi, A, Lab, aut=None):
    """V_phi: vertices all of whose paths satisfy phi."""
    if aut is None:
        aut = translate(('not', phi))
    return ~violating(aut, A, Lab)


def limits(phi, ps, aut=None):
    """(lim P(K^+ |= phi), lim P(K^1 |= phi), valid-minterm vector) via U_k."""
    k = len(ps)
    A, Lab = universal_structure(k)
    vec = sat(phi, A, Lab, aut)          # vec[s]: sigma_s -> phi is valid
    c1 = float(minterm_probs(ps)[vec].sum())
    cplus = 1.0 if vec.all() else 0.0
    return cplus, c1, vec
