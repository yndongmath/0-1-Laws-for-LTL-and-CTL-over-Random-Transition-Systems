"""
CTL: explicit-state model checking and the limit algorithm of Theorem 5.4.

`sat(phi, A, Lab)` returns the boolean vector V_phi (vertices satisfying phi)
using the standard fixpoint characterisations.

`reduce_to_prop(phi, k)` implements the bottom-up algorithm behind Theorem 5.4:
every CTL formula is exponentially equivalent to a propositional formula alpha.
Propositional formulas are represented by their truth tables over the 2^k
minterms (an int bitmask), so each rewrite step of Lemma 5.7 costs O(2^k).
"""
import numpy as np


# ------------------------------------------------------------ model checking

def _ex(A_int, x):
    # v |= EX x  iff  some successor satisfies x
    return (A_int @ x.astype(np.int32)) > 0


def sat(phi, A, Lab, _cache=None):
    """V_phi as a boolean vector of length n."""
    if _cache is None:
        _cache = {}
    if '__A' not in _cache:
        _cache['__A'] = A.astype(np.int32)
    A_int = _cache['__A']
    key = phi
    if key in _cache:
        return _cache[key]
    n = A.shape[0]
    op = phi[0]
    if op == 'true':
        r = np.ones(n, dtype=bool)
    elif op == 'false':
        r = np.zeros(n, dtype=bool)
    elif op == 'ap':
        r = Lab[:, phi[1]].copy()
    elif op == 'not':
        r = ~sat(phi[1], A, Lab, _cache)
    elif op == 'and':
        r = sat(phi[1], A, Lab, _cache) & sat(phi[2], A, Lab, _cache)
    elif op == 'or':
        r = sat(phi[1], A, Lab, _cache) | sat(phi[2], A, Lab, _cache)
    elif op == 'imp':
        r = ~sat(phi[1], A, Lab, _cache) | sat(phi[2], A, Lab, _cache)
    elif op == 'EX':
        r = _ex(A_int, sat(phi[1], A, Lab, _cache))
    elif op == 'AX':
        r = ~_ex(A_int, ~sat(phi[1], A, Lab, _cache))
    elif op == 'EU':
        r = _eu(A_int, sat(phi[1], A, Lab, _cache), sat(phi[2], A, Lab, _cache))
    elif op == 'EF':
        r = _eu(A_int, np.ones(n, dtype=bool), sat(phi[1], A, Lab, _cache))
    elif op == 'EG':
        r = _eg(A_int, sat(phi[1], A, Lab, _cache))
    elif op == 'AF':
        r = ~_eg(A_int, ~sat(phi[1], A, Lab, _cache))
    elif op == 'AG':
        r = ~_eu(A_int, np.ones(n, dtype=bool), ~sat(phi[1], A, Lab, _cache))
    elif op == 'AU':
        a = sat(phi[1], A, Lab, _cache)
        b = sat(phi[2], A, Lab, _cache)
        # A(a U b) = !( E(!b U (!a & !b)) | EG !b )
        r = ~(_eu(A_int, ~b, ~a & ~b) | _eg(A_int, ~b))
    else:
        raise ValueError(f'not a CTL operator: {op}')
    _cache[key] = r
    return r


def _eu(A_int, a, b):
    # least fixpoint Z = b | (a & EX Z)
    Z = b.copy()
    while True:
        Znew = b | (a & _ex(A_int, Z))
        if (Znew == Z).all():
            return Z
        Z = Znew


def _eg(A_int, a):
    # greatest fixpoint Z = a & EX Z
    Z = a.copy()
    while True:
        Znew = a & _ex(A_int, Z)
        if (Znew == Z).all():
            return Z
        Z = Znew


# ------------------------------------------- limit algorithm (Theorem 5.4)

def reduce_to_prop(phi, k):
    """Truth table (bitmask over the 2^k minterms) of the propositional formula
    alpha with phi ==_exp alpha, computed bottom-up with Lemma 5.7."""
    m = 2 ** k
    full = (1 << m) - 1

    def ap_mask(i):
        return sum(1 << s for s in range(m) if (s >> i) & 1)

    def rec(f):
        op = f[0]
        if op == 'true':
            return full
        if op == 'false':
            return 0
        if op == 'ap':
            return ap_mask(f[1])
        if op == 'not':
            return full & ~rec(f[1])
        if op == 'and':
            return rec(f[1]) & rec(f[2])
        if op == 'or':
            return rec(f[1]) | rec(f[2])
        if op == 'imp':
            return (full & ~rec(f[1])) | rec(f[2])
        if op in ('EX', 'EF'):                 # Lemma 5.7 (1)
            return 0 if rec(f[1]) == 0 else full
        if op in ('AX', 'AG'):                 # Lemma 5.7 (2)
            return full if rec(f[1]) == full else 0
        if op in ('AF', 'EG'):                 # Lemma 5.7 (3)
            return rec(f[1])
        if op == 'AU':                         # Lemma 5.7 (3): A(a2 U a1) == a1
            rec(f[1])
            return rec(f[2])
        if op == 'EU':                         # Lemma 5.7 (4)
            a1, a2 = rec(f[1]), rec(f[2])
            return 0 if a2 == 0 else (a1 | a2)
        raise ValueError(f'not a CTL operator: {op}')
    return rec(phi)


def minterm_probs(ps):
    """pi(sigma) for every minterm code sigma (bit i = a_{i+1})."""
    k = len(ps)
    pr = np.ones(2 ** k)
    for s in range(2 ** k):
        for i in range(k):
            pr[s] *= ps[i] if (s >> i) & 1 else 1 - ps[i]
    return pr


def mask_to_vec(mask, k):
    return np.array([(mask >> s) & 1 for s in range(2 ** k)], dtype=bool)


def limits(phi, ps):
    """(lim P(K^+ |= phi), lim P(K^1 |= phi), truth-table mask of alpha)."""
    k = len(ps)
    mask = reduce_to_prop(phi, k)
    vec = mask_to_vec(mask, k)
    c1 = float(minterm_probs(ps)[vec].sum())
    cplus = 1.0 if vec.all() else 0.0
    return cplus, c1, vec
