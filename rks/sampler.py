"""
Exact samplers for the random structures of the paper.

The transition relation R and the labelling L are sampled from the random
labelled digraph and conditioned on R being left-total.  Since the rows of R
are independent and left-totality is a property of each row separately, the
conditioning factorises over vertices: we resample every empty row until it is
non-empty.  This is exact rejection sampling, done row by row.

The initial states are *not* sampled here.  Both models are obtained from the
same (R, L):
  * K^1_{n,p}: I = {v} with v uniform in V (independent of R, L);
  * K^+_{n,p}: v in I independently with probability p_I, conditioned on
    I != {} (independent of R, L).
Given the satisfaction set V_phi of a formula, the conditional probabilities
of K |= phi are available in closed form (see estimators.py), which is what the
experiments use; `sample_initial_*` are provided for the plain estimator.
"""
import numpy as np


def sample_relation(n, p_R, rng):
    """Boolean n x n adjacency matrix of a left-total Erdos-Renyi digraph
    (self-loops allowed), conditioned on every row being non-empty."""
    A = rng.random((n, n)) < p_R
    empty = ~A.any(axis=1)
    while empty.any():
        idx = np.flatnonzero(empty)
        A[idx] = rng.random((len(idx), n)) < p_R
        empty[idx] = ~A[idx].any(axis=1)
    return A


def sample_labels(n, ps, rng):
    """Boolean n x k matrix; entry (v, i) says v is in L_i (prob ps[i])."""
    ps = np.asarray(ps, dtype=float)
    return rng.random((n, len(ps))) < ps[None, :]


def label_codes(Lab):
    """Integer code of the minterm satisfied by each vertex (bit i = a_{i+1})."""
    k = Lab.shape[1]
    return (Lab.astype(np.int64) << np.arange(k)[None, :]).sum(axis=1)


def sample_initial_multiple(n, p_I, rng):
    while True:
        I = rng.random(n) < p_I
        if I.any():
            return I


def sample_initial_single(n, rng):
    I = np.zeros(n, dtype=bool)
    I[rng.integers(n)] = True
    return I


def universal_structure(k):
    """The complete Kripke structure U_k: one vertex per minterm over k atomic
    propositions and all 4^k edges (including self-loops).  Every infinite word
    over 2^AP is the trace of a path of U_k, so an LTL formula holds on all
    paths from the vertex labelled sigma iff  sigma -> phi  is valid."""
    m = 2 ** k
    A = np.ones((m, m), dtype=bool)
    Lab = ((np.arange(m)[:, None] >> np.arange(k)[None, :]) & 1).astype(bool)
    return A, Lab
