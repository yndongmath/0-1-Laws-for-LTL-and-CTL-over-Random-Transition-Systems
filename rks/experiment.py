"""
Monte Carlo estimation of P(K^+_{n,p} |= phi) and P(K^1_{n,p} |= phi).

For every sampled pair (R, L) we compute the satisfaction set V_phi once and
evaluate *both* models from it:

  * conditional (Rao-Blackwellised) estimators.  With m = |V_phi| and
    q = 1 - p_I, the probability over the initial states is exact:
        P(K^1 |= phi | R, L) = m / n,
        P(K^+ |= phi | R, L) = P(I subset V_phi | I != {})
                             = q^(n-m) (1 - q^m) / (1 - q^n);
    averaging these over the samples is unbiased and has lower variance than
    the plain frequency;
  * a control variate for K^1: let beta be the propositional formula whose
    truth table is the limit (Theorem 4.6 / Theorem 5.4).  Then
    E[|V_beta| / n] = pi(beta) = lim P(K^1 |= phi) exactly, so
        P(K^1 |= phi) - lim = E[(|V_phi| - |V_beta|) / n],
    which is estimated with a variance that vanishes as V_phi -> V_beta;
  * plain estimators (one sampled initial state / initial set per sample), so
    the results can be compared with the original experiment.
"""
import numpy as np

from . import ctl, ltl
from .sampler import sample_relation, sample_labels, label_codes


class Formula:
    def __init__(self, f, logic, ps):
        self.f = f
        self.logic = logic
        if logic == 'LTL':
            self.aut = ltl.translate(('not', f))
            self.cplus, self.c1, self.vec = ltl.limits(f, ps, self.aut)
        else:
            self.aut = None
            self.cplus, self.c1, self.vec = ctl.limits(f, ps)


STATS = ['p1', 'pplus', 'oneminus_pplus', 'dev1', 'raw1', 'rawplus', 'frac_beta']


def run_point(n, N, p_R, p_I, ps, forms, seed):
    """Returns dict stat -> (sum, sumsq) arrays over formulas, for N samples."""
    rng = np.random.default_rng(seed)
    F = len(forms)
    acc = {s: np.zeros((2, F)) for s in STATS}
    q = 1.0 - p_I
    lq = np.log(q)
    denom = -np.expm1(n * lq)                  # 1 - q^n = P(I != {})
    for _ in range(N):
        A = sample_relation(n, p_R, rng)
        Lab = sample_labels(n, ps, rng)
        codes = label_codes(Lab)
        src, dst = np.nonzero(A)
        cache = {'__A': A.astype(np.int32)}
        # plain estimators: draw the initial state(s) once per sample
        v0 = rng.integers(n)
        while True:
            I = rng.random(n) < p_I
            if I.any():
                break
        for j, fm in enumerate(forms):
            if fm.logic == 'LTL':
                V = ~ltl.violating(fm.aut, A, Lab, src, dst)
            else:
                V = ctl.sat(fm.f, A, Lab, cache)
            m = int(V.sum())
            mb = int(fm.vec[codes].sum())
            p1 = m / n
            # q^(n-m) (1 - q^m) / (1 - q^n), computed in a cancellation-free way
            pplus = 0.0 if m == 0 else np.exp((n - m) * lq) * (-np.expm1(m * lq)) / denom
            om = -np.expm1((n - m) * lq) / denom   # 1 - pplus
            vals = {
                'p1': p1, 'pplus': pplus, 'oneminus_pplus': om,
                'dev1': (m - mb) / n, 'raw1': float(V[v0]),
                'rawplus': float(V[I].all()), 'frac_beta': mb / n,
            }
            for s, x in vals.items():
                acc[s][0, j] += x
                acc[s][1, j] += x * x
    return acc


def summarise(acc, N):
    out = {}
    for s, (S, SS) in acc.items():
        mean = S / N
        var = np.maximum(SS / N - mean ** 2, 0.0) * N / max(N - 1, 1)
        out[s] = mean.tolist()
        out[s + '_se'] = np.sqrt(var / N).tolist()
    return out
