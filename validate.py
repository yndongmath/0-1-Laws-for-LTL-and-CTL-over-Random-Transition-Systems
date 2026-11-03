"""Cross-validation of the model checkers and of the limit algorithms."""
import sys, time
import numpy as np
from rks import ctl, ltl, ltl_ref, formulas as F
from rks.sampler import sample_relation, sample_labels, universal_structure
from rks.randformula import random_formula

rng = np.random.default_rng(12345)
k = 3

def check_ltl_vs_ref(trials=3000):
    bad = 0
    for t in range(trials):
        n = int(rng.integers(1, 6))
        A = sample_relation(n, 0.4, rng)
        Lab = sample_labels(n, [0.5]*k, rng)
        f = random_formula(int(rng.integers(1, 9)), k, 'LTL', rng)
        v1 = ltl.violating(ltl.translate(('not', f)), A, Lab)
        v2 = ltl_ref.violating_ref(f, A, Lab)
        if not (v1 == v2).all():
            bad += 1
            if bad < 5: print('LTL mismatch', F.to_str(f), v1, v2)
    return bad

def check_ctl_vs_pymc(trials=1500):
    from pyModelChecking import Kripke
    import pyModelChecking.CTL as CTL
    bad = 0
    names = 'pqrstuvw'
    for t in range(trials):
        n = int(rng.integers(1, 7))
        A = sample_relation(n, 0.4, rng)
        Lab = sample_labels(n, [0.5]*k, rng)
        f = random_formula(int(rng.integers(1, 9)), k, 'CTL', rng)
        mine = ctl.sat(f, A, Lab)
        R = [(int(i), int(j)) for i, j in zip(*np.nonzero(A))]
        L = {v: set(names[i] for i in range(k) if Lab[v, i]) for v in range(n)}
        K = Kripke(S=list(range(n)), S0=list(range(n)), R=R, L=L)
        ref = CTL.modelcheck(K, F.to_pymc(f, 'CTL'))
        refv = np.array([v in ref for v in range(n)])
        if not (mine == refv).all():
            bad += 1
            if bad < 5: print('CTL mismatch', F.to_str(f), mine, refv)
    return bad

def check_ltl_vs_pymc(trials=400):
    from pyModelChecking import Kripke
    import pyModelChecking.LTL as LTL
    bad = 0; examples = []
    names = 'pqrstuvw'
    for t in range(trials):
        n = int(rng.integers(1, 5))
        A = sample_relation(n, 0.5, rng)
        Lab = sample_labels(n, [0.5]*k, rng)
        f = random_formula(int(rng.integers(1, 6)), k, 'LTL', rng)
        mine = ltl.sat(f, A, Lab)
        R = [(int(i), int(j)) for i, j in zip(*np.nonzero(A))]
        L = {v: set(names[i] for i in range(k) if Lab[v, i]) for v in range(n)}
        K = Kripke(S=list(range(n)), S0=list(range(n)), R=R, L=L)
        try:
            ref = LTL.modelcheck(K, F.to_pymc(f, 'LTL'))
        except Exception as e:
            continue
        refv = np.array([v in ref for v in range(n)])
        if not (mine == refv).all():
            bad += 1
            examples.append((F.to_str(f), n, mine, refv))
    return bad, examples

def check_ctl_limit_vs_universal(trials=3000):
    # Theorem 5.4 algorithm (truth tables) vs CTL model checking on U_k
    A, Lab = universal_structure(k)
    bad = 0
    for t in range(trials):
        f = random_formula(int(rng.integers(1, 15)), k, 'CTL', rng)
        vec = ctl.mask_to_vec(ctl.reduce_to_prop(f, k), k)
        if not (vec == ctl.sat(f, A, Lab)).all():
            bad += 1
            if bad < 5: print('limit mismatch', F.to_str(f))
    return bad

t=time.time(); print('LTL GPVW vs atom reference mismatches:', check_ltl_vs_ref(), f'({time.time()-t:.1f}s)')
t=time.time(); print('CTL vs pyModelChecking mismatches:', check_ctl_vs_pymc(), f'({time.time()-t:.1f}s)')
t=time.time(); print('CTL limit alg vs U_k mismatches:', check_ctl_limit_vs_universal(), f'({time.time()-t:.1f}s)')
t=time.time(); b, ex = check_ltl_vs_pymc(); print('LTL vs pyModelChecking mismatches:', b, f'({time.time()-t:.1f}s)')
for e in ex[:8]: print('   ', e)
