"""
Driver for all experiments of Section 6.

  python run_experiments.py validate   # cross-validation of the checkers
  python run_experiments.py fixed      # Table 2 formulas, dense model, p_R = 0.1
  python run_experiments.py random     # random LTL/CTL formula benchmark
  python run_experiments.py sparse     # scope: sparse regime p_R = c / n
  python run_experiments.py sens       # sensitivity: p_R in {0.05, 0.2}
  python run_experiments.py timing     # model checking time per formula

Results are written as JSON to results/.  Everything is seeded.
"""
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

from rks import formulas as Fm
from rks.experiment import Formula, run_point, summarise, STATS
from rks.paper_formulas import LTL_FIXED, CTL_FIXED
from rks.randformula import benchmark

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(OUT, exist_ok=True)
WORKERS = int(os.environ.get('WORKERS', '2'))
CHUNK = 500


def _task(args):
    n, N, p_R, p_I, ps, forms, seed = args
    t = time.time()
    acc = run_point(n, N, p_R, p_I, ps, forms, seed)
    return n, N, acc, time.time() - t


def sweep(name, forms, ps, p_I, grid, p_R_of_n, base_seed):
    """grid: list of (n, N).  p_R_of_n: function n -> p_R."""
    tasks = []
    ss = np.random.SeedSequence(base_seed)
    for (n, N) in grid:
        chunks = [CHUNK] * (N // CHUNK) + ([N % CHUNK] if N % CHUNK else [])
        for c, child in zip(chunks, ss.spawn(len(chunks))):
            tasks.append((n, c, p_R_of_n(n), p_I, ps, forms, child))
    # largest n first for load balancing
    tasks.sort(key=lambda t: -t[0])
    total = {}
    cpu = {}
    t0 = time.time()
    last_save = time.time()
    with Pool(WORKERS) as pool:
        for i, (n, N, acc, dt) in enumerate(pool.imap_unordered(_task, tasks)):
            if n not in total:
                total[n] = [0, {s: np.zeros_like(acc[s]) for s in STATS}]
                cpu[n] = 0.0
            total[n][0] += N
            for s in STATS:
                total[n][1][s] += acc[s]
            cpu[n] += dt
            if i % 20 == 0:
                print(f'[{name}] {i + 1}/{len(tasks)} tasks, {time.time() - t0:.0f}s', flush=True)
            if time.time() - last_save > 120:
                _save(name + '.partial', forms, ps, p_I, total, cpu, p_R_of_n, t0)
                last_save = time.time()
    return _save(name, forms, ps, p_I, total, cpu, p_R_of_n, t0)


def _save(name, forms, ps, p_I, total, cpu, p_R_of_n, t0):
    res = {
        'name': name,
        'p_I': p_I, 'ps': ps,
        'formulas': [{'logic': fm.logic, 'str': Fm.to_str(fm.f), 'latex': Fm.to_latex(fm.f),
                      'size': Fm.size(fm.f), 'depth': Fm.temporal_depth(fm.f),
                      'cplus': fm.cplus, 'c1': fm.c1,
                      'aut_states': (fm.aut.nstates if fm.aut is not None else None)}
                     for fm in forms],
        'points': [],
        'wall_seconds': time.time() - t0,
    }
    for n in sorted(total):
        N, acc = total[n]
        res['points'].append({'n': n, 'N': N, 'p_R': p_R_of_n(n), 'cpu_seconds': cpu[n],
                              **summarise(acc, N)})
    with open(os.path.join(OUT, name + '.json'), 'w') as fh:
        json.dump(res, fh)
    if not name.endswith('.partial'):
        print(f'[{name}] done in {time.time() - t0:.0f}s', flush=True)
    return res


def fixed_forms(ps):
    return [Formula(f, 'LTL', ps) for f in LTL_FIXED] + [Formula(f, 'CTL', ps) for f in CTL_FIXED]


def cmd_fixed():
    ps = [0.5] * 5
    grid = [(n, 10000) for n in list(range(2, 101, 2)) + list(range(105, 251, 5))] + \
           [(n, 1000) for n in (300, 400, 500, 750, 1000)]
    sweep('fixed_pR0.1', fixed_forms(ps), ps, 0.1, grid, lambda n: 0.1, 20251)


def random_forms(ps, seed=2026):
    rng = np.random.default_rng(seed)
    k = len(ps)
    L = benchmark('LTL', k, range(3, 16), 8, rng)
    C = benchmark('CTL', k, range(3, 16), 8, rng)
    return [Formula(f, 'LTL', ps) for f in L] + [Formula(f, 'CTL', ps) for f in C]


def cmd_random():
    ps = [0.5] * 4
    grid = [(n, 1000) for n in (4, 8, 12, 16, 24, 32, 48, 64, 96, 128)] + \
           [(n, 500) for n in (192, 256, 384)]
    sweep('random_pR0.1', random_forms(ps), ps, 0.1, grid, lambda n: 0.1, 20252)


def cmd_sparse():
    ps = [0.5] * 5
    c = 3.0
    grid = [(n, 1000) for n in (8, 16, 32, 64, 128, 256, 512, 1024)]
    sweep('sparse_c3', fixed_forms(ps), ps, 0.1, grid, lambda n: min(1.0, c / n), 20253)


def cmd_sens():
    ps = [0.5] * 5
    grid = [(n, 1000) for n in (4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512)]
    for p_R, seed in ((0.05, 20254), (0.2, 20255)):
        sweep(f'fixed_pR{p_R}', fixed_forms(ps), ps, 0.1, grid, lambda n, p=p_R: p, seed)


def cmd_timing():
    from rks import ltl, ctl
    from rks.sampler import sample_relation, sample_labels
    ps = [0.5] * 5
    forms = fixed_forms(ps)
    rng = np.random.default_rng(7)
    reps = 200
    res = {}
    for n in (250, 1000):
        times = np.zeros(len(forms))
        for _ in range(reps):
            A = sample_relation(n, 0.1, rng)
            Lab = sample_labels(n, ps, rng)
            src, dst = np.nonzero(A)
            for j, fm in enumerate(forms):
                t = time.perf_counter()
                if fm.logic == 'LTL':
                    ltl.violating(fm.aut, A, Lab, src, dst)
                else:
                    ctl.sat(fm.f, A, Lab, {})
                times[j] += time.perf_counter() - t
        res[n] = (times / reps).tolist()
    res['formulas'] = [Fm.to_str(fm.f) for fm in forms]
    with open(os.path.join(OUT, 'timing.json'), 'w') as fh:
        json.dump(res, fh)
    print('timing done', flush=True)


def cmd_validate():
    from rks import ltl, ctl, ltl_ref
    from rks.sampler import sample_relation, sample_labels, universal_structure
    from rks.randformula import random_formula
    from pyModelChecking import Kripke
    import pyModelChecking.CTL as PCTL
    import pyModelChecking.LTL as PLTL
    rng = np.random.default_rng(99)
    k = 3
    names = 'pqrstuvw'
    out = {}
    # 1. LTL (GPVW) vs independent atom-based checker
    bad = 0
    T = 10000
    for _ in range(T):
        n = int(rng.integers(1, 7))
        A = sample_relation(n, float(rng.uniform(0.2, 0.8)), rng)
        Lab = sample_labels(n, [0.5] * k, rng)
        f = random_formula(int(rng.integers(1, 10)), k, 'LTL', rng)
        if not (ltl.violating(ltl.translate(('not', f)), A, Lab) == ltl_ref.violating_ref(f, A, Lab)).all():
            bad += 1
    out['ltl_vs_atoms'] = {'trials': T, 'mismatches': bad}
    # 2. CTL vs pyModelChecking
    bad = 0
    T = 5000
    for _ in range(T):
        n = int(rng.integers(1, 9))
        A = sample_relation(n, float(rng.uniform(0.2, 0.8)), rng)
        Lab = sample_labels(n, [0.5] * k, rng)
        f = random_formula(int(rng.integers(1, 12)), k, 'CTL', rng)
        R = [(int(i), int(j)) for i, j in zip(*np.nonzero(A))]
        L = {v: set(names[i] for i in range(k) if Lab[v, i]) for v in range(n)}
        K = Kripke(S=list(range(n)), S0=list(range(n)), R=R, L=L)
        ref = PCTL.modelcheck(K, Fm.to_pymc(f, 'CTL'))
        if not (ctl.sat(f, A, Lab) == np.array([v in ref for v in range(n)])).all():
            bad += 1
    out['ctl_vs_pymc'] = {'trials': T, 'mismatches': bad}
    # 3. Theorem 5.4 algorithm vs CTL model checking on U_k (k = 3, 4)
    bad = 0
    T = 0
    for kk in (3, 4):
        A, Lab = universal_structure(kk)
        for _ in range(5000):
            f = random_formula(int(rng.integers(1, 20)), kk, 'CTL', rng)
            T += 1
            if not (ctl.mask_to_vec(ctl.reduce_to_prop(f, kk), kk) == ctl.sat(f, A, Lab)).all():
                bad += 1
    out['ctl_limit_vs_Uk'] = {'trials': T, 'mismatches': bad}
    # 4. pyModelChecking LTL vs ours (documents the discrepancy)
    bad = 0
    T = 2000
    by_op = {}
    for _ in range(T):
        n = int(rng.integers(1, 6))
        A = sample_relation(n, float(rng.uniform(0.2, 0.8)), rng)
        Lab = sample_labels(n, [0.5] * k, rng)
        f = random_formula(int(rng.integers(1, 7)), k, 'LTL', rng)
        R = [(int(i), int(j)) for i, j in zip(*np.nonzero(A))]
        L = {v: set(names[i] for i in range(k) if Lab[v, i]) for v in range(n)}
        K = Kripke(S=list(range(n)), S0=list(range(n)), R=R, L=L)
        try:
            ref = PLTL.modelcheck(K, Fm.to_pymc(f, 'LTL'))
        except Exception:
            continue
        mism = not (ltl.sat(f, A, Lab) == np.array([v in ref for v in range(n)])).all()
        bad += mism
        hasG = 'G' in Fm.to_str(f)
        by_op.setdefault('with_G' if hasG else 'without_G', [0, 0])
        by_op['with_G' if hasG else 'without_G'][0] += 1
        by_op['with_G' if hasG else 'without_G'][1] += mism
    out['ltl_vs_pymc'] = {'trials': T, 'mismatches': int(bad), 'by_G': by_op}
    with open(os.path.join(OUT, 'validation.json'), 'w') as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1), flush=True)


if __name__ == '__main__':
    for cmd in sys.argv[1:]:
        globals()['cmd_' + cmd]()
