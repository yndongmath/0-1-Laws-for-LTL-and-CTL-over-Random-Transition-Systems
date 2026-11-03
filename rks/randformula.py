"""
Random formula generation in the style of Spot's `randltl`: a random syntax
tree with a prescribed number of nodes, operators chosen uniformly among the
admissible ones and leaves chosen uniformly among the atomic propositions.
"""
import numpy as np

LTL_UN = ['not', 'X', 'F', 'G']
LTL_BIN = ['and', 'or', 'imp', 'U', 'R']
CTL_UN = ['not', 'EX', 'AX', 'EF', 'AF', 'EG', 'AG']
CTL_BIN = ['and', 'or', 'imp', 'EU', 'AU']


def random_formula(size, k, logic, rng):
    un, bi = (LTL_UN, LTL_BIN) if logic == 'LTL' else (CTL_UN, CTL_BIN)

    def gen(s):
        if s == 1:
            return ('ap', int(rng.integers(k)))
        if s == 2:
            return (un[rng.integers(len(un))], gen(1))
        # unary with probability |un| / (|un| + |bi|), as in randltl with equal priorities
        if rng.random() < len(un) / (len(un) + len(bi)):
            return (un[rng.integers(len(un))], gen(s - 1))
        s1 = int(rng.integers(1, s - 1))
        return (bi[rng.integers(len(bi))], gen(s1), gen(s - 1 - s1))
    return gen(size)


def has_double_negation(f):
    if f[0] in ('ap', 'true', 'false'):
        return False
    if f[0] == 'not' and f[1][0] == 'not':
        return True
    return any(has_double_negation(g) for g in f[1:])


def benchmark(logic, k, sizes, per_size, rng, max_tries=100000):
    """Distinct random formulas: `per_size` formulas of each size in `sizes`,
    without double negations (which randltl also removes by default)."""
    out, seen = [], set()
    for s in sizes:
        got, tries = 0, 0
        while got < per_size and tries < max_tries:
            tries += 1
            f = random_formula(s, k, logic, rng)
            if f in seen or has_double_negation(f):
                continue
            seen.add(f)
            out.append(f)
            got += 1
    return out
