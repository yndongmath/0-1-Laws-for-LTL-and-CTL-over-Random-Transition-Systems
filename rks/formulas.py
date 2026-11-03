"""
Formula representation for propositional logic, LTL and CTL.

Formulas are nested tuples.  Atomic propositions are indexed 0..k-1 and
printed as a_1..a_k (matching the paper).

  ('true',) ('false',) ('ap', i)
  ('not', f) ('and', f, g) ('or', f, g) ('imp', f, g)
LTL temporal operators (path formulas):
  ('X', f) ('F', f) ('G', f) ('U', f, g) ('R', f, g)
CTL temporal operators (state formulas):
  ('EX', f) ('AX', f) ('EF', f) ('AF', f) ('EG', f) ('AG', f)
  ('EU', f, g) ('AU', f, g)
"""

TRUE = ('true',)
FALSE = ('false',)

BOOL_OPS = {'not', 'and', 'or', 'imp'}
LTL_UNARY = {'X', 'F', 'G'}
LTL_BINARY = {'U', 'R'}
CTL_UNARY = {'EX', 'AX', 'EF', 'AF', 'EG', 'AG'}
CTL_BINARY = {'EU', 'AU'}


def ap(i):
    return ('ap', i)


def size(f):
    """Number of nodes of the syntax tree (the usual |phi|)."""
    if f[0] in ('true', 'false', 'ap'):
        return 1
    return 1 + sum(size(g) for g in f[1:])


def temporal_depth(f):
    """Nesting depth of temporal operators."""
    if f[0] in ('true', 'false', 'ap'):
        return 0
    inner = max(temporal_depth(g) for g in f[1:])
    if f[0] in BOOL_OPS:
        return inner
    return 1 + inner


def num_aps(f):
    s = set()

    def rec(g):
        if g[0] == 'ap':
            s.add(g[1])
        elif g[0] not in ('true', 'false'):
            for h in g[1:]:
                rec(h)
    rec(f)
    return len(s)


# ----------------------------------------------------------------- printing

def to_str(f):
    """Plain-text rendering, e.g. G(a1 -> F a2)."""
    op = f[0]
    if op == 'true':
        return 'true'
    if op == 'false':
        return 'false'
    if op == 'ap':
        return f'a{f[1] + 1}'
    if op == 'not':
        return '!' + _wrap(f[1])
    if op in ('and', 'or', 'imp'):
        sym = {'and': ' & ', 'or': ' | ', 'imp': ' -> '}[op]
        return _wrap(f[1]) + sym + _wrap(f[2])
    if op in LTL_UNARY or op in CTL_UNARY:
        return op + _wrap(f[1], tight=True)
    if op in ('U', 'R'):
        return '(' + to_str(f[1]) + f' {op} ' + to_str(f[2]) + ')'
    if op in ('EU', 'AU'):
        return op[0] + '(' + to_str(f[1]) + ' U ' + to_str(f[2]) + ')'
    raise ValueError(op)


def _wrap(f, tight=False):
    s = to_str(f)
    if f[0] in ('true', 'false', 'ap', 'not') or f[0] in LTL_UNARY or f[0] in CTL_UNARY \
            or f[0] in ('U', 'R', 'EU', 'AU'):
        return (' ' + s) if (tight and f[0] in ('ap', 'true', 'false')) else (s if not tight else s)
    return '(' + s + ')'


def to_latex(f):
    """LaTeX rendering using the paper's macros (\\X, \\F, \\G, \\U, \\A, \\E)."""
    op = f[0]
    if op == 'true':
        return r'\true'
    if op == 'false':
        return r'\false'
    if op == 'ap':
        return f'a_{{{f[1] + 1}}}'
    if op == 'not':
        return r'\neg ' + _lwrap(f[1])
    if op in ('and', 'or', 'imp'):
        sym = {'and': r' \wedge ', 'or': r' \vee ', 'imp': r' \rightarrow '}[op]
        return _lwrap(f[1]) + sym + _lwrap(f[2])
    if op in LTL_UNARY:
        return '\\' + op + ' ' + _lwrap(f[1])
    if op in CTL_UNARY:
        return '\\' + op[0] + '\\' + op[1] + ' ' + _lwrap(f[1])
    if op == 'U':
        return r'\U(' + to_latex(f[1]) + ', ' + to_latex(f[2]) + ')'
    if op == 'R':
        return r'\R(' + to_latex(f[1]) + ', ' + to_latex(f[2]) + ')'
    if op in ('EU', 'AU'):
        return '\\' + op[0] + r'\U(' + to_latex(f[1]) + ', ' + to_latex(f[2]) + ')'
    raise ValueError(op)


def _lwrap(f):
    s = to_latex(f)
    if f[0] in ('and', 'or', 'imp'):
        return '(' + s + ')'
    return s


# ------------------------------------------------- conversion for checking

def to_pymc(f, logic):
    """Convert to a pyModelChecking formula (used only for cross-validation)."""
    if logic == 'CTL':
        import pyModelChecking.CTL as M
    else:
        import pyModelChecking.LTL as M
    names = 'pqrstuvw'

    def rec(g):
        op = g[0]
        if op == 'true':
            return M.Bool(True)
        if op == 'false':
            return M.Bool(False)
        if op == 'ap':
            return M.AtomicProposition(names[g[1]])
        if op == 'not':
            return M.Not(rec(g[1]))
        if op == 'and':
            return M.And(rec(g[1]), rec(g[2]))
        if op == 'or':
            return M.Or(rec(g[1]), rec(g[2]))
        if op == 'imp':
            return M.Imply(rec(g[1]), rec(g[2]))
        if op in ('X', 'F', 'G'):
            return getattr(M, op)(rec(g[1]))
        if op == 'U':
            return M.U(rec(g[1]), rec(g[2]))
        if op == 'R':
            return M.R(rec(g[1]), rec(g[2]))
        if op in CTL_UNARY:
            return getattr(M, op[0])(getattr(M, op[1])(rec(g[1])))
        if op in CTL_BINARY:
            return getattr(M, op[0])(M.U(rec(g[1]), rec(g[2])))
        raise ValueError(op)
    phi = rec(f)
    if logic == 'LTL':
        phi = M.A(phi)
    return phi
