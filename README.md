# Experiments for "0-1 Laws for LTL and CTL over Random Transition Systems" (STTT revision)

This directory replaces the experiment code of the submitted version. Everything
in Section 6 of the revised paper is produced by the scripts here.

## Why the code was rewritten

* **LTL results of the submitted version were wrong.** The LTL procedure of
  `pyModelChecking` 1.3.4 returns incorrect results, mostly for formulae
  containing `G`. For example, `A G p` is reported false in a single `p`-labelled
  state with a self-loop. On 2,000 random instances it disagreed with two
  independent implementations in 189 cases (see `results/validation.json`). This
  explains why ψ5 and ψ7 were 0 throughout in the old Fig. 3, and why one curve
  converged to 1 in the multiple-initial-states panel, which contradicts
  Theorem 4.3. The CTL procedure of `pyModelChecking` is correct.
* **The formulae in the old code did not match Table 2.** `formulae.py` had no
  `G a1` but did have `!a1 U F a2`.
* **The sampler was not exact for K+.** When no initial state was drawn, the old
  generator added a random one instead of conditioning on `I != {}`.
* **Scale.** LTL was sampled only 50 or 10 times per n.

## Layout

| file | content |
|---|---|
| `rks/sampler.py` | exact samplers: left-totality by resampling empty rows; the complete structure `U_k` |
| `rks/ctl.py` | CTL fixpoint model checker; limit algorithm of Theorem 5.4 (truth tables) |
| `rks/ltl.py` | LTL model checker: GPVW tableau, product, SCCs; limits via Theorems 4.3/4.6 on `U_k` |
| `rks/ltl_ref.py` | independent LTL checker (Lichtenstein–Pnueli atoms), only for validation |
| `rks/experiment.py` | conditional (Rao–Blackwellised) estimators and the control variate for K^1 |
| `rks/randformula.py` | random formulae (randltl-style random syntax trees) |
| `rks/paper_formulas.py` | ψ1–ψ8 and φ1–φ10 of Table 2 |
| `run_experiments.py` | driver: `validate fixed random sparse sens timing` |
| `make_figures.py`, `make_tables.py` | figures and tables of Section 6 from `results/*.json` |
| `validate.py` | quick cross-check (a few seconds) |
| `results/` | raw results (JSON) of all runs reported in the paper |

## Running

```
pip install numpy scipy matplotlib pyModelChecking   # pyModelChecking only for validation
python validate.py                                   # quick sanity check
WORKERS=2 python run_experiments.py validate fixed random sparse sens timing
python make_figures.py figs && python make_tables.py figs
```

All runs are seeded and therefore reproducible. On two cores, the complete set
takes a few hours; the hand-picked formulae (`fixed`) take the largest share.

## Estimators (Section 6.1)

For each sampled (R, L) and formula φ the satisfaction set V_φ is computed once.
Both models are then evaluated from it in closed form, using m = |V_φ| and
q = 1 − p_I:

* P(K^1 |= φ | R, L) = m / n
* P(K^+ |= φ | R, L) = q^(n−m) (1 − q^m) / (1 − q^n)

For K^1 the deviation from the limit is estimated with the control variate
|V_β| / n, whose expectation is exactly the predicted limit π(β). The plain
estimators (one sampled initial state, or one sampled initial set) are stored
as well, as `raw1` and `rawplus`.
