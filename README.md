# 0-1 Laws for LTL and CTL over Random Transition Systems: experiments

This repository contains the code and the raw data for the experiments in
Section 6 of

> Yanni Dong, Milan Lopuhaä-Zwakenberg, Denis Timofeev Fornasov and Mariëlle
> Stoelinga. *0-1 Laws for LTL and CTL over Random Transition Systems.*
> Extended journal version of the SPIN 2025 paper, for the STTT special issue
> on SPIN 2025.

The paper shows that LTL and CTL obey a 0-1 law on random Kripke structures
with multiple initial states (K⁺ₙ,ₚ) and a convergence law on random Kripke
structures with a single initial state (K¹ₙ,ₚ̂), and that in both cases the
convergence is exponentially fast. The experiments estimate the probability
that a formula holds on structures with up to 1000 states and compare it with
the limits predicted by the theorems.

## Experiments

All experiments are run through `run_experiments.py <command> ...`.

| command | what it computes | in the paper |
|---|---|---|
| `validate` | cross-validation of the model checkers and of the limit algorithms | Section 6.1 |
| `fixed` | the 8 LTL formulae ψ1–ψ8 and the 10 CTL formulae φ1–φ10 of Table 2; k = 5 atomic propositions, p_R = p_I = 0.1, p_i = 0.5; N = 10⁴ samples for every even n ≤ 100 and every n ∈ {105, 110, …, 250}, N = 10³ for n ∈ {300, 400, 500, 750, 1000} | Section 6.2, Table 2, Figs. 3 and 4 |
| `timing` | average model checking time per formula for n = 250 and n = 1000 | Section 6.2 |
| `random` | 104 random LTL and 104 random CTL formulae (sizes 3–15, eight per size); k = 4; N = 1000 for n ∈ {4, 8, …, 128}, N = 500 for n ∈ {192, 256, 384} | Section 6.3, Fig. 5 |
| `sens` | as `fixed`, for p_R ∈ {0.05, 0.2} | additional, not in the paper |
| `sparse` | as `fixed`, for p_R = 3/n | additional, not in the paper |

The paper assumes that p_R is a constant, so `sparse` lies outside the scope
of its theorems. In that regime, conditioning on left-totality (see below)
changes the distribution considerably.

## Layout

| file | content |
|---|---|
| `rks/sampler.py` | exact samplers for K⁺ and K¹; the complete Kripke structure U_k |
| `rks/ltl.py` | LTL model checker (GPVW tableau, product, SCCs); limits via Theorems 4.3 and 4.6 |
| `rks/ltl_ref.py` | independent LTL model checker (Lichtenstein–Pnueli atoms), used only for validation |
| `rks/ctl.py` | CTL model checker (fixpoints); limits via Theorem 5.4 |
| `rks/experiment.py` | conditional estimators and the control variate for K¹ (see below) |
| `rks/formulas.py` | formulae as nested tuples: size, temporal depth, printing (text and LaTeX), conversion to `pyModelChecking` |
| `rks/randformula.py` | random formulae (random syntax trees in the style of `randltl`) |
| `rks/paper_formulas.py` | ψ1–ψ8 and φ1–φ10 of Table 2 |
| `run_experiments.py` | driver for all experiments; writes `results/*.json` |
| `make_figures.py` | figures and the numbers quoted in Section 6, from `results/*.json` |
| `make_tables.py` | LaTeX table bodies, from `figs/numbers.json` |
| `validate.py` | quick cross-check of the model checkers (a few seconds) |
| `results/` | raw results of all runs |
| `figs/` | generated figures and tables |

## Requirements

Tested with Python 3.13, NumPy 2.5, SciPy 1.18 and Matplotlib 3.11.
`pyModelChecking` 1.3.4 is needed only for the validation.

```
pip install numpy scipy matplotlib
pip install pyModelChecking==1.3.4
```

## Reproducing the results

Run all commands from the repository root, i.e. the directory that contains
`rks/`.

```
python validate.py
python run_experiments.py validate fixed timing random
python make_figures.py figs
python make_tables.py figs
```

`make_figures.py` writes `fig_ltl.pdf` (Fig. 3), `fig_ctl.pdf` (Fig. 4),
`fig_random.pdf` (Fig. 5) and `numbers.json` (all numbers quoted in Section 6)
to `figs/`. `make_tables.py` then writes the body of Table 2 to
`figs/tab_fixed.tex`. To regenerate only the figures from the results in
`results/`, the last two commands suffice.

The number of worker processes is set with the environment variable `WORKERS`
(default 2):

```
WORKERS=8 python run_experiments.py fixed                 # Linux, macOS
$env:WORKERS=8; python run_experiments.py fixed           # Windows PowerShell
```

On two cores, `fixed` takes about 1.3 CPU hours and `random` about 1.0 CPU
hours; the other commands take a few minutes. Every run is seeded, with one
seed per chunk of samples, so the results do not depend on the number of
workers (up to floating-point rounding in the order of summation). The timings
of `timing` depend on the machine.

## Method

**Sampling.** The relation is sampled with independent edges of probability
p_R. Since a structure is left-total if and only if no row of its adjacency
matrix is empty, and rows are independent, resampling empty rows samples
exactly from the distribution conditioned on left-totality. Labels are
independent with probability p_i per atomic proposition.

**Estimators.** For every sampled pair (R, L) and formula φ, the set V_φ of
vertices satisfying φ is computed once. Both models are then evaluated from it
in closed form, with m = |V_φ| and q = 1 − p_I:

* P(K¹ ⊨ φ | R, L) = m / n
* P(K⁺ ⊨ φ | R, L) = q^(n−m) (1 − q^m) / (1 − q^n)

Averaging these conditional probabilities over the samples is unbiased and has
smaller variance than sampling the initial states. For K¹, the deviation from
the limit is estimated with the control variate |V_β| / n, whose expectation is
exactly the predicted limit π(β). The plain estimators (one sampled initial
vertex, or one sampled set of initial vertices, per sample) are stored as well,
as `raw1` and `rawplus`.

**Limits.** For CTL, the limits are computed with the algorithm of Theorem 5.4,
representing propositional formulae by their truth tables over the 2^k
minterms. For LTL, they follow from Theorems 4.3 and 4.6: whether σ → φ is a
tautology is decided by model checking φ on the complete Kripke structure U_k,
which has one vertex per minterm and all 4^k edges.

**Validation.** `run_experiments.py validate` checks that

1. the LTL model checker agrees with the independent atom-based implementation
   on 10,000 random pairs of formulae and structures;
2. the CTL model checker agrees with the CTL procedure of `pyModelChecking` on
   5,000 random pairs;
3. the CTL limit algorithm agrees with CTL model checking on U_k for 10,000
   random formulae.

The LTL procedure of `pyModelChecking` 1.3.4 is not used: on 2,000 random pairs
it disagreed with both of our implementations in 189 cases, mostly for
formulae containing `G`. For instance, it reports that `G a1` does not hold in
a single `a1`-labelled state with a self-loop. The results are in
`results/validation.json`.

## Citation

If you use this code, please cite the paper. The conference version is

```bibtex
@inproceedings{DongSPIN2025,
  author    = {Dong, Yanni and Lopuha{\"a}-Zwakenberg, Milan and Stoelinga, Mari{\"e}lle},
  title     = {0-1 Laws for {LTL} and {CTL} over Random Transition Systems},
  booktitle = {Model Checking Software -- 31st International Symposium, SPIN 2025},
  editor    = {Ernst, Gidon and Rozier, Kristin Yvonne},
  series    = {Lecture Notes in Computer Science},
  volume    = {15945},
  pages     = {66--87},
  publisher = {Springer},
  year      = {2025},
  doi       = {10.1007/978-3-032-06847-7_4}
}
```

A BibTeX entry for the journal version will be added once it is published.
