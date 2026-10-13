# poc-bounds

Closed-form bounds on **probabilities of causation** for multivalued treatments
and outcomes in Structural Causal Models.

Reference implementation for:

- X. Shu, S. Wang, A. Li. *Identification of Probabilities of Causation: From
  Recursive to Closed-Form Bounds.* IJCAI-ECAI 2026, 4063–4070.

Probabilities of causation — the probability of necessity (PN), of sufficiency
(PS), and of necessity and sufficiency (PNS) — are counterfactual quantities and
so are generally not point identifiable; they must be bounded. Classical results
give tight bounds only for binary treatments and outcomes. This library computes
bounds for the multivalued case, in closed form.

## Install

```bash
git clone https://github.com/<your-username>/poc-bounds.git
cd poc-bounds
pip install -r requirements.txt
```

Python 3.9 or newer.

## Quick start

```python
from pocbounds import ConjecturePNS, generate_prob_with_constraint

# sample an SCM with a mediator and read off its observational and
# experimental distributions
prob = generate_prob_with_constraint("mediator", n_x=4, n_y=4, n_z=3)

# bound PNS(k) and the three related quantities for k = 2,
# at treatment levels x_1, x_3 paired with outcome levels y_1, y_3
b = ConjecturePNS(n=4, k=2, p=1, q=3, prob=prob, yi=[1, 3], xj=[1, 3])

print(b.lb,   b.ub)     # PNS(k)
print(b.lbx,  b.ubx)    # PSub(k, p)
print(b.lby,  b.uby)    # PRep(k, q)
print(b.lbxy, b.ubxy)   # PN(k, p, q)
```

The recursive (Li–Pearl) bounds used as the comparison baseline live in
`pocbounds.recursive.RecursivePNS`, which additionally takes `m`.

## Layout

```
pocbounds/
  bounds.py        ConjecturePNS -- the closed-form bounds
  recursive.py     RecursivePNS  -- Li-Pearl recursive baseline
  sampling.py      random SCM samplers with rejection sampling on validity
  symbolic.py      GaussianSolver -- symbolic tightness verification
  style.py         matplotlib styling for the paper figures
experiments/       drivers that reproduce the figures
tests/             the test suite
docs/              open-issues.md
data/              equ_33.txt, informer_33_lb.txt
```

## Reproducing the figures

```bash
python -m experiments.tightness_graph
python experiments/symbolic_tightness.py --equ data/equ_33.txt --k 3
```

## Tests

```bash
pip install pytest pytest-cov
pytest tests/ -q
pytest tests/ --cov=pocbounds
```

1,020 tests, 97% statement coverage. Four groups:

- **`test_golden_regression.py`** — 1,920 bound values frozen from the original
  research scripts before the refactor, so the restructuring cannot have moved a
  published number.
- **`test_bound_validity.py`** — properties that must hold for any valid input:
  `lb <= ub`, bounds within `[0, 1]`, and the closed-form bounds dominating the
  recursive baseline — checked both in the canonical index ordering the theorems
  are stated in and over arbitrary index subsets with `p` and `q` swept over the
  full range.
- **`test_sampling.py`** — the samplers emit genuine distributions (marginals sum
  to one, non-negativity, the rejection constraint actually holds), including a
  test that the validity checker rejects an invalid input.
- **`test_symbolic.py`** — the symbolic solver's output is verified by
  substituting the solution back into `A x = b`, rather than by comparing text.
  SymPy picks different free variables in different releases, so the original
  text comparison was not reproducible across versions; `requirements.txt` pins
  the version as well.

## Design notes

[`docs/design-notes.md`](docs/design-notes.md) covers the places where the code
has to be read alongside the papers: how conditional terms are expressed for an
arbitrary index ordering, the position-wise index convention, and why the
dependencies are pinned.

## Publishing this repository

```bash
git remote add origin https://github.com/<your-username>/poc-bounds.git
git branch -M main
git push -u origin main
```

## License

MIT — see [LICENSE](LICENSE).
