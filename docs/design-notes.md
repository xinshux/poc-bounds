# Design notes

Points where the implementation needs to be read alongside the papers.

## Index orderings and conditional terms

The theorems are stated for the **canonical ordering**, in which the $k$ factors
of the joint counterfactual carry treatment levels $x_1,\dots,x_k$ — as the
paper puts it, "the values of X and Y are arranged in increasing order within
the probability expression". Theorem 4 of the paper writes the
condition on the first term of the PRep($k,q$) upper bound accordingly, as
$q \in \{1,\dots,k\}$:

```
      ⎧ P(y_q|do(x_q)),                            if q ∈ {1,...,k}
min   ⎨ P(x_q, y_q) + Σ_{k+1≤j≤n, j≠q} P(x_j, y_q)
      ⎩ P(y_j|do(x_j)) − P(x_j, y_j),              j ∈ {1,...,k}, j ≠ q
```

The library is more general than that phrasing: `xj` may be any index vector,
not only `[1, ..., k]`. Conditions of this kind are therefore written as
membership in the index vector — `if q in xj` for the first term, `xj[j] != q`
for the third — so that they express the same thing for any ordering. In the
canonical ordering the two spellings coincide. The lower bound's corresponding
condition is written the same way.

`tests/test_bound_validity.py` exercises both regimes: the canonical ordering,
and arbitrary index subsets of `1..n` with `p` and `q` swept over the full
range. In both it asserts that all four quantities stay at least as tight as the
recursive baseline of Li and Pearl.

## Index conventions

`yi[t]` and `xj[t]` are the outcome and treatment level of the t-th factor of
the joint counterfactual, so the two lists are read position-wise rather than as
independent sets. The samplers and tests use that convention throughout.

The constructors do not validate their arguments: `k > n`, duplicate index
entries, an index outside `1..n`, or `len(yi) != k` are either accepted or raise
a bare `KeyError`/`IndexError` from a probability lookup rather than a message
naming the offending argument. Adding a validation helper is a natural next step
and would make misuse fail loudly at the call site.

## Dependency pinning

`requirements.txt` pins exact versions, SymPy in particular. SymPy chooses which
variables are free when it row-reduces, and that choice has changed between
releases, so an unpinned install reproduces the same mathematics but prints a
different — equally valid — parameterisation of the solution. The symbolic tests
therefore verify the solution by substituting it back into `A x = b` rather than
by comparing text.
