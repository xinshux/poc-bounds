"""Semantic properties the bounds must satisfy for *any* valid input.

These tests do not depend on frozen numbers, so they keep working when the
implementation changes -- they check the mathematics, not the output text.

Note on index conventions: `yi[t]` and `xj[t]` are the outcome and treatment
level of the t-th factor of the joint counterfactual, so the two lists are
position-wise paired.  The tests below draw them with that convention.
"""
import random

import numpy as np
import pytest

from pocbounds import ConjecturePNS, generate_prob_with_constraint
from pocbounds.recursive import RecursivePNS

PAIRS = [("lb", "ub"), ("lbx", "ubx"), ("lby", "uby"), ("lbxy", "ubxy")]
CLOSED_FORM = [ConjecturePNS]



N_CASES = 40
TOL = 1e-9


def _cases(seed, n_cases=N_CASES):
    rng = random.Random(seed)
    np.random.seed(seed)
    out = []
    for _ in range(n_cases):
        n = rng.choice([3, 4, 5])
        k = rng.randint(2, n)
        scm = rng.choice(["mediator", "confounder", "complex"])
        prob = generate_prob_with_constraint(scm, n_x=n, n_y=n, n_z=3)
        idx = rng.sample(range(1, n + 1), k)
        out.append(
            dict(n=n, k=k, p=rng.randint(1, n), q=rng.randint(1, n),
                 prob=prob, yi=idx[:], xj=idx[:])
        )
    return out


CASES = _cases(7)


@pytest.mark.parametrize("c", CASES, ids=range(len(CASES)))
@pytest.mark.parametrize("cls", CLOSED_FORM, ids=[c.__name__ for c in CLOSED_FORM])
def test_lower_bound_never_exceeds_upper_bound(cls, c):
    obj = cls(c["n"], c["k"], c["p"], c["q"], c["prob"], c["yi"], c["xj"])
    for lo, hi in PAIRS:
        assert getattr(obj, lo) <= getattr(obj, hi) + TOL, (
            f"{cls.__name__}: {lo}={getattr(obj, lo)} > {hi}={getattr(obj, hi)}"
        )


@pytest.mark.parametrize("c", CASES, ids=range(len(CASES)))
@pytest.mark.parametrize("cls", CLOSED_FORM, ids=[c.__name__ for c in CLOSED_FORM])
def test_bounds_lie_in_the_unit_interval(cls, c):
    obj = cls(c["n"], c["k"], c["p"], c["q"], c["prob"], c["yi"], c["xj"])
    for lo, hi in PAIRS:
        assert -TOL <= getattr(obj, lo) <= 1 + TOL
        assert -TOL <= getattr(obj, hi) <= 1 + TOL


@pytest.mark.parametrize("c", CASES, ids=range(len(CASES)))
@pytest.mark.parametrize("bound", ["lb", "lbx", "lby", "lbxy"])
def test_closed_form_lower_bounds_dominate_recursive(bound, c):
    """A tighter lower bound is a larger one."""
    conj = ConjecturePNS(c["n"], c["k"], c["p"], c["q"], c["prob"], c["yi"], c["xj"])
    rec = RecursivePNS(c["n"], c["n"], c["k"], c["p"], c["q"], c["prob"], c["yi"], c["xj"])
    assert getattr(conj, bound) >= getattr(rec, bound) - TOL


@pytest.mark.parametrize("c", CASES, ids=range(len(CASES)))
@pytest.mark.parametrize("bound", ["ub", "ubx", "uby", "ubxy"])
def test_closed_form_upper_bounds_dominate_recursive(bound, c):
    """A tighter upper bound is a smaller one."""
    conj = ConjecturePNS(c["n"], c["k"], c["p"], c["q"], c["prob"], c["yi"], c["xj"])
    rec = RecursivePNS(c["n"], c["n"], c["k"], c["p"], c["q"], c["prob"], c["yi"], c["xj"])
    assert getattr(conj, bound) <= getattr(rec, bound) + TOL


@pytest.mark.parametrize("c", _cases(5, n_cases=200), ids=range(200))
def test_dominance_holds_for_arbitrary_index_orderings(c):
    """All four quantities must stay at least as tight as the recursive baseline.

    The theorems are stated for the canonical ordering, where the treatment
    levels of the k factors are x_1..x_k; the published experiments use it
    throughout.  This sweep draws the index vector as an arbitrary subset of
    1..n instead, which is where the guards on the conditional terms have to be
    expressed in terms of membership in `xj` rather than in terms of k.
    """
    conj = ConjecturePNS(c["n"], c["k"], c["p"], c["q"], c["prob"], c["yi"], c["xj"])
    rec = RecursivePNS(c["n"], c["n"], c["k"], c["p"], c["q"], c["prob"], c["yi"], c["xj"])
    for lo, hi in PAIRS:
        assert getattr(conj, lo) >= getattr(rec, lo) - TOL, (
            f"closed-form {lo} looser than baseline "
            f"(n={c['n']}, k={c['k']}, q={c['q']}, xj={c['xj']})"
        )
        assert getattr(conj, hi) <= getattr(rec, hi) + TOL, (
            f"closed-form {hi} looser than baseline "
            f"(n={c['n']}, k={c['k']}, q={c['q']}, xj={c['xj']})"
        )


@pytest.mark.parametrize("c", _cases(11, n_cases=120), ids=range(120))
def test_dominance_holds_in_the_canonical_ordering(c):
    """The ordering the theorems are stated in, and the one every figure uses."""
    xj = list(range(1, c["k"] + 1))
    conj = ConjecturePNS(c["n"], c["k"], c["p"], c["q"], c["prob"], xj[:], xj[:])
    rec = RecursivePNS(c["n"], c["n"], c["k"], c["p"], c["q"], c["prob"], xj[:], xj[:])
    for lo, hi in PAIRS:
        assert getattr(conj, lo) >= getattr(rec, lo) - TOL
        assert getattr(conj, hi) <= getattr(rec, hi) + TOL


@pytest.mark.parametrize("c", CASES[:10], ids=range(10))
def test_bounds_are_stable_across_repeated_construction(c):
    """Guards the shadowing bug fixed during the refactor: in the original the
    bound attributes overwrote the methods that produced them, so reordering the
    assignments in __init__ raised AttributeError."""
    a = ConjecturePNS(c["n"], c["k"], c["p"], c["q"], c["prob"], c["yi"], c["xj"])
    b = ConjecturePNS(c["n"], c["k"], c["p"], c["q"], c["prob"], c["yi"], c["xj"])
    for lo, hi in PAIRS:
        assert getattr(a, lo) == getattr(b, lo)
        assert getattr(a, hi) == getattr(b, hi)
