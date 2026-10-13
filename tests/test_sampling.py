"""The random SCM samplers must emit genuinely valid distributions."""
import numpy as np
import pytest

from pocbounds import (
    check_px_y_ineq,
    generate_prob_complex,
    generate_prob_confounder,
    generate_prob_mediator,
    generate_prob_with_constraint,
)

GENERATORS = [generate_prob_mediator, generate_prob_confounder, generate_prob_complex]
TOL = 1e-9


@pytest.mark.parametrize("gen", GENERATORS, ids=[g.__name__ for g in GENERATORS])
@pytest.mark.parametrize("n", [3, 4, 5])
def test_joint_distribution_sums_to_one(gen, n):
    np.random.seed(0)
    prob = gen(n_x=n, n_y=n, n_z=3)
    total = sum(prob[f"P(x{i+1},y{j+1})"] for i in range(n) for j in range(n))
    assert total == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("gen", GENERATORS, ids=[g.__name__ for g in GENERATORS])
@pytest.mark.parametrize("n", [3, 4, 5])
def test_interventional_distribution_sums_to_one_per_treatment(gen, n):
    np.random.seed(1)
    prob = gen(n_x=n, n_y=n, n_z=3)
    for i in range(n):
        total = sum(prob[f"P(y{j+1}_x{i+1})"] for j in range(n))
        assert total == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("gen", GENERATORS, ids=[g.__name__ for g in GENERATORS])
@pytest.mark.parametrize("n", [3, 4])
def test_all_probabilities_are_non_negative(gen, n):
    np.random.seed(2)
    prob = gen(n_x=n, n_y=n, n_z=3)
    assert all(v >= -TOL for v in prob.values())


@pytest.mark.parametrize("scm", ["mediator", "confounder", "complex"])
@pytest.mark.parametrize("n", [3, 4])
def test_rejection_sampler_output_satisfies_the_constraint(scm, n):
    """generate_prob_with_constraint promises
    P(x_i,y_j) <= P(y_j|do(x_i)) <= 1 - P(x_i, not y_j)."""
    np.random.seed(3)
    for _ in range(10):
        prob = generate_prob_with_constraint(scm, n_x=n, n_y=n, n_z=3)
        assert check_px_y_ineq(prob, n, n)


def test_constraint_checker_rejects_an_invalid_distribution():
    """A checker that never returns False would make the sampler meaningless."""
    prob = {
        "P(x1,y1)": 0.5, "P(x1,y2)": 0.5,
        "P(x2,y1)": 0.0, "P(x2,y2)": 0.0,
        "P(y1_x1)": 0.0, "P(y2_x1)": 1.0,   # violates P(x1,y1) <= P(y1_x1)
        "P(y1_x2)": 0.5, "P(y2_x2)": 0.5,
    }
    assert check_px_y_ineq(prob, 2, 2) is False


def test_unknown_scm_type_is_rejected():
    with pytest.raises(ValueError):
        generate_prob_with_constraint("not-a-structure", n_x=3)
