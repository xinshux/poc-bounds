"""Version-independent checks for the symbolic tightness verification.

The original script compared solver output as text, which is not reproducible:
SymPy chooses which variables are free when it row-reduces, and that choice has
changed between releases, so the same mathematics prints differently.  These
tests check the structure (shape, rank, number of free variables) and verify the
solution by substituting it back into A x = b, which is stable across versions.
"""
import os

import pytest

sp = pytest.importorskip("sympy")

from pocbounds.symbolic import GaussianSolver, x_syms

HERE = os.path.dirname(__file__)
EQU = os.path.join(HERE, os.pardir, "data", "equ_33.txt")

N_UNKNOWNS = 81
EXPECTED_INFORMERS = 5          # 1 + 1 + k, with k = 3
EXPECTED_ROWS = 16              # 15 equations from equ_33.txt plus one informer
EXPECTED_RANK = 16
EXPECTED_FREE_VARS = 65


@pytest.fixture(scope="module")
def solver(tmp_path_factory):
    out = tmp_path_factory.mktemp("solutions")
    return GaussianSolver(equ_path=EQU, k=3, out_dir=str(out))


def _system(solver, idx):
    informer_A, informer_b = solver.informer_list[idx]
    A = sp.Matrix(solver.equ_list_A + [informer_A])
    b = sp.Matrix(solver.equ_list_b + [informer_b])
    return A, b


def test_solver_builds_the_expected_number_of_systems(solver):
    assert len(solver.informer_list) == EXPECTED_INFORMERS


@pytest.mark.parametrize("idx", range(EXPECTED_INFORMERS))
def test_system_has_the_expected_shape_and_rank(solver, idx):
    A, b = _system(solver, idx)
    assert A.shape == (EXPECTED_ROWS, N_UNKNOWNS)
    assert A.rank() == EXPECTED_RANK
    assert A.shape[1] - A.rank() == EXPECTED_FREE_VARS


@pytest.mark.parametrize("idx", range(EXPECTED_INFORMERS))
def test_system_is_consistent(solver, idx):
    """rank([A|b]) == rank(A), i.e. a solution exists."""
    A, b = _system(solver, idx)
    assert A.row_join(b).rank() == A.rank()


@pytest.mark.parametrize("idx", range(EXPECTED_INFORMERS))
def test_solution_substituted_back_satisfies_the_system(solver, idx):
    """The real correctness check: whatever parameterisation SymPy happens to
    pick, substituting it into A x - b must give the zero vector."""
    A, b = _system(solver, idx)
    sol_set = sp.linsolve((A, b), x_syms)
    assert len(sol_set) == 1

    sol = list(sol_set)[0]
    assert len(sol) == N_UNKNOWNS

    subs = {x_syms[i]: sol[i] for i in range(N_UNKNOWNS)}
    residual = sp.simplify((A * sp.Matrix(list(x_syms)) - b).subs(subs))
    assert residual == sp.zeros(EXPECTED_ROWS, 1)


def test_run_writes_one_report_per_informer(solver, tmp_path):
    out = tmp_path / "reports"
    GaussianSolver(equ_path=EQU, k=3, out_dir=str(out)).run()
    produced = sorted(p.name for p in out.iterdir())
    assert produced == [f"sol{i+1}.txt" for i in range(EXPECTED_INFORMERS)]
    for name in produced:
        text = (out / name).read_text(encoding="utf-8")
        assert "rank(A) = %d" % EXPECTED_RANK in text
        assert text.count("=") > 10
