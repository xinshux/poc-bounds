"""Closed-form bounds for multivalued probabilities of causation.

Reference implementation for:
  X. Shu, S. Wang, A. Li. "Identification of Probabilities of Causation: From
  Recursive to Closed-Form Bounds." IJCAI-ECAI 2026, 4063-4070.
"""

from .bounds import ConjecturePNS
from .recursive import RecursivePNS
from .sampling import (
    generate_prob_mediator,
    generate_prob_confounder,
    generate_prob_complex,
    check_px_y_ineq,
    generate_prob_with_constraint,
)
from .style import set_times_paper_style

__version__ = "0.1.0"

__all__ = [
    "ConjecturePNS",
    "RecursivePNS",
    "generate_prob_mediator",
    "generate_prob_confounder",
    "generate_prob_complex",
    "check_px_y_ineq",
    "generate_prob_with_constraint",
    "set_times_paper_style",
]
