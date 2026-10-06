"""Numerical helpers that paper over NumPy API changes.

NumPy 2.0 removed ``np.trapz`` in favour of ``np.trapezoid``. The pipeline
integrates time courses in several layers, so it would otherwise fail outright
on any environment resolving ``numpy>=2`` from requirements.txt.
"""

from __future__ import annotations

import numpy as np

__all__ = ["trapezoid"]

# np.trapezoid exists from NumPy 2.0; np.trapz is its pre-2.0 spelling and was
# removed in 2.0. Both compute the same composite trapezoidal integral.
trapezoid = getattr(np, "trapezoid", None)
if trapezoid is None:  # pragma: no cover - NumPy < 2.0
    trapezoid = np.trapz
