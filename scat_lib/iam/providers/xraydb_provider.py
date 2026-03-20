"""Optional provider using the `xraydb` Python package (if installed).

This gives access to energy-dependent corrections f1, f2 and alternative f0 coefficients.
We only import at call-time to avoid a hard dependency.
"""
from __future__ import annotations
from typing import Optional
import numpy as np


def _coerce_f0_value(value):
    """Normalize xraydb.f0 output across scalar, ndarray, and tuple variants."""
    if isinstance(value, tuple):
        value = value[0]
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0 or arr.size == 1:
        return float(arr.reshape(-1)[0])
    return arr


def xraydb_fx(element: str, s: float | np.ndarray) -> Optional[float | np.ndarray]:
    """Return f0(s) using xraydb if available, else None."""
    try:
        import xraydb
    except Exception:
        return None
    return _coerce_f0_value(xraydb.f0(element, s))
