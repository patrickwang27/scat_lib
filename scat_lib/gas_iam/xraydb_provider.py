from __future__ import annotations
from typing import Optional, Mapping
import re
import numpy as np

def _normalize_label(label: str, ion_map: Optional[Mapping[str, str]] = None) -> str:
    """Normalize label to a form xraydb understands. 
    - Applies user-supplied ion_map first.
    - Accepts 'Fe2+' / 'O1-' style ionic labels.
    - Falls back to bare element symbol if parsing succeeds but charge is unsupported.
    """
    if ion_map and label in ion_map:
        return ion_map[label]
    # direct acceptance (Fe, Fe2+, O1-, etc.)
    if re.match(r'^[A-Z][a-z]?([0-9]+[+-])?$', label):
        return label
    # Try to strip 'val'/'v' suffixes (e.g., Cval, Siv, Sival) => base element (C, Si)
    m = re.match(r'^([A-Z][a-z]?)(?:val|v|valence)?$', label)
    if m:
        return m.group(1)
    # default: return as-is to let xraydb error out
    return label

def _coerce_f0_value(value):
    """Normalize xraydb.f0 output across API variants.

    Some xraydb versions return a scalar-like ndarray even for scalar input,
    while older code paths may return ``(f0, metadata)``. This helper converts
    singleton results to a Python float and preserves true vector outputs.
    """
    if isinstance(value, tuple):
        value = value[0]
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0 or arr.size == 1:
        return float(arr.reshape(-1)[0])
    return arr


def fx_xraydb(symbol: str, s: float | np.ndarray, *, ion_map: Optional[Mapping[str, str]] = None) -> float | np.ndarray:
    """
    Return atomic form factor f_xraydb(symbol, s) using xraydb.f0.
    
    Parameters
    ----------
    symbol : str
        Atomic symbol or label, e.g., 'C', 'O1-', 'Fe2+', 'Cval', 'Siv'.
    s : float | np.ndarray
        Scattering vector magnitude in 1/Angstrom.
    ion_map : Optional[Mapping[str, str]], optional
        Optional mapping of labels to standard symbols,
        e.g., {'Cval':'C', 'Siv':'Si4+'}.
        Default is None.
    
    Returns
    -------
    float | np.ndarray
        Atomic form factor f(s). Scalar inputs produce floats; array inputs
        produce NumPy arrays.

    """
    try:
        import xraydb
    except Exception as exc:
        raise ImportError("xraydb is not installed. Install with: pip install xraydb") from exc
    sym = _normalize_label(symbol, ion_map=ion_map)
    return _coerce_f0_value(xraydb.f0(sym, s))
