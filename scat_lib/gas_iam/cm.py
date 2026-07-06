from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

@dataclass(frozen=True)
class CMCoeffs:
    a: np.ndarray  # (4,)
    b: np.ndarray  # (4,)
    c: float

class CromerMannTable:
    """
    Table of Cromer–Mann coefficients loaded from affl.txt (4 Gaussians + constant).
    See: International Tables for Crystallography, Vol. C, 2006, Table 4.2.4.3

    Attributes
    ----------
    path : str
        Path to affl.txt file.
    _d : Dict[str, CMCoeffs]
        Dictionary mapping element symbols to their Cromer-Mann coefficients.
    keys : List[str]
        List of element symbols available in the table.
    Methods
    -------
    __contains__(k: str) -> bool
        Check if element symbol k is in the table.
    get(k: str) -> CMCoeffs
        Get the Cromer-Mann coefficients for element symbol k.
    load_cm_table(path: str | None = None) -> CromerMannTable
        Load a Cromer-Mann table from the specified path or default location.
    fx_cromer_mann(symbol: str, s: float, table: CromerMannTable) -> float
        Evaluate f_x(s) for the given element symbol and s value using the provided table.
    """
    def __init__(self, path: str | None = None):
        self.path = path or os.path.join(DATA_DIR, "affl.txt")
        self._d: Dict[str, CMCoeffs] = {}
        with open(self.path, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln or ln.startswith("#"):
                    continue
                parts = ln.split()
                sym = parts[0]
                vals = [float(x) for x in parts[1:]]
                if len(vals) != 9:
                    raise ValueError(f"affl.txt line for {sym} has {len(vals)} numeric fields; expected 9")
                a = np.array(vals[0:8:2], dtype=float)
                b = np.array(vals[1:8:2], dtype=float)
                c = float(vals[8])
                self._d[sym] = CMCoeffs(a=a, b=b, c=c)

    def __contains__(self, k: str) -> bool:
        return k in self._d

    def get(self, k: str) -> CMCoeffs:
        try:
            return self._d[k]
        except KeyError:
            keys = ", ".join(sorted(self._d.keys())[:10]) + ("..." if len(self._d) > 10 else "")
            raise KeyError(f"Element label '{k}' not found in affl table at {self.path}. Example keys: {keys}")

    @property
    def keys(self) -> List[str]:
        return list(self._d.keys())

def load_cm_table(path: str | None = None) -> CromerMannTable:
    return CromerMannTable(path)

def fx_cromer_mann(symbol: str, s: "float | np.ndarray", table: CromerMannTable) -> "float | np.ndarray":
    """Evaluate f_x(s) from CM coefficients for `symbol` (e.g., 'C', 'Cval', 'Si', 'Siv', 'O1-').
    s is sin(theta)/lambda in Å^-1; scalar input returns a float, array input an array
    of the same shape.
    """
    coeffs = table.get(symbol)
    s_arr = np.asarray(s, dtype=float)
    s2 = np.atleast_1d(s_arr).ravel() ** 2
    # f(s) = sum_i a_i * exp(-b_i s^2) + c
    vals = coeffs.a @ np.exp(-np.outer(coeffs.b, s2)) + coeffs.c
    if s_arr.ndim == 0:
        return float(vals[0])
    return vals.reshape(s_arr.shape)
