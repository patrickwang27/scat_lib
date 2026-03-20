from __future__ import annotations
from scipy.interpolate import CubicSpline
import os
import re
from functools import lru_cache
from typing import Callable, List, Mapping, Optional, Tuple
import numpy as np
from .cm import CromerMannTable, fx_cromer_mann
from .constants import PI
from ..iam.constants import ATOMIC_NUMBERS

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
_INELASTIC_GRID = np.array([0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 1.50, 2.00], dtype=float)
_LABEL_BASE_RE = re.compile(r"^([A-Z][a-z]?)")
_LABEL_ION_RE = re.compile(r"^([A-Z][a-z]?)([0-9]*)([+-])$")

# type of a provider function: fx(symbol: str, s: float) -> float
FXFunc = Callable[[str, float], float]


def _resolve_label(label: str, ion_map: Optional[Mapping[str, str]] = None) -> str:
    """Apply ion_map if provided; otherwise return label unchanged."""
    if ion_map and label in ion_map:
        return ion_map[label]
    return label


def _base_element(label: str) -> str:
    """Extract base element symbol from a label (e.g., 'Cval' -> 'C', 'Fe2+' -> 'Fe')."""
    m = _LABEL_BASE_RE.match(label)
    return m.group(1) if m else label


def _parse_charge(label: str) -> int:
    """Parse charge magnitude/sign from ionic labels like 'Fe2+' or 'O1-'.
    Returns positive int for cations, negative for anions, zero otherwise."""
    m = _LABEL_ION_RE.match(label)
    if not m:
        return 0
    magnitude = int(m.group(2)) if m.group(2) else 1
    return magnitude if m.group(3) == "+" else -magnitude


def _electron_count(label: str, resolved: str, base: str, cm: Optional[CromerMannTable], charge: Optional[int] = None) -> float:
    """Estimate electron count for a label using CM coefficients if available, else atomic number and charge."""
    if cm and label in cm:
        coeffs = cm.get(label)
        return float(np.sum(coeffs.a) + coeffs.c)
    if cm and resolved in cm:
        coeffs = cm.get(resolved)
        return float(np.sum(coeffs.a) + coeffs.c)
    Z = ATOMIC_NUMBERS.get(base)
    if Z is None:
        raise ValueError(f"Unknown element '{base}' for inelastic lookup.")
    charge_val = charge if charge is not None else _parse_charge(resolved)
    return float(Z - charge_val)


@lru_cache(maxsize=1)
def _load_inelastic_table() -> Mapping[str, np.ndarray]:
    """Load inelastic (Z - f) data from isfl.txt."""
    table: dict[str, np.ndarray] = {}
    path = os.path.join(DATA_DIR, "isfl.txt")
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            symbol = parts[0]
            vals = np.array([float(x) for x in parts[1:]], dtype=float)
            table[symbol] = vals
    return table


def _inelastic_lookup(base_symbol: str, s: float) -> Optional[float]:
    """Return interpolated Z - f from the tabulated data if possible."""
    table = _load_inelastic_table()
    data = table.get(base_symbol)
    if data is None:
        return None
    if _INELASTIC_GRID[0] <= s <= _INELASTIC_GRID[-1]:
        return float(CubicSpline(_INELASTIC_GRID, data)(s))
    return None

def _sinc(x: np.ndarray) -> np.ndarray:
    """Return sinc(x) = sin(x)/x, handling x=0 case."""
    out = np.empty_like(x, dtype=float)
    small = np.abs(x) < 1e-12
    out[small] = 1.0
    xs = x[~small]
    out[~small] = np.sin(xs)/xs
    return out

def _fx_from_backend(backend: str | FXFunc, cm: Optional[CromerMannTable] = None, ion_map: Optional[Mapping[str, str]] = None) -> FXFunc:
    """Return a function fx(symbol: str, s: float) -> float according to backend."""
    if callable(backend):
        return backend
    be = (backend or 'affl').lower()
    if be == 'xraydb':
        from .xraydb_provider import fx_xraydb as _fx
        return lambda sym, s: _fx(sym, s, ion_map=ion_map)
    elif be in ('affl','cm','cromer-mann'):
        cm = cm or CromerMannTable()
        return lambda sym, s: fx_cromer_mann(sym, s, cm)
    else:
        raise ValueError(f"Unknown backend '{backend}'. Use 'affl' or 'xraydb' or pass a callable.")


def intensity_components_xray(positions: np.ndarray, labels: List[str], q: np.ndarray, cm: Optional[CromerMannTable] = None,
                              *, backend: str | FXFunc = 'affl', ion_map: Optional[Mapping[str, str]] = None,
                              inelastic: bool | str = False) -> Tuple[np.ndarray, ...]:
    """
    Return total, self, and cross I(q) from atomic positions and labels.
    
    Parameters
    ----------
    positions : np.ndarray
        Array of shape (N, 3) with atomic positions in Angstrom.
    labels : List[str]
        List of length N with atomic symbols or labels.
    q : np.ndarray
        1D array of q values (in 1/Angstrom) at which to
        compute the scattering intensity.
    cm : Optional[CromerMannTable], optional
        Optional Cromer-Mann table to use if backend is 'affl'.
        If None, a default table will be used. Default is None.
    backend : str | FXFunc, optional
        Backend to use for atomic form factors.
        'affl' (default) for internal Cromer–Mann table,
        or 'xraydb' to use xraydb.f0,
        or a callable function of signature fx(symbol: str, s: float) -> float.
        Default is 'affl'.
    ion_map : Optional[Mapping[str, str]], optional
        Optional mapping of labels to standard symbols,
        e.g., {'Cval':'C', 'Siv':'Si4+'}.
        Only used if backend is 'xraydb'. Default is None.
    inelastic : bool | str, optional
        Controls computation of the inelastic (incoherent) scattering component.
        False (default) disables it. True selects 'auto' mode (use tabulated data
        when available, otherwise backend-specific fallback). Explicit strings
        'table' or 'xraydb' force the source for Z - f lookups.
    
    Returns
    -------
    Tuple[np.ndarray, ...]
        Three 1D arrays of I(q): total, self, and cross terms.
        When `inelastic` is truthy, a fourth array containing the inelastic
        component is appended.
    """
    R = np.asarray(positions, float)
    q = np.asarray(q, float)
    N = R.shape[0]
    diffs = R[:,None,:] - R[None,:,:]
    rij = np.linalg.norm(diffs, axis=2)  # (N,N)

    I_tot = np.zeros(q.shape, float)
    I_self = np.zeros(q.shape, float)
    I_cross = np.zeros(q.shape, float)
    include_inelastic = bool(inelastic)
    inelastic_mode: Optional[str]
    if not include_inelastic:
        inelastic_mode = None
    else:
        if inelastic is True:
            inelastic_mode = 'auto'
        elif isinstance(inelastic, str):
            inelastic_mode = inelastic.lower()
        else:
            raise TypeError("inelastic must be False, True, or one of {'table','xraydb','auto'}.")
        if inelastic_mode == 'auto':
            if isinstance(backend, str) and backend.lower() == 'xraydb':
                inelastic_mode = 'xraydb'
            else:
                inelastic_mode = 'table'
        if inelastic_mode not in ('table', 'xraydb'):
            raise ValueError("inelastic mode must be 'table', 'xraydb', 'auto', True, or False.")
    if isinstance(backend, str) and backend.lower() in ('affl', 'cm', 'cromer-mann'):
        cm = cm or CromerMannTable()
    fx = _fx_from_backend(backend, cm=cm, ion_map=ion_map)
    I_inelastic = np.zeros(q.shape, float) if include_inelastic else None
    resolved_labels = [_resolve_label(lbl, ion_map=ion_map) for lbl in labels] if include_inelastic else []
    base_labels = [_base_element(lbl) for lbl in resolved_labels] if include_inelastic else []
    electron_counts: dict[str, float] = {}
    charges_by_label: dict[str, int] = {}
    if include_inelastic:
        cm_for_inelastic = None
        if inelastic_mode == 'table':
            cm_for_inelastic = cm or CromerMannTable()
        for raw, resolved, base in zip(labels, resolved_labels, base_labels):
            charge = _parse_charge(resolved)
            charges_by_label[raw] = charge
            electron_counts[raw] = _electron_count(raw, resolved, base, cm_for_inelastic, charge=charge)

    # unique labels to avoid repeated lookups at each q
    labels = list(labels)
    uniq = sorted(set(labels))
    # For each q, compute f0 for each unique symbol once
    for k, qk in enumerate(q):
        s = qk / (4.0*PI)
        f_by_sym = {sym: float(fx(sym, s)) for sym in uniq}
        w = np.array([f_by_sym[sym] for sym in labels], float)
        I_self[k] = float(np.sum(w*w))
        S = _sinc(qk * rij)
        I_tot[k] = float(w @ (S @ w))
        I_cross[k] = I_tot[k] - I_self[k]
        if include_inelastic and I_inelastic is not None:
            if inelastic_mode == 'table':
                total_inelastic = 0.0
                for raw, base in zip(labels, base_labels):
                    use_table = charges_by_label.get(raw, 0) == 0
                    z_minus = _inelastic_lookup(base, s) if use_table else None
                    if z_minus is None:
                        z_minus = electron_counts[raw] - f_by_sym[raw]
                    total_inelastic += max(0.0, z_minus)
                I_inelastic[k] = total_inelastic
            else:  # xraydb
                total_inelastic = 0.0
                for raw in labels:
                    total_inelastic += max(0.0, electron_counts[raw] - f_by_sym[raw])
                I_inelastic[k] = total_inelastic
    if include_inelastic and I_inelastic is not None:
        return I_tot, I_self, I_cross, I_inelastic
    return I_tot, I_self, I_cross

def intensity_molecular_xray(positions: np.ndarray, labels: List[str], q: np.ndarray, cm: Optional[CromerMannTable] = None,
                             *, backend: str | FXFunc = 'affl', ion_map: Optional[Mapping[str, str]] = None,
                             inelastic: bool | str = False):
    """
    Return I(q) from atomic positions and labels.

    Parameters
    ----------
    positions : np.ndarray
        Array of shape (N, 3) with atomic positions in Angstrom.
    labels : List[str]
        List of length N with atomic symbols or labels.
    q : np.ndarray
        1D array of q values (in 1/Angstrom) at which to
        compute the scattering intensity.
    cm : Optional[CromerMannTable], optional
        Optional Cromer-Mann table to use if backend is 'affl'.
        If None, a default table will be used. Default is None.
    backend : str | FXFunc, optional
        Backend to use for atomic form factors.
        'affl' (default) for internal Cromer–Mann table,
        or 'xraydb' to use xraydb.f0,
        or a callable function of signature fx(symbol: str, s: float) -> float.
        Default is 'affl'.
    ion_map : Optional[Mapping[str, str]], optional
        Optional mapping of labels to standard symbols,
        e.g., {'Cval':'C', 'Siv':'Si4+'}.
        Only used if backend is 'xraydb'. Default is None.
    inelastic : bool | str, optional
        Same semantics as in :func:`intensity_components_xray`.

    Returns
    -------
    np.ndarray or Tuple[np.ndarray, np.ndarray]
        If `inelastic` is falsy, returns the molecular intensity array.
        Otherwise, returns a tuple `(I_total, I_inelastic)`.
    """
    comps = intensity_components_xray(positions, labels, q, cm, backend=backend, ion_map=ion_map, inelastic=inelastic)
    if inelastic:
        return comps[0] + comps[-1]
    return comps[0]

def intensity_pyscf(mol: "gto.Mole", q: np.ndarray, cm: Optional[CromerMannTable] = None,
                     *, backend: str | FXFunc = 'affl', ion_map: Optional[Mapping[str, str]] = None,
                     inelastic: bool | str = False):
    """
    Return I(q) from a PySCF gto.Mole object.

    Parameters
    ----------
    mol : gto.Mole
        PySCF molecule with atom positions and labels.
    q : np.ndarray
        1D array of q values (in 1/Angstrom).
    cm, backend, ion_map, inelastic :
        Same semantics as :func:`intensity_molecular_xray`.

    Returns
    -------
    np.ndarray or Tuple[np.ndarray, np.ndarray]
        Matches the return type of :func:`intensity_molecular_xray`.
    """
    from .pyscf_bridge import positions_and_labels_from_mole
    positions, labels = positions_and_labels_from_mole(mol)
    return intensity_molecular_xray(positions, labels, q, cm, backend=backend, ion_map=ion_map, inelastic=inelastic)

    