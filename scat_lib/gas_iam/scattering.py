from __future__ import annotations
import os
import re
import warnings
from collections import Counter
from functools import lru_cache
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.special import spherical_jn

from .cm import CromerMannTable, fx_cromer_mann
from .constants import PI
from ..iam.constants import ATOMIC_NUMBERS

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
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
    """Extract the base element symbol from a label (e.g. 'Cval' -> 'C',
    'Siv' -> 'Si', 'Fe2+' -> 'Fe').

    The leading ``[A-Z][a-z]?`` match is greedy, so a valence/custom label such
    as ``'Cval'`` first yields ``'Cv'``. When that two-letter candidate is not a
    real element but its first letter is, fall back to the single-letter symbol
    so labels like ``'Cval'`` resolve to carbon rather than the non-element
    ``'Cv'``. Genuine two-letter elements ('Fe', 'Si') are left untouched.
    """
    m = _LABEL_BASE_RE.match(label)
    if not m:
        return label
    cand = m.group(1)
    if len(cand) == 2 and cand not in ATOMIC_NUMBERS and cand[0] in ATOMIC_NUMBERS:
        return cand[0]
    return cand


def _parse_charge(label: str) -> int:
    """Parse charge magnitude/sign from ionic labels like 'Fe2+' or 'O1-'.
    Returns positive int for cations, negative for anions, zero otherwise."""
    m = _LABEL_ION_RE.match(label)
    if not m:
        return 0
    magnitude = int(m.group(2)) if m.group(2) else 1
    return magnitude if m.group(3) == "+" else -magnitude


def _effective_electron_count(label: str, ion_map: Optional[Mapping[str, str]] = None) -> Optional[float]:
    """Return the exact electron count Z - charge for a label, or None if the
    base element is unknown."""
    resolved = _resolve_label(label, ion_map)
    Z = ATOMIC_NUMBERS.get(_base_element(resolved))
    if Z is None:
        return None
    return float(Z - _parse_charge(resolved))


def _electron_count(label: str, resolved: str, base: str, cm: Optional[CromerMannTable], charge: Optional[int] = None) -> float:
    """Legacy electron-count estimate used when ``normalize=False``: prefer the
    Cromer-Mann f(0) if the label is tabulated, else fall back to Z - charge."""
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
def _load_inelastic_data() -> Tuple[np.ndarray, Mapping[str, np.ndarray]]:
    """Load the incoherent scattering function table from isfl.txt.

    Returns the s grid parsed from the header line and a mapping
    element -> S(s) values on that grid. The data are validated on load:
    every row must have one value per grid point, and S(s) must be
    non-negative and non-decreasing in s.
    """
    path = os.path.join(DATA_DIR, "isfl.txt")
    table: dict[str, np.ndarray] = {}
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().split()
        try:
            grid = np.array([float(x) for x in header[1:]], dtype=float)
        except ValueError as exc:
            raise ValueError(f"Malformed header in {path}: {' '.join(header)!r}") from exc
        if grid.size < 2 or grid[0] <= 0.0 or np.any(np.diff(grid) <= 0.0):
            raise ValueError(f"Header of {path} must list strictly increasing positive s values.")
        for lineno, line in enumerate(f, start=2):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            symbol = parts[0]
            try:
                vals = np.array([float(x) for x in parts[1:]], dtype=float)
            except ValueError as exc:
                raise ValueError(f"Malformed row for '{symbol}' at {path}:{lineno}") from exc
            if vals.size != grid.size:
                raise ValueError(
                    f"Row for '{symbol}' at {path}:{lineno} has {vals.size} values; expected {grid.size}."
                )
            if symbol in table:
                raise ValueError(f"Duplicate element '{symbol}' at {path}:{lineno}.")
            if np.any(vals < 0.0):
                raise ValueError(f"Negative S(s) value for '{symbol}' at {path}:{lineno}.")
            if np.any(np.diff(vals) < 0.0):
                warnings.warn(
                    f"S(s) for '{symbol}' in {path} is not monotonically increasing; "
                    "the table may contain a transcription error."
                )
            table[symbol] = vals
    return grid, table


@lru_cache(maxsize=None)
def _inelastic_interpolator(base_symbol: str) -> Optional[Callable[[np.ndarray], np.ndarray]]:
    """Return S(s) for a tabulated element as a smooth function of s >= 0, or
    None if the element is not in the table.

    A monotone (PCHIP) interpolant is built through (0, 0) - the exact
    incoherent sum rule S(0) = 0 - and the tabulated points. Beyond the last
    tabulated s the curve continues with a C^1 exponential approach to the
    S(s -> inf) = Z asymptote, so there is no discontinuity at either end of
    the tabulated range.
    """
    grid, table = _load_inelastic_data()
    vals = table.get(base_symbol)
    if vals is None:
        return None
    knots = np.concatenate(([0.0], grid))
    svals = np.concatenate(([0.0], vals))
    pchip = PchipInterpolator(knots, svals, extrapolate=False)
    s_max = float(knots[-1])
    S_max = float(svals[-1])
    Z = float(ATOMIC_NUMBERS.get(base_symbol, S_max))
    gap = Z - S_max
    slope = max(float(pchip.derivative()(s_max)), 0.0)

    def S_of_s(s: np.ndarray) -> np.ndarray:
        s = np.abs(np.asarray(s, dtype=float))
        out = np.asarray(pchip(np.minimum(s, s_max)), dtype=float)
        above = s > s_max
        if np.any(above):
            if gap > 1e-9:
                # S(s) = Z - (Z - S_max) exp(-k (s - s_max)) with k matching
                # the interpolant's slope at s_max: continuous and C^1.
                out[above] = Z - gap * np.exp(-(slope / gap) * (s[above] - s_max))
            else:
                out[above] = min(S_max, Z)
        return np.clip(out, 0.0, None)

    return S_of_s


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


def _to_float(value) -> float:
    """Coerce a scalar-like fx return (float, 0-d, or size-1 array) to a float."""
    arr = np.asarray(value, dtype=float)
    return float(arr.reshape(-1)[0])


def _eval_fx(fx: FXFunc, symbol: str, s: np.ndarray) -> np.ndarray:
    """Evaluate fx(symbol, s) over an array of s values.

    The documented contract is a scalar function ``fx(symbol, s) -> float``. Both
    built-in backends also accept arrays, so a single vectorised call is tried
    first and accepted only if it has the right shape *and* agrees with scalar
    probes at the first and last grid points; the probes guard against a callable
    that ignores s and returns a constant array of the right length. Otherwise
    every point is evaluated individually, which supports scalar-only callables.
    A callable that works on neither array nor scalar input raises a clear error
    rather than a cryptic one.
    """
    s = np.asarray(s, dtype=float)
    if s.size == 0:
        return s.copy()
    flat = s.reshape(-1)
    probe_idx = (0,) if flat.size == 1 else (0, flat.size - 1)
    try:
        probes = {i: _to_float(fx(symbol, float(flat[i]))) for i in probe_idx}
        scalar_ok = True
    except Exception:
        probes = None
        scalar_ok = False
    try:
        vals = np.asarray(fx(symbol, s), dtype=float)
        vflat = vals.reshape(-1)
        if vals.shape == s.shape and (probes is None or
                all(np.isclose(vflat[i], p, rtol=1e-9, atol=1e-12) for i, p in probes.items())):
            return vals
    except Exception:
        pass
    if not scalar_ok:
        raise TypeError(
            f"Form-factor backend failed for symbol {symbol!r} on both array and scalar input."
        )
    return np.array([_to_float(fx(symbol, float(sk))) for sk in flat],
                    dtype=float).reshape(s.shape)


def _form_factors(fx: FXFunc, labels: Sequence[str], s: np.ndarray, *,
                  normalize: bool, ion_map: Optional[Mapping[str, str]] = None
                  ) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """Evaluate form factors for every unique label over the s grid.

    Returns (F, e0) where F maps label -> f(s) array and e0 maps label -> the
    value of the (possibly normalised) form factor at s = 0.

    With ``normalize=True`` each form factor is rescaled by
    (Z - charge) / f(0) so that f(0) equals the electron count exactly. The
    tabulated parameterisations (Cromer-Mann, Waasmaier-Kirfel/xraydb) are
    least-squares fits whose value at s = 0 misses Z by up to ~0.06 e-, which
    otherwise breaks the elastic sum rule I(0) = N_e^2.
    """
    F: Dict[str, np.ndarray] = {}
    e0: Dict[str, float] = {}
    for sym in sorted(set(labels)):
        vals = _eval_fx(fx, sym, s)
        if normalize:
            f0 = float(_eval_fx(fx, sym, np.array([0.0]))[0])
            n_e = _effective_electron_count(sym, ion_map)
            scale = 1.0
            if n_e is None:
                warnings.warn(
                    f"Cannot determine the electron count for label '{sym}'; "
                    "its form factor is left unnormalised."
                )
                n_e = f0
            elif not np.isfinite(f0) or f0 <= 0.0:
                warnings.warn(f"Form factor for '{sym}' is {f0} at s=0; leaving it unnormalised.")
                n_e = f0
            else:
                scale = n_e / f0
                if abs(scale - 1.0) > 0.02:
                    warnings.warn(
                        f"f(0) for '{sym}' deviates from its electron count "
                        f"({f0:.4f} vs {n_e:.1f}); check the form-factor data."
                    )
            F[sym] = vals * scale
            e0[sym] = float(n_e)
        else:
            F[sym] = vals
    return F, e0


def _resolve_inelastic_mode(inelastic: bool | str, backend: str | FXFunc) -> Optional[str]:
    """Validate the ``inelastic`` argument and resolve it to None/'table'/'xraydb'."""
    if not inelastic:
        return None
    if inelastic is True:
        mode = 'auto'
    elif isinstance(inelastic, str):
        mode = inelastic.lower()
    else:
        raise TypeError("inelastic must be False, True, or one of {'table','xraydb','auto'}.")
    if mode == 'auto':
        if isinstance(backend, str) and backend.lower() == 'xraydb':
            mode = 'xraydb'
        else:
            mode = 'table'
    if mode not in ('table', 'xraydb'):
        raise ValueError("inelastic mode must be 'table', 'xraydb', 'auto', True, or False.")
    return mode


def _inelastic_intensity(labels: Sequence[str], s: np.ndarray, F: Mapping[str, np.ndarray],
                         e0: Mapping[str, float], *, mode: str,
                         ion_map: Optional[Mapping[str, str]], normalize: bool,
                         cm: Optional[CromerMannTable]) -> np.ndarray:
    """Sum the atomic incoherent scattering functions S(s) over all atoms.

    In 'table' mode neutral atoms use the tabulated Waller-Hartree S(s)
    (monotone interpolation, exact S(0) = 0, smooth tail to the Z asymptote).
    Ions and elements missing from the table - and everything in 'xraydb'
    mode - use the crude approximation S(s) = N_e - f(s), clipped at zero.
    """
    counts = Counter(labels)
    I_inel = np.zeros_like(s)
    cm_for_fallback = None
    if not normalize and mode == 'table':
        cm_for_fallback = cm or CromerMannTable()
    for raw, n_atoms in counts.items():
        resolved = _resolve_label(raw, ion_map)
        base = _base_element(resolved)
        charge = _parse_charge(resolved)
        contrib: Optional[np.ndarray] = None
        if mode == 'table' and charge == 0:
            S_of_s = _inelastic_interpolator(base)
            if S_of_s is not None:
                contrib = S_of_s(s)
        if contrib is None:
            if normalize:
                e_count = e0[raw]
            else:
                e_count = _electron_count(raw, resolved, base, cm_for_fallback, charge=charge)
            contrib = np.clip(e_count - F[raw], 0.0, None)
        I_inel += n_atoms * contrib
    return I_inel


def _validate_inputs(positions: np.ndarray, labels: List[str], q: np.ndarray) -> None:
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"positions must have shape (N, 3); got {positions.shape}.")
    if len(labels) != positions.shape[0]:
        raise ValueError(f"Got {len(labels)} labels for {positions.shape[0]} atoms.")
    if q.ndim != 1:
        raise ValueError(f"q must be a 1D array; got shape {q.shape}.")


def intensity_components_xray(positions: np.ndarray, labels: List[str], q: np.ndarray, cm: Optional[CromerMannTable] = None,
                              *, backend: str | FXFunc = 'affl', ion_map: Optional[Mapping[str, str]] = None,
                              inelastic: bool | str = False, normalize: bool = True) -> Tuple[np.ndarray, ...]:
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
        'table' or 'xraydb' force the source for the atomic S(s) lookups.
    normalize : bool, optional
        When True (default), rescale each atomic form factor by
        (Z - charge) / f(0) so that f(0) equals the electron count exactly.
        This enforces the elastic sum rule I_total(0) = N_e^2 (and
        I_inelastic(0) = 0), which the raw fitted parameterisations miss by
        up to ~0.06 e- per atom. Set False to use the tabulated
        parameterisations unmodified.

    Returns
    -------
    Tuple[np.ndarray, ...]
        Three 1D arrays of I(q): total, self, and cross terms.
        When `inelastic` is truthy, a fourth array containing the inelastic
        component is appended.
    """
    R = np.asarray(positions, float)
    q = np.asarray(q, float)
    labels = list(labels)
    _validate_inputs(R, labels, q)
    inelastic_mode = _resolve_inelastic_mode(inelastic, backend)
    if isinstance(backend, str) and backend.lower() in ('affl', 'cm', 'cromer-mann'):
        cm = cm or CromerMannTable()
    fx = _fx_from_backend(backend, cm=cm, ion_map=ion_map)

    s = q / (4.0 * PI)
    F, e0 = _form_factors(fx, labels, s, normalize=normalize, ion_map=ion_map)
    W = np.array([F[lbl] for lbl in labels], dtype=float)  # (N, nq)
    I_self = np.einsum('ak,ak->k', W, W)

    diffs = R[:, None, :] - R[None, :, :]
    rij = np.linalg.norm(diffs, axis=2)  # (N,N)
    I_tot = np.empty_like(q)
    for k, qk in enumerate(q):
        w = W[:, k]
        I_tot[k] = w @ (_sinc(qk * rij) @ w)
    I_cross = I_tot - I_self

    if inelastic_mode is None:
        return I_tot, I_self, I_cross
    I_inelastic = _inelastic_intensity(labels, s, F, e0, mode=inelastic_mode,
                                       ion_map=ion_map, normalize=normalize, cm=cm)
    return I_tot, I_self, I_cross, I_inelastic


def intensity_molecular_xray(positions: np.ndarray, labels: List[str], q: np.ndarray, cm: Optional[CromerMannTable] = None,
                             *, backend: str | FXFunc = 'affl', ion_map: Optional[Mapping[str, str]] = None,
                             inelastic: bool | str = False, normalize: bool = True):
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
    normalize : bool, optional
        Same semantics as in :func:`intensity_components_xray`.

    Returns
    -------
    np.ndarray or Tuple[np.ndarray, np.ndarray]
        If `inelastic` is falsy, returns the molecular (elastic) intensity array.
        Otherwise, returns a tuple `(I_total, I_inelastic)` where `I_total` is
        the elastic molecular intensity.
    """
    comps = intensity_components_xray(positions, labels, q, cm, backend=backend, ion_map=ion_map,
                                      inelastic=inelastic, normalize=normalize)
    if inelastic:
        return comps[0], comps[-1]
    return comps[0]

# Y_2^0 normalisation of the anisotropic component: n * P_2(cos theta) = Y_2^0(theta)
_Y20_NORM = np.sqrt(5.0 / PI) / 2.0


def intensity_j2_xray(positions: np.ndarray, labels: List[str], q: np.ndarray, cm: Optional[CromerMannTable] = None,
                      *, backend: str | FXFunc = 'affl', ion_map: Optional[Mapping[str, str]] = None,
                      normalize: bool = True) -> np.ndarray:
    """
    Return the anisotropic j2 component of the elastic scattering, I_j2(q).

    Within the IAM the second (l = 2) term of the Legendre expansion of the
    elastic scattering is

        I_j2(q) = n * sum_{a != b} f_a(q) f_b(q) j_2(q r_ab) P_2(z_ab / r_ab)

    where n = sqrt(5/pi)/2 (so that n P_2(cos theta) = Y_2^0(theta)), j_2 is
    the spherical Bessel function of order 2, and P_2 is the second Legendre
    polynomial evaluated at the normalised z-component of the interatomic
    vector r_ab. Unlike the isotropic (j0) component, the sum runs only over
    distinct atom pairs (no self terms) and the component is purely elastic:
    inelastic scattering enters only the isotropic component within the IAM.

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
    normalize : bool, optional
        Same semantics as in :func:`intensity_components_xray`; the same
        form factors are used for all components, so keep the setting
        consistent between the j0 and j2 parts of one calculation.

    Returns
    -------
    np.ndarray
        1D array of the anisotropic elastic component I_j2(q).
    """
    R = np.asarray(positions, float)
    q = np.asarray(q, float)
    labels = list(labels)
    _validate_inputs(R, labels, q)
    diffs = R[:,None,:] - R[None,:,:]
    rij = np.linalg.norm(diffs, axis=2)  # (N,N)
    # P_2(cos theta_ab) directly from the normalised z-component of r_ab.
    # Entries with r_ab = 0 carry no weight since j2(0) = 0; the guarded
    # divide only keeps them finite.
    zr = np.divide(diffs[:,:,2], rij, out=np.zeros_like(rij), where=rij > 0.0)
    P2 = 1.5*zr*zr - 0.5
    np.fill_diagonal(P2, 0.0)  # no self terms in the anisotropic component

    if isinstance(backend, str) and backend.lower() in ('affl', 'cm', 'cromer-mann'):
        cm = cm or CromerMannTable()
    fx = _fx_from_backend(backend, cm=cm, ion_map=ion_map)

    s = q / (4.0 * PI)
    F, _ = _form_factors(fx, labels, s, normalize=normalize, ion_map=ion_map)
    W = np.array([F[lbl] for lbl in labels], dtype=float)  # (N, nq)
    I_j2 = np.empty_like(q)
    for k, qk in enumerate(q):
        w = W[:, k]
        K = spherical_jn(2, qk * rij) * P2
        I_j2[k] = _Y20_NORM * (w @ (K @ w))
    return I_j2


def intensity_pyscf(mol: "gto.Mole", q: np.ndarray, cm: Optional[CromerMannTable] = None,
                     *, backend: str | FXFunc = 'affl', ion_map: Optional[Mapping[str, str]] = None,
                     inelastic: bool | str = False, normalize: bool = True):
    """
    Return I(q) from a PySCF gto.Mole object.

    Parameters
    ----------
    mol : gto.Mole
        PySCF molecule with atom positions and labels.
    q : np.ndarray
        1D array of q values (in 1/Angstrom).
    cm, backend, ion_map, inelastic, normalize :
        Same semantics as :func:`intensity_molecular_xray`.

    Returns
    -------
    np.ndarray or Tuple[np.ndarray, np.ndarray]
        Matches the return type of :func:`intensity_molecular_xray`.
    """
    from .pyscf_bridge import positions_and_labels_from_mole
    positions, labels = positions_and_labels_from_mole(mol)
    return intensity_molecular_xray(positions, labels, q, cm, backend=backend, ion_map=ion_map,
                                    inelastic=inelastic, normalize=normalize)


def intensity_pyscf_j2(mol: "gto.Mole", q: np.ndarray, cm: Optional[CromerMannTable] = None,
                       *, backend: str | FXFunc = 'affl', ion_map: Optional[Mapping[str, str]] = None,
                       normalize: bool = True) -> np.ndarray:
    """
    Return the anisotropic j2 elastic component I_j2(q) from a PySCF gto.Mole object.

    Parameters
    ----------
    mol : gto.Mole
        PySCF molecule with atom positions and labels.
    q : np.ndarray
        1D array of q values (in 1/Angstrom).
    cm, backend, ion_map, normalize :
        Same semantics as :func:`intensity_j2_xray`.

    Returns
    -------
    np.ndarray
        1D array of the anisotropic elastic component I_j2(q).
    """
    from .pyscf_bridge import positions_and_labels_from_mole
    positions, labels = positions_and_labels_from_mole(mol)
    return intensity_j2_xray(positions, labels, q, cm, backend=backend, ion_map=ion_map,
                             normalize=normalize)
