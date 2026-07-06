"""Tests for the gas_iam normalisation and inelastic-continuity fixes.

Covers:
- the elastic sum rule I(0) = N_e^2 (exact with normalize=True, the default);
- the incoherent sum rule S(0) = 0;
- continuity/monotonicity of the inelastic component across the tabulated
  range and its former fallback boundaries (s = 0.1 and s = 2.0);
- integrity of the bundled isfl.txt (Hubbell et al. 1975) table;
- backend equivalence and scalar-callable fallbacks.
"""
import os
import warnings

import numpy as np
import pytest

from scat_lib.gas_iam import scattering
from scat_lib.gas_iam.cm import CromerMannTable, fx_cromer_mann
from scat_lib.iam.constants import ATOMIC_NUMBERS

GLYCINE_POS = np.array([
    [0.000, 0.000, 0.000],
    [1.451, 0.000, 0.000],
    [2.100, 1.400, 0.000],
    [1.500, 2.500, 0.000],
    [3.400, 1.450, 0.000],
    [-0.350, -0.950, 0.000],
    [-0.350, 0.500, 0.800],
    [1.800, -0.550, 0.870],
    [1.800, -0.550, -0.870],
    [3.750, 2.350, 0.000],
])
GLYCINE_LABELS = ["N", "C", "C", "O", "O", "H", "H", "H", "H", "H"]
GLYCINE_NEL = sum(ATOMIC_NUMBERS[l] for l in GLYCINE_LABELS)


def _has_xraydb():
    try:
        import xraydb  # noqa: F401
        return True
    except ImportError:
        return False


BACKENDS = ["affl"] + (["xraydb"] if _has_xraydb() else [])


@pytest.mark.parametrize("backend", BACKENDS)
def test_elastic_sum_rule_at_q0(backend):
    """I_total(0) equals (sum_a Z_a)^2 exactly with the default normalisation."""
    q = np.array([0.0])
    I_tot, I_self, I_cross = scattering.intensity_components_xray(
        GLYCINE_POS, GLYCINE_LABELS, q, backend=backend)
    np.testing.assert_allclose(I_tot[0], GLYCINE_NEL**2, rtol=1e-12)
    expected_self = sum(ATOMIC_NUMBERS[l]**2 for l in GLYCINE_LABELS)
    np.testing.assert_allclose(I_self[0], expected_self, rtol=1e-12)


@pytest.mark.parametrize("backend", BACKENDS)
def test_inelastic_sum_rule_at_q0(backend):
    """The incoherent component vanishes at q = 0 for both lookup modes."""
    q = np.array([0.0])
    for mode in ("table", "xraydb"):
        out = scattering.intensity_components_xray(
            GLYCINE_POS, GLYCINE_LABELS, q, backend=backend, inelastic=mode)
        np.testing.assert_allclose(out[-1][0], 0.0, atol=1e-10)


def test_raw_form_factors_break_sum_rule():
    """Sanity check that normalize=False reproduces the historic behaviour
    (raw Cromer-Mann f(0) != Z, so I(0) misses N_e^2 slightly)."""
    q = np.array([0.0])
    I_raw = scattering.intensity_molecular_xray(
        GLYCINE_POS, GLYCINE_LABELS, q, normalize=False)
    assert abs(I_raw[0] - GLYCINE_NEL**2) > 0.1
    np.testing.assert_allclose(I_raw[0], 1599.2928781456005, rtol=1e-12)


def test_ionic_labels_sum_rule():
    """Ionic labels count Z - charge electrons: Na1+ (10) + Cl1- (18) -> 784."""
    positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.8]])
    I = scattering.intensity_molecular_xray(positions, ["Na1+", "Cl1-"], np.array([0.0]))
    np.testing.assert_allclose(I[0], (10 + 18)**2, rtol=1e-12)


@pytest.mark.parametrize("element", ["H", "C", "O", "Fe", "I", "Cs"])
def test_inelastic_interpolant_is_smooth_and_monotone(element):
    """S(s) starts at exactly 0, increases monotonically, has no jumps at the
    old fallback boundaries (s = 0.1, s = 2.0), and tends to Z from below."""
    S_of_s = scattering._inelastic_interpolator(element)
    assert S_of_s is not None
    s = np.linspace(0.0, 3.0, 6001)
    S = S_of_s(s)
    assert S[0] == 0.0
    assert np.all(np.isfinite(S))
    steps = np.diff(S)
    assert np.all(steps >= -1e-12), "S(s) must be non-decreasing"
    # No discontinuities: on this grid (ds = 5e-4) steps must be tiny.
    assert steps.max() < 0.05
    Z = ATOMIC_NUMBERS[element]
    assert np.all(S <= Z + 1e-9)


def test_inelastic_matches_table_at_knots():
    """The interpolant reproduces the tabulated values exactly at the knots."""
    grid, table = scattering._load_inelastic_data()
    for element in ("C", "N", "O", "S", "Fe", "Xe"):
        S_of_s = scattering._inelastic_interpolator(element)
        np.testing.assert_allclose(S_of_s(grid), table[element], rtol=0, atol=1e-12)


def test_isfl_table_is_monotonic_everywhere():
    """After the Hubbell-1975 typo fixes, every row must be non-decreasing in s
    (S is a cumulative quantity) and bounded by the atomic number."""
    grid, table = scattering._load_inelastic_data()
    for element, vals in table.items():
        assert np.all(np.diff(vals) >= 0.0), f"non-monotonic S(s) for {element}"
        assert vals[-1] <= ATOMIC_NUMBERS[element] + 1e-9, f"S exceeds Z for {element}"


def test_isfl_hydrogen_matches_analytic():
    """For hydrogen the Waller-Hartree S(s) = 1 - f(s)^2 is exact, with
    f(s) = (1 + (2 pi a0 s)^2)^-2. The shipped row must match to 3 decimals."""
    a0 = 0.529177210903
    grid, table = scattering._load_inelastic_data()
    f = (1.0 + (2.0 * np.pi * a0 * grid) ** 2) ** -2
    np.testing.assert_allclose(table["H"], np.round(1.0 - f * f, 3), atol=5.1e-4)


def test_inelastic_continuous_into_tail():
    """Full pipeline check: I_inelastic(q) sampled densely through q = 4*pi*0.1
    and q = 8*pi (the former fallback edges) shows no jump."""
    positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.2]])
    labels = ["Fe", "O"]
    for s_edge in (0.1, 2.0):
        q = 4.0 * np.pi * np.linspace(s_edge - 0.01, s_edge + 0.01, 201)
        out = scattering.intensity_components_xray(positions, labels, q, inelastic="table")
        I_inel = out[-1]
        assert np.abs(np.diff(I_inel)).max() < 0.02, f"jump near s = {s_edge}"


def test_scalar_only_callable_backend():
    """A custom scalar-only fx callable still works (per-point fallback)."""
    def fx(symbol, s):
        if not np.isscalar(s):
            raise TypeError("scalar only")
        return {"C": 6.0, "O": 8.0}[symbol] * np.exp(-s)

    positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.1]])
    q = np.linspace(0.0, 5.0, 7)
    I_tot, I_self, I_cross = scattering.intensity_components_xray(
        positions, ["C", "O"], q, backend=fx)
    np.testing.assert_allclose(I_tot[0], (6 + 8)**2, rtol=1e-12)


@pytest.mark.skipif(not _has_xraydb(), reason="xraydb not installed")
def test_backends_agree_after_normalisation():
    """Normalised CM and xraydb form factors are independent fits of the same
    underlying atomic data; the resulting I(q) should agree to well under 1%."""
    q = np.linspace(0.0, 12.0, 25)
    I_cm = scattering.intensity_molecular_xray(GLYCINE_POS, GLYCINE_LABELS, q, backend="affl")
    I_xdb = scattering.intensity_molecular_xray(GLYCINE_POS, GLYCINE_LABELS, q, backend="xraydb")
    np.testing.assert_allclose(I_cm, I_xdb, rtol=5e-3, atol=0.3)


def test_vectorised_cm_matches_scalar():
    """fx_cromer_mann accepts arrays and matches per-point scalar evaluation."""
    cm = CromerMannTable()
    s = np.linspace(0.0, 2.0, 41)
    vec = fx_cromer_mann("Fe", s, cm)
    scalar = np.array([fx_cromer_mann("Fe", float(sk), cm) for sk in s])
    assert isinstance(fx_cromer_mann("Fe", 0.3, cm), float)
    np.testing.assert_allclose(vec, scalar, rtol=1e-15)


def test_molecular_tuple_return_with_inelastic():
    """intensity_molecular_xray returns (I_elastic, I_inelastic) arrays when
    the inelastic component is requested (regression test for the summed-array
    return bug)."""
    positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    q = np.array([0.2, 0.4])
    total, inelastic = scattering.intensity_molecular_xray(
        positions, ["C", "O"], q, inelastic=True)
    assert isinstance(total, np.ndarray) and isinstance(inelastic, np.ndarray)
    assert total.shape == q.shape == inelastic.shape
    comps = scattering.intensity_components_xray(
        positions, ["C", "O"], q, inelastic=True)
    np.testing.assert_allclose(total, comps[0], rtol=0, atol=0)
    np.testing.assert_allclose(inelastic, comps[-1], rtol=0, atol=0)


def test_input_validation():
    positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    q = np.array([0.1])
    with pytest.raises(ValueError, match="labels"):
        scattering.intensity_components_xray(positions, ["C"], q)
    with pytest.raises(ValueError, match="shape"):
        scattering.intensity_components_xray(positions[:, :2], ["C", "O"], q)
    with pytest.raises(ValueError, match="1D"):
        scattering.intensity_components_xray(positions, ["C", "O"], np.array(1.0))


@pytest.mark.parametrize("label,base", [("Cval", "C"), ("Siv", "Si"), ("Sival", "Si"),
                                        ("Fe2+", "Fe"), ("O1-", "O")])
def test_base_element_disambiguation(label, base):
    assert scattering._base_element(label) == base


def test_valence_label_normalises_without_warning():
    """A documented valence label ('Cval', 6 electrons) must normalise to Z=6
    like plain 'C' rather than tripping the unknown-element warning."""
    positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.2]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        I = scattering.intensity_molecular_xray(positions, ["Cval", "O"], np.array([0.0]))
    np.testing.assert_allclose(I[0], (6 + 8) ** 2, rtol=1e-12)


def test_eval_fx_rejects_s_ignoring_callable():
    """A callable whose vectorised output ignores s (constant array of matching
    length) must not be trusted; the per-point scalar path is used instead."""
    def bad_fx(symbol, s):
        return np.array([1.0, 2.0, 3.0])  # ignores s entirely

    out = scattering._eval_fx(bad_fx, "C", np.array([0.5, 0.6, 0.7]))
    assert not np.allclose(out, [1.0, 2.0, 3.0])


def test_eval_fx_vectorised_matches_scalar_loop():
    """A genuine elementwise callable is accepted on the fast path and equals
    the per-point evaluation."""
    def good_fx(symbol, s):
        return {"C": 6.0, "O": 8.0}[symbol] * np.exp(-np.asarray(s, float))

    s = np.linspace(0.0, 2.0, 9)
    fast = scattering._eval_fx(good_fx, "O", s)
    slow = np.array([good_fx("O", float(sk)) for sk in s])
    np.testing.assert_allclose(fast, slow, rtol=1e-12)


def test_eval_fx_broken_callable_raises_clear_error():
    def broken(symbol, s):
        raise RuntimeError("backend down")

    with pytest.raises((TypeError, RuntimeError)):
        scattering._eval_fx(broken, "C", np.array([0.1, 0.2]))


def test_unknown_label_warns_and_stays_unnormalised():
    """A label whose element cannot be identified is left unnormalised (with a
    warning) instead of crashing."""
    def fx(symbol, s):
        return 5.5 * np.exp(-np.asarray(s, dtype=float))

    positions = np.array([[0.0, 0.0, 0.0]])
    with pytest.warns(UserWarning, match="electron count"):
        I = scattering.intensity_molecular_xray(positions, ["Qq"], np.array([0.0]), backend=fx)
    np.testing.assert_allclose(I[0], 5.5**2, rtol=1e-12)
