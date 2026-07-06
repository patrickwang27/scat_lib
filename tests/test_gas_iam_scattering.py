import os
import numpy as np
from scat_lib.gas_iam import scattering


def _load_isfl_values(symbol):
    path = os.path.join(os.path.dirname(scattering.__file__), "data", "isfl.txt")
    grid = np.array([0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 1.50, 2.00], dtype=float)
    with open(path, "r", encoding="utf-8") as fh:
        fh.readline()
        for line in fh:
            parts = line.strip().split()
            if not parts or parts[0] != symbol:
                continue
            values = np.array([float(x) for x in parts[1:]], dtype=float)
            return grid, values
    raise ValueError(f"Missing symbol {symbol} in isfl table")


def test_inelastic_table_matches_tabulated_values():
    positions = np.array([[0.0, 0.0, 0.0],
                          [0.0, 0.0, 1.2]])
    labels = ["C", "O"]
    s_vals = np.array([0.10, 0.50, 1.00], dtype=float)
    q = s_vals * (4.0 * np.pi)

    I_tot, I_self, I_cross, I_inel = scattering.intensity_components_xray(
        positions, labels, q, inelastic="table"
    )

    # Sanity checks on coherent components
    assert I_tot.shape == q.shape
    np.testing.assert_allclose(I_tot, I_self + I_cross, rtol=1e-10, atol=1e-10)

    grid_c, vals_c = _load_isfl_values("C")
    grid_o, vals_o = _load_isfl_values("O")
    assert np.allclose(grid_c, grid_o)

    expected = np.interp(s_vals, grid_c, vals_c) + np.interp(s_vals, grid_o, vals_o)
    np.testing.assert_allclose(I_inel, expected, rtol=1e-7, atol=1e-9)


def test_inelastic_xraydb_mode_uses_electron_difference():
    positions = np.array([[0.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0]])
    labels = ["C", "O"]
    q = np.array([0.2, 0.6], dtype=float)
    electron_counts = {"C": 6.0, "O": 8.0}

    def fake_fx(symbol, s):
        return electron_counts[symbol] - 0.5 * s

    I_tot, I_self, I_cross, I_inel = scattering.intensity_components_xray(
        positions, labels, q, backend=fake_fx, inelastic="xraydb"
    )
    np.testing.assert_allclose(I_tot, I_self + I_cross, rtol=1e-10, atol=1e-10)

    s_vals = q / (4.0 * scattering.PI)
    expected = np.array(
        [
            sum(electron_counts[label] - fake_fx(label, s) for label in labels)
            for s in s_vals
        ]
    )
    np.testing.assert_allclose(I_inel, expected, rtol=1e-8, atol=1e-9)


def test_intensity_molecular_returns_tuple_when_inelastic_requested():
    positions = np.array([[0.0, 0.0, 0.0],
                          [0.0, 0.0, 1.0]])
    labels = ["C", "O"]
    q = np.array([0.2, 0.4], dtype=float)

    total, inelastic = scattering.intensity_molecular_xray(
        positions, labels, q, inelastic=True
    )
    assert isinstance(total, np.ndarray)
    assert isinstance(inelastic, np.ndarray)
    assert total.shape == q.shape == inelastic.shape
