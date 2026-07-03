"""Tests for the anisotropic j2 elastic component in scat_lib.gas_iam.

The reference implementation `_xelj2_reference` is a literal translation of the
Mathematica code XElJ2 (M. Simmermacher):

    n = Sqrt[5/Pi]/2
    sig = Sum_{a<b} 2*n*AFF[ela,q]*AFF[elb,q]*SphericalBesselJ[2, q*rab]*P2(zr)
    with zr = vab_z / rab and P2(x) = 3/2 x^2 - 1/2
"""
import os
import sys

import numpy as np
from scipy.special import spherical_jn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scat_lib.gas_iam import (  # noqa: E402
    CromerMannTable,
    fx_cromer_mann,
    intensity_j2_xray,
)
from scat_lib.gas_iam.constants import PI  # noqa: E402

N_Y20 = np.sqrt(5.0 / PI) / 2.0


def _xelj2_reference(positions, labels, q, cm):
    """Direct double-loop translation of the Mathematica XElJ2 function."""
    positions = np.asarray(positions, float)
    n_atoms = len(labels)
    sig = np.zeros_like(np.asarray(q, float))
    for k, qk in enumerate(q):
        s = qk / (4.0 * PI)
        for a in range(n_atoms - 1):
            for b in range(a + 1, n_atoms):
                vab = positions[a] - positions[b]
                rab = np.linalg.norm(vab)
                zr = vab[2] / rab
                lp2 = 1.5 * zr**2 - 0.5
                fa = fx_cromer_mann(labels[a], s, cm)
                fb = fx_cromer_mann(labels[b], s, cm)
                sig[k] += 2.0 * N_Y20 * fa * fb * spherical_jn(2, qk * rab) * lp2
    return sig


def test_matches_mathematica_reference():
    """Vectorised implementation reproduces the literal Mathematica loop."""
    rng = np.random.default_rng(42)
    positions = rng.uniform(-2.0, 2.0, size=(5, 3))
    labels = ["C", "O", "H", "N", "S"]
    q = np.linspace(0.0, 15.0, 200)
    cm = CromerMannTable()

    expected = _xelj2_reference(positions, labels, q, cm)
    actual = intensity_j2_xray(positions, labels, q, cm)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_diatomic_along_z_analytic():
    """For a diatomic along z, I_j2(q) = 2 n f_a f_b j2(q r) since P2(1) = 1."""
    r = 1.128
    positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, r]])
    labels = ["C", "O"]
    q = np.linspace(0.0, 12.0, 100)
    cm = CromerMannTable()

    s = q / (4.0 * PI)
    fC = np.array([fx_cromer_mann("C", sk, cm) for sk in s])
    fO = np.array([fx_cromer_mann("O", sk, cm) for sk in s])
    expected = 2.0 * N_Y20 * fC * fO * spherical_jn(2, q * r)

    actual = intensity_j2_xray(positions, labels, q, cm)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_diatomic_perpendicular_is_minus_half():
    """P2(0) = -1/2: bond along x gives -1/2 of the along-z signal."""
    r = 1.4
    q = np.linspace(0.1, 10.0, 50)
    along_z = intensity_j2_xray(
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, r]]), ["N", "N"], q)
    along_x = intensity_j2_xray(
        np.array([[0.0, 0.0, 0.0], [r, 0.0, 0.0]]), ["N", "N"], q)
    np.testing.assert_allclose(along_x, -0.5 * along_z, rtol=1e-12, atol=1e-12)


def test_magic_angle_is_zero():
    """P2 vanishes at cos(theta) = 1/sqrt(3): no anisotropic signal there."""
    r = 2.0
    z = r / np.sqrt(3.0)
    xy = np.sqrt(r * r - z * z) / np.sqrt(2.0)
    positions = np.array([[0.0, 0.0, 0.0], [xy, xy, z]])
    q = np.linspace(0.0, 10.0, 50)
    I_j2 = intensity_j2_xray(positions, ["C", "C"], q)
    np.testing.assert_allclose(I_j2, 0.0, atol=1e-12)


def test_rotation_about_z_invariance():
    """I_j2 depends only on angles w.r.t. z, so rotations about z leave it unchanged."""
    rng = np.random.default_rng(7)
    positions = rng.uniform(-2.0, 2.0, size=(4, 3))
    labels = ["C", "O", "N", "H"]
    q = np.linspace(0.0, 10.0, 50)

    phi = 1.234
    Rz = np.array([[np.cos(phi), -np.sin(phi), 0.0],
                   [np.sin(phi), np.cos(phi), 0.0],
                   [0.0, 0.0, 1.0]])
    I_orig = intensity_j2_xray(positions, labels, q)
    I_rot = intensity_j2_xray(positions @ Rz.T, labels, q)
    np.testing.assert_allclose(I_rot, I_orig, rtol=1e-10, atol=1e-12)


def test_orientational_average_vanishes():
    """Averaging the bond direction over the sphere gives zero anisotropy.

    For a diatomic, I_j2 is proportional to P2(cos(theta)) of the bond
    orientation; Gauss-Legendre quadrature over cos(theta) integrates it
    exactly, and the integral of P2 over the sphere is zero.
    """
    r = 1.1
    q = np.linspace(0.5, 8.0, 20)
    nodes, weights = np.polynomial.legendre.leggauss(8)
    avg = np.zeros_like(q)
    for ct, wgt in zip(nodes, weights):
        st = np.sqrt(1.0 - ct * ct)
        positions = np.array([[0.0, 0.0, 0.0], [r * st, 0.0, r * ct]])
        avg += wgt * intensity_j2_xray(positions, ["O", "O"], q)
    avg /= 2.0  # normalise the weight integral over [-1, 1]
    np.testing.assert_allclose(avg, 0.0, atol=1e-12)


def test_single_atom_and_q_zero():
    """No pairs -> zero signal; j2(0) = 0 -> zero at q = 0."""
    q = np.linspace(0.0, 10.0, 20)
    I_single = intensity_j2_xray(np.array([[0.1, -0.2, 0.3]]), ["Xe"], q)
    np.testing.assert_allclose(I_single, 0.0, atol=0.0)

    I_co = intensity_j2_xray(
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.1]]), ["C", "O"],
        np.array([0.0]))
    np.testing.assert_allclose(I_co, 0.0, atol=0.0)


def test_coincident_atoms_no_nan():
    """Coincident atoms (r_ab = 0) contribute nothing and produce no NaN."""
    positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.5]])
    q = np.linspace(0.0, 10.0, 20)
    I_j2 = intensity_j2_xray(positions, ["H", "H", "F"], q)
    assert np.all(np.isfinite(I_j2))
    # The coincident H pair adds nothing: same as one H plus the F.
    I_two = intensity_j2_xray(
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.5]]), ["H", "F"], q)
    # Both H atoms pair with F identically, so the H-F contribution doubles.
    np.testing.assert_allclose(I_j2, 2.0 * I_two, rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print(f"PASS {name}")
    print(f"{len(tests)} tests passed.")
