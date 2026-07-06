import numpy as np
import pytest

from scat_lib.gas_iam.pyscf_bridge import positions_and_labels_from_mole


class DummyMole:
    def __init__(self, xyz: str):
        self._xyz = xyz

    def tostring(self, format: str = "cart"):
        if format != "xyz":
            raise ValueError("Unexpected format")
        return self._xyz


def test_positions_and_labels_from_mole_parses_xyz():
    xyz = """3
water molecule
O  0.000000  0.000000  0.000000
H  0.757000  0.586000  0.000000
H -0.757000  0.586000  0.000000
"""
    mol = DummyMole(xyz)

    pos, labels = positions_and_labels_from_mole(mol)

    assert labels == ["O", "H", "H"]
    expected = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.757, 0.586, 0.0],
            [-0.757, 0.586, 0.0],
        ]
    )
    assert pos.shape == (3, 3)
    np.testing.assert_allclose(pos, expected)


def test_positions_and_labels_from_mole_requires_pyscf_like_object():
    class NoTostring:
        pass

    with pytest.raises(TypeError):
        positions_and_labels_from_mole(NoTostring())

    with pytest.raises(TypeError):
        positions_and_labels_from_mole(None)
