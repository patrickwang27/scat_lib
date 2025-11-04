from __future__ import annotations
from typing import Tuple, List, Sequence
import numpy as np

def read_xyz(path: str) -> Tuple[np.ndarray, List[str]]:
    """
    Read positions/labels from an XYZ file.

    Parameters
    ----------
    path : str
        Path to the XYZ file.
    Returns
    -------
    Tuple[np.ndarray, List[str]]
        A tuple (positions, labels) where positions is an (N, 3) array of atomic
        positions in Angstrom, and labels is a list of length N with atomic symbols
        or labels.
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    return parse_xyz_lines(lines)

def parse_xyz_string(xyz: str) -> Tuple[np.ndarray, List[str]]:
    """Parse positions/labels from an XYZ string."""
    lines = [ln.strip() for ln in xyz.splitlines() if ln.strip()]
    return parse_xyz_lines(lines)

def parse_xyz_lines(lines: Sequence[str]) -> Tuple[np.ndarray, List[str]]:
    """Parse positions/labels from pre-split XYZ lines."""
    if not lines:
        raise ValueError("Empty XYZ data")
    try:
        N = int(lines[0].split()[0])
    except (ValueError, IndexError) as exc:
        raise ValueError(f"Invalid XYZ atom count line: {lines[0]!r}") from exc
    if len(lines) < N + 2:
        raise ValueError(f"XYZ data has {len(lines) - 2} atom lines, expected {N}")
    body = lines[2:2+N]
    labels: List[str] = []
    pos = np.zeros((N,3), dtype=float)
    for i, ln in enumerate(body):
        parts = ln.split()
        if len(parts) < 4:
            raise ValueError(f"Invalid XYZ atom line (expected label + 3 coords): {ln!r}")
        labels.append(parts[0])
        pos[i,0] = float(parts[1])
        pos[i,1] = float(parts[2])
        pos[i,2] = float(parts[3])
    return pos, labels

def read_xyz_frames(path: str, contain_velocity: bool = True) -> Tuple[List[str], List[np.ndarray]]:
    '''
    Read multiple frames from an XYZ file that may contain velocity information.

    Parameters
    ----------
    path : str
        Path to the XYZ file.
    contain_velocity : bool
        Whether each atom line also stores velocity components (label + 6 numbers).

    Returns
    -------
    Tuple[List[str], List[np.ndarray]]
        The common atom labels and a list of (N, 3) position arrays, one per frame.
    '''
    positions_per_frame: List[np.ndarray] = []
    labels: List[str] | None = None

    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    i = 0
    total_lines = len(lines)
    while i < total_lines:
        # Skip any blank lines between frames
        while i < total_lines and not lines[i].strip():
            i += 1
        if i >= total_lines:
            break

        count_line = lines[i].strip()
        try:
            n_atoms = int(count_line.split()[0])
        except (ValueError, IndexError) as exc:
            raise ValueError(f"Invalid XYZ atom count line: {lines[i]!r}") from exc
        i += 1

        if i >= total_lines:
            raise ValueError("Unexpected end of file while reading XYZ comment line")
        # Skip the comment line (may be blank)
        i += 1

        if total_lines < i + n_atoms:
            raise ValueError(
                f"XYZ data has insufficient atom lines for frame starting near line {i+1}"
            )
        frame_lines = lines[i:i + n_atoms]
        i += n_atoms

        frame_labels: List[str] = []
        frame_positions = np.zeros((n_atoms, 3), dtype=float)

        min_expected_columns = 1 + 3 + (3 if contain_velocity else 0)
        for idx, ln in enumerate(frame_lines):
            parts = ln.split()
            if len(parts) < min_expected_columns:
                raise ValueError(
                    f"Invalid XYZ atom line (expected at least {min_expected_columns - 1} numeric columns): {ln!r}"
                )
            frame_labels.append(parts[0])
            try:
                frame_positions[idx, 0] = float(parts[1])
                frame_positions[idx, 1] = float(parts[2])
                frame_positions[idx, 2] = float(parts[3])
            except ValueError as exc:
                raise ValueError(f"Invalid XYZ coordinate values: {ln!r}") from exc

        if labels is None:
            labels = frame_labels
        elif labels != frame_labels:
            raise ValueError("Atom labels differ between frames; a single label set is expected.")

        positions_per_frame.append(frame_positions)

    return (labels or [], positions_per_frame)


def pair_distance_matrix(positions: np.ndarray) -> np.ndarray:
    """Return NxN matrix of pair distances r_ij [Å]."""
    R = np.asarray(positions, float)
    diffs = R[:,None,:] - R[None,:,:]
    rij = np.linalg.norm(diffs, axis=2)
    return rij
