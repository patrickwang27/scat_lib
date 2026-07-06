Gas IAM Guide
=============

The ``scat_lib.gas_iam`` subpackage provides a minimal, fast implementation of gas-phase
independent-atom model (IAM) X-ray scattering.

Key characteristics
-------------------

- No Debye–Waller factors or damping.
- Uses Cromer–Mann form factors with elemental/ionic labels that must match the bundled table.
- Computes the molecular (total) intensity via the Debye expression with a ``sinc`` kernel.
- Form factors are normalised so that ``f(0)`` equals the electron count exactly (see
  `Normalisation and sum rules`_ below); pass ``normalize=False`` for the raw tables.

Quickstart
----------

.. code-block:: python

   import numpy as np
   from scat_lib.gas_iam import intensity_molecular_xray, intensity_pyscf
   from pyscf import gto

   mol = gto.M(atom = 'Li 0 0 0; F 0 0 1.5639', basis = 'sto-3g')
   q = np.linspace(0, 10, 100)  # q in Ang
   I_el, I_inel = intensity_pyscf(mol, q, inelastic=True)  # elastic + incoherent components


One can also specify the positions and labels directly with the ``intensity_molecular_xray``
method. To help with that workflow, ``scat_lib.gas_iam.geometry.read_xyz_frames`` reads
multi-frame XYZ files produced by molecular dynamics and returns a tuple
``(labels, [positions_per_frame])`` where each ``positions_per_frame`` is an ``(N, 3)``
array. A typical usage pattern looks like:

.. code-block:: python

   import numpy as np
   from scat_lib.gas_iam import intensity_molecular_xray
   from scat_lib.gas_iam.geometry import read_xyz_frames

   labels, positions = read_xyz_frames('out.xyz')
   q = np.linspace(0, 10, 100)  # q in Ang
   for pos in positions:
       I_el, I_inel = intensity_molecular_xray(pos, labels, q, inelastic=True)

   # Or use a list comprehension if you just need the total intensities collected
   Iqs = np.array([np.sum(intensity_molecular_xray(pos, labels, q, inelastic=True), axis=0)
                   for pos in positions])
   # Average over frames if desired
   Iq_avg = np.mean(Iqs, axis=0)


Normalisation and sum rules
---------------------------

The tabulated form-factor parameterisations (the bundled Cromer–Mann table and
xraydb's Waasmaier–Kirfel fits) are least-squares fits whose value at
:math:`q = 0` misses the electron count by up to ~0.06 electrons per atom. By
default all intensity functions therefore rescale each atomic form factor by
:math:`(Z - \text{charge})/f(0)` (a relative change of order :math:`10^{-3}`),
which enforces the exact sum rules

.. math::

   I_\text{elastic}(0) = N_e^2, \qquad I_\text{inelastic}(0) = 0,

where :math:`N_e` is the total number of electrons. Pass ``normalize=False``
(or use ``--no-normalize`` on the command line) to reproduce the raw tabulated
values, e.g. for bit-for-bit parity with the original F90 outputs.

Inelastic (incoherent) component
--------------------------------

The tabulated atomic incoherent scattering functions :math:`S(s)` in
``data/isfl.txt`` are the Waller–Hartree values of Hubbell *et al.*,
J. Phys. Chem. Ref. Data **4**, 471 (1975), Table I, for H–Cs on the grid
:math:`s = \sin\theta/\lambda = 0.10 \dots 2.00` Å\ :sup:`-1`. Lookups build a
monotone (PCHIP) interpolant anchored at the exact :math:`S(0) = 0` limit, so
the full range :math:`s \in [0, 2]` is covered without any fallback seam;
beyond :math:`s = 2` the curve continues smoothly towards the
:math:`S(\infty) = Z` asymptote. Ions and elements heavier than Cs fall back
to the cruder :math:`S(s) \approx N_e - f(s)` approximation. The inelastic
component can also be requested from the command line via
``--inelastic table`` (or ``xraydb``/``auto``), which adds ``I_inel(q)`` and
``I_tot(q)`` columns to the output.



Anisotropic j2 component
------------------------

In addition to the isotropic (j0) intensity, the subpackage computes the anisotropic
j2 component of the elastic scattering,

.. math::

   I_{j_2}(q) = \frac{\sqrt{5/\pi}}{2} \sum_{a \neq b} f_a(q)\, f_b(q)\,
                j_2(q\, r_{ab})\, P_2\!\left(\frac{z_{ab}}{r_{ab}}\right),

where :math:`j_2` is the spherical Bessel function of order 2 and :math:`P_2` is the
second Legendre polynomial evaluated at the normalised z-component of the interatomic
vector. The prefactor makes the angular kernel the spherical harmonic
:math:`Y_2^0`. The sum runs only over distinct atom pairs (no self terms), and the
component is purely elastic — within the IAM, inelastic scattering enters only the
isotropic component.

.. code-block:: python

   import numpy as np
   from scat_lib.gas_iam import intensity_j2_xray, intensity_pyscf_j2
   from pyscf import gto

   mol = gto.M(atom = 'Li 0 0 0; F 0 0 1.5639', basis = 'sto-3g')
   q = np.linspace(0, 10, 100)  # q in inverse Ang
   I_j2 = intensity_pyscf_j2(mol, q)

   # Or directly from positions and labels:
   positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.5639]])
   I_j2 = intensity_j2_xray(positions, ['Li', 'F'], q)

The command-line interface exposes the same component via the ``--j2`` flag:

.. code-block:: bash

   python -m scat_lib.gas_iam --xyz molecule.xyz --j2 --out Ij2.txt

Notes
-----

- Element/ion labels must exactly match the keys in the bundled table (e.g., ``'C'``, ``'Cval'``, ``'Siv'``, ``'O1-'``).
- Units: Positions are in Ångstrom, ``q`` is in inverse Å, and the Debye expression uses ``s = q / (4π)``.
- Pass ``inelastic=True`` (or ``'table'`` / ``'xraydb'``) to ``intensity_molecular_xray`` /
  ``intensity_components_xray`` to obtain the incoherent contribution alongside the coherent terms;
  ``intensity_molecular_xray`` then returns the ``(I_elastic, I_inelastic)`` tuple.
- ``normalize=False`` disables the ``f(0) = Z - charge`` rescaling on any of the intensity
  functions (see `Normalisation and sum rules`_).

See also
--------

- :mod:`scat_lib.iam` for a structured IAM implementation with crystals, Debye–Waller factors, and electron scattering.
