IAM Guide
=========

This guide introduces the Independent Atom Model (IAM) utilities available under
``scat_lib.iam`` for computing X-ray and electron scattering quantities.

What’s Included
---------------

- Kinematics helpers: ``s``, ``g``, and ``q`` conversions
- X-ray form factors: Cromer–Mann ``f_x(s)`` and tabulated ``Z - f_x(s)``
- Electron form factors via Mott–Bethe (with optional relativistic scaling)
- Minimal structure/atom-site classes and structure-factor evaluators

.. note::

   The bundled ``data/isfl.txt`` table (incoherent scattering functions of
   Hubbell *et al.*, J. Phys. Chem. Ref. Data **4**, 471 (1975)) had six
   transcription typos corrected against the primary source (H at
   s = 0.7/0.8, Pd at 0.6, Te at 0.1/0.7, I at 0.5). Tabulated
   ``Z - f_x(s)`` values and Mott–Bethe electron form factors for those
   elements change accordingly.

Quick Start
-----------

Install the library in your environment and import the IAM submodule:

.. code-block:: python

   import numpy as np
   from scat_lib import iam

   # Kinematics
   s = 0.5                     # sin(theta)/lambda in Å^-1
   q = iam.q_from_s(s)         # momentum transfer magnitude [Å^-1]
   g = iam.g_from_s(s)         # electron convention: g = 2*s [Å^-1]

   # X-ray form factor f_x(s) using Cromer–Mann coefficients
   fx_c = iam.fx_cromer_mann('C', s)

   # Electron form factor via Mott–Bethe
   fe_c = iam.fe_mott_bethe('C', s)
   fe_c_rel = iam.fe_mott_bethe_relativistic('C', s, beam_energy_keV=200.0)

Structure Factors
-----------------

Define a simple structure with two carbon atoms and compute structure factors
for a few ``q`` vectors. Positions are in Cartesian Å.

.. code-block:: python

   from scat_lib import iam
   import numpy as np

   struct = iam.Structure([
       iam.AtomSite('C', (0.0, 0.0, 0.0)),
       iam.AtomSite('C', (1.0, 0.0, 0.0), debye_waller=iam.DebyeWaller(B_iso=0.5)),
   ])

   q_vecs = np.array([
       [1.0, 0.0, 0.0],
       [0.0, 1.0, 0.0],
       [0.0, 0.0, 1.0],
   ], dtype=float)

   F_x = iam.structure_factor_xray(struct, q_vecs)
   F_e = iam.structure_factor_electron(struct, q_vecs, beam_energy_keV=200.0)

Notes
-----

- ``DebyeWaller(B_iso)`` applies an isotropic Debye–Waller factor
  ``T(s) = exp(-B_iso * s^2)`` where ``s = |q|/(4π)``.
- Electron ``f_e(s)`` diverges as ``1/s^2`` as ``s -> 0``; avoid ``q=0`` in practice.
- Data files for Cromer–Mann and ``Z - f_x`` tables are bundled in ``scat_lib/iam/data``.

See Also
--------

- API reference: :mod:`scat_lib.iam`
- Modules: :mod:`scat_lib.iam.constants`, :mod:`scat_lib.iam.kinematics`,
  :mod:`scat_lib.iam.form_factors`, :mod:`scat_lib.iam.electron`, :mod:`scat_lib.iam.structure`
 - For a minimal gas-phase implementation mirroring an F90 workflow, see :mod:`scat_lib.gas_iam` and the Gas IAM Guide.
