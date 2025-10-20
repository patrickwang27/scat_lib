.. scat_lib documentation master file, created by
   sphinx-quickstart on Mon Jun  9 14:53:21 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

scat_lib documentation
======================

This is a python library with a collection of tools for X-ray scattering calculations developed at the Kirrander Group, University of Oxford. (https://kirrander.web.ox.ac.uk)

The library provides a wrapper for PyXSCAT library. It can compute X-ray scattering amplitudes (total and elastic) from ab initio calculations done by PySCF.

The library also provides a suite of tools in calculating scattering, in the gas phase, using the independent atom model.

Inquiries and issues can be directed to the GitHub issues page, or via email to Patrick Wang (patrick.wang@chem.ox.ac.uk)



.. toctree::
   :maxdepth: 2
   :caption: Contents:

   iam_guide
   gas_iam_guide
   installation_guide
   modules

API Reference
=============

.. autosummary::
   :toctree: _autosummary
   :caption: API Reference

   scat_lib.pyscf_scat
   scat_lib.pyscf_scat.ci_to_2rdm
   scat_lib.pyscf_scat.fit_utils
   scat_lib.pyscf_scat.makerdm
   scat_lib.molecule
   scat_lib.pyscf_scat.rdm_tools
   scat_lib.pyscf_scat.reduced_ci
   scat_lib.pyscf_scat.scat_calc
   scat_lib.iam
   scat_lib.gas_iam

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
