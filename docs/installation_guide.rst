Installation Guide
===================

This guide provides a step-by-step walkthrough for installing the Python library.

Prerequisites
-------------
Before you begin the installation process, ensure that you have the following prerequisites:
- Python 3.8 or higher
- pip (Python package installer)
- git installed on your system
- Intel oneAPI base toolkit
- Intel Fortran compiler (ifort is preferred over ifx)
- Intel HPC toolkit

Installation Steps
------------------
1. **Install from git**
   Open your terminal and run the following command to clone the repository:

   .. code-block:: bash

      python -m pip install git+https://github.com/KirranderGroup/scat_lib.git

2. **Locating the Executables**
   The package comes supplied with precompiled executables for Intel 64-bit processors. These executables are located in the `scat_lib/bin` directory of the installed package. You can find the installation path by running the following Python commands:

   .. code-block:: python

      import scat_lib
      import os
      print(os.path.join(os.path.dirname(scat_lib.__file__), 'bin'))

3. **Setting Environment Variables**
   To ensure that the executables can be found by your system, you need to add the `bin` directory to your system's PATH environment variable. You can do this by adding the following line to your shell configuration file (e.g., `.bashrc`, `.zshrc`):

   .. code-block:: bash

      export PATH="$PATH:/path/to/scat_lib/bin"

   Replace `/path/to/scat_lib/bin` with the actual path obtained in the previous step. After adding the line, reload your shell configuration by running:

   .. code-block:: bash

      source ~/.bashrc  # or source ~/.zshrc

4. **Compile from Source**
   If you prefer to compile the executables from source, after pip installation, the following entry point script is made available:

   .. code-block:: bash

      scat-lib-build

   Ensure that you have the Intel Fortran compiler and Intel HPC toolkit properly installed and configured in your environment before running the build script. 
