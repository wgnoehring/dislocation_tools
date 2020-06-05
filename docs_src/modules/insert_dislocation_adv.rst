Script for inserting dislocations
=================================

The script :code:`insert_dislocation_adv.py` can be used to insert one or more
straight dislocations into a configuration given as Lammps dump or data file.

Usage
-----

.. code :: bash

    python insert_dislocation_adv.py CONFIGURATION_FILE

Where CONFIGURATION_FILE is a plain text file which contains
information on files, dislocations, etc. See the documentation below.

.. autofunction:: scripts.insert_dislocation_adv.insert

.. autofunction:: scripts.insert_dislocation_adv.get_m_direction
