#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Insert dislocations into an anisotropic medium.
insert_dislocation_adv.py configfile

Read atom coordinates from a Lammps file and insert one or more dislocations by
applying their anisotropic-elastic displacement  field. The field is calculated
with  Stroh's advanced  straight  dislocation formalism,  or  with Barnett  and
Lothe's integral  formalism [2].  The two formalisms  are closely  related, see
Hirth and Lothe's book [3], and the review of Bacon etal. [4].

The variables of the elastic problem are given similar names as in [3] and [4].

Parameters
----------
configfile (str): configuration file in the format of python-configparser.
    The file contains three mandatory sections:
    [simulation cell]
        x, y, z (ndarray): The orientation of  the crystal relative to the x, y
            and z directions of the simulation  cell The vectors do not need to
            be  normalized.  For each  vector,  the  three components  must  be
            written as three floating point numbers on the same line, separated
            by whitespace.
        boundary_style (str): Boundary  style as used by  the Lammps 'boundary'
            command, e.g. 's s p'.
    [elastic  constants]
        c11, c12, c44  (float): cubic elastic stiffnesses. These  should be the
            components  relative to  the  [100]-[010]-[001] CRYSTAL  coordinate
            system; NOT relative  to the simulation cell.  The program performs
            all the required tensor rotations!

    [files]
        format (str): 'dump' or 'data'
        input, output (str): paths to the input and output files.
        append_displacements (bool):  if True,  the displacement field  will be
            appended to the output file (as the last three columns). 'ux uy uz'
            will be appended to the ITEM:ATOMS header line.

    The mandatory sections are followed by an arbitrary number of sections with
    the name [dislocationX], where X is an integer. Each [dislocationX] section
    defines  a  dislocation. If  there  are  several  such sections,  then  the
    integers X decide in which sequence the displacement fields will be applied
    (ascending order, i.e. dislocation1 would be inserted before dislocation2).
    Parameters in a dislocation-section:
        b (ndarray): Burgers vector (distance units)
        xi (ndarray): Line direction, does not need to be normalized
        center (ndarray): Center in simulation cell coordinates
        m (ndarray,  optional): first  direction in the  dislocation coordinate
            system, expressed  in crystal coordinates. This  vector is parallel
            to the  normal of the  plane along which the  cut would be  made to
            insert the dislocation.
        solution_method  (str):  Method  for  solving  for  the  displacements.
            Recognized   choices   are:   "stroh"  (for   Stroh's   formalism),
            "symbolical_integral_method"  (for   the  Barnett-Lothe  formalism,
            using symbolical math), as well as "numerical_integral_method" (for
            the Barnett-Lothe formalism, using  numerical math). The symbolical
            backend is faster than the numerical one.


    Example file:
        [simulation cell]
        x =  1 -2  1
        y = -1 -1 -1
        z =  1  0 -1
        boundary_style = s s p
        [elastic constants]
        c11 = 170.221549146026319
        c12 = 122.600531649638015
        c44 = 75.848200859164038
        [files]
        format = dump
        input  = Cu_parallelepiped.atom
        output = Cu_parallelepiped_with_screw_dislocation.atom
        append_displacements = True
        [dislocation1]
        b   =  -1.807500856188893   0.000000000000000   1.807500856188893
        xi  =  -1 0 1
        m   =   1 2 1
        center = 125.444552873685 88.702693999897 0.0
        solution_method = stroh

Notes
-----
Currently, Lammps  dump and  data files  are supported.  Input dump  files must
contain the columns 'id',  'type', 'x', 'y', and 'z', in  this order. They must
contain a single snapshot. Data files  are currently read and written using the
Lammps Python API, i.e. Lammps is actually called.

Todo: a data file  parser which does not rely on  Lammps should be implemented.
The file reading / writing functionality should be outsourced into a module.

References
----------
[1] Stroh, A.N. J. Math. Phys., 41: 77 (1962)
[2] Barnett, D.M.; Lothe, J. Phys. Norvegica, 7: 13 (1973)
[3] Hirth, J.P.; Lothe, J. Theory of Dislocations, 2nd Edition;
    John Wiley and Sons, 1982. pp 467
[4] Bacon, D. J.; Barnett, D. M.; Scattergood, R. O.
    Progress in Materials Science 1978, 23, 51-262.
"""

import sys
import configparser
import numpy as np
from scipy import isclose

from ..io.lammps_formats import read_data, write_data, read_dump, write_dump
from ..backend.stroh import 
from ..tensor.tensor import voigt2tensor
from ..backend.stroh import (
    solve_sextic_equations, 
    calculate_displacements_from_eigensystem
)
from ..backend.integral import (
        calculate_displacements_with_numerical_integrals
        calculate_displacements_with_symbolical_integrals,
        calculate_cylindrical_coordinates
)

__author__ = "Wolfram Georg Nöhring"
__copyright__ = """\
© All rights reserved. ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, 
Switzerland, Laboratory for Multiscale Mechanics Modelling, 2015"""
__license__ = "GNU General Public License"
__email__ = "wolfram.nohring@imtek.uni-freiburg.de"

def main():
    # Parse coordinate system, elastic constants, and files
    configfile = sys.argv[1]
    config = configparser.ConfigParser()
    config.read(configfile)
    x = np.array(config.get('simulation cell', 'x').split(), dtype=float)
    y = np.array(config.get('simulation cell', 'y').split(), dtype=float)
    z = np.array(config.get('simulation cell', 'z').split(), dtype=float)
    x /= np.linalg.norm(x)
    y /= np.linalg.norm(y)
    z /= np.linalg.norm(z)
    boundary_style = config.get('simulation cell', 'boundary_style').strip()
    c11 = config.getfloat('elastic constants', 'c11')
    c12 = config.getfloat('elastic constants', 'c12')
    c44 = config.getfloat('elastic constants', 'c44')
    file_format = config.get('files', 'format').strip()
    infile = config.get('files', 'input')
    outfile = config.get('files', 'output')
    append_displacements = config.getboolean('files', 'append_displacements')

    # Read input coordinates
    if file_format == 'dump':
        (header, atomdata) = read_dump(infile)
        coordinates = atomdata[:, 2:5]
    elif file_format == 'data':
        coordinates = read_data(infile, boundary_style)
        if append_displacements:
            raise ValueError(
                'Appending displacements to data files ' +
                'is currently not supported.'
            )
    else:
        raise ValueError('Wrong file format.')

    # Parse dislocations and sort them according to the specified rank
    list_of_dislocations = []
    for section_name in config.sections():
        if section_name.startswith('dislocation'):
            rank = int(section_name[11::])
            if (sys.version_info.major > 3):
                list_of_dislocations.append([rank, config[section_name]])
            else:
                list_of_dislocations.append(
                    [rank, config._sections[section_name]]
                )
    list_of_dislocations.sort(key=lambda x: x[0])
    list_of_dislocations = [sublist[1] for sublist in list_of_dislocations]

    # Check that we can solve for all dislocations
    for dislocation in list_of_dislocations:
        try:
            solution_method = dislocation["solution_method"]
        except KeyError:
            solution_method = "stroh"
            dislocation["solution_method"] = "stroh"
            print("No solution method specified. Using Stroh's formalism.")
        recognized_solution_methods = [
            "stroh",
            "symbolical_integral_method",
            "numerical_integral_method"
        ]
        if solution_method not in recognized_solution_methods:
            raise ValueError(
                "Unknown solution method. Valid choices are: " +
                ("{:s}, " * len(recognized_solution_methods)).format(
                        *recognized_solution_methods
                    ).rstrip(", ")
                )

    # Define the elastic constants
    c_voigt = np.zeros((6, 6), dtype=float)
    c_voigt[0:3, 0:3] = c12
    for i in range(0, 3):
        c_voigt[i, i] = c11
    for i in range(3, 6):
        c_voigt[i, i] = c44
    c = voigt2tensor(c_voigt)
    # Rotate stiffness tensor to the laboratory coordinate system
    r_crys_lab = np.zeros((3, 3), dtype=float)
    r_crys_lab[0, :] = x / np.linalg.norm(x)
    r_crys_lab[1, :] = y / np.linalg.norm(y)
    r_crys_lab[2, :] = z / np.linalg.norm(z)
    assert(isclose(np.linalg.det(r_crys_lab), 1.0))
    c = np.einsum(
        'ig,jh,ghmn,km,ln',
        r_crys_lab, r_crys_lab, c, r_crys_lab, r_crys_lab
    )
    if append_displacements:
        reference_coordinates = np.copy(coordinates)
    for dislocation in list_of_dislocations:
        b = np.array(dislocation['b'].split(), dtype=float)
        xi = np.array(dislocation['xi'].split(), dtype=float)
        xi /= np.linalg.norm(xi)
        direction_m_set_by_user = False
        if 'm' in dislocation.keys():
            m = np.array(dislocation['m'].split(), dtype=float)
            m /= np.linalg.norm(m)
            direction_m_set_by_user = True
        center = np.array(dislocation['center'].split(), dtype=float)

        # Rotate vectors associated with the dislocation
        # to the laboratory coordinate system
        b = np.einsum('ij,j', r_crys_lab, b)
        xi = np.einsum('ij,j', r_crys_lab, xi)
        if direction_m_set_by_user:
            m = np.einsum('ij,j', r_crys_lab, m)

        # Define the dislocation coordinate system
        zeros = isclose(xi, 0.0)
        assert(not np.all(zeros))
        if not direction_m_set_by_user:
            m = get_m_direction(xi)
        n = np.cross(xi, m)
        assert(isclose(np.dot(xi, m), 0.0))
        assert(isclose(np.dot(xi, n), 0.0))
        r = np.zeros((3, 3), dtype=float)
        r[0, :] = m
        r[1, :] = n
        r[2, :] = xi
        assert(isclose(np.linalg.det(r), 1.0))

        coordinates -= center
        solution_method = dislocation["solution_method"]
        if solution_method == "stroh":
            # Solve the sextic equations and apply displacements
            (Np, Nv) = solve_sextic_equations(m, n, c)
            coordinates += calculate_displacements_from_eigensystem(
                coordinates, b, m, n, Np, Nv
            )
        elif solution_method == "symbolical_integral_method":
            radii, angles = calculate_cylindrical_coordinates(
                coordinates, xi, m
            )
            coordinates += calculate_displacements_with_symbolical_integrals(
                radii, angles, b, m, n, xi, c
            )
        elif solution_method == "numerical_integral_method":
            radii, angles = calculate_cylindrical_coordinates(
                coordinates, xi, m
            )
            coordinates += calculate_displacements_with_numerical_integrals(
                radii, angles, b, m, n, xi, c
            )
        else:
            raise ValueError(
                "Unknown solution method. Valid choices are: " +
                ("{:s}, " * len(recognized_solution_methods)).format(
                        *recognized_solution_methods
                    ).rstrip(", ")
                )
        coordinates += center

    if (file_format == 'dump'):
        if append_displacements:
            atomdata = np.append(
                atomdata, np.zeros((atomdata.shape[0], 3), dtype=float), 1
            )
            header[-1].append('ux uy uz')
            # Calculate the total displacements
            atomdata[:, -3::] = reference_coordinates - coordinates
        write_dump(outfile, header, atomdata)
    elif (file_format == 'data'):
        write_data(outfile, boundary_style, infile, coordinates)


def get_m_direction(xi):
    """ Find two directions which are perpendicular to the line direction.

    Parameters
    ----------
    xi (numpy.ndarray): dislocation line direction

    Returns
    -------
    m (numpy.ndarray): vector perpendicular to xi
    """
    m = np.zeros(3, dtype=float)
    zeros = isclose(xi, 0.0)
    if np.any(zeros):
        for i in range(0, 3):
            if(zeros[i]):
                m[i] = 1.0
                break
    else:
        m[0] = -xi[1]
        m[1] = xi[0]
        m[2] = 0.0
    return m


if __name__ == '__main__':
    main()
