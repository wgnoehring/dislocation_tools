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
scipy_available = True
try:
    import sympy as sp
    from sympy.utilities.autowrap import ufuncify
except ImportError:
    scipy_available = False
from scipy import isclose
from scipy.integrate import quad

__author__ = "Wolfram Georg Nöhring"
__copyright__ = "Copyright 2015, EPFL"
__license__ = "GNU General Public License"
__email__ = "wolfram.nohring@epfl.ch"

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


def calculate_cylindrical_coordinates(
        coordinates, line_direction, perpendicular_direction):
    """Calculate cylindrical coordinates

    Given a  list of coordinates  in 3D space,  calculate the radii  and angles
    that  determine the  perpendicular  positions  relative to  a  line in  the
    direction  line_direction through  the  origin. The  datum  from which  the
    angles are measured is given by perpendicular_direction.

    Parameters
    ----------
    coordinates (numpy.ndarray): Nx3 array of coordinates
    line_direction (numpy.ndarray): line direction relative to which the
        perpendicular distance is calculated
    perpendicular_direction (numpy.ndarray): direction perpendicular to
        line_direction; serves as reference datum for calculating angles

    Returns
    -------
    radii (numpy.ndarray): Nx1 array of perpendicular distances of the
        coordinates from the line direction
    angles (numpy.ndarray): Nx1 array of angles in the plane perpendicular
        to line_direction, and relative to perpendicular_direction
    """
    nrows = coordinates.shape[0]
    p1 = perpendicular_direction / np.linalg.norm(perpendicular_direction)
    l = line_direction / np.linalg.norm(line_direction)
    p2 = np.cross(l, p1)
    projection_on_line = np.dot(coordinates, l)
    perpendicular_coordinates = (
        coordinates - projection_on_line[:, np.newaxis] * l
    )
    radii = np.linalg.norm(perpendicular_coordinates, axis=1)
    adjacent = np.dot(perpendicular_coordinates, p1)
    opposite = np.dot(perpendicular_coordinates, p2)
    adjacent /= radii
    opposite /= radii
    angles = np.arctan2(opposite, adjacent)
    return (radii, angles)


def check_S_and_B(S, B, m, n, c):
    """Check the matrices S and B of the integral formalism.

    S and B  can be computed both from Barnett  and Lothe's integral formalism,
    as  well  as from  Stroh's  formalism,  see equations  (13-211),  (13-213),
    (13-215), and  (13-217) in Hirth and  Lothe's book. This function  checks S
    and B  as obtained  from the  integral formalism by  comparing them  to the
    corresponding prediction from Stroh's formalism.

    Parameters
    ----------
    S (nump.ndarray): matrix S, see equ. 13-211 in Hirth and Lothe's book
    B (nump.ndarray): matrix B, see equ. 13-213 in Hirth and Lothe's book
    m (numpy.ndarray): x-direction of the dislocation coordinate system
    n (numpy.ndarray): y-direction of the dislocation coordinate system
    c (numpy.ndarray): 3x3x3x3 array representation of elastic stiffness,
        given in laboratory coordinates
    """
    Np, Nv = solve_sextic_equations(m, n, c)
    signs = np.sign(np.imag(Np))
    signs[np.where(signs == 0.0)] = 1.0
    A = Nv[0:3, :]
    L = Nv[3:6, :]
    S_check = np.zeros((3, 3), dtype=complex)
    for k in range(3):
        for s in range(3):
            for alpha in range(6):
                S_check[k, s] += (
                    1j * A[k, alpha] * L[s, alpha] * signs[alpha]
                )
    assert(isclose(S-S_check, 0.0).all())
    B_check = np.zeros((3, 3), dtype=complex)
    for k in range(3):
        for s in range(3):
            for alpha in range(6):
                B_check[k, s] += (
                    L[k, alpha] * L[s, alpha] * signs[alpha]
                )
    B_check /= (-4.0 * np.pi * 1j)
    assert(isclose(B-B_check, 0.0).all())


def generate_numerical_rotation_matrix(axis, angle):
    """Generate numeric rotation matrix for rotation about an axis by an angle

    Parameters
    ----------
    axis (numpy.ndarray): rotation axis
    angle (float): rotation angle (in the sense of the right hand rule)

    Returns
    -------
    rotation_matrix (numpy.ndarray): rotation matrix
    """
    tensor_product = np.eye(3)
    for i in range(3):
        for j in range(3):
            tensor_product[i, j] = axis[i] * axis[j]
    cross_product_matrix = np.zeros((3, 3), dtype=float)
    cross_product_matrix[0, 1] = -1.0 * axis[2]
    cross_product_matrix[1, 0] = axis[2]
    cross_product_matrix[0, 2] = axis[1]
    cross_product_matrix[2, 0] = -1.0 * axis[1]
    cross_product_matrix[1, 2] = -1.0 * axis[0]
    cross_product_matrix[2, 1] = axis[0]
    rotation_matrix = (
        np.cos(angle) * np.eye(3) +
        np.sin(angle) * cross_product_matrix +
        (1.0 - np.cos(angle)) * tensor_product
    )
    return rotation_matrix


def generate_symbolical_rotation_matrix(axis):
    """Generate symbolic rotation matrix for rotation about an axis by an angle

    Parameters
    ----------
    axis (sympy.Matrix): rotation axis

    Returns
    -------
    rotation_matrix (sympy.Matrix): rotation matrix; the free symbol is "omega"
    """
    angle = sp.Symbol("omega")
    tensor_product = sp.eye(3)
    for i in range(3):
        for j in range(3):
            tensor_product[i, j] = axis[i] * axis[j]
    cross_product_matrix = sp.zeros(3)
    cross_product_matrix[0, 1] = -1.0 * axis[2]
    cross_product_matrix[1, 0] = axis[2]
    cross_product_matrix[0, 2] = axis[1]
    cross_product_matrix[2, 0] = -1.0 * axis[1]
    cross_product_matrix[1, 2] = -1.0 * axis[0]
    cross_product_matrix[2, 1] = axis[0]
    rotation_matrix = (
        sp.cos(angle) * sp.eye(3) +
        sp.sin(angle) * cross_product_matrix +
        (1.0 - sp.cos(angle)) * tensor_product
    )
    return rotation_matrix


def ufuncify_angular_function(symbolic_function_matrix):
    """Convert a matrix of symbolical functions into numerical functions

    Parameters
    ----------
    symbolic_function_matrix (numpy.ndarray): matrix of symbolical functions

    Returns
    -------
    numerical_function_matrix (numpy.ndarray): matrix of numerical functions
        (numpy ufunc-like functions)
    """
    if len(symbolic_function_matrix.free_symbols) == 1:
        my_symbol = symbolic_function_matrix.free_symbols.pop()
    else:
        raise ValueError("Number of free symbols must be one")
    num_rows, num_cols = symbolic_function_matrix.shape
    numerical_function_matrix = np.empty((num_rows, num_cols), dtype=object)
    for i in range(num_rows):
        for j in range(num_cols):
            numerical_function_matrix[i, j] = ufuncify(
                [my_symbol], symbolic_function_matrix[i, j]
            )
    return numerical_function_matrix


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


def solve_sextic_equations(m, n, c):
    """ Compute the eigenvalues and eigenvectors of the problem.

    Parameters
    ----------
    m (numpy.ndarray): x-direction of the dislocation coordinate system
    n (numpy.ndarray): y-direction of the dislocation coordinate system
    c (numpy.ndarray): 3x3x3x3 array representation of elastic stiffness,
        given in laboratory coordinates

    The equations below refer to:
    cf. pp. 467 in J.P. Hirth and J. Lothe, Theory of Dislocations, 2nd ed.

    Returns
    -------
    Np (numpy.ndarray): eigenvalues
    Nv (numpy.ndarray): eigenvectors
    """
    mm = numerical_ab_contraction(m, m, c)
    nn = numerical_ab_contraction(n, n, c)
    mn = numerical_ab_contraction(m, n, c)
    nm = numerical_ab_contraction(n, m, c)
    # Define the matrix N, see equation 13-168 and the footnote
    # on page 468. Note that there is an error in the footnote:
    # the sign of the lower right element is wrong.
    nninv = np.linalg.inv(nn)
    mn_nninv = np.dot(mn, nninv)
    N = np.zeros((6, 6), dtype=float)
    N[0:3, 0:3] = -np.dot(nninv, nm)
    N[0:3, 3:6] = -nninv
    N[3:6, 0:3] = -(np.dot(mn_nninv, nm) - mm)
    N[3:6, 3:6] = -mn_nninv
    # Matrices U, V, and I: see equation 13-169
    I = np.eye(3, dtype=float)
    U = np.zeros((6, 6), dtype=float)
    U[0:3, 0:3] = I
    U[3:6, 3:6] = I
    V = np.zeros((6, 6), dtype=float)
    V[0:3, 3:6] = I
    V[3:6, 0:3] = I
    # Assert that N has the required symmetries, see equation 13-172
    assert(isclose(N.T - np.dot(V, np.dot(N, V)), 0.0).all())
    assert(isclose(np.dot(N.T, V) - np.dot(V, N), 0.0).all())
    # Solve the |N-pU| for p (equation 13-170)
    Np, Nv = np.linalg.eig(N)
    # The eigenvector Nv contains the vectors A and L.
    for i in range(0, 6):
        # Assert that L can be computed from A as specified by equ. 13-167:
        assert(
            isclose(
                np.dot(-(nm + Np[i] * nn), Nv[0:3, i]) - Nv[3:6, i], 0.0
            ).all()
        )
        # Normalize A and L, such that 2*A*L=1 (equation 13-178)
        norm = 2.0 * np.dot(Nv[0:3, i], Nv[3:6, i])
        Nv[0:3, i] /= np.sqrt(norm)
        Nv[3:6, i] /= np.sqrt(norm)
        assert(isclose(2.0 * np.dot(Nv[0:3, i], Nv[3:6, i]), 1.0))
    for i in range(0, 6):
        for j in range(0, 6):
            # Assert that equation 13-177 is satisfied:
            assert(
                isclose(
                    np.real(
                        np.dot(Nv[0:3, i], Nv[3:6, j]) +
                        np.dot(Nv[0:3, j], Nv[3:6, i])
                    ),
                    float(i == j)
                )
            )
            # Assert the orthogonality relation, equ 13-176:
            assert(
                isclose(np.dot(Nv[:, j], np.dot(V, Nv[:, i])), float(i == j))
            )
    return (Np, Nv)


def calculate_displacements_from_eigensystem(coordinates, b, m, n, Np, Nv):
    """Compute the displacements from the solution of the sextic equations.

    Parameters
    ----------
    b (numpy.ndarray): Burgers vector (in laboratory coordinate system)
    m (numpy.ndarray): x-direction of the dislocation coordinate system
    n (numpy.ndarray): y-direction of the dislocation coordinate system
    c (numpy.ndarray): 3x3x3x3 array representation of elastic stiffness,
        given in laboratory coordinates
    Np (numpy.ndarray): eigenvalues
    Nv (numpy.ndarray): eigenvectors

    Returns
    -------
    u (numpy.ndarray): displacements in laboratory coordinate system
    """
    # Calculate the constant factors in the summation for the displacements
    signs = np.sign(np.imag(Np))
    signs[np.where(signs == 0.0)] = 1.0
    constant_factor = (signs * Nv[0:3, :] * np.einsum('i,ij', b, Nv[3:6, :]))
    # Apply the displacements
    eta = (
        np.expand_dims(np.einsum('i,ji', m, coordinates), axis=1) +
        np.outer(np.einsum('i,ji', n, coordinates), Np)
    )
    u = (
        (1.0 / (2.0 * np.pi * 1.0j)) *
        np.einsum('ij,kj', np.log(eta), constant_factor)
    )
    assert(isclose(np.imag(u), 0.0).all())
    return np.real(u)


def calculate_displacements_with_numerical_integrals(
        radii, angles, b, m, n, xi, c):
    """Compute the displacements with integral formalism and numerical math

    Calculate  the displacement  using Barnett  and Lothe's  integral formalism
    [1],  see [2]  and [3].  The  variables of  the elastic  problem are  given
    similar  names  as in  [2]  and  [3].  The solution  involves  integration,
    see  equ. 4.1.25  in  [3].  The integrands  are  combinations of  matrices,
    which,  in   turn  are  contractions   of  the  elastic   stiffness  tensor
    with  two  perpendicular  vectors.  The numerical  integration  used  here,
    scipy.integrate.quad, currently  does not support  simultaneous integration
    of  matrix components.  Therefore,  the components  have  to be  integrated
    independently,  and the  same contractions  have  to be  performed 9  times
    per  matrix.  The  integral  given  by   equ.  4.1.25  has  to  be  carried
    out  for   each  atom;  therefore  the   current  numerical  implementation
    is  relatively  slow.  For  a  faster  solution  using  symbolic  math  see
    calculate_displacements_with_symbolical_integrals.

    Parameters
    ----------
    radii (numpy.ndarray): Nx1 array of perpendicular distances of the
        atoms from the line direction
    angles (numpy.ndarray): Nx1 array of angles in the plane perpendicular
        to xi, and relative to m
    b (numpy.ndarray): Burgers vector (in laboratory coordinate system)
    m (numpy.ndarray): x-direction of the dislocation coordinate system
    n (numpy.ndarray): y-direction of the dislocation coordinate system
    xi (numpy.ndarray): dislocation line direction; z-direction of the
        dislocation coordinate system
    c (numpy.ndarray): 3x3x3x3 array representation of elastic stiffness,
        given in laboratory coordinates

    Returns
    -------
    u (numpy.ndarray): displacements in laboratory coordinate system

    References
    ----------
    [1] Barnett, D.M.; Lothe, J. Phys. Norvegica, 7: 13 (1973)
    [2] Hirth, J.P.; Lothe, J. Theory of Dislocations, 2nd Edition;
        John Wiley and Sons, 1982. pp 467
    [3] Bacon, D. J.; Barnett, D. M.; Scattergood, R. O.
        Progress in Materials Science 1978, 23, 51-262.
    """

    def nninv_integrand(angle, i, j):
        """Component ij of the first integrand of equ. 4.1.25 in [3]"""
        rotation_matrix = generate_numerical_rotation_matrix(xi, angle)
        n_rot = np.einsum('ij, j', rotation_matrix, n)
        nn = numerical_ab_contraction(n_rot, n_rot, c)
        return np.linalg.inv(nn)[i, j]

    def S_integrand(angle, i, j):
        """Component ij of the integrand of equ. 3.6.6 in [3]

        This is the same as the second integrand of equ. 4.1.25 in [3].
        """
        rotation_matrix = generate_numerical_rotation_matrix(xi, angle)
        m_rot = np.einsum('ij, j', rotation_matrix, m)
        n_rot = np.einsum('ij, j', rotation_matrix, n)
        nn = numerical_ab_contraction(n_rot, n_rot, c)
        nninv = np.linalg.inv(nn)
        nm = numerical_ab_contraction(n_rot, m_rot, c)
        return np.dot(nninv, nm)[i, j]

    def B_integrand(angle, i, j):
        """Component js of the integrand of equ. 3.6.9 in [3]
        """
        rotation_matrix = generate_numerical_rotation_matrix(xi, angle)
        m_rot = np.einsum('ij, j', rotation_matrix, m)
        n_rot = np.einsum('ij, j', rotation_matrix, n)
        nn = numerical_ab_contraction(n_rot, n_rot, c)
        nninv = np.linalg.inv(nn)
        nm = numerical_ab_contraction(n_rot, m_rot, c)
        mn = numerical_ab_contraction(m_rot, n_rot, c)
        mm = numerical_ab_contraction(m_rot, m_rot, c)
        integrand = np.dot(nninv, nm)
        integrand = mm - np.dot(mn, integrand)
        return integrand[i, j]

    # Compute the matrices S and B, see equ. 3.6.6 and 3.6.9 in [3]
    S = np.zeros((3, 3), dtype=float)
    B = np.zeros((3, 3), dtype=float)
    for i in range(3):
        for j in range(3):
            S_val_half, S_err = quad(
                S_integrand, 0.0, np.pi, args=(i, j)
            )
            S[i, j] = S_val_half * 2.0
            B_val_half, B_err = quad(
                B_integrand, 0.0, np.pi, args=(i, j)
            )
            B[i, j] = B_val_half * 2.0
            print(
                "S[{:d}, {:d}]/2.0, error: {:16.8f} {:16.8f}".format(
                        i, j, S_val_half, S_err
                )
            )
            print(
                "B[{:d}, {:d}]/2.0, error: {:16.8f} {:16.8f}".format(
                        i, j, B_val_half, S_err
                )
            )
    S /= (-2.0 * np.pi)
    B /= (8.0 * np.pi**2.0)

    # For debugging: check S and B by computing them from Stroh's solution
    if False:
        check_S_and_B(S, B, m, n, c)

    # Calculate the displacements from equ. 4.1.25 in [3]
    u = np.zeros((angles.shape[0], 3))
    for atom_index in range(u.shape[0]):
        # Calculate the integrals
        nninv_integral = np.zeros((3, 3), dtype=float)
        Slike_integral = np.zeros((3, 3), dtype=float)
        for i in range(3):
            for j in range(3):
                # Note: if the upper limit is 2*pi, then the Burgers
                # vector must result!
                value, error = quad(
                    S_integrand, 0.0, angles[atom_index], args=(i, j)
                )
                Slike_integral[i, j] = value
                value, error = quad(
                    nninv_integrand, 0.0, angles[atom_index], args=(i, j)
                )
                nninv_integral[i, j] = value
        matrix_1 = -1.0 * S * np.log(radii[atom_index])
        matrix_2 = 4.0 * np.pi * np.einsum('ks,ik', B, nninv_integral)
        matrix_3 = np.einsum('rs,ir', S, Slike_integral)
        matrix_4 = (matrix_1 + matrix_2 + matrix_3)
        u[atom_index, :] = (
            np.einsum('s,is', b, matrix_4) / (2.0 * np.pi)
        )
    return u


def calculate_displacements_with_symbolical_integrals(
        radii, angles, b, m, n, xi, c):
    """Compute the displacements with integral formalism and symbolical math

    Calculate  the displacement  using Barnett  and Lothe's  integral formalism
    [1],  see [2]  and [3].  The  variables of  the elastic  problem are  given
    similar names  as in [2]  and [3].  The solution involves  integration, see
    equ. 4.1.25  in [3].  The integrands are  combinations of  matrices, which,
    in  turn  are  contractions  of  the  elastic  stiffness  tensor  with  two
    perpendicular  vectors. A  naive,  fully numerical  solution  is slow,  see
    calculate_displacements_with_numerical_integrals. Here,  the integrands are
    made  callable functions  of  the  angle omega,  which  is the  integration
    variable. This  allows fast  evaluation. However,  problems can  arise when
    attempting to perform trigonomic  simplifications on the symbolic matrices,
    as well as when symbolically inverting the matrix nn. This might be a sympy
    issue or  an issue  of number overflow.  Currently, no  simplifications are
    performed  and nn  is inverted  using LU-decomposition,  which seems  to be
    stable.

    Parameters
    ----------
    radii (numpy.ndarray): Nx1 array of perpendicular distances of the
        atoms from the line direction
    angles (numpy.ndarray): Nx1 array of angles in the plane perpendicular
        to xi, and relative to m
    b (numpy.ndarray): Burgers vector (in laboratory coordinate system)
    m (numpy.ndarray): x-direction of the dislocation coordinate system
    n (numpy.ndarray): y-direction of the dislocation coordinate system
    xi (numpy.ndarray): dislocation line direction; z-direction of the
        dislocation coordinate system
    c (numpy.ndarray): 3x3x3x3 array representation of elastic stiffness,
        given in laboratory coordinates

    Returns
    -------
    u (numpy.ndarray): displacements in laboratory coordinate system

    References
    ----------
    [1] Barnett, D.M.; Lothe, J. Phys. Norvegica, 7: 13 (1973)
    [2] Hirth, J.P.; Lothe, J. Theory of Dislocations, 2nd Edition;
        John Wiley and Sons, 1982. pp 467
    [3] Bacon, D. J.; Barnett, D. M.; Scattergood, R. O.
        Progress in Materials Science 1978, 23, 51-262.
    """
    print("Constructing angular function matrices")
    rotation_matrix = generate_symbolical_rotation_matrix(sp.Matrix(xi))
    n_rot = rotation_matrix * n
    m_rot = rotation_matrix * m
    # It would perhaps be useful to simplify the expressions,
    # e.g. via trigsimp. Currently, however, this introduces
    # numerical error.
    # nn = sp.trigsimp(ab_contraction_symbolic(n_rot, n_rot, c))
    # mm = sp.trigsimp(ab_contraction_symbolic(m_rot, m_rot, c))
    # nm = sp.trigsimp(ab_contraction_symbolic(n_rot, m_rot, c))
    # mn = sp.trigsimp(ab_contraction_symbolic(m_rot, n_rot, c))
    nn = symbolical_ab_contraction(n_rot, n_rot, c)
    mm = symbolical_ab_contraction(m_rot, m_rot, c)
    nm = symbolical_ab_contraction(n_rot, m_rot, c)
    mn = symbolical_ab_contraction(m_rot, n_rot, c)
    # inv() and inverge_GE() seem to suffer from a
    # loss of precision; don't use!
    nninv = nn.inverse_LU()
    # Convert to numerical functions
    print("Converting symbolic functions to callables")
    nn_numerical = ufuncify_angular_function(nn)
    mm_numerical = ufuncify_angular_function(mm)
    nm_numerical = ufuncify_angular_function(nm)
    mn_numerical = ufuncify_angular_function(mn)
    nninv_numerical = ufuncify_angular_function(nninv)

    # Compute the matrices S and B, see equ. 3.6.6 and 3.6.9 in [3]
    print("calculating S and B")
    # construct the integrands
    S_integrand = np.tensordot(nninv, nm, axes=([1], [0]))
    S_integrand = sp.Matrix(S_integrand)
    B_integrand = np.tensordot(nninv, nm, axes=([1], [0]))
    B_integrand = np.tensordot(mn, B_integrand, axes=([1], [0]))
    B_integrand = mm - sp.Matrix(B_integrand)
    S_integrand = ufuncify_angular_function(S_integrand)
    B_integrand = ufuncify_angular_function(B_integrand)
    # integrate; exploit the fact that the integrands have period w
    S = np.zeros((3, 3), dtype=float)
    B = np.zeros((3, 3), dtype=float)
    for i in range(3):
        for j in range(3):
            S_val_half, S_err = quad(S_integrand[i, j], 0.0, np.pi)
            S[i, j] = S_val_half * 2.0
            B_val_half, B_err = quad(B_integrand[i, j], 0.0, np.pi)
            B[i, j] = B_val_half * 2.0
            print(
                "S[{:d}, {:d}]/2.0, error: {:16.8f} {:16.8f}".format(
                        i, j, S_val_half, S_err
                )
            )
            print(
                "B[{:d}, {:d}]/2.0, error: {:16.8f} {:16.8f}".format(
                        i, j, B_val_half, S_err
                )
            )
    S /= (-2.0 * np.pi)
    B /= (8.0 * np.pi**2.0)
    # For debugging: check S and B by computing them from Stroh's solution
    if False:
        check_S_and_B(S, B, m, n, c)

    # Calculate the displacements from equ. 4.1.25 in [3]
    print("calculating atomic displacements")
    # Calculate radii and angles
    displacements = np.zeros((angles.shape[0], 3))
    for atom_index in range(displacements.shape[0]):
        # Calculate the integrals
        nninv_integral = np.zeros((3, 3), dtype=float)
        Slike_integral = np.zeros((3, 3), dtype=float)
        for i in range(3):
            for j in range(3):
                nninv_integral[i, j], integration_error = quad(
                    nninv_numerical[i, j], 0.0, angles[atom_index]
                )
                Slike_integral[i, j], integration_error = quad(
                    S_integrand[i, j], 0.0, angles[atom_index]
                )
        matrix_1 = -1.0 * S * np.log(radii[atom_index])
        matrix_2 = 4.0 * np.pi * np.einsum('ks,ik', B, nninv_integral)
        matrix_3 = np.einsum('rs,ir', S, Slike_integral)
        matrix_4 = (matrix_1 + matrix_2 + matrix_3)
        displacements[atom_index] = (
            np.einsum('s,is', b, matrix_4) / (2.0 * np.pi)
        )
    return displacements


def tensor2voigt(t):
    """Convert a fourth order tensor into a Voigt matrix.

    Parameters
    ----------
    t (np.ndarray): elastic stiffness tensor in 3x3x3x3 array representation

    Returns
    -------
    v (np.ndarray): elastic stiffness in 6x6 (Voigt-) matrix representation
    """
    v = np.zeros((6, 6), dtype=float)
    for i in range(1, 4):
        for j in range(1, 4):
            for k in range(1, 4):
                for l in range(1, 4):
                    dij = int(i - 1 == j - 1)
                    dkl = int(k - 1 == l - 1)
                    p = i * dij + (1 - dij)*(9 - i - j) - 1
                    q = k * dkl + (1 - dkl)*(9 - k - l) - 1
                    v[p, q] = t[i - 1, j - 1, k - 1, l - 1]
    return v


def voigt2tensor(v):
    """Convert a Voigt matrix into a fourth order tensor.

    Parameters
    ----------
    v (np.ndarray): elastic stiffness in 6x6 (Voigt-) matrix representation

    Returns
    -------
    t (np.ndarray): elastic stiffness tensor in 3x3x3x3 array representation
    """
    t = np.zeros((3, 3, 3, 3), dtype=float)
    for i in range(1, 4):
        for j in range(1, 4):
            for k in range(1, 4):
                for l in range(1, 4):
                    dij = int(i - 1 == j - 1)
                    dkl = int(k - 1 == l - 1)
                    p = i * dij + (1 - dij) * (9 - i - j) - 1
                    q = k * dkl + (1 - dkl) * (9 - k - l) - 1
                    t[i - 1, j - 1, k - 1, l - 1] = v[p, q]
    return t


def symbolical_ab_contraction(a, b, t):
    """Return a contraction of the fourth order tensor t and the
       vectors a and b, see equation 13-162 in Hirth & Lothe's book.
    In component notation: (ab)_ij = a_i*c_ijkl*b_l.
    This function performs the contraction symbolically.
    """
    contraction_1 = np.tensordot(t, b, axes=([3], [0]))
    contraction_2 = np.tensordot(a, contraction_1, axes=([0], [0]))
    contraction_2 = contraction_2[0, :, :, 0]
    contraction_2 = sp.Matrix(contraction_2)
    return contraction_2


def numerical_ab_contraction(a, b, t):
    """Return a contraction of the fourth order tensor t and the
       vectors a and b, see equation 13-162 in Hirth & Lothe's book.
    In component notation: (ab)_ij = a_i*c_ijkl*b_l.
    This function performs the contraction numerically.
    """
    return np.einsum('i,ijkl,l', a, t, b, dtype=float, casting='safe')


def calc_image_distances(img_chunk, periodic_boundary, box_size):
    """Calculate the lengths for wrapping or unwrapping atomic coordinates
    across periodic boundaries.

    Parameters
    ----------
    img_chunk (ndarray): image flags
    periodic_boundary (list): element i = True if boundary periodic in i
    box_size (ndarray): simulation box size

    Returns
    -------
    image_distances (list, len=3): distance that  must be subtracted from x, y,
        or z-coordinates to  wrap them across the periodic  boundaries. None if
        boundary is not periodic.

    Notes
    -----
    Triclinic boxes are not fully supported. In this case, only the
    non-inclinced direction can be periodic.

    imgmax = 512 and img2bits = 20 only if Lammps has been compiled
    with LAMMPS_SMALLBIG
    """
    # Bit mask values for decoding Lammps image flags:
    imgmask = np.array(1023, dtype=img_chunk.dtype)
    imgmax = np.array(512, dtype=img_chunk.dtype)
    imgbits = np.array(10, dtype=img_chunk.dtype)
    img2bits = np.array(20, dtype=img_chunk.dtype)
    image_distances = [None] * 3
    if periodic_boundary[0]:
        image_distances[0] = np.bitwise_and(img_chunk, imgmask)
        image_distances[0] -= imgmax
        image_distances[0] = image_distances[0].astype(float)
        image_distances[0] *= box_size[0]
    if periodic_boundary[1]:
        image_distances[1] = np.right_shift(img_chunk, imgbits)
        image_distances[1] &= imgmask
        image_distances[1] -= imgmax
        image_distances[1] = image_distances[1].astype(float)
        image_distances[1] *= box_size[1]
    if periodic_boundary[2]:
        image_distances[2] = np.right_shift(img_chunk, img2bits)
        image_distances[2] -= imgmax
        image_distances[2] = image_distances[2].astype(float)
        image_distances[2] *= box_size[2]
    return image_distances


def apply_pbc(dof, periodic_boundary, image_distances, mode):
    """Apply periodic boundary conditions.

    Parameters
    ----------
    dof_chunk (ndarray): atomic coordinates
    periodic_boundary (list): element i = True if boundary in
        direction i is periodic
    image_distances (list, len=3): distance that  must be subtracted from x, y,
        or z-coordinates to  wrap them across the periodic  boundaries. None if
        boundary is not periodic.
    mode (string): 'wrap' to wrap coordinates, 'unwrap' to unwrap

    Returns
    -------
    dof_chunk (ndarray): atomic coordinates with pbc applied

    Notes
    -----
    Triclinic boxes are not fully supported. In this case, only the
    non-inclinced direction can be periodic.
    """
    mode = str(mode)
    directions = range(3)
    if mode == 'unwrap':
        for i in directions:
            if periodic_boundary[i]:
                dof[i::3] += image_distances[i]
    elif mode == 'wrap':
        for i in directions:
            if periodic_boundary[i]:
                dof[i::3] -= image_distances[i]
    else:
        raise ValueError('Wrong mode: {:s}'.format(mode))
    return dof


def read_data(infile, boundary_style):
    """Read a Lammps data file.

    Parameters
    ----------
        infile (str): input data file
        boundary_style (str): Lammps boundary style (e.g. 's s p' or 'p p p')

    Returns
    -------
    coordinates  (ndarray):   coordinates  of  the  atoms.   Periodic  boundary
        conditions  are undone,  i.e.  the  atoms are  not  wrapped around  the
        periodic boundary.
    """
    from lammps import lammps
    my_lammps = lammps()
    my_lammps.command('atom_style atomic')
    my_lammps.command('units metal')
    my_lammps.command('boundary ' + boundary_style)
    my_lammps.command('read_data ' + infile)
    coordinates = np.asarray(my_lammps.gather_atoms("x", 1, 3))
    image_flags = np.asarray(my_lammps.gather_atoms("image", 0, 1))
    box_size = np.zeros(3)
    box_size[0] += my_lammps.extract_global("boxxhi", 1)
    box_size[0] -= my_lammps.extract_global("boxxlo", 1)
    box_size[1] += my_lammps.extract_global("boxyhi", 1)
    box_size[1] -= my_lammps.extract_global("boxylo", 1)
    box_size[2] += my_lammps.extract_global("boxzhi", 1)
    box_size[2] -= my_lammps.extract_global("boxzlo", 1)
    my_lammps.close()
    periodic_boundary = [None] * 3
    for i, flag in enumerate(boundary_style.split()):
        if flag == 'p':
            periodic_boundary[i] = True
        else:
            periodic_boundary[i] = False
    image_distances = calc_image_distances(
        image_flags, periodic_boundary, box_size
    )
    coordinates = apply_pbc(
        coordinates, periodic_boundary, image_distances, 'unwrap'
    )
    coordinates = coordinates.reshape((int(coordinates.shape[0]/3), 3))
    return coordinates


def write_data(outfile, boundary_style, infile, coordinates):
    """Write a Lammps data file.

    Parameters
    ----------
    outfile (str): output data file
    boundary_style (str): Lammps boundary style (e.g. 's s p' or 'p p p')
    infile (str): input data file
    coordinates (ndarray): coordinates  of  the  atoms.

    Notes
    -----
    The  input  data  file  must  be  given  and  is  actually  re-read  before
    writing, so  that scatter_atoms can be  used to overwrite a  current set of
    coordinates.
    """
    from lammps import lammps
    my_lammps = lammps()
    my_lammps.command('atom_style atomic')
    my_lammps.command('atom_modify map array')
    my_lammps.command('units metal')
    my_lammps.command('boundary ' + boundary_style)
    my_lammps.command('read_data ' + infile)
    coordinates = np.ravel(coordinates)
    # Zero all image flags --- we previously unwrapped atomic coordinates
    all_zero_bit = 537395712 
    image = np.zeros((coordinates.shape[0], 1), np.int32) + all_zero_bit
    my_lammps.scatter_atoms(
        "image", 0, 1, np.ctypeslib.as_ctypes(image)
    )
    my_lammps.scatter_atoms(
        "x", 1, 3, np.ctypeslib.as_ctypes(coordinates)
    )
    my_lammps.command('change_box all set')
    my_lammps.command('change_box all remap')
    my_lammps.command('write_data ' + outfile)
    my_lammps.close()
    return None


def read_dump(path_to_dump_file):
    """Read a Lammps dump file.

    Parameters
    ----------
    path_to_dump_file (str): path to the dump file

    Returns
    -------
    header (list): header of the dump file
    atoms_section  (ndarray): ATOMS  section  of  the dump  file  as 2D  array.
        Contains one  row of data for  each atom. Number of  columns and column
        contents depend must be inferred from the last line in the header
    """
    with open(path_to_dump_file, 'r') as file:
        list_of_lines = file.readlines()
    list_of_lines = [
        line.rstrip().split() for line in list_of_lines
        if not line.startswith('#')
    ]
    header = list_of_lines[0:9]
    if (header[-1][0] != 'ITEM:' or header[-1][1] != 'ATOMS'):
        raise ValueError('cannot read ITEM: ATOMS header')
    atoms_section = list_of_lines[9::]
    atoms_section = np.asarray(atoms_section, dtype=float)
    return (header, atoms_section)


def write_dump(outfile, header, atomdata):
    """Write a Lammps dump file.

    Parameters
    ----------
    outfile (str): file to write to
    header (list): header of the dump file
    atomdata (ndarray): one row of data for each atom.
    """
    with open(outfile, 'wb') as file:
        if (sys.version_info.major > 3):
            for line in header:
                file.write(bytes(' '.join(line) + '\n', 'UTF-8'))
        else:
            for line in header:
                file.write(bytes(' '.join(line) + '\n'))
    # Define the number format in the "ATOMS" section
    fmt = []
    integer_columns = ['id', 'type', 'ix', 'iy', 'iz']
    for column in header[-1][2:]:
        if column in integer_columns:
            fmt.append('%d')
        else:
            fmt.append('%.14e')
    with open(outfile, 'ab') as file:
        np.savetxt(file, atomdata, fmt=fmt)


if __name__ == '__main__':
    main()
