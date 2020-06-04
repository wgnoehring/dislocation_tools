#!/usr/bin/env python
# -*- coding: utf-8 -*-
from scipy.integrate import quad
from scipy import isclose
import sympy as sp
from sympy.utilities.autowrap import ufuncify
from ..tensor import (
    numerical_ab_contraction,
    symbolical_ab_contraction
)

__author__ = "Wolfram Georg Nöhring"
__copyright__ = """\
© All rights reserved. ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, 
Switzerland, Laboratory for Multiscale Mechanics Modelling, 2015"""
__license__ = "GNU General Public License"
__email__ = "wolfram.nohring@imtek.uni-freiburg.de"


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
