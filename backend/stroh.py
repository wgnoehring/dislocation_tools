#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from scipy import isclose
from ..tensor import numerical_ab_contraction

__author__ = "Wolfram Georg Nöhring"
__copyright__ = """\
© All rights reserved. ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, 
Switzerland, Laboratory for Multiscale Mechanics Modelling, 2015"""
__license__ = "GNU General Public License"
__email__ = "wolfram.nohring@imtek.uni-freiburg.de"


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
