#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

__author__ = "Wolfram Georg Nöhring"
__copyright__ = """\
© All rights reserved. ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, 
Switzerland, Laboratory for Multiscale Mechanics Modelling, 2015"""
__license__ = "GNU General Public License"
__email__ = "wolfram.nohring@imtek.uni-freiburg.de"

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
