#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
References
----------
[1] Hirth, J. P.; Lothe, J. Theory of Dislocations, 2nd Edition;
    John Wiley and Sons, 1982. pp 467
[2] Bacon, D. J.; Barnett, D. M.; Scattergood, R. O.
    Prog. Mater. Sci. 1979, 23, 51–262.
"""

import sys
import configparser
import numpy as np
from scipy import isclose

# new dependencies
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
    x = normalize(x)
    y = normalize(y)
    z = normalize(z)
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
            raise ValueError('Appending displacements to data files'
                +' is currently not supported.')
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
                list_of_dislocations.append([rank, config._sections[section_name]])
    list_of_dislocations.sort(key =lambda x: x[0])
    list_of_dislocations = [sublist[1] for sublist in list_of_dislocations]

    # Define elastic constants
    c_voigt = np.zeros((6, 6), dtype=float)
    c_voigt[0:3, 0:3] = c12
    for i in range(0, 3):
        c_voigt[i, i] = c11
    for i in range(3, 6):
        c_voigt[i, i] = c44
    c = voigt2tensor(c_voigt)
    # Rotate stiffness tensor to the laboratory coordinate system
    r_crys_lab = np.zeros((3, 3), dtype=float)
    r_crys_lab[0, :] = normalize(x)
    r_crys_lab[1, :] = normalize(y)
    r_crys_lab[2, :] = normalize(z)
    assert(isclose(np.linalg.det(r_crys_lab), 1.0))
    c = np.einsum('ig,jh,ghmn,km,ln',
        r_crys_lab, r_crys_lab, c, r_crys_lab, r_crys_lab)

    for dislocation in list_of_dislocations:
        b  = np.array(dislocation['b'].split(), dtype=float)
        xi = np.array(dislocation['xi'].split(), dtype=float)
        xi = normalize(xi)
        direction_m_set_by_user = False
        if 'm' in dislocation.keys():
            m = np.array(dislocation['m'].split(), dtype=float)
            m = normalize(m)
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
        r = np.zeros((3,3), dtype=float)
        r[0,:] = m
        r[1,:] = n
        r[2,:] = xi
        assert(isclose(np.linalg.det(r), 1.0))

        # Solve the integral problem
        S = np.zeros((3, 3), dtype=float)
        B = np.zeros((3, 3), dtype=float)
        for i in range(3):
            for j in range(3):
                S_val_half, S_err = quad(
                    S_integrand, 0.0, np.pi,
                    args=(xi, m, n, c, i, j)
                )
                S[i, j] = S_val_half * 2.0
                B_val_half, B_err = quad(
                    B_integrand, 0.0, np.pi,
                    args=(xi, m, n, c, i, j)
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
            Np, Nv = solve_sextic_equations(m, n, c)
            signs = np.sign(np.imag(Np))
            signs[np.where(signs==0.0)] = 1.0
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

        # Calculate the displacements
        coordinates -= center
        # Calculate radii and angles
        radii = np.linalg.norm(coordinates, axis=1)
        tmp_x = coordinates[:, 0] / radii
        tmp_y = coordinates[:, 1] / radii
        angles = np.arctan2(tmp_y, tmp_x)

        # Calculate the displacements
        displacements = np.zeros_like(coordinates)
        for atom_index in range(displacements.shape[0]):
            # Calculate the integrals
            nninv_integral = np.zeros((3, 3), dtype=float)
            Slike_integral = np.zeros((3, 3), dtype=float)
            for i in range(3):
                for j in range(3):
                    # Note: if the upper limit is 2*pi, then the Burgers
                    # vector must result!
                    value, error = quad(
                        S_integrand, 0.0, angles[atom_index],
                        args=(xi, m, n, c, i, j)
                    )
                    Slike_integral[i, j] = value
                    value, error = quad(
                        nninv_integrand, 0.0, angles[atom_index],
                        args=(xi, n, c, i, j)
                    )
                    nninv_integral[i, j] = value
            matrix_1 = -1.0 * S * np.log(radii[atom_index])
            matrix_2 = 4.0 * np.pi * np.einsum('ks,ik', B, nninv_integral)
            matrix_3 = np.einsum('rs,ir', S, Slike_integral)
            matrix_4 = (matrix_1 + matrix_2 + matrix_3)
            displacements[atom_index, :] = (
                np.einsum('s,is', b, matrix_4) / (2.0 * np.pi)
            )
        coordinates += displacements
        coordinates += center

    # Write output file
    if file_format == 'dump':
        if append_displacements:
            atomdata = np.append(
                atomdata, np.zeros((atomdata.shape[0], 3), dtype=float), 1
            )
            header[-1].append('ux uy uz')
            # Calculate the total displacements
            atomdata[:, -3::] = reference_coordinates - coordinates
        write_dump(outfile, header, atomdata)
    elif file_format == 'data':
        write_data(outfile, boundary_style, infile, coordinates)

def nninv_integrand(angle, xi, n, c, i, j):
    rotation_matrix = construct_rotation_matrix_numeric(xi, angle)
    n_rot = np.einsum('ij, j', rotation_matrix, n)
    nn = ab_contraction(n_rot, n_rot, c)
    return np.linalg.inv(nn)[i, j]

def S_integrand(angle, xi, m, n, c, i, j):
    rotation_matrix = construct_rotation_matrix_numeric(xi, angle)
    m_rot = np.einsum('ij, j', rotation_matrix, m)
    n_rot = np.einsum('ij, j', rotation_matrix, n)
    nn = ab_contraction(n_rot, n_rot, c)
    nninv = np.linalg.inv(nn)
    nm = ab_contraction(n_rot, m_rot, c)
    #return np.einsum('kj, js', nninv, nm)[i, j]
    return np.dot(nninv, nm)[i, j]

def B_integrand(angle, xi, m , n, c, i, j):
    rotation_matrix = construct_rotation_matrix_numeric(xi, angle)
    m_rot = np.einsum('ij, j', rotation_matrix, m)
    n_rot = np.einsum('ij, j', rotation_matrix, n)
    nn = ab_contraction(n_rot, n_rot, c)
    nninv = np.linalg.inv(nn)
    nm = ab_contraction(n_rot, m_rot, c)
    mn = ab_contraction(m_rot, n_rot, c)
    mm = ab_contraction(m_rot, m_rot, c)
    #integrand = np.einsum('rk,ks', nninv, nm)
    #integrand = mm - np.einsum('jr,rs', mn, integrand)
    #return integrand[i, j]
    integrand = np.dot(nninv, nm)
    integrand = mm - np.dot(mn, integrand)
    return integrand[i, j]

def construct_rotation_matrix_numeric(axis, angle):
    """construct a rotation about an axis by an angle
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
        m[1] =  xi[0]
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
    mm = ab_contraction(m, m, c)
    nn = ab_contraction(n, n, c)
    mn = ab_contraction(m, n, c)
    nm = ab_contraction(n, m, c)
    # Define the matrix N, see equation 13-168 and the footnote
    # on page 468. Note that there is an error in the footnote:
    # the sign of the lower right element is wrong.
    nninv = np.linalg.inv(nn)
    mn_nninv = np.dot(mn, nninv)
    N = np.zeros((6,6), dtype=float)
    N[0:3,0:3] = -np.dot(nninv, nm)
    N[0:3,3:6] = -nninv
    N[3:6,0:3] = -(np.dot(mn_nninv, nm) - mm)
    N[3:6,3:6] = -mn_nninv
    # Matrices U, V, and I: see equation 13-169
    I = np.eye(3, dtype=float)
    U = np.zeros((6,6), dtype=float)
    U[0:3,0:3] = I
    U[3:6,3:6] = I
    V = np.zeros((6,6), dtype=float)
    V[0:3,3:6] = I
    V[3:6,0:3] = I
    # Assert that N has the required symmetries, see equation 13-172
    assert(isclose(N.T-np.dot(V, np.dot(N, V)), 0.0).all())
    assert(isclose(np.dot(N.T, V)-np.dot(V, N), 0.0).all())
    # Solve the |N-pU| for p (equation 13-170)
    Np, Nv = np.linalg.eig(N)
    # The eigenvector Nv contains the vectors A and L.
    for i in range(0,6):
        # Assert that L can be computed from A as specified by equ. 13-167:
        assert(isclose(np.dot(-(nm+Np[i]*nn), Nv[0:3, i]) - Nv[3:6, i], 0.0).all())
        # Normalize A and L, such that 2*A*L=1 (equation 13-178)
        norm = 2.0 * np.dot(Nv[0:3, i], Nv[3:6, i])
        Nv[0:3, i] /= np.sqrt(norm)
        Nv[3:6, i] /= np.sqrt(norm)
        assert(isclose(2.0 * np.dot(Nv[0:3, i], Nv[3:6, i]), 1.0))
    for i in range(0,6):
        for j in range(0,6):
            # Assert that equation 13-177 is satisfied:
            assert(isclose(np.real(
                 np.dot(Nv[0:3, i], Nv[3:6, j])
                +np.dot(Nv[0:3, j], Nv[3:6, i])
                ), kron6(i, j, float)))
             # Assert the orthogonality relation, equ 13-176:
            assert(isclose(np.dot(Nv[:, j], np.dot(V, Nv[:, i])), kron6(i, j, float)))
    return (Np, Nv)

def calculate_displacements(coordinates, b, m, n, Np, Nv):
    """ Compute the displacements from the solution of the sextic equations.

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
    signs[np.where(signs==0.0)] = 1.0
    constant_factor = (signs * Nv[0:3,:] * np.einsum('i,ij', b, Nv[3:6,:]))
    # Apply the displacements
    eta = (np.expand_dims(np.einsum('i,ji', m, coordinates), axis=1)
        + np.outer(np.einsum('i,ji', n, coordinates), Np))
    u = ((1.0/(2.0 * np.pi * 1.0j))
        * np.einsum('ij,kj', np.log(eta), constant_factor))
    assert(isclose(np.imag(u),0.0).all())
    return np.real(u)

def tensor2voigt(t):
    """ Convert a fourth order tensor into a Voigt matrix.

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
                    dij = kron(i-1, j-1, int)
                    dkl = kron(k-1, l-1, int)
                    p = i*dij + (1-dij)*(9-i-j) - 1
                    q = k*dkl + (1-dkl)*(9-k-l) - 1
                    v[p, q] = t[i-1, j-1, k-1, l-1]
    return v

def voigt2tensor(v):
    """ Convert a Voigt matrix into a fourth order tensor.

    Parameters
    ----------
    v (np.ndarray): elastic stiffness in 6x6 (Voigt-) matrix representation

    Returns
    -------
    t (np.ndarray): elastic stiffness tensor in 3x3x3x3 array representation
    """
    t = np.zeros((3,3,3,3), dtype=float)
    for i in range(1,4):
        for j in range(1,4):
            for k in range(1,4):
                for l in range(1,4):
                    dij = kron(i-1, j-1, int)
                    dkl = kron(k-1, l-1, int)
                    p = i*dij + (1-dij)*(9-i-j) - 1
                    q = k*dkl + (1-dkl)*(9-k-l) - 1
                    t[i-1, j-1, k-1, l-1] = v[p, q]
    return t

def kron(i, j, dtype):
    """ Compute the Kronecker delta of numbers i,j.
    """
    return np.eye(3, dtype=dtype)[i, j]

def kron6(i, j, dtype):
    """ Compute the Kronecker delta of numbers i,j.
    """
    return np.eye(6, dtype=dtype)[i, j]

def normalize(v):
    """ Normalize a vector. """
    return v/np.linalg.norm(v)

def ab_contraction(a, b, t):
    """ Return a contraction of the fourth order tensor t and the
        vectors a and b, see equation 13-162 in Hirth & Lothe's book.
    In component notation: (ab)_ij = a_i*c_ijkl*b_l.
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
    img2bits = np.array(20, dtype=img_chunk.dtype)
    image_distances = [None] * 3
    if periodic_boundary[0]:
        image_distances[0]  = np.bitwise_and(img_chunk, imgmask)
        image_distances[0] -= imgmax
        image_distances[0]  = image_distances[0].astype(float)
        image_distances[0] *= box_size[0]
    if periodic_boundary[1]:
        image_distances[1]  = np.right_shift(img_chunk, imgbits)
        image_distances[1] &= imgmask
        image_distances[1] -= imgmax
        image_distances[1]  = image_distances[1].astype(float)
        image_distances[1] *= box_size[1]
    if periodic_boundary[2]:
        image_distances[2]  = np.right_shift(img_chunk, img2bits)
        image_distances[2] -= imgmax
        image_distances[2]  = image_distances[2].astype(float)
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
    my_lammps.command('atom_modify map array')
    my_lammps.command('atom_modify sort 0 0.0')
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
    coordinates = coordinates.reshape((coordinates.shape[0]/3, 3))
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
    my_lammps.scatter_atoms(
        "x", 1, 3, np.ctypeslib.as_ctypes(coordinates)
    )
    my_lammps.command('change_box all set')
    my_lammps.command('change_box all remap')
    my_lammps.command('write_data ' + outfile)
    my_lammps.close()
    return None

def read_dump(infile):
    """Read a Lammps dump file.

    Parameters
    ----------
    infile (str): dump file

    Returns
    -------
    header (list): header of the dump file
    atomdata (ndarray):  one row of data  for each atom. Number  of columns and
        column  contents depend  must be  inferred from  the last  line in  the
        header
    """
    atomdata = to_list(infile)
    (header, atomdata) = strip_header(atomdata)
    atomdata = np.asarray(atomdata, dtype=float)
    return (header, atomdata)

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

def to_list(infile):
    """ Reads a file and returns the contents as a list of lists."""
    with open(infile, 'r') as file:
        list_of_lines = file.readlines()
    list_of_lines = [
        line.rstrip().split() for line in list_of_lines
        if not line.startswith('#')
    ]
    return list_of_lines

def strip_header(config):
    """ Strip the header from the contents of a .atom-file."""
    header = config[0:9]
    if (header[-1][0] != 'ITEM:' or header[-1][1] != 'ATOMS'):
        raise ValueError('cannot read ITEM: ATOMS header')
    config = config[9::]
    return (header, config)

if __name__ == '__main__':
    main()
