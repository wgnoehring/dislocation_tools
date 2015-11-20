#!/usr/bin/python3.4
""" Insert dislocations into an anisotropic medium.

Usage
-----
python3.4 insert_dislocation_adv.py <configuration-file>

Synopsis
--------
This program reads atom coordinates from a LAMMPS dump file and inserts one or
more dislocations by applying their anisotropic-elastic displacement field.
The displacements are calculated with the advanced straight dislocation
formalism, see:
pp. 467 in J.P. Hirth and J. Lothe, Theory of Dislocations, 2nd ed.
This book section will be referred to frequently. The variables involved in the
elastic problem are given similar names as in the book. (This is un-pythonic,
but makes it easier to compare code and text.)

Limitations
-----------
Currently, only LAMMPS dump files are supported. The files must contain the
columns 'id', 'type', 'x', 'y', and 'z', in this order. More columns are not
allowed. If you have scaled coordinates ('xs', 'ys', 'zs'), first re-scale!

Files
-----
The program reads two files:
1) A LAMMPS dump file, which contains the atomic coordinates
2) A configuration file, which contains information on the elastic
    constants, the dislocations, etc.
The file formats are explained below:

1) Dump file
The atomic coordinates must be stored as a LAMMPS dump file. It must not
contain more than one snapshot.

A typical dump file starts with nine header lines. Here is an example:
    ITEM: TIMESTEP
    0
    ITEM: NUMBER OF ATOMS
    443760
    ITEM: BOX BOUNDS xy xz yz ss ss pp
    -0.0189274 252.384 63.0912
    -0.0178449 178.467 0
    0 153.372 0
    ITEM: ATOMS id type x y z
The header is followed by the list of atomic properties, ordered according to
what has been specified on the ITEM:ATOMS line.

2) Configuration file
The configuration file is a plain ASCII text file, which is divided into
several sections. The sections are formatted as follows:
[section name]
parameter1 = value1
parameter2 = value2
(...)
The components of vector values must be written on the same line and must be
separated by whitespace. There are three mandatory sections: [simulation cell],
[elastic constants], and [files].
Section [simulation cell] contains the vectors "x", "y", and "z". They indicate
the orientation of the crystal relative to the x, y, and z directions of the
simulation cell. The vectors do not need to be normalized.
Section [elastic constants] contains the scalars "c11", "c12", and "c44", the
cubic elastic stiffnesses. These should be the components relative to the
[100]-[010]-[001] CRYSTAL coordinate system; NOT relative to the simulation
cell. The program performs all the required tensor rotations!
Section [files] contains the strings "input", "output", and
"append_displacements". The first two are the pathds to the input and output
files. "append_displacements" must be "True" or "False". If it is true, then
the script will append the displacement field to the output file (as the last
three columns). The string 'ux uy uz' will be appended to the ITEM: ATOMS
header line.

The mandatory sections are followed by an arbitrary number of sections with the
name [dislocationX], where X is an integer. Each [dislocationX] section defines
a dislocation. If there are several such sections, then the integers X decide
in which sequence the displacement fields will be applied (ascending order,
i.e. dislocation1 would be inserted before dislocation2).

Each [dislocation] section contains the mandatory parameters "b", "xi",
and "center". "b" and "xi" are the Burgers vector and the line direction,
respectively. Both must be given in the crystal coordinate system. "b" has
distance units. Its magnitude is the Burgers vector magnitude. "xi" does not
need to be normalized.
"center" is the center of the dislocation simulation cell coordinates.
Additionally, a vector "m" can be specified. "m" is the first direction of the
dislocation coordinate system, expressed in crystal coordinates. This vector is
parallel to the normal of the plane along which the cut would be made to insert
the dislocation.

Here is an example configuration file:
    [simulation cell]
    x =  1 -2  1
    y = -1 -1 -1
    z =  1  0 -1
    [elastic constants]
    c11 = 170.221549146026319
    c12 = 122.600531649638015
    c44 = 75.848200859164038
    [files]
    input  = Cu_parallelepiped.atom
    output = Cu_parallelepiped_with_screw_dislocation.atom
    append_displacements = True
    [dislocation1]
    b   =  -1.807500856188893   0.000000000000000   1.807500856188893
    xi  =  -1 0 1
    m   =   1 2 1
    center = 125.444552873685 88.702693999897 0.0

Output
------
The output is written to a LAMMPS dump file. If append_displacements is true,
the displacements will be appended.

Author
------
Wolfram Georg Noehring, wolfram.nohring@epfl.ch
Last modification: Wed Apr 15 22:33:26 CEST 2015
"""

import numpy as np
import pathlib
import sys
import configparser
from scipy import isclose

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
    c11 = float(config.get('elastic constants', 'c11'))
    c12 = float(config.get('elastic constants', 'c12'))
    c44 = float(config.get('elastic constants', 'c44'))
    infile = pathlib.Path(config.get('files', 'input'))
    outfile = pathlib.Path(config.get('files', 'output'))
    append_displacements = bool(config.get('files', 'append_displacements'))

    # Read input .atom-file
    if not infile.exists():
        raise ValueError('could not find input coordinates')
    else:
        atomdata = to_list(infile)
        (header, atomdata) = strip_header(atomdata)
        atomdata = np.asarray(atomdata, dtype=float)
        num_atoms = atomdata.shape[0]

    # Parse dislocations and sort them according to the specified rank
    list_of_dislocations = []
    for section_name in config.sections():
        if section_name.startswith('dislocation'):
            rank = int(section_name[11::])
            list_of_dislocations.append([rank, config[section_name]])
    list_of_dislocations.sort(key =lambda x: x[0])
    list_of_dislocations = [sublist[1] for sublist in list_of_dislocations]

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
    r_crys_lab[0, :] = normalize(x)
    r_crys_lab[1, :] = normalize(y)
    r_crys_lab[2, :] = normalize(z)
    assert(isclose(np.linalg.det(r_crys_lab), 1.0))
    c = np.einsum('ig,jh,ghmn,km,ln',
        r_crys_lab, r_crys_lab, c, r_crys_lab, r_crys_lab)

    coordinates = atomdata[:, 2:5]
    if append_displacements:
        reference_coordinates = np.copy(coordinates)
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

        # Solve the sextic equations and apply displacements
        (Np, Nv) = solve_sextic_equations(m, n, c)

        coordinates -= center
        coordinates += calculate_displacements(coordinates, b, m, n, Np, Nv)
        coordinates += center

    # Calculate the total displacements
    if append_displacements:
        atomdata = np.append(atomdata, np.zeros((num_atoms, 3), dtype=float), 1)
        header[-1].append('ux uy uz')
        atomdata[:, 5:8] = reference_coordinates - coordinates

    # Write output file
    with outfile.open('wb') as file:
        for line in header:
            file.write(bytes(' '.join(line) + '\n', 'UTF-8'))

    fmt=["%d", "%d", "%.14e", "%.14e", "%.14e"]
    if append_displacements:
        fmt.extend(["%.14e", "%.14e", "%.14e"])
    with outfile.open('ab') as file:
        np.savetxt(file, atomdata, fmt=fmt)

def get_m_direction(xi):
    """ Find two directions which are perpendicular to the line direction.

    Arguments:
    ----------
    xi (numpy.ndarray): dislocation line direction

    Returns:
    --------
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

    Arguments:
    ----------
    m (numpy.ndarray): x-direction of the dislocation coordinate system
    n (numpy.ndarray): y-direction of the dislocation coordinate system
    c (numpy.ndarray): 3x3x3x3 array representation of elastic stiffness,
        given in laboratory coordinates

    The equations below refer to:
    cf. pp. 467 in J.P. Hirth and J. Lothe, Theory of Dislocations, 2nd ed.

    Returns:
    --------
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

    Arguments:
    ----------
    b (numpy.ndarray): Burgers vector (in laboratory coordinate system)
    m (numpy.ndarray): x-direction of the dislocation coordinate system
    n (numpy.ndarray): y-direction of the dislocation coordinate system
    c (numpy.ndarray): 3x3x3x3 array representation of elastic stiffness,
        given in laboratory coordinates
    Np (numpy.ndarray): eigenvalues
    Nv (numpy.ndarray): eigenvectors

    Returns:
    --------
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

    Argument:
    ---------
    t (np.ndarray): elastic stiffness tensor in 3x3x3x3 array representation

    Returns:
    --------
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

    Argument:
    ---------
    v (np.ndarray): elastic stiffness in 6x6 (Voigt-) matrix representation

    Returns:
    --------
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

def to_list(infile):
    """ Reads a file and returns the contents as a list of lists."""
    infile = pathlib.Path(infile)
    with infile.open() as file:
        list_of_lines = file.readlines()
    list_of_lines = [line.rstrip().split() for line in list_of_lines
        if not line.startswith('#')]
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
