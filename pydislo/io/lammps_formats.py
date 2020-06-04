#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Handles file formats understood by Lammps"""
import sys
import numpy as np

__author__ = "Wolfram Georg Nöhring"
__copyright__ = """\
© All rights reserved. ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, 
Switzerland, Laboratory for Multiscale Mechanics Modelling, 2015"""
__license__ = "GNU General Public License"
__email__ = "wolfram.nohring@imtek.uni-freiburg.de"

def read_data(infile, boundary_style):
    """Read a Lammps data file.

    Parameters
    ----------
    infile : str
        input data file
    boundary_style : str
        Lammps boundary style (e.g. 's s p' or 'p p p')

    Returns
    -------
    coordinates : numpy.ndarray
        coordinates of the atoms. Periodic boundary conditions are undone,
        i.e. the atoms are not wrapped around the periodic boundary.
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
    outfile : str
        output data file
    boundary_style: str
        Lammps boundary style (e.g. 's s p' or 'p p p')
    infile : str
        input data file
    coordinates : array-like
        coordinates  of  the  atoms.

    Notes
    -----
    The input data file must be given and is actually re-read before writing, so
    that scatter_atoms can be used to overwrite a current set of coordinates.
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
    path_to_dump_file: str
        path to the dump file

    Returns
    -------
    header : list
        header of the dump file
    atoms_section : numpy.ndarray
        ATOMS section of the dump file as 2D array. Contains one row
        of data for each atom. Number of columns and column contents
        depend must be inferred from the last line in the header
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
    outfile: str 
        file to write to
    header: list
        header of the dump file
    atomdata : array-like
        one row of data for each atom.
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


def calc_image_distances(img_chunk, periodic_boundary, box_size):
    """Calculate the lengths for wrapping or unwrapping atomic coordinates
    across periodic boundaries.

    Parameters
    ----------
    img_chunk : array-like
        image flags
    periodic_boundary : array-like
        element i = True if boundary periodic in i
    box_size : array-like
        simulation box size

    Returns
    -------
    image_distances : list 
        distance that must be subtracted from x, y, or z-coordinates to wrap
        them across the periodic boundaries. None if boundary is not periodic.

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
    dof_chunk : array-like
        atomic coordinates
    periodic_boundary : list
        element i = True if boundary in direction i is periodic
    image_distances : list 
        distance that must be subtracted from x, y, or z-coordinates to wrap
        them across the periodic boundaries. None if boundary is not periodic.
    mode : str
        'wrap' to wrap coordinates, 'unwrap' to unwrap

    Returns
    -------
    dof_chunk : numpy.ndarray
        atomic coordinates with pbc applied

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
