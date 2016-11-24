def calculate_displacements_with_symbolical_integrals(radii, angles, b, m, n, xi, c):
    # Solve the integral problem
    print("Constructing angular function matrices")
    rotation_matrix = generate_symbolical_rotation_matrix(
        sp.matrices.Matrix(xi)
    )
    n_rot = rotation_matrix * n
    m_rot = rotation_matrix * m
    # It would perhaps be useful to simplify the expressions,
    # e.g. via trigsimp. Currently, however, this introduces
    # numerical error.
    #nn = sp.trigsimp(ab_contraction_symbolic(n_rot, n_rot, c))
    #mm = sp.trigsimp(ab_contraction_symbolic(m_rot, m_rot, c))
    #nm = sp.trigsimp(ab_contraction_symbolic(n_rot, m_rot, c))
    #mn = sp.trigsimp(ab_contraction_symbolic(m_rot, n_rot, c))
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

    # calculate the matrices S and B
    print("calculating S and B")
    # construct the integrands
    S_integrand = np.tensordot(nninv, nm, axes=([1], [0]))
    S_integrand = sp.matrices.Matrix(S_integrand)
    B_integrand = np.tensordot(nninv, nm, axes=([1], [0]))
    B_integrand = np.tensordot(mn, B_integrand, axes=([1], [0]))
    B_integrand = mm - sp.matrices.Matrix(B_integrand)
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
    if False: check_S_and_B(S, B, m, n, c)

    radii, angles = calculate_cylindrical_coordinates(coordinates, xi, m)

    # Calculate the displacements
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
        displacements[atom_index] = np.einsum(
            's,is', b, matrix_4) / (2.0 * np.pi
        )
    return displacements
