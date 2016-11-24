def calculate_displacements_with_numerical_integrals(radii, angles, b, m, n, xi, c):
    def nninv_integrand(angle, i, j):
        rotation_matrix = generate_numerical_rotation_matrix(xi, angle)
        n_rot = np.einsum('ij, j', rotation_matrix, n)
        nn = numerical_ab_contraction(n_rot, n_rot, c)
        return np.linalg.inv(nn)[i, j]

    def S_integrand(angle, i, j):
        rotation_matrix = generate_numerical_rotation_matrix(xi, angle)
        m_rot = np.einsum('ij, j', rotation_matrix, m)
        n_rot = np.einsum('ij, j', rotation_matrix, n)
        nn = numerical_ab_contraction(n_rot, n_rot, c)
        nninv = np.linalg.inv(nn)
        nm = numerical_ab_contraction(n_rot, m_rot, c)
        return np.dot(nninv, nm)[i, j]

    def B_integrand(angle, i, j):
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
    if False: check_S_and_B(S, B, m, n, c)

    radii, angles = calculate_cylindrical_coordinates(coordinates, xi, m)

    # Calculate the displacements
    displacements = np.zeros((angles.shape[0], 3))
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
    return displacements
