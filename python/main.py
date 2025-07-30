import numpy.typing as npt
import cupy as cp
import time
from constants import num_of_particle, alpha, k, omega, mu_0, mu_b, L
from functions import (
    calculate_gradient_forces,
    create_G_mnij,
    create_G_mnij_scatter,
    print_condition_number,
    gen_Einc_mi,
)

if __name__ == "__main__":
    # Generate positions and calculate incident field
    # polarizabilities = cp.full((num_of_particle, 3), alpha)
    polarizabilities = cp.full((num_of_particle, 3), alpha)
    # init_positions_arr = cp.random.uniform(-L / 2, L / 2, size=(num_of_particle, 3))
    init_positions_arr = cp.asarray(
        [
            [-9888.3, -1688.19, 10996.0],
            [-11854.3, -4905.48, -2754.85],
            [1644.37, 6523.84, 2778.41],
        ]
    )
    positions_arr = init_positions_arr.copy()
    # Incident Field
    Einc_mi = gen_Einc_mi(positions_arr)

    # Calculate dyadic Green's function and solve for polarizabilities
    G_nmij = create_G_mnij(positions_arr, polarizabilities)

    G_flattened = G_nmij.transpose(0, 2, 1, 3).reshape(
        3 * num_of_particle, 3 * num_of_particle
    )
    E_flattened = Einc_mi.reshape(3 * num_of_particle)
    p_i = (-cp.linalg.inv(G_flattened) @ E_flattened).reshape(num_of_particle, 3)

    # Calculate scattering dyadic Green's function and find Escatterd
    G_mnij_scatter = create_G_mnij_scatter(positions_arr)
    Escattered_ni = cp.einsum("nmij,mj->ni", G_mnij_scatter, p_i)

    # Calculate magnetic field
    Etot = Einc_mi + Escattered_ni
    Htot = (
        1j / (omega * mu_0 * mu_b) * cp.cross(cp.asarray([0, 0, k]), Etot)
    )  # BUG: Analytic solution required, work out by hand

    # Calculate forces
    grad_forces = calculate_gradient_forces(positions_arr, p_i, Escattered_ni)

    # NOTE: Print line debugging

    # print(G_nmij[1, 0])  # Passed
    # print(G_nmij[1, 1])  # Passed
    # print(G_nmij[0, 2])  # Passed

    # Test for symmetry: Passed
    # for i in range(G_nmij.shape[0]):
    #     for j in range(G_nmij.shape[1]):
    #         print(cp.isclose(G_nmij[i, j], G_nmij[j, i]))

    # print(Einc_mi[2])  # Passed
    # print(p_i[0])  # BUG: Failed
    # print(1e5 * p_i[0])  # Passed?

    print(Escattered_ni[0])  # Passed? Wrong order of magnitude
