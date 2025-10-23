# import cupy as cp
import numpy as cp
from constants import num_of_particle, alpha, gamma, mass, L, dt, maxstep, lam
from functions import (
    create_G_mnij,
    create_G_mnij_scatter,
    gen_Einc_mi,
    gen_dx_Escat_vec,
    gen_F_grad,
)


if __name__ == "__main__":
    # Generate positions and calculate incident field
    # polarizabilities = cp.full((num_of_particle, 3), alpha)
    polarizabilities = cp.full((num_of_particle, 3), alpha)
    rng = cp.random.default_rng(42)
    init_pos_arr = rng.uniform(-L / 2, L / 2, size=(num_of_particle, 3))
    # init_pos_arr = cp.asarray(
    #     [
    #         [-9888.3, -1688.19, 10996.0],
    #         [-11854.3, -4905.48, -2754.85],
    #         [1644.37, 6523.84, 2778.41],
    #     ]
    # )
    # init_pos_arr = cp.asarray([[2 * lam, 2 * lam, 0], [2 * lam, 0, 0]])

    pos_arr = init_pos_arr.copy()

    Einc_mi = gen_Einc_mi(pos_arr)

    # NOTE: Calculate dyadic Green's function and solve for polarizabilities
    G_nmij = create_G_mnij(pos_arr, polarizabilities)

    G_flattened = G_nmij.transpose(0, 2, 1, 3).reshape(
        3 * num_of_particle, 3 * num_of_particle
    )
    E_flattened = Einc_mi.reshape(3 * num_of_particle)
    p_i = (-cp.linalg.inv(G_flattened) @ E_flattened).reshape(
        num_of_particle, 3
    )  # (N, 3)

    # NOTE: Calculate scattering dyadic Green's function and find Escatterd
    G_mnij_scatter = create_G_mnij_scatter(pos_arr)
    Escattered_ni = cp.einsum("nmij,mj->ni", G_mnij_scatter, p_i)

    F_grad = gen_F_grad(pos_arr, p_i)

    # NOTE: Time Evolution
    # dt = 10 / gamma
    # dt = 10 / gamma
    # Γ = 2 * gamma * kB * T / mass
    # ΔB = Γ * dt
    # maxstep = 1200

    velocity_arr = cp.zeros((maxstep + 1, num_of_particle, 3))
    full_pos_arr = cp.zeros((maxstep + 1, num_of_particle, 3))
    forces_arr = cp.zeros((maxstep + 1, num_of_particle, 3))

    # initialize
    full_pos_arr[0] = init_pos_arr
    velocity_arr[0] = cp.zeros((num_of_particle, 3))

    # Loop
    for step in range(1, maxstep + 1):
        Einc_mi = gen_Einc_mi(full_pos_arr[step - 1])
        G_nmij = create_G_mnij(full_pos_arr[step - 1], polarizabilities)
        G_flattened = G_nmij.transpose(0, 2, 1, 3).reshape(
            3 * num_of_particle, 3 * num_of_particle
        )
        p_i = (-cp.linalg.inv(G_flattened) @ E_flattened).reshape(
            num_of_particle, 3
        )  # (N, 3)
        forces_arr[step - 1] = gen_F_grad(full_pos_arr[step - 1], p_i)

        full_pos_arr[step] = full_pos_arr[step - 1] + velocity_arr[step - 1] * dt

        # Brownian kick goes here

        velocity_arr[step] = velocity_arr[step - 1] * cp.exp(-gamma * dt) + forces_arr[
            step - 1
        ] * (dt / mass)

    # SAVE DATA AND PLOT IT
    cp.save("./data/position_data.npy", full_pos_arr)
    cp.save("./data/velocity_data.npy", velocity_arr)
    cp.save("./data/forces_data.npy", forces_arr)
    print("saved data")
