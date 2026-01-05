# import cupy as cp
import numpy as cp
from constants import (
    num_of_particle,
    alpha,
    gamma,
    mass,
    dt,
    maxstep,
    lam,
    ΔB,
)
from functions import (
    create_G_mnij,
    create_G_mnij_scatter,
    gen_Einc_mi,
    gen_F_grad,
)


if __name__ == "__main__":
    # Generate positions and calculate incident field
    # polarizabilities = cp.full((num_of_particle, 3), alpha)
    polarizabilities = cp.full((num_of_particle, 3), alpha)

    r = 2 * lam
    x1 = r * cp.cos(0)
    y1 = r * cp.sin(0)

    x2 = r * cp.cos(2 * cp.pi * (1 / 5))
    y2 = r * cp.sin(2 * cp.pi * (1 / 5))

    x3 = r * cp.cos(2 * cp.pi * (2 / 5))
    y3 = r * cp.sin(2 * cp.pi * (2 / 5))

    x4 = r * cp.cos(2 * cp.pi * (3 / 5))
    y4 = r * cp.sin(2 * cp.pi * (3 / 5))

    # x5 = r * cp.cos(2 * cp.pi * (4 / 5))
    # y5 = r * cp.sin(2 * cp.pi * (4 / 5))

    init_pos_arr = cp.asarray(
        [
            [x1, y1, 0],
            [x2, y2, 0],
            [x3, y3, 0],
            [x4, y4, 0],
            # [x5, y5, 0],
        ]
    )

    pos_arr = init_pos_arr.copy()

    # Einc_mi = gen_Einc_Gaussian_mi(pos_arr)  # guassian beam
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
        # Einc_mi = gen_Einc_Gaussian_mi(full_pos_arr[step - 1])
        Einc_mi = gen_Einc_mi(full_pos_arr[step - 1])
        E_flattened = E_flattened = Einc_mi.reshape(3 * num_of_particle)
        G_nmij = create_G_mnij(full_pos_arr[step - 1], polarizabilities)
        G_flattened = G_nmij.transpose(0, 2, 1, 3).reshape(
            3 * num_of_particle, 3 * num_of_particle
        )

        ## extract eignevalues of G matrix
        # eigenvalues, _ = cp.linalg.eig(G_flattened)
        # print(eigenvalues.shape)
        # cp.save("./eigenvalues_unmodified_G.npy", eigenvalues)

        p_i = (-cp.linalg.inv(G_flattened) @ E_flattened).reshape(
            num_of_particle, 3
        )  # (N, 3)
        forces_arr[step - 1] = gen_F_grad(full_pos_arr[step - 1], p_i)

        full_pos_arr[step] = full_pos_arr[step - 1] + velocity_arr[step - 1] * dt

        # Brownian kick goes here
        # randomDeltaV = (
        #     cp.sqrt(ΔB) * cp.random.normal(0, 1, (num_of_particle, 3)) / cp.sqrt(2)
        # )

        velocity_arr[step] = (
            velocity_arr[step - 1] * cp.exp(-gamma * dt)
            # + randomDeltaV
            + forces_arr[step - 1] * (dt / mass)
        )

        # velocity_arr[step, 0] = cp.asarray(
        #     [0, 0, 0]
        # )  # set velocity to zero on the first particle

        for i in range(num_of_particle):
            velocity_arr[step, i, 2] = 0

        print(f"step: {step} / {maxstep} ")

    # SAVE DATA AND PLOT IT
    cp.save("./data/position_data.npy", full_pos_arr)
    cp.save("./data/velocity_data.npy", velocity_arr)
    cp.save("./data/forces_data.npy", forces_arr)
    cp.save("./data/induced_dipoles.npy", p_i)
    print("saved data")
