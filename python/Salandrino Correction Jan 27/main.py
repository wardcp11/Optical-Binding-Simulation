# import cupy as cp
import numpy as cp
from scipy.integrate import solve_ivp
from tqdm.auto import trange
from constants import (
    init_pos_arr,
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
    gen_dx_Einc,
)


# def rk45_step_x(x, v, t0, dt):
#     x0 = x.reshape(-1)
#     v0 = v.reshape(-1)
#
#     def rhs(t, x):
#         return v0
#
#     sol = solve_ivp(
#         rhs, (t0, t0 + dt), x0, method="RK45", t_eval=[t0 + dt], rtol=1e-10, atol=1e-12
#     )
#     return sol.y[:, -1].reshape(x.shape)


if __name__ == "__main__":
    # Generate positions and calculate incident field
    polarizabilities = cp.full((num_of_particle, 3), alpha)

    pos_arr = init_pos_arr.copy()

    # Einc_mi = gen_Einc_Gaussian_mi(pos_arr)  # guassian beam
    Einc_mi = gen_Einc_mi(pos_arr)

    # NOTE: Calculate dyadic Green's function and solve for polarizabilities
    G_nmij = create_G_mnij(pos_arr, polarizabilities)

    G_flattened = G_nmij.transpose(0, 2, 1, 3).reshape(
        3 * num_of_particle, 3 * num_of_particle
    )  # Transpose due to tensor strange default tensor ordering in numpy
    E_flattened = Einc_mi.reshape(3 * num_of_particle)

    inv = cp.linalg.inv(alpha * G_flattened)
    p_i = ((inv @ E_flattened) / alpha).reshape(num_of_particle, 3)

    # NOTE: Calculate scattering dyadic Green's function and find Escatterd
    G_mnij_scatter = create_G_mnij_scatter(pos_arr)

    Escattered_ni = cp.einsum("nmij,mj->ni", G_mnij_scatter, p_i)
    E_tot = Escattered_ni + Einc_mi

    F_grad = gen_F_grad(pos_arr, p_i)  # type: ignore

    velocity_arr = cp.zeros((maxstep + 1, num_of_particle, 3))
    full_pos_arr = cp.zeros((maxstep + 1, num_of_particle, 3))
    forces_arr = cp.zeros((maxstep + 1, num_of_particle, 3))
    p_i_arr = cp.zeros((maxstep + 1, num_of_particle, 3))
    E_n_arr = cp.zeros((maxstep + 1, num_of_particle, 3))
    det_G_arr = cp.zeros((maxstep + 1))
    max_G_arr = cp.zeros((maxstep + 1))

    eigenvalue_arr = []

    # initialize
    full_pos_arr[0] = init_pos_arr
    velocity_arr[0] = cp.zeros((num_of_particle, 3))

    # Loop
    # for step in range(1, maxstep + 1):
    for step in trange(1, maxstep + 1, desc="Simulating"):
        # Einc_mi = gen_Einc_Gaussian_mi(full_pos_arr[step - 1])
        Einc_mi = gen_Einc_mi(full_pos_arr[step - 1])
        E_flattened = E_flattened = Einc_mi.reshape(3 * num_of_particle)
        G_nmij = create_G_mnij(full_pos_arr[step - 1], polarizabilities)

        G_flattened = G_nmij.transpose(0, 2, 1, 3).reshape(
            3 * num_of_particle, 3 * num_of_particle
        )

        det_G_arr[step - 1] = cp.linalg.det(G_flattened)
        max_G_arr[step - 1] = cp.max(G_flattened)

        ## extract eigenvalues of G_matrix
        # eigenvalues, _ = cp.linalg.eig(G_flattened)
        # eigenvalue_arr.append(cp.abs(eigenvalues))

        inv = cp.linalg.inv(alpha * G_flattened)
        p_i = ((inv @ E_flattened) / alpha).reshape(num_of_particle, 3)

        p_i_arr[step - 1] = p_i

        G_mnij_scatter = create_G_mnij_scatter(full_pos_arr[step - 1])
        Escattered_ni = cp.einsum("nmij,mj->ni", G_mnij_scatter, p_i_arr[step - 1])
        E_n_arr[step - 1] = Einc_mi + Escattered_ni

        forces_arr[step - 1] = gen_F_grad(full_pos_arr[step - 1], p_i)  # type: ignore

        full_pos_arr[step] = full_pos_arr[step - 1] + velocity_arr[step - 1] * dt
        # full_pos_arr[step] = rk45_step_x(
        #     full_pos_arr[step - 1], velocity_arr[step - 1], t0=step * dt, dt=dt
        # )

        # Brownian kick goes here
        # randomDeltaV = (
        #     cp.sqrt(ΔB) * cp.random.normal(0, 1, (num_of_particle, 3)) / cp.sqrt(2)
        # )

        velocity_arr[step] = (
            velocity_arr[step - 1] * cp.exp(-gamma * dt)
            # + randomDeltaV
            + forces_arr[step - 1] * (dt / mass)
        )

        # NOTE: Velocity clamp
        # velocity_max = 2.5e3
        # for N in range(num_of_particle):
        #     velocity_x = velocity_arr[step, N, 0]
        #     velocity_y = velocity_arr[step, N, 1]
        #
        #     velocity_mag = cp.sqrt(velocity_x**2 + velocity_y**2)
        #
        #     theta = cp.atan2(velocity_y, velocity_x)
        #
        #     if velocity_mag > velocity_max:
        #         velocity_arr[step, N, 0] = velocity_max * cp.cos(theta)
        #         velocity_arr[step, N, 1] = velocity_max * cp.sin(theta)

        # velocity_arr[step, 0] = cp.asarray(
        #     [0, 0, 0]
        # )  # set velocity to zero on the first particle

        for i in range(num_of_particle):
            forces_arr[step, i, 2] = 0
            velocity_arr[step, i, 2] = 0
            full_pos_arr[step, i, 2] = 0

        # print(f"step: {step} / {maxstep} ")

    # SAVE DATA
    cp.save("./data/p_i_data.npy", cp.asarray(p_i_arr))
    cp.save("./data/E_n_data.npy", cp.asarray(E_n_arr))
    cp.save("./data/position_data.npy", cp.asarray(full_pos_arr))
    cp.save("./data/velocity_data.npy", cp.asarray(velocity_arr))
    cp.save("./data/forces_data.npy", cp.asarray(forces_arr))
    cp.save("./data/eigenvalues_data.npy", cp.asarray(eigenvalue_arr))
    cp.save("./data/G_det.npy", cp.asarray(det_G_arr))
    cp.save("./data/G_max.npy", cp.asarray(max_G_arr))
    print("saved data")
