# import cupy as cp
import numpy as cp
import matplotlib.pyplot as plt
from constants import (
    num_of_particle,
    alpha,
    L,
    lam,
    k,
    w0,
    E0,
)
from functions import (
    create_G_field,
    create_G_mnij,
    create_G_mnij_scatter,
    gen_Einc_mi,
    gen_F_grad,
)


if __name__ == "__main__":
    # Generate positions and calculate incident field
    polarizabilities = cp.full((num_of_particle, 3), alpha)

    # init_pos_arr = cp.asarray([[0, 0, 0], [10 * lam, 0, 0]])
    init_pos_arr = cp.asarray([[0, 0, 0]])

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

    sample_points = 1000
    x = cp.linspace(-L / 2, L / 2, sample_points)
    y = x
    z = lam / 2
    # z = 0
    X, Y = cp.meshgrid(x, y, indexing="ij")
    Z = cp.full_like(X, z)
    obs_pts = cp.stack((X.ravel(), Y.ravel(), Z.ravel()), axis=1)

    G = create_G_field(pos_arr, obs_pts)
    E = cp.einsum("mnij,nj->mi", G, p_i)
    E = E.reshape((sample_points, sample_points, 3))
    E_mag = cp.linalg.norm(E, axis=2)
    I_scat = E_mag**2

    Einc = cp.zeros((sample_points, sample_points, 3), dtype=complex)
    Einc[:, :, 0] = E0 * cp.exp(1j * k * z) * cp.exp(-(X**2 + Y**2) / w0**2)

    I_inc = (cp.linalg.norm(Einc, axis=2)) ** 2

    superposition = E + Einc
    superposition_mag = cp.linalg.norm(superposition, axis=2)
    I = superposition_mag**2

    # plt.figure()
    # plt.imshow(10 * cp.log10(I), cmap="jet")
    # plt.show()
    # exit()
    plt.figure()
    plt.plot(x, I_inc[500, :], label="x-cross section")
    plt.plot(x, I_inc[500, :], label="y-cross section")

    I_inc = cp.linalg.norm(Einc, axis=2) ** 2

    I_scat = (cp.abs(E_mag) ** 2).reshape(X.shape)

    # plt.plot(x, I_inc[500, :], label="I_inc cross section")
    # plt.plot(x, I_scat[:, 500], label="I_scat cross section")

    # plt.plot(x, )

    plt.legend()
    plt.show()

    plt.figure()
    plt.imshow(I_inc)

    # plt.figure(figsize=(6, 5))
    # plt.pcolormesh(X, Y, I, shading="auto", cmap="jet")
    # plt.xlabel("x (nm)")
    # plt.ylabel("y (nm)")
    # plt.title(r"$|E|^2$ superposition intensity cross-section at z = %.2f µm" % (z))
    # plt.colorbar(label="Intensity")
    # plt.axis("equal")
    # plt.show()

    # Ex = superposition[:, :, 0]
    # Ey = superposition[:, :, 1]
    # Ez = superposition[:, :, 2]

    dx = (x.max() - x.min()) / (sample_points - 1)
    dy = (x.max() - x.min()) / (sample_points - 1)

    dx_Emag, dy_Emag = cp.gradient(superposition_mag**2, dx, dy)

    mag = cp.sqrt(dx_Emag**2 + dy_Emag**2)

    # NOTE: Quiver plot below
    # fig, ax = plt.subplots()
    # ax.quiver(X, Y, dx_Emag / mag, dy_Emag / mag, mag, cmap="magma")
    # # ax.quiver(X, Y, dx_Emag, dy_Emag, mag, cmap="magma")
    # ax.set_aspect("equal")

    max_Ex = cp.max(dx_Emag)
    max_Ey = cp.max(dy_Emag)

    masked_dx_Emag = cp.where(cp.abs(dx_Emag) < 0.01 * max_Ex, 1, 0).astype(cp.uint8)
    masked_dy_Emag = cp.where(cp.abs(dy_Emag) < 0.01 * max_Ey, 1, 0).astype(cp.uint8)

    # NOTE: Sign plots
    fig, axs = plt.subplots(nrows=1, ncols=2)
    axs[0].imshow(masked_dx_Emag, cmap="jet", extent=(-L / 2, L / 2, -L / 2, L / 2))
    axs[1].imshow(masked_dy_Emag, cmap="jet", extent=(-L / 2, L / 2, -L / 2, L / 2))

    axs[0].set_title(r"$\nabla_x |E|^2$ w/ 0.0001 max threshold ")
    axs[1].set_title(r"$\nabla_y |E|^2$ w/ 0.0001 max threshold ")

    # NOTE: Logical and plots
    logical_and_plot = masked_dx_Emag & masked_dy_Emag
    fig, ax = plt.subplots()
    ax.imshow(logical_and_plot, cmap="jet", extent=(-L / 2, L / 2, -L / 2, L / 2))

    plt.show()
