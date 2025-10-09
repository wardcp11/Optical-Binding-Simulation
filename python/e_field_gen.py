import cupy as cp
from constants import num_of_particle, alpha, L
from functions import (
    create_G_mnij,
    create_G_mnij_scatter,
    gen_Einc_mi,
    gen_Escat,
    gen_dx_Escat_vec,
    gen_F_grad,
    create_G_field,
)
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # Generate positions and calculate incident field
    polarizabilities = cp.full((num_of_particle, 3), alpha)
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
    p_i = (-cp.linalg.inv(G_flattened) @ E_flattened).reshape(
        num_of_particle, 3
    )  # (N, 3)
    # Calculate scattering dyadic Green's function and find Escatterd
    G_mnij_scatter = create_G_mnij_scatter(positions_arr)
    Escattered_ni = cp.einsum("nmij,mj->ni", G_mnij_scatter, p_i)

    x = cp.linspace(-L / 2, L / 2, 200)
    y = cp.linspace(-L / 2, L / 2, 200)
    z = 10000

    X, Y = cp.meshgrid(x, y, indexing="ij")
    Z = cp.full_like(X, z)
    obs_pts = cp.stack((X.ravel(), Y.ravel(), Z.ravel()), axis=1)

    G = create_G_field(positions_arr, obs_pts)
    E = cp.einsum("mnij,nj->mi", G, p_i)  # (M,3)
    E_mag = cp.linalg.norm(E, axis=1)
    I = 10 * cp.log10(cp.abs(E_mag) ** 2)  # field intensity

    I_cpu = cp.asnumpy(I).reshape(X.shape)

    plt.figure(figsize=(6, 5))
    plt.pcolormesh(
        cp.asnumpy(X * 1e6), cp.asnumpy(Y * 1e6), I_cpu, shading="auto", cmap="inferno"
    )
    plt.xlabel("x (µm)")
    plt.ylabel("y (µm)")
    plt.title(r"$|E|^2$ intensity cross-section at z = %.2f µm" % (z))
    plt.colorbar(label="Intensity")
    plt.axis("equal")
    plt.show()

    dxEscat = gen_dx_Escat_vec(positions_arr, p_i)
