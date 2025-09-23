import cupy as cp
from cupy import float64, complex128
from cupy.testing import assert_allclose
from numpy.typing import NDArray
from constants import (
    k,
    epsilon_0,
    epsilon_b,
    E0,
    w0,
    alpha,
    alpha_real,
    num_of_particle,
)


def create_G_mnij(
    pos_arr: NDArray[float64], pol_arr: NDArray[complex128]
) -> NDArray[complex128]:
    """
    Computes the full (N, N, 3, 3) dyadic Green's tensor Gₘₙᵢⱼ for a system of N dipoles.

    For dipoles at positions `pos_arr[n]`, this function returns a 4D array where
    G[n, m, i, j] contains the dyadic Green's function tensor that describes the
    electromagnetic coupling between dipole `m` and dipole `n`. Self-interactions
    (where n == m) are set to -I / alphaₘ.

    Parameters
    ----------
    pos_arr : cp.ndarray
        Array of particle positions with shape (N, 3).

    pol_arr : cp.ndarray
        Array of particle polarizabilities with shape (N,).

    Returns
    -------
    cp.ndarray
        The dyadic Green's tensor with shape (N, N, 3, 3), where the last two indices
        represent the tensor components and the first two index the dipole pair (n, m).
        Self-terms (n == m) are set to -1 / alpha * I₃.

    Notes
    -----
    - Computation is vectorized and runs entirely on the GPU using CuPy.
    - Based on the free-space dyadic Green's function in the frequency domain.
    """
    N = pos_arr.shape[0]
    pos_diff_arr = pos_arr[:, None, :] - pos_arr[None, :, :]  # shape of (N, N, 3)

    R = cp.linalg.norm(pos_diff_arr, axis=2)  # (N, N)
    kR = k * R  # (N, N)
    G = cp.exp(1j * kR) / (4 * cp.pi * R * epsilon_0 * epsilon_b)  # (N, N)
    outer_prod_arr = cp.einsum(
        "...i,...j->...ij", pos_diff_arr, pos_diff_arr
    )  # (N, N, 3, 3)
    block_struct = cp.eye(3)[None, None, :, :]  # (1, 1, 3, 3)

    main_diag_terms = (
        block_struct
        * G[:, :, None, None]
        * R[:, :, None, None] ** 2
        * (kR * (kR + 1j) - 1)[:, :, None, None]
    )  # (N, N, 3, 3)
    off_diag_terms = (
        G[:, :, None, None] * (-3 + kR * (kR + 3j))[:, :, None, None] * outer_prod_arr
    )  # (N, N, 3, 3)
    scaling_terms = 1 / (R[:, :, None, None] ** 4)
    dyadic = scaling_terms * (main_diag_terms - off_diag_terms)

    mask = cp.eye(N, dtype=bool)  # (N,N)
    dyadic[mask] = (-1 / pol_arr[:, None]) * cp.eye(3)[None, :, :]
    return dyadic


def create_G_mnij_scatter(
    pos_arr: NDArray[float64],
) -> NDArray[complex128]:
    """
    Computes the full (N, N, 3, 3) dyadic Green's tensor Gₘₙᵢⱼ for a system of N dipoles.

    For dipoles at positions `pos_arr[n]`, this function returns a 4D array where
    G[n, m, i, j] contains the dyadic Green's function tensor that describes the
    electromagnetic coupling between dipole `m` and dipole `n`. Self-interactions
    (where n == m) are set to 0.

    Parameters
    ----------
    pos_arr : cp.ndarray
        Array of particle positions with shape (N, 3).

    pol_arr : cp.ndarray
        Array of particle polarizabilities with shape (N,).

    Returns
    -------
    cp.ndarray
        The dyadic Green's tensor with shape (N, N, 3, 3), where the last two indices
        represent the tensor components and the first two index the dipole pair (n, m).
        Self-terms (n == m) are set to 0.

    Notes
    -----
    - Computation is vectorized and runs entirely on the GPU using CuPy.
    - Based on the free-space dyadic Green's function in the frequency domain.
    """
    N = pos_arr.shape[0]
    pos_diff_arr = pos_arr[:, None, :] - pos_arr[None, :, :]  # shape of (N, N, 3)

    R = cp.linalg.norm(pos_diff_arr, axis=2)  # (N, N)
    kR = k * R  # (N, N)
    G = cp.exp(1j * kR) / (4 * cp.pi * R * epsilon_0 * epsilon_b)  # (N, N)
    outer_prod_arr = cp.einsum(
        "...i,...j->...ij", pos_diff_arr, pos_diff_arr
    )  # (N, N, 3, 3)
    block_struct = cp.eye(3)[None, None, :, :]  # (1, 1, 3, 3)

    main_diag_terms = (
        block_struct
        * G[:, :, None, None]
        * R[:, :, None, None] ** 2
        * (kR * (kR + 1j) - 1)[:, :, None, None]
    )  # (N, N, 3, 3)
    off_diag_terms = (
        G[:, :, None, None] * (-3 + kR * (kR + 3j))[:, :, None, None] * outer_prod_arr
    )  # (N, N, 3, 3)
    scaling_terms = 1 / (R[:, :, None, None] ** 4)
    dyadic = scaling_terms * (main_diag_terms - off_diag_terms)

    mask = cp.eye(N, dtype=bool)  # (N,N)
    dyadic[mask] = 0.0 * cp.eye(3)[None, :, :]
    return dyadic


def print_condition_number(A: NDArray):
    """
    Calculates the condition number for a given matrix `A`. The condition number for a matrix is defined as k = λmax / λmin, then prints it to console.

    Parameters
    ----------
    A : cp.ndarray
        Matrix to be evaluated

    Returns
    float
        Condition number

    Notes
    ----
    - Requires a square matrix input
    """
    sigma = cp.linalg.svd(A, compute_uv=False)
    lam_max = cp.max(sigma)
    lam_min = cp.min(sigma)
    if cp.isclose(lam_min, 0, atol=1e-15):
        print(f"Condition number: Inf")
    else:
        print(
            f"Condition number: {float(lam_max / lam_min)} | {10 * cp.log10(float(lam_max/lam_min))} dB"
        )


def gen_Einc_mi(pos_arr: NDArray[float64]) -> NDArray[complex128]:
    """
    Generates the incident electric field. Assumes x-polarized Gaussian beam.

    Parameters
    ----------
    pos_arr: cp.ndarray
        Array of particle positons

    Returns
    cp.ndarray
        Einc_mi with a shape of (N, 3), where N is the number of particles
    """

    N = pos_arr.shape[0]
    x = pos_arr[:, 0]
    y = pos_arr[:, 1]
    z = pos_arr[:, 2]

    Einc_mi = cp.zeros((N, 3), dtype=complex128)
    Einc_mi[:, 0] = E0 * cp.exp(1j * k * z) * cp.exp(-(x**2 + y**2) / w0**2)

    return Einc_mi  # (N, 3)


def gen_Escat(
    pos_arr: NDArray[float64], pol_arr: NDArray[complex128]
) -> NDArray[complex128]:
    G_mnij = create_G_mnij_scatter(pos_arr)
    return cp.einsum("nmij,mj->ni", G_mnij, pol_arr)


def gen_dx_Escat(
    pos_arr: NDArray[float64], pol_arr: NDArray[complex128]
) -> NDArray[complex128]:
    # NOTE: This is a first pass not using array broadcasting
    π = cp.pi
    N = pos_arr.shape[0]

    dx_Escat = cp.zeros((N, 3, 3), dtype=complex128)
    for n in range(N):
        for m in range(N):
            if n == m:
                pass
            else:
                xi = pos_arr[n] - pos_arr[m]
                r = cp.linalg.norm(xi, axis=0)
                kr = k * r
                r_sq = r**2

                G = cp.exp(1j * kr) / (4 * π * r)
                # Calculate d/dxl d/dxj d/dxi G
                kron = cp.eye(3)  # (3, 3)
                xiδjl = cp.einsum("i,jl->ijl", xi, kron)  # (3, 3, 3)
                xjδil = cp.einsum("j,il->ijl", xi, kron)  # (3, 3, 3)
                xlδij = cp.einsum("l,ij->ijl", xi, kron)  # (3, 3, 3)
                xixjxl = cp.einsum("i,j,l->ijl", xi, xi, xi)  # (3, 3, 3)

                kron_terms = (r_sq * (kr**2 + 3j * kr - 3)) * (
                    xiδjl + xjδil + xlδij
                )  # (3, 3, 3)

                dxldxjdxiG = (-G / r**6) * (
                    kron_terms + xixjxl * (1j * kr**3 - 6 * kr**2 - 15j * kr + 15)
                )  # (3, 3, 3) # Passed

                full_terms = (
                    (G * (k**2) * (r**-2) * xlδij * (1j * kr - 1) + dxldxjdxiG) / (epsilon_0 * epsilon_b)
                )  # (3, 3, 3) # Passed
                
                dx_Escat[n] = cp.einsum("ijl,l->ij", full_terms, pol_arr[n])
    return dx_Escat


def gen_dx_Einc(pos_arr: NDArray[float64]) -> NDArray[complex128]:
    N = num_of_particle

    dx_Einc = cp.zeros((N, 3, 3), dtype=complex128)

    for n in range(N):
        x, y, z = pos_arr[n]
        coeff = E0 * cp.exp(1j * k * z) * cp.exp(-(x**2 + y**2) / w0**2)

        matrix_term = cp.zeros((3, 3), dtype=complex128)
        matrix_term[0, 0] = -2 * (w0**-2) * x
        matrix_term[0, 1] = -2 * (w0**-2) * y
        matrix_term[0, 2] = 1j * k

        dx_Einc[n] = coeff * matrix_term

    return dx_Einc


def gen_F_grad(
    pos_arr: NDArray[float64], pol_arr: NDArray[complex128]
) -> NDArray[complex128]:
    Einc = gen_Einc_mi(pos_arr)  # (N, 3)
    dx_Einc = gen_dx_Einc(pos_arr)  # (N, 3, 3)

    Escat = gen_Escat(pos_arr, pol_arr)  # (N, 3)
    dx_Escat = gen_dx_Escat(pos_arr, pol_arr)  # (N, 3, 3)

    F_grad = cp.zeros((num_of_particle, 3))
    for n in range(num_of_particle):
        prod_rule1 = dx_Escat[n] @ Escat[n].conj() + dx_Escat[n].conj() @ Escat[n]
        prod_rule2 = dx_Escat[n] @ Einc[n].conj() + dx_Einc[n].conj() @ Escat[n]
        prod_rule3 = dx_Escat[n].conj() @ Einc[n] + dx_Einc[n] @ Escat[n].conj()
        prod_rule4 = dx_Einc[n] @ Einc[n].conj() + dx_Einc[n].conj() @ Einc[n]

        F_grad[n] = (alpha_real / 4) * cp.real(
            prod_rule1 + prod_rule2 + prod_rule3 + prod_rule4
        )
    return F_grad



if __name__ == "__main__":
    pos_arr = cp.asarray(
        [
            [-9888.3, -1688.19, 10996.0],
            [-11854.3, -4905.48, -2754.85],
            [1644.37, 6523.84, 2778.41],
        ]
    )

    pol_arr = cp.full((num_of_particle, 3), alpha)

    # gen_dx_Escat(pos_arr, pol_arr)
    print(gen_F_grad(pos_arr, pol_arr))
