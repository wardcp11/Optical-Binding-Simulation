import cupy as cp
import cupyx.scipy.linalg as cpx
import numpy.typing as npt
from constants import k, epsilon_0, epsilon_b, E0, w0, alpha_real, num_of_particle


def create_G_mnij(
    pos_arr: npt.NDArray[cp.float64], pol_arr: npt.NDArray[cp.complex128]
) -> npt.NDArray[cp.complex128]:
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
    # pos_arr: npt.NDArray[cp.float64], pol_arr: npt.NDArray[cp.complex128]


def create_G_mnij_scatter(
    pos_arr: npt.NDArray[cp.float64],
) -> npt.NDArray[cp.complex128]:
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


# def gen_Escat_mi(Gscat_mnij: cp.ndarray[cp.complex128], pi: cp.ndarray[cp.complex128]) -> cp.ndarray[cp.complex128]:
#     Gflattened = Gscat_mnij.reshape(3 * num_of_particle, 3 * num_of_particle)
#     pflattened =


def print_condition_number(A: npt.NDArray):
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


def gen_Einc_mi(pos_arr: npt.NDArray[cp.float64]) -> npt.NDArray[cp.complex128]:
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

    Einc_mi = cp.zeros((N, 3), dtype=cp.complex128)
    Einc_mi[:, 0] = E0 * cp.exp(1j * k * z) * cp.exp(-(x**2 + y**2) / w0**2)

    return Einc_mi  # (N, 3)


def gen_Escat_di_mi(
    pos_arr: npt.NDArray[cp.float64], pol_arr: npt.NDArray[cp.complex128]
) -> npt.NDArray[cp.complex128]:
    N = pos_arr.shape[0]
    # x = pos_arr[:, 0]
    # y = pos_arr[:, 1]
    # z = pos_arr[:, 2]

    pos_diff_arr = pos_arr[:, None, :] - pos_arr[None, :, :]  # shape of (N, N, 3)
    R = cp.linalg.norm(pos_diff_arr, axis=2)  # (N, N)
    kR = k * R  # (N, N)
    G = cp.exp(1j * kR) / (4 * cp.pi * R * epsilon_0 * epsilon_b)  # (N, N)

    X_mniX_mnj = cp.einsum(
        "...i,...j->...ij", pos_diff_arr, pos_diff_arr
    )  # (N, N, 3, 3)

    delta = cp.eye(3)[None, None, :, :]  # (1, 1, 3, 3)
    first_term = (
        G[:, :, None]
        * ((1j * kR - 1)[:, :, None] / R[:, :, None])
        * (pos_diff_arr / R[:, :, None])
    )

    a = (G[:, :, None] * pos_diff_arr**3 / R**4) * (
        k * (6j * R + k * R**2 * (3 - 1j * kR) - 6)
    )
    b = (
        (
            G[:, :, None, None]
            * (2 - kR[:, :, None, None] ** 2 - 2j * kR[:, :, None, None])
            / (R[:, :, None, None] ** 2)
        )
        * ((delta / R[:, :, None, None]) + (X_mniX_mnj / R[:, :, None, None] ** 3))
        * (pos_diff_arr[:, :, :, None] / R[:, :, None])
    )
    second_term = a[:, :, :, None] + 2 * b

    third_term = (G[:, :, None, None] / R[:, :, None, None] ** 8) * (
        1j
        * (1j + kR[:, :, None, None])
        * (
            R[:, :, None, None] ** 4 * pos_diff_arr[:, :, :, None]
            - X_mniX_mnj * pos_diff_arr[:, :, :, None]
        )
        - R[:, :, None, None] ** 2
        * (kR[:, :, None, None] ** 2 - 2 + 2j * kR[:, :, None, None])
        * pos_diff_arr[:, :, :, None]
        * (X_mniX_mnj + R[:, :, None, None] ** 2 * delta)
    )

    derivative_terms = first_term[:, :, :, None] + second_term + third_term
    Escat_di_mi = cp.einsum("nmij,mi->nj", derivative_terms, pol_arr)
    return Escat_di_mi

def calculate_gradient_forces(
    pos_arr: npt.NDArray[cp.float64],
    pol_arr: npt.NDArray[cp.complex128],
    Escat_mi: npt.NDArray[cp.complex128],
) -> npt.NDArray:
    N = pos_arr.shape[0]
    pos_diff_arr = pos_arr[:, None, :] - pos_arr[None, :, :]  # shape of (N, N, 3)
    x = pos_arr[:, 0]  # (N, 1)
    y = pos_arr[:, 1]
    z = pos_arr[:, 2]
    mag_Einc_sq_di_mi = cp.zeros((N, 3), dtype=cp.complex128)
    Einc_di_mi = cp.zeros((N, 3), dtype=cp.complex128)

    R = cp.linalg.norm(pos_diff_arr, axis=2)  # (N, N)
    # G = cp.exp(1j * k * R) / (4 * cp.pi * R * epsilon_0 * epsilon_b)  # (N, N)

    # Calculate all the terms present in gradient force equation
    Einc_mi = gen_Einc_mi(pos_arr)
    Einc_di_mi[:, 0] = (
        -(2 * E0 / w0**2) * x * cp.exp(1j * k * z) * cp.exp(-(x**2 + y**2) / w0**2)
    )
    Escat_di_mi = gen_Escat_di_mi(pos_arr, pol_arr)
    mag_Einc_sq_di_mi[:, 0] = (
        -4 * E0**2 / w0**2 * cp.exp(-2 * (x**2 + y**2) / w0**2)
    ) * x

    output = (
        alpha_real
        / 4
        * (
            mag_Einc_sq_di_mi * Escat_mi
            + Einc_di_mi * cp.conjugate(Escat_mi)
            + cp.conjugate(Einc_mi) * Escat_di_mi
            + Escat_di_mi * cp.conjugate(Escat_mi)
            + Escat_mi * cp.conjugate(Escat_di_mi)
        )
    )
    # print(output.shape)

    return output
