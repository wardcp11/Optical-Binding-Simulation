import cupy as cp
from cupy import float64, complex128, full
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


def create_G_field(
    src_pos_arr: NDArray[float64],  # (N, 3) source dipoles
    obs_pos_arr: NDArray[float64],  # (M, 3) observation grid points
) -> NDArray[complex128]:
    """
    Computes the dyadic Green's function Gₘₙᵢⱼ for fields at arbitrary observation points.

    Parameters
    ----------
    src_pos_arr : cp.ndarray
        Positions of N source dipoles (N, 3).
    obs_pos_arr : cp.ndarray
        Positions of M observation points (M, 3).

    Returns
    -------
    G_field : cp.ndarray
        Dyadic Green's function array of shape (M, N, 3, 3), where
        G_field[m, n, i, j] gives the (i,j)-component of the Green’s tensor
        from dipole n to observation point m.
    """
    obs_pos_arr = cp.asarray(obs_pos_arr)
    src_pos_arr = cp.asarray(src_pos_arr)

    # (M, N, 3) difference vector: observation - source
    Rmn = obs_pos_arr[:, None, :] - src_pos_arr[None, :, :]

    R = cp.linalg.norm(Rmn, axis=2)  # (M, N)
    kR = k * R

    # Free-space scalar Green’s function
    G_scalar = cp.exp(1j * kR) / (4 * cp.pi * R * epsilon_0 * epsilon_b)

    # Tensor structure
    I = cp.eye(3)[None, None, :, :]  # (1,1,3,3)
    outer = cp.einsum("mni,mnj->mnij", Rmn, Rmn)  # (M,N,3,3)

    # Dyadic terms
    main = (
        I
        * G_scalar[:, :, None, None]
        * R[:, :, None, None] ** 2
        * (kR * (kR + 1j) - 1)[:, :, None, None]
    )
    off = G_scalar[:, :, None, None] * (-3 + kR * (kR + 3j))[:, :, None, None] * outer

    G_field = (main - off) / (R[:, :, None, None] ** 4)

    return G_field


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


# def gen_dx_Escat(
#     pos_arr: NDArray[float64], pol_arr: NDArray[complex128]
# ) -> NDArray[complex128]:
#     # NOTE: This is a first pass not using array broadcasting
#     π = cp.pi
#     N = pos_arr.shape[0]
#
#     dx_Escat = cp.zeros((N, N, 3, 3), dtype=complex128)
#     for n in range(N):
#         for m in range(N):
#             if n == m:
#                 dx_Escat[n, m] = cp.zeros((3, 3))
#             else:
#                 xi = pos_arr[n] - pos_arr[m]
#                 r = cp.linalg.norm(xi, axis=0)
#                 kr = k * r
#                 r_sq = r**2
#
#                 G = cp.exp(1j * kr) / (4 * π * r)
#                 # Calculate d/dxl d/dxj d/dxi G
#                 kron = cp.eye(3)  # (3, 3)
#                 xiδjl = cp.einsum("i,jl->ijl", xi, kron)  # (3, 3, 3)
#                 xjδil = cp.einsum("j,il->ijl", xi, kron)  # (3, 3, 3)
#                 xlδij = cp.einsum("l,ij->ijl", xi, kron)  # (3, 3, 3)
#                 xixjxl = cp.einsum("i,j,l->ijl", xi, xi, xi)  # (3, 3, 3)
#
#                 kron_terms = (r_sq * (kr**2 + 3j * kr - 3)) * (
#                     xiδjl + xjδil + xlδij
#                 )  # (3, 3, 3)
#
#                 dxldxjdxiG = (-G / r**6) * (
#                     kron_terms + xixjxl * (1j * kr**3 - 6 * kr**2 - 15j * kr + 15)
#                 )  # (3, 3, 3) # Passed
#
#                 full_terms = (
#                     G * (k**2) * (r**-2) * xlδij * (1j * kr - 1) + dxldxjdxiG
#                 ) / (
#                     epsilon_0 * epsilon_b
#                 )  # (3, 3, 3) # Passed
#
#                 dx_Escat[n, m] = cp.einsum("ijl,l->ij", full_terms, pol_arr[n])
#     return dx_Escat.sum(axis=1)


def gen_dx_Escat_vec(
    pos_arr: NDArray[float64], pol_arr: NDArray[complex128]
) -> NDArray[complex128]:
    π = cp.pi
    N = num_of_particle

    xi = pos_arr[:, None, :] - pos_arr[None, :, :]  # shape of (N, N, 3) (m, n, i)
    r = cp.linalg.norm(
        xi, axis=2
    )  # shape of (N, N) (m, n) # give 2 particle indices ill give you the distance
    kr = k * r  # (m, n)
    r_sq = r**2  # (m, n)

    G = cp.exp(1j * kr) / (4 * π * r)  # (m, n)

    kron = cp.eye(3)
    xiδjl = cp.einsum("mni,jl->mnijl", xi, kron)
    xjδil = cp.einsum("mnj,il->mnijl", xi, kron)
    xlδij = cp.einsum("mnl,ij->mnijl", xi, kron)
    xixjxl = cp.einsum("mni,mnj,mnl->mnijl", xi, xi, xi)

    # make everything a rank 5 tensor
    r = r[:, :, None, None, None]
    kr = kr[:, :, None, None, None]
    r_sq = r_sq[:, :, None, None, None]
    G = G[:, :, None, None, None]

    kron_terms = (r_sq * (kr**2 + 3j * kr - 3)) * (xiδjl + xjδil + xlδij)

    dxldxjdxiG = (-G / r**6) * (
        kron_terms + xixjxl * (1j * kr**3 - 6 * kr**2 - 15j * kr + 15)
    )

    full_terms = (G * (k**2) * (r**-2) * xlδij * (1j * kr - 1) + dxldxjdxiG) / (
        epsilon_0 * epsilon_b
    )

    mask = cp.eye(N, N, dtype=bool)  # creates identity boolean mask
    mask = mask[:, :, None, None, None]  # resizes mask to (m,n,i,j,l)

    masked_full_terms = cp.where(mask, 0, full_terms)
    dx_Escat = cp.einsum("mnijl,nj->mil", masked_full_terms, pol_arr)
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
    dx_Escat = gen_dx_Escat_vec(pos_arr, pol_arr)  # (N, 3, 3)

    prod_rule1 = cp.einsum("nj,njl->nl", Escat, dx_Escat.conj()) + cp.einsum(
        "nj,njl->nl", Escat.conj(), dx_Escat
    )

    prod_rule2 = cp.einsum("nj,njl->nl", Escat, dx_Einc.conj()) + cp.einsum(
        "nj,njl->nl", Einc.conj(), dx_Escat
    )

    prod_rule3 = cp.einsum("nj,njl->nl", Einc, dx_Escat.conj()) + cp.einsum(
        "nj,njl->nl", Escat.conj(), dx_Einc
    )

    prod_rule4 = cp.einsum("nj,njl->nl", Einc, dx_Einc.conj()) + cp.einsum(
        "nj,njl->nl", Einc.conj(), dx_Einc
    )

    F_grad = (alpha_real / 4) * cp.real(
        prod_rule1 + prod_rule2 + prod_rule3 + prod_rule4
    )

    return F_grad


if __name__ == "__main__":
    print("Hello world")
