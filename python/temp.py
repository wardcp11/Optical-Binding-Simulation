
def gen_F_grad(
    pos_arr: NDArray[float64], pol_arr: NDArray[complex128]
) -> NDArray[complex128]:
    Einc = gen_Einc_mi(pos_arr)  # (N, 3)
    dx_Einc = gen_dx_Einc(pos_arr)  # (N, 3, 3)

    Escat = gen_Escat(pos_arr, pol_arr)  # (N, 3)
    dx_Escat = gen_dx_Escat_vec(pos_arr, pol_arr)  # (N, 3, 3)

    prod_rule1 = cp.einsum("ni,nij->nj", Escat, dx_Escat.conj()) + cp.einsum(
        "ni,nij->nj", Escat.conj(), dx_Escat
    )

    prod_rule2 = cp.einsum("ni,nij->nj", Escat, dx_Einc.conj()) + cp.einsum(
        "ni,nij->nj", Einc.conj(), dx_Escat
    )

    prod_rule3 = cp.einsum("ni,nij->nj", Einc, dx_Escat.conj()) + cp.einsum(
        "ni,nij->nj", Escat.conj(), dx_Einc
    )

    prod_rule4 = cp.einsum("ni,nij->nj", Einc, dx_Einc.conj()) + cp.einsum(
        "ni,nij->nj", Einc.conj(), dx_Einc
    )

    F_grad = (alpha_real / 4) * cp.real(
        prod_rule1 + prod_rule2 + prod_rule3 + prod_rule4
    )

    return F_grad

