import json
import cupy as cp
from cupy.testing import assert_allclose
from functions import create_G_mnij, create_G_mnij_scatter, gen_Einc_mi
from constants import alpha, num_of_particle


def test_create_G_mnij():
    with open("../test_files/starting_pol_arr.json") as f:
        pol_arr = json.load(f)
    pol_arr = cp.asarray(pol_arr, dtype=cp.complex128)

    with open("../test_files/starting_pos_arr.json") as f:
        pos_arr = json.load(f)
    pos_arr = cp.asarray(pos_arr, dtype=cp.float64)

    with open("../test_files/G_mnij_real.json") as f:
        G_mnij_real = json.load(f)
    G_mnij_real = cp.asarray(G_mnij_real, dtype=cp.complex128)

    with open("../test_files/G_mnij_imag.json") as f:
        G_mnij_imag = json.load(f)
    G_mnij_imag = cp.asarray(G_mnij_imag, dtype=cp.complex128)

    G_mnij_ground_truth = G_mnij_real + 1j * G_mnij_imag
    G_mnij = create_G_mnij(pos_arr, pol_arr)

    # Allows any value to be within .1% off and an absolute difference of 1e-7
    assert_allclose(G_mnij, G_mnij_ground_truth, rtol=0.1e-2, atol=1e-7)  # type: ignore


def test_create_G_mnij_scatter():
    with open("../test_files/starting_pos_arr.json") as f:
        pos_arr = json.load(f)
    pos_arr = cp.asarray(pos_arr, dtype=cp.float64)

    with open("../test_files/G_mnij_real.json") as f:
        G_mnij_real = json.load(f)
    G_mnij_real = cp.asarray(G_mnij_real, dtype=cp.complex128)

    with open("../test_files/G_mnij_imag.json") as f:
        G_mnij_imag = json.load(f)
    G_mnij_imag = cp.asarray(G_mnij_imag, dtype=cp.complex128)

    G_mnij_ground_truth = G_mnij_real + 1j * G_mnij_imag

    for m in range(G_mnij_ground_truth.shape[0]):
        for n in range(G_mnij_ground_truth.shape[1]):
            if m == n:
                G_mnij_ground_truth[m, n] = cp.zeros((3, 3))

    print(G_mnij_ground_truth.shape)

    G_mnij = create_G_mnij_scatter(pos_arr)

    assert_allclose(G_mnij, G_mnij_ground_truth, rtol=0.1e-2, atol=1e-7)  # type: ignore


def test_gen_Einc_mi():
    with open("../test_files/starting_pos_arr.json") as f:
        pos_arr = json.load(f)
    pos_arr = cp.asarray(pos_arr, dtype=cp.float64)

    with open("../test_files/Einc_mi_real.json") as f:
        Einc_mi_real = json.load(f)
    Einc_mi_real = cp.asarray(Einc_mi_real)

    with open("../test_files/Einc_mi_imag.json") as f:
        Einc_mi_imag = json.load(f)
    Einc_mi_imag = cp.asarray(Einc_mi_imag)

    Einc_mi_ground_truth = Einc_mi_real + 1j * Einc_mi_imag
    Einc_mi = gen_Einc_mi(pos_arr)

    assert_allclose(Einc_mi, Einc_mi_ground_truth, rtol=0.1e-2, atol=1e-7)  # type: ignore


def test_dipole_moment_calcs():
    with open("../test_files/starting_pol_arr.json") as f:
        pol_arr = json.load(f)
    pol_arr = cp.asarray(pol_arr, dtype=cp.complex128)

    with open("../test_files/starting_pos_arr.json") as f:
        pos_arr = json.load(f)
    pos_arr = cp.asarray(pos_arr, dtype=cp.float64)

    with open("../test_files/p_i_real.json") as f:
        p_i_real = json.load(f)
    p_i_real = cp.asarray(p_i_real, dtype=cp.float64)

    with open("../test_files/p_i_imag.json") as f:
        p_i_imag = json.load(f)
    p_i_imag = cp.asarray(p_i_imag, dtype=cp.float64)

    p_i_ground_truth = (p_i_real + 1j * p_i_imag).reshape((3, 3))

    Einc_mi = gen_Einc_mi(pos_arr)
    G_nmij = create_G_mnij(pos_arr, pol_arr)

    G_flattened = G_nmij.transpose(0, 2, 1, 3).reshape(
        3 * num_of_particle, 3 * num_of_particle
    )
    E_flattened = Einc_mi.reshape(3 * num_of_particle)
    p_i = (-cp.linalg.inv(G_flattened) @ E_flattened).reshape(num_of_particle, 3)

    assert_allclose(p_i, p_i_ground_truth, rtol=0.2e-2, atol=1e-7)  # type:ignore


def test_Escatter_calcs():
    with open("../test_files/starting_pos_arr.json") as f:
        pos_arr = json.load(f)
    pos_arr = cp.asarray(pos_arr, dtype=cp.float64)

    with open("../test_files/starting_pol_arr.json") as f:
        pol_arr = json.load(f)
    pol_arr = cp.asarray(pol_arr, dtype=cp.complex128)

    with open("../test_files/Esct_mi_real.json") as f:
        Esc_mi_real = json.load(f)
    Esc_mi_real = cp.asarray(Esc_mi_real, dtype=cp.float64)

    with open("../test_files/Esct_mi_imag.json") as f:
        Esc_mi_imag = json.load(f)
    Esc_mi_imag = cp.asarray(Esc_mi_imag, dtype=cp.float64)

    Esc_mi_ground_truth = Esc_mi_real + 1j * Esc_mi_imag
    Einc_mi = gen_Einc_mi(pos_arr)
    G_nmij = create_G_mnij(pos_arr, pol_arr)

    G_flattened = G_nmij.transpose(0, 2, 1, 3).reshape(
        3 * num_of_particle, 3 * num_of_particle
    )
    E_flattened = Einc_mi.reshape(3 * num_of_particle)
    p_i = (-cp.linalg.inv(G_flattened) @ E_flattened).reshape(num_of_particle, 3)

    G_mnij_scatter = create_G_mnij_scatter(pos_arr)
    Escattered_ni = cp.einsum("nmij,mj->ni", G_mnij_scatter, p_i)

    assert_allclose(
        Escattered_ni, Esc_mi_ground_truth, rtol=0.2e-2, atol=1e-7  # type:ignore
    )


if __name__ == "__main__":
    print("Fuzzy Wuzzy was a bear.")
