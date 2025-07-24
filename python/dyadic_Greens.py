import numpy as np
import numpy.typing as npt

k = 0.0139277
epsilon_0 = 8.85e-6
epsilon_b = 1.7689


def dyadicGreensFunc(r: npt.NDArray, rp: npt.NDArray) -> npt.NDArray:
    R = np.linalg.norm(r - rp)
    G = np.exp(1j * k * R) / (4 * np.pi * R * epsilon_0 * epsilon_b)

    delta_ij = np.eye(3)
    diff = r - rp
    outer_prod = np.einsum("i,j->ij", diff, diff)

    dyadicGreens = (
        1
        / (R**4)
        * (
            delta_ij * G * R**2 * (k * R * (k * R + 1j) - 1)
            - G * (-3 + k * R * (k * R + 3j)) * outer_prod
        )
    )
    return dyadicGreens


if __name__ == "__main__":
    r = np.asarray([2, 3, 5])
    rp = np.asarray([0, 0, 0])

    dyadGreensFunc = dyadicGreensFunc(r, rp)
    print(dyadGreensFunc)
