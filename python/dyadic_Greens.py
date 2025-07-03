from dataclasses import dataclass
import numpy as np
import numpy.typing as npt


# @dataclass
# class Point3D:
#     x: float
#     y: float
#     z: float


def dyadicGreensFunc(r: npt.NDArray, rp: npt.NDArray, k: float) -> npt.NDArray:
    R = np.linalg.norm(r - rp)
    G = np.exp(1j * k * R) / (4 * np.pi * R)

    delta_ij = np.eye(3)
    diff = r - rp
    outer_prod = np.einsum("i,j->ij", diff, diff)

    dyadicGreens = (
        1
        / (k**2 * R**4)
        * (
            delta_ij * G * R**2 * (k * R * (k * R + 1j) - 1)
            - G * (-3 + k * R * (k * R + 3j)) * outer_prod
        )
    )
    return dyadicGreens


if __name__ == "__main__":
    r = np.asarray([1, 1, 1])
    rp = np.asarray([1, 1, 0])

    dyadGreensFunc = dyadicGreensFunc(r, rp, 1)
    print(dyadGreensFunc)
