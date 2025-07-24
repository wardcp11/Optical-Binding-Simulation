import cupy as cp
from numpy.typing import NDArray


def is_symmetric(A: NDArray, epsilon: float = 1e-8) -> bool:
    return cp.allclose(A, A.T.conj(), atol=epsilon)


data: NDArray = cp.load("./G.npy")  # type: ignore

m, n, i, j = data.shape
for mi in range(m):
    for ni in range(n):
        symmetry = is_symmetric(data[mi][ni])
        print(f"({mi+1},{ni+1}) is symmetric:{symmetry}")

print(f"Is flattened array symmetric: {is_symmetric(data.reshape(m * i, n * j))}")
