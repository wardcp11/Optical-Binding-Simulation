import cupy as cp

I = cp.eye(3)
ones = cp.ones(3)

δ_ij = I[:, :, None] * ones[None, None, :]  # (3, 3)
δ_ik = I[:, None, :] * ones[None, :, None]  # (3, 3)
δ_jk = I[None, :, :] * ones[:, None, None]  # (3, 3)
print(δ_ij.shape)

xi = cp.zeros((2, 2, 3))  # (N, N, 3)
δ_ijx_k = cp.einsum("...ij,...k->...ijk", δ_ij, xi)  # (3, 3, 3)
print(δ_ijx_k.shape)
