import matplotlib.pyplot
import numpy as np
import numpy.typing as npt
import cupy as cp

# NOTE:  Constants

# Unit vectors
xx = cp.asarray([1.0, 0.0, 0.0])
yy = cp.asarray([0.0, 1.0, 0.0])
zz = cp.asarray([0.0, 0.0, 1.0])

# Units
microsecond = 1.0
microgram = 1.0
nanometer = 1.0
femtocoulomb = 1.0
kelvin = 1.0
second = microsecond * 1e6
kg = microgram * 1e9
meter = nanometer * 1e9
coulomb = femtocoulomb * 1e15
joule = kg * meter**2 * second**2
volt = joule / coulomb
farad = coulomb / volt
ampere = coulomb / second
watt = joule / second
henry = second**2 / farad

# Fundamental Constants
pi = np.pi
epsilon_0 = 8.854187817e-12 * farad / meter
mu_0 = pi * (4e-7) * henry / meter
eta_0 = np.sqrt(mu_0 / epsilon_0)
c = 1.0 / (np.sqrt(epsilon_0 * mu_0))
kB = 1.38e-23 * kg * meter**2 / ((second**2) * kelvin)

# Colloid Properties
a = 500e-9 * meter
vol = pi * a**3 * (4 / 3)
rhoSIO2 = 2320 * kg / meter**3
mass = vol * rhoSIO2
mu = 1.6e-3 * kg / (meter * second)
gamma = 6.0 * pi * mu * a / mass
T = 300

# Laser Properties
lam = 600e-9 * meter
area = 100 * (1e-6 * meter) ** 2
power = 1000 * watt
I_0 = power / area

# Polarizability
epsilon_p = -3 * 1.33**2
mu_p = 1.0
epsilon_b = 1.33**2
mu_b = 1.0
eta_b = np.sqrt((mu_b / epsilon_b))
k0 = 2.0 * pi / lam
omega = k0 * c
k = k0 * np.sqrt(epsilon_b)
alpha_sr = complex(
    4.0
    * pi
    * epsilon_0
    * epsilon_b
    * a**3
    * ((epsilon_p - epsilon_b) / (epsilon_p + 2 * epsilon_b))
)
alpha_s = ((1 / alpha_sr) - 1j * (k**3 / (6 * pi * epsilon_0 * epsilon_b))) ** -1
alpha_real = alpha_sr.real
alpha_imag = alpha_sr.imag

# Physical Setup
step_number = 100
L = 25e-6 * meter
num_of_particle = 10
alpha = alpha_real + alpha_imag
polarizabilities = cp.full((num_of_particle, 3), alpha)
init_positions_arr = cp.random.uniform(-L / 2, L / 2, size=(num_of_particle, 3))
# init_positions_arr = cp.asarray([[-4388.63, 12006.1, 0.0], [5618.52, 12135.7, 0.0]])
positions_arr = init_positions_arr.copy()


def dyadicGreensFunc(r: cp.ndarray, rp: cp.ndarray) -> cp.ndarray:
    R = cp.linalg.norm(r - rp)
    G = cp.exp(1j * k * R) / (4 * cp.pi * R)

    delta_ij = cp.eye(3)
    diff = r - rp
    outer_prod = cp.einsum("i,j->ij", diff, diff)

    dyadicGreens = (
        1
        / (k**2 * R**4 * epsilon_0 * epsilon_b)
        * (
            delta_ij * G * R**2 * (k * R * (k * R + 1j) - 1)
            - G * (-3 + k * R * (k * R + 3j)) * outer_prod
        )
    )
    return dyadicGreens


def GG(n: int, m: int) -> cp.ndarray:
    if n == m:
        return (-1 / polarizabilities[m]) * cp.eye(3)
    else:
        return dyadicGreensFunc(positions_arr[n], positions_arr[m])


# Incident Field
w0 = L / 4
E0 = np.sqrt(2 * eta_0 * eta_b * I_0)


def EIncident(n: int) -> cp.ndarray:
    # Could come back and do a single shot for tensor processing later?
    x, y, z = positions_arr[n]
    return E0 * xx * cp.exp(1j * k * z) * cp.exp(-(x**2 + y**2) / w0**2)


E_ni = cp.zeros((num_of_particle, 3), dtype=cp.complex128)
for n in range(num_of_particle):
    E_ni[n] = EIncident(n)

# Unknown Variables

# Creates full (N, N, 3, 3) tensor to solve the system of equations
G_nmij = cp.zeros((num_of_particle, num_of_particle, 3, 3), dtype=cp.complex128)
for n in range(num_of_particle):
    for m in range(num_of_particle):
        G_nmij[n][m] = GG(n, m)
# NOTE: Calling transpose here is to reorder how the object is stored in memory and the reshape takes it from a tensor to a matrix so we can use cp.la.solve()
G_flattened = G_nmij.transpose(0, 2, 3, 1).reshape(
    3 * num_of_particle, 3 * num_of_particle
)
E_flattened = E_ni.reshape(3 * num_of_particle)
pp = -cp.linalg.solve(G_flattened, E_flattened)

if __name__ == "__main__":
    print(f"{E0=}")
    print(f"{GG(1,1)=}")
