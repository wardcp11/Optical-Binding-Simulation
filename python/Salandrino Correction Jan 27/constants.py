import numpy as np

use_gaussian_beam = True
use_circular_polarization = False
pol_angle = 0

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
joule = kg * meter**2 * (second**-2)
volt = joule / coulomb
farad = coulomb / volt
ampere = coulomb / second
watt = joule / second
henry = second**2 / farad


# Fundamental Constants
pi = np.pi
# epsilon_0 = 8.854187817e-12 * farad / meter
epsilon_0 = 8.854187817e-12 * farad / meter
mu_0 = pi * (4e-7) * henry / meter
eta_0 = np.sqrt(mu_0 / epsilon_0)
c = 1.0 / (np.sqrt(epsilon_0 * mu_0))
kB = 1.38e-23 * kg * meter**2 / ((second**2) * kelvin)

# Colloid Properties
a = 75e-9 * meter
vol = pi * a**3 * (4 / 3)
rhoSIO2 = 2320 * kg / (meter**3)
mass = vol * rhoSIO2
# mass = 1
mu = 1.6e-3 * kg / (meter * second)
gamma = 6.0 * pi * mu * a / mass
T = 300 * kelvin

# Laser Properties
lam = 600e-9 * meter
area = 100 * (1e-6 * meter) ** 2
power = 20 * watt
I_0 = power / area * 0.1

# Polarizability
# epsilon_p = -3 * (1.33**2)
# epsilon_p = -3
epsilon_p = 2.1 * (1.33) ** 2
mu_p = 1.0
epsilon_b = 1
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
alpha = alpha_s
alpha_real = alpha_s.real

# Physical Setup
# step_number = 100
L = 10 * lam


r = lam / 2
x1 = 0
y1 = r

x2 = r
y2 = 0

x3 = -r
y3 = 0

x4 = 0
y4 = -r

x5 = r
y5 = r

# x1 = r * np.cos(0)
# y1 = r * np.sin(0)
#
# x2 = r * np.cos(2 * np.pi * (1 / 5))
# y2 = r * np.sin(2 * np.pi * (1 / 5))
#
# x3 = r * np.cos(2 * np.pi * (2 / 5))
# y3 = r * np.sin(2 * np.pi * (2 / 5))

# x4 = r * np.cos(2 * np.pi * (3 / 5))
# y4 = r * np.sin(2 * np.pi * (3 / 5))


init_pos_arr = np.asarray(
    [
        # [-2 * r, r, 0],
        # [r, 0, 0],
        [0, 0, 0],
        [r, r, r],
        # [2 * r, r, 0],
        # [-r, -r, 0],
        # [-2 * r, -2 * r, 0],
        # [451.93, 26.3571, 0],
        # [-496.532, -146.504, 0],
        # [-586.027, 512.719, 0],
        # [500, 500, 0],
        # [-53, 573, 0],
        # [532, 554, 0],
        # [-237, -40, 0],
        # [-526, -137, 0],
    ]
)

# np.random.seed(128)
# init_pos_arr = np.random.uniform(low=-L / 2, high=L / 2, size=(10, 3))
# init_pos_arr[:, 2] = 0

# init_pos_arr = np.load("./data/2_9_26_tests/stable_pos.npy")
# init_pos_arr = np.append(init_pos_arr, [[10 * r, -3 * r, 0]], axis=0)
# init_pos_arr = np.append(init_pos_arr, [[2 * r, -3 * r, 0]], axis=0)
# init_pos_arr = np.append(init_pos_arr, [[-7 * r, -6 * r, 0]], axis=0)
# init_pos_arr = np.append(init_pos_arr, [[-8 * r, 10 * r, 0]], axis=0)
# init_pos_arr = np.append(init_pos_arr, [[6 * r, 11 * r, 0]], axis=0)


num_of_particle = init_pos_arr.shape[0]


w0 = 400 * lam
E0 = np.sqrt(2 * eta_0 * eta_b * I_0)

q0 = 3e-5
# q0 = 0

# dt = 0.0000001 * 10 / gamma
dt = 1
Γ = 2 * gamma * kB * T / mass
ΔB = Γ * dt
maxstep = 100000 * 5
# maxstep = 100000

if __name__ == "__main__":
    print(f"{I_0=}")
    print(f"{E0=}")
    print(f"{epsilon_0=}")
    print(f"{epsilon_p=}")
    print(f"{epsilon_b=}")
    print(f"{alpha_sr=}")
    print(f"{alpha_s=}")
