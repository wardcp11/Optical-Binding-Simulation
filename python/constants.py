import numpy as np

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
epsilon_0 = 8.854187817e-12 * farad / meter
mu_0 = pi * (4e-7) * henry / meter
eta_0 = np.sqrt(mu_0 / epsilon_0)
c = 1.0 / (np.sqrt(epsilon_0 * mu_0))
kB = 1.38e-23 * kg * meter**2 / ((second**2) * kelvin)

# Colloid Properties
a = 500e-9 * meter
vol = pi * a**3 * (4 / 3)
rhoSIO2 = 2320 * kg / (meter**3)
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
epsilon_p = -3 * (1.33**2)
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
# num_of_particle = 3
num_of_particle = 3
alpha = alpha_real + alpha_imag

w0 = L / 4
E0 = np.sqrt(2 * eta_0 * eta_b * I_0)

if __name__ == "__main__":
    print(f"{epsilon_0=}")
    print(f"{epsilon_b=}")
    print(f"{k=}")
    print(f"{E0=}")
