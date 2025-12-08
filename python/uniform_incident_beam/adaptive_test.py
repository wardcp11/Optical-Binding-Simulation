import numpy as cp
from constants import (
    num_of_particle,
    alpha,
    gamma,
    mass,
    dt as dt_initial,
    maxstep,
    lam,
    ΔB,
)
from functions import (
    create_G_mnij,
    create_G_mnij_scatter,
    gen_Einc_mi,
    gen_F_grad,
)


def compute_force_and_dipoles(pos, pol):
    Einc_mi = gen_Einc_mi(pos)
    G_nmij = create_G_mnij(pos, pol)
    G_flattened = G_nmij.transpose(0, 2, 1, 3).reshape(
        3 * num_of_particle, 3 * num_of_particle
    )

    E_flattened = Einc_mi.reshape(3 * num_of_particle)
    p_i = (-cp.linalg.inv(G_flattened) @ E_flattened).reshape(
        num_of_particle, 3
    )  # (N, 3)

    F_grad = gen_F_grad(pos, p_i)

    return F_grad, p_i


def step_damped_euler(pos, vel, dt, pol):
    """
    Single step of your current integrator with step size dt:

        pos_{n+1} = pos_n + vel_n * dt
        vel_{n+1} = vel_n * exp(-gamma * dt) + F(pos_n) * dt / mass

    F is recomputed inside from pos_n.
    """
    F, _ = compute_force_and_dipoles(pos, pol)

    pos_new = pos + vel * dt
    vel_new = vel * cp.exp(-gamma * dt) + F * (dt / mass)

    vel_new[:, 2] = 0  # restrics all movement in z

    return pos_new, vel_new


def error_norm(pos_big, vel_big, pos_two, vel_two, rtol=1e-3, atol=1e-6):
    """
    Compute a scaled RMS error norm between the "big step"
    and "two half steps" solutions.
    """

    y_big = cp.concatenate([pos_big.ravel(), vel_big.ravel()])
    y_two = cp.concatenate([pos_two.ravel(), vel_two.ravel()])
    err = y_two - y_big

    scale = atol + rtol * cp.maximum(cp.abs(y_two), cp.abs(y_big))
    return cp.sqrt(cp.mean((err / scale) ** 2))


if __name__ == "__main__":
    # Generate positions and calculate incident field
    polarizabilities = cp.full((num_of_particle, 3), alpha)

    r = 2 * lam
    x1 = r * cp.cos(0)
    y1 = r * cp.sin(0)

    x2 = r * cp.cos(2 * cp.pi * (1 / 5))
    y2 = r * cp.sin(2 * cp.pi * (1 / 5))

    x3 = r * cp.cos(2 * cp.pi * (2 / 5))
    y3 = r * cp.sin(2 * cp.pi * (2 / 5))

    x4 = r * cp.cos(2 * cp.pi * (3 / 5))
    y4 = r * cp.sin(2 * cp.pi * (3 / 5))

    x5 = r * cp.cos(2 * cp.pi * (4 / 5))
    y5 = r * cp.sin(2 * cp.pi * (4 / 5))

    init_pos_arr = cp.asarray(
        [
            [x1, y1, 0],
            [x2, y2, 0],
            [x3, y3, 0],
            [x4, y4, 0],
            [x5, y5, 0],
        ]
    )

    # Initial conditions
    pos = init_pos_arr.copy()
    vel = cp.zeros((num_of_particle, 3))

    # Initial force and dipoles (t = 0)
    F0, p0 = compute_force_and_dipoles(pos, polarizabilities)

    ## Time evolution with adaptive dt

    # Interpret maxstep * dt_initial as the "target phsyical time"
    T_final = maxstep * dt_initial
    t = 0.0
    dt = dt_initial

    # tolerances for adaptability
    rtol = 1e-6
    atol = 1e-9

    # step size control param
    p_order = 1
    safety = 0.9
    min_factor = 0.2
    max_factor = 5.0
    dt_min = 1e-15

    # storage
    t_list = [t]
    pos_list = [pos.copy()]
    vel_list = [vel.copy()]
    force_list = [F0.copy()]
    dipole_list = [p0.copy()]
    dt_list = []

    step_count = 0

    while t < T_final and step_count < maxstep:
        if t + dt > T_final:
            dt = T_final - t

        while True:
            # One big step of size dt
            pos_big, vel_big = step_damped_euler(pos, vel, dt, polarizabilities)

            # Two half steps of size dt/2
            h = dt / 2.0
            pos_half, vel_half = step_damped_euler(pos, vel, h, polarizabilities)
            pos_two, vel_two = step_damped_euler(
                pos_half, vel_half, h, polarizabilities
            )

            # Error estimate
            err = float(
                error_norm(pos_big, vel_big, pos_two, vel_two, rtol=rtol, atol=atol)
            )

            if err == 0.0:
                factor = max_factor
            else:
                # local error ~ O(dt^(p+1)), p=1 => exponent -1/(p+1) = -1/2
                factor = safety * err ** (-1.0 / (p_order + 1))
                factor = min(max_factor, max(min_factor, factor))

            dt_new = dt * factor

            if err <= 1.0:
                # Accept step: use the more accurate two-half-steps solution
                t = t + dt
                pos = pos_two
                vel = vel_two

                # Compute force and dipoles at the new accepted state for output
                F, p_i = compute_force_and_dipoles(pos, polarizabilities)

                t_list.append(t)
                pos_list.append(pos.copy())
                vel_list.append(vel.copy())
                force_list.append(F.copy())
                dipole_list.append(p_i.copy())
                dt_list.append(dt)

                step_count += 1
                dt = dt_new  # updated dt for next step
                print(
                    f"accepted step {step_count}, t = {t:.4e}, dt = {dt:.4e}, err = {err:.3e}"
                )
                break  # exit inner while and move forward in time
            else:
                # Reject step and shrink dt, try again from same (pos, vel, t)
                dt = dt_new
                print(f"rejected step, shrinking dt to {dt:.4e}, err = {err:.3e}")
                if dt < dt_min:
                    raise RuntimeError(
                        "dt underflow: cannot make progress with given tolerances"
                    )

    # Stack lists into arrays
    full_pos_arr = cp.stack(pos_list, axis=0)  # (n_steps+1, N, 3)
    velocity_arr = cp.stack(vel_list, axis=0)  # (n_steps+1, N, 3)
    forces_arr = cp.stack(force_list, axis=0)  # (n_steps+1, N, 3)
    dipoles_arr = cp.stack(dipole_list, axis=0)  # (n_steps+1, N, 3)
    dt_history = cp.asarray(dt_list)  # (n_steps,)

    # SAVE DATA
    cp.save("./data/position_data.npy", full_pos_arr)
    cp.save("./data/velocity_data.npy", velocity_arr)
    cp.save("./data/forces_data.npy", forces_arr)
    cp.save("./data/induced_dipoles.npy", dipoles_arr)
    cp.save("./data/dt_history.npy", dt_history)

    print("saved data")
