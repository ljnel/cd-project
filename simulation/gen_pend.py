import pinocchio as pin
import numpy as np


def create_model():
    model = pin.Model()

    mass = 1.0   # kg
    length = 1.0  # meters

    joint_id = model.addJoint(0, 
                              pin.JointModelRX(),
                              pin.SE3.Identity(),
                              "pendulum_joint"
                              )
    inertia = pin.Inertia(mass,
                          np.array([0., 0., length / 2]),
                          pin.Inertia.FromCylinder(mass, length / 20, length).inertia)
    model.appendBodyToJoint(joint_id, 
                            inertia,
                            pin.SE3.Identity())
    model.gravity.linear[2] = -9.81

    return model


def simulate_inverted_pendulum(model, physical_params, simulation_time, dt):
    init_q, init_v, damping_coeff = physical_params  # initial angle, angular velocity and damping coefficient

    data = model.createData()

    q = np.array([init_q])
    v = np.array([init_v])

    trajectory = []

    num_steps = int(simulation_time / dt)
    for i in range(num_steps):
        trajectory.append([q.copy(), v.copy()])

        tau = np.array([-damping_coeff * v])
        a = pin.aba(model, data, q, v, tau)

        v += a * dt
        q = pin.integrate(model, q, v * dt)

    return np.array(trajectory).T


def gen_trajs(n_simulations, t, dt, q_lim, v_lim, b_lim):

    times = np.arange(0., t, step=dt)
    trajs = np.zeros((n_simulations, 2, int(t / dt)))
    model = create_model()

    for i in range(n_simulations):

        q = np.random.uniform(-q_lim, q_lim)
        v = np.random.uniform(-v_lim, v_lim)
        b = np.random.uniform(0, b_lim)
        params = (q, v, b)

        traj = simulate_inverted_pendulum(
            model, params, t, dt
        )
        trajs[i, :, :] = traj

    trajs = np.array(trajs).reshape(n_simulations, 2, -1)
    return times.astype(np.float32), trajs.astype(np.float32)


def gen_one_traj(t, dt, q, v, b):

    times = np.arange(0., t, step=dt)
    trajs = np.zeros((1, 2, int(t / dt)))
    model = create_model()

    for i in range(1):

        q = np.array(q)
        v = np.array(v)
        b = np.array(b)
        params = (q, v, b)

        traj = simulate_inverted_pendulum(
            model, params, t, dt
        )
        trajs[i, :, :] = traj

    trajs = np.array(trajs).reshape(1, 2, -1)
    return times.astype(np.float32), trajs.astype(np.float32)
