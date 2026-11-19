import numpy as np


def calculate_normalization_stats(raw_dataset, train_trajectory_ids):
    all_states = []
    all_derivatives = []

    for trajectory_id in train_trajectory_ids:
        trajectory = raw_dataset.load_trajectory(trajectory_id)
        qpos = np.asarray(trajectory.qpos, dtype=np.float64)
        qvel = np.asarray(trajectory.qvel, dtype=np.float64)
        time = np.asarray(trajectory.time, dtype=np.float64)

        # [cart_position, pendulum_angle, cart_velocity, angular_velocity]
        states = np.concatenate((qpos, qvel), axis=1)

        dt = np.diff(time)

        if np.any(dt <= 0):
            raise ValueError("Time vector is not monotonously increasing.")

        #Approximate derivatives with finite diff. for all states.
        derivatives = np.diff(states, axis=0) / dt[:, None]
        all_states.append(states)
        all_derivatives.append(derivatives)

    #Get std for the entire dataset.
    all_states = np.concatenate(all_states, axis=0)
    all_derivatives = np.concatenate(all_derivatives, axis=0)
    state_std = np.std(all_states, axis=0)
    deriv_std = np.std(all_derivatives, axis=0)

    #If the state is somehow constant, std will be 0. Later while normalizing we will have to divide by 0. Here is a safeguard for that.
    eps = 1e-6
    state_std = np.maximum(state_std, eps)
    deriv_std = np.maximum(deriv_std, eps)

    return state_std, deriv_std