import numpy as np

def check_trajectories(traj1, traj2):
    qpos1 = np.asarray(traj1.qpos)
    qpos2 = np.asarray(traj2.qpos)

    qvel1 = np.asarray(traj1.qvel)
    qvel2 = np.asarray(traj2.qvel)

    if traj1.num_samples() != traj2.num_samples():
        raise ValueError("Different number of samples.")

    if not np.array_equal(traj1.time, traj2.time):
        raise ValueError("Different time vectors.")

    if qpos1.shape != qpos2.shape:
        raise ValueError("Position arrays have different shapes.")

    if qvel1.shape != qvel2.shape:
        raise ValueError("Velocity arrays have different shapes.")



def calculate_metrics(traj_true, traj_pred):
    check_trajectories(traj_true, traj_pred)

    x_true = np.concatenate((traj_true.qpos, traj_true.qvel),axis=1)
    x_pred = np.concatenate((traj_pred.qpos, traj_pred.qvel),axis=1)

    error = x_pred - x_true #calculate error vector state wise.
    mse = np.mean(error**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(error))

    return {
        "MSE": float(mse),
        "RMSE": float(rmse),
        "MAE": float(mae)
    }