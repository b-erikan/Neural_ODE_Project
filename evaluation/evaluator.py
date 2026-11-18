import numpy as np
import torch

from utils.trajectory import Trajectory
from evaluation.metrics import calculate_metrics


def generate_initial_state(config, seed=None):
    #Generate random initials within specified range.
    rng = np.random.default_rng(seed)
    qpos_ranges = config["dataset"]["qpos_init_range"]
    qvel_ranges = config["dataset"]["qvel_init_range"]
    initial_qpos = np.array([
        rng.uniform(*qpos_ranges["cart_position"]),
        rng.uniform(*qpos_ranges["pendulum_angle"])
    ])
    initial_qvel = np.array([
        rng.uniform(*qvel_ranges["cart_velocity"]),
        rng.uniform(*qvel_ranges["pendulum_angular_velocity"])
    ])
    return initial_qpos, initial_qvel


def generate_true_trajectory(simulator, initial_state, config):
    #Simulate to get ground truth.
    trajectory_length = config["dataset"]["raw_trajectory_size"]
    trajectory = simulator.generate_trajectory_from_initial(
        trajectory_length_samples=trajectory_length,
        initial_state=initial_state
    )
    return trajectory


def predict_trajectory(model, true_trajectory, simulator, device):
    x0 = np.concatenate((
        np.asarray(true_trajectory.qpos[0]),
        np.asarray(true_trajectory.qvel[0])
    ))
    time = np.asarray(true_trajectory.time, dtype=np.float32)
    time = time - time[0]


    #Evaluate on GPU, even if it is just one trajectory. Because why not?
    x0 = torch.as_tensor(
        x0,
        dtype=torch.float32,
        device=device
    )
    time = torch.as_tensor(
        time,
        dtype=torch.float32,
        device=device
    )
    model.eval()

    with torch.no_grad():
        prediction = model.integrate(x0, time)
    #Move back to CPU to convert to numpy.
    prediction = prediction.cpu().numpy()

    #Extract pos/vel
    qpos_pred = prediction[:, :2]
    qvel_pred = prediction[:, 2:]
    predicted_trajectory = Trajectory()
    #Convert to Trajectory class.
    for i in range(len(time)):
        predicted_trajectory.append_sample(
            true_trajectory.time[i],
            qpos_pred[i],
            qvel_pred[i],
            true_trajectory.ctrl[i]
        )

    return predicted_trajectory


def evaluate(model, simulator, config, device, seed=None):
    initial_state = generate_initial_state(config, seed)
    true_trajectory = generate_true_trajectory(simulator, initial_state, config)
    predicted_trajectory = predict_trajectory(model, true_trajectory, simulator, device)
    metrics = calculate_metrics(true_trajectory, predicted_trajectory)

    print("\nEvaluation metrics:")

    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    return {
        "true_trajectory": true_trajectory,
        "predicted_trajectory": predicted_trajectory,
        "metrics": metrics
    }