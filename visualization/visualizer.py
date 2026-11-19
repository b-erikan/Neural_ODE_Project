import numpy as np
import matplotlib.pyplot as plt


class Visualizer:
    #Completely AI generated visualization class. I'm not trying to learn how to plot stuff so it's better to let AI handle the work here.
    def __init__(self):
        pass

    def plot_trajectories(self, true_trajectory, predicted_trajectory):
        """Plot ground-truth and predicted trajectories in the same figure."""
        
        # Convert trajectory data to NumPy arrays.
        true_time = np.asarray(true_trajectory.time)
        true_qpos = np.asarray(true_trajectory.qpos)
        true_qvel = np.asarray(true_trajectory.qvel)

        predicted_time = np.asarray(predicted_trajectory.time)
        predicted_qpos = np.asarray(predicted_trajectory.qpos)
        predicted_qvel = np.asarray(predicted_trajectory.qvel)

        # Create figure with four subplots.
        fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

        # Cart position.
        axes[0].plot(true_time, true_qpos[:, 0], label="MuJoCo")
        axes[0].plot(predicted_time, predicted_qpos[:, 0], "--", label="Neural ODE")
        axes[0].set_ylabel("Position [m]")
        axes[0].set_title("Cart Position")
        axes[0].legend()
        axes[0].grid(True)

        # Pendulum angle.
        axes[1].plot(true_time, true_qpos[:, 1], label="MuJoCo")
        axes[1].plot(predicted_time, predicted_qpos[:, 1], "--", label="Neural ODE")
        axes[1].set_ylabel("Angle [rad]")
        axes[1].set_title("Pendulum Angle")
        axes[1].legend()
        axes[1].grid(True)

        # Cart velocity.
        axes[2].plot(true_time, true_qvel[:, 0], label="MuJoCo")
        axes[2].plot(predicted_time, predicted_qvel[:, 0], "--", label="Neural ODE")
        axes[2].set_ylabel("Velocity [m/s]")
        axes[2].set_title("Cart Velocity")
        axes[2].legend()
        axes[2].grid(True)

        # Pendulum angular velocity.
        axes[3].plot(true_time, true_qvel[:, 1], label="MuJoCo")
        axes[3].plot(predicted_time, predicted_qvel[:, 1], "--", label="Neural ODE")
        axes[3].set_ylabel("Angular velocity [rad/s]")
        axes[3].set_xlabel("Time [s]")
        axes[3].set_title("Pendulum Angular Velocity")
        axes[3].legend()
        axes[3].grid(True)

        fig.suptitle("MuJoCo vs Neural ODE Trajectory Comparison", fontsize=15)
        fig.tight_layout()

        plt.show()

    def plot_phase_portraits(self, true_trajectory, predicted_trajectory):
        """Plot cart and pendulum states in their 2D phase planes."""
        true_qpos = np.asarray(true_trajectory.qpos)
        true_qvel = np.asarray(true_trajectory.qvel)
        predicted_qpos = np.asarray(predicted_trajectory.qpos)
        predicted_qvel = np.asarray(predicted_trajectory.qvel)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].plot(true_qpos[:, 0], true_qvel[:, 0], label="MuJoCo")
        axes[0].plot(
            predicted_qpos[:, 0],
            predicted_qvel[:, 0],
            "--",
            label="Neural ODE",
        )
        axes[0].set_xlabel("Cart position [m]")
        axes[0].set_ylabel("Cart velocity [m/s]")
        axes[0].set_title("Cart Phase Portrait")
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(true_qpos[:, 1], true_qvel[:, 1], label="MuJoCo")
        axes[1].plot(
            predicted_qpos[:, 1],
            predicted_qvel[:, 1],
            "--",
            label="Neural ODE",
        )
        axes[1].set_xlabel("Pendulum angle [rad]")
        axes[1].set_ylabel("Angular velocity [rad/s]")
        axes[1].set_title("Pendulum Phase Portrait")
        axes[1].legend()
        axes[1].grid(True)

        fig.suptitle("MuJoCo vs Neural ODE Phase Portraits", fontsize=15)
        fig.tight_layout()

        plt.show()