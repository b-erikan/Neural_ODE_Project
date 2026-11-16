import numpy as np
import matplotlib.pyplot as plt

from dataset.neural_ode_dataset import NeuralODEDataset
from dataset.trajectory_dataset import TrajectoryDataset


class Visualizer:
    #Some AI generated class for plotting stuff.

    def visualize(self, dataset):
        if isinstance(dataset, NeuralODEDataset):
            print("Neural ODE Dataset Passed")
            return

        if not isinstance(dataset, TrajectoryDataset):
            raise TypeError("data type mismatch")

        import h5py

        with h5py.File(dataset.output_path, "r") as f:
            num_trajectories = int(f.attrs.get("num_trajectories", 0))

        if num_trajectories == 0:
            print("Dataset is empty.")
            return

        trajectory_ids = np.random.choice(
            num_trajectories,
            size=10,
            replace=num_trajectories < 10,
        )
        fig, axes = plt.subplots(10, 4, figsize=(18, 30), sharex=True)
        state_plots = (
            ("Cart Position", "Position [m]", "qpos", 0),
            ("Pendulum Angle", "Angle [rad]", "qpos", 1),
            ("Cart Velocity", "Velocity [m/s]", "qvel", 0),
            ("Pendulum Angular Velocity", "Angular Velocity [rad/s]", "qvel", 1),
        )

        for row, trajectory_id in enumerate(trajectory_ids):
            trajectory = dataset.load_trajectory(int(trajectory_id))
            for column, (title, ylabel, state_name, state_index) in enumerate(state_plots):
                axis = axes[row, column]
                state = getattr(trajectory, state_name)
                axis.plot(trajectory.time, np.asarray(state)[:, state_index])
                if row == 0:
                    axis.set_title(title)
                if column == 0:
                    axis.set_ylabel(f"Trajectory {trajectory_id}\n{ylabel}")
                axis.grid(True)

        for axis in axes[-1, :]:
            axis.set_xlabel("Time [s]")

        fig.suptitle("Randomly Selected Raw Dataset Trajectories", fontsize=16)
        fig.tight_layout()
        plt.show()

    # --------------------------------------------------
    # Plot individual trajectory
    # --------------------------------------------------

    def plot_trajectory(self, trajectory):

        time = np.asarray(trajectory.time)
        qpos = np.asarray(trajectory.qpos)
        qvel = np.asarray(trajectory.qvel)
        ctrl = np.asarray(trajectory.ctrl)

        fig = plt.figure(figsize=(14, 10))

        gs = fig.add_gridspec(3, 2)

        # Cart position
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(time, qpos[:, 0])
        ax1.set_title("Cart Position")
        ax1.set_xlabel("Time [s]")
        ax1.set_ylabel("Position [m]")
        ax1.grid(True)

        # Cart velocity
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(time, qvel[:, 0])
        ax2.set_title("Cart Velocity")
        ax2.set_xlabel("Time [s]")
        ax2.set_ylabel("Velocity [m/s]")
        ax2.grid(True)

        # Pendulum angle
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(time, qpos[:, 1])
        ax3.set_title("Pendulum Angle")
        ax3.set_xlabel("Time [s]")
        ax3.set_ylabel("Angle [rad]")
        ax3.grid(True)

        # Pendulum angular velocity
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(time, qvel[:, 1])
        ax4.set_title("Pendulum Angular Velocity")
        ax4.set_xlabel("Time [s]")
        ax4.set_ylabel("Angular Velocity [rad/s]")
        ax4.grid(True)

        # Control input
        ax5 = fig.add_subplot(gs[2, :])

        if ctrl.ndim == 1:
            ctrl = ctrl[:, None]

        for i in range(ctrl.shape[1]):
            ax5.plot(time, ctrl[:, i], label=f"Control {i}")

        ax5.set_title("Control Input")
        ax5.set_xlabel("Time [s]")
        ax5.set_ylabel("Control")
        ax5.grid(True)
        ax5.legend()

        fig.suptitle("Trajectory Simulation Results", fontsize=16)

        plt.tight_layout()
        plt.show()
