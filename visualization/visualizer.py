from datetime import datetime
from pathlib import Path
import time

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
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = Path("results/graphics") / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / "trajectory_comparison.png")

        plt.show()

    def run_realtime(
        self,
        simulator,
        neural_ode,
        initial_state,
        duration=5.0,
        control_sequence=None,
        plot_every=5,
        one_step=False,
    ):
        """Run MuJoCo and the Neural ODE side by side with a live viewer and plot."""
        import mujoco.viewer
        import torch

        dt = simulator.model.opt.timestep
        if duration <= 0 or plot_every < 1:
            raise ValueError("duration and plot_every must be positive.")

        n_steps = int(round(duration / dt))
        if n_steps < 1:
            raise ValueError("duration must be at least one MuJoCo timestep.")

        if control_sequence is not None:
            control_sequence = np.asarray(control_sequence)
            expected_shape = (n_steps, simulator.model.nu)
            if control_sequence.shape != expected_shape:
                raise ValueError(
                    f"control_sequence must have shape {expected_shape}, "
                    f"got {control_sequence.shape}."
                )

        initial_qpos, initial_qvel = initial_state
        initial_state_vector = np.concatenate((initial_qpos, initial_qvel))
        if initial_state_vector.shape != (4,):
            raise ValueError("initial_state must contain two positions and two velocities.")

        simulator.reset()
        simulator.set_state(initial_qpos, initial_qvel)
        neural_ode.eval()
        device = next(neural_ode.parameters()).device
        y_nn = torch.as_tensor(initial_state_vector, dtype=torch.float32, device=device)
        time_pair = torch.tensor([0.0, dt], dtype=torch.float32, device=device)

        labels = ("Cart position [m]", "Pendulum angle [rad]",
                  "Cart velocity [m/s]", "Pendulum angular velocity [rad/s]")
        fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
        lines = []
        for index, ax in enumerate(axes.flat):
            true_line, = ax.plot([], [], "g-", label="MuJoCo")
            neural_line, = ax.plot([], [], "r--", label="Neural ODE")
            ax.set_ylabel(labels[index])
            ax.set_xlim(0, n_steps * dt)
            ax.grid(True)
            lines.append((true_line, neural_line))
        axes[0, 0].legend(loc="upper right")
        axes[1, 0].set_xlabel("Time [s]")
        axes[1, 1].set_xlabel("Time [s]")
        fig.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)

        times = [0.0]
        true_states = [initial_state_vector.copy()]
        neural_states = [initial_state_vector.copy()]
        late_steps = 0

        with mujoco.viewer.launch_passive(simulator.model, simulator.data) as viewer:
            viewer.sync()
            input("Press Enter to start the real-time simulation...")
            start_time = time.perf_counter()

            for step_index in range(1, n_steps + 1):
                if not viewer.is_running():
                    break

                neural_input = y_nn
                if one_step:
                    qpos, qvel = simulator.get_state()
                    neural_input = torch.as_tensor(
                        np.concatenate((qpos, qvel)),
                        dtype=torch.float32,
                        device=device,
                    )
                with torch.no_grad():
                    y_nn = neural_ode.integrate(neural_input, time_pair)[-1]

                with viewer.lock():
                    control = (
                        None
                        if control_sequence is None
                        else control_sequence[step_index - 1]
                    )
                    simulator.step(control)
                viewer.sync()

                qpos, qvel = simulator.get_state()
                times.append(step_index * dt)
                true_states.append(np.concatenate((qpos, qvel)))
                neural_states.append(y_nn.detach().cpu().numpy().copy())

                if step_index % plot_every == 0:
                    true_array = np.asarray(true_states)
                    neural_array = np.asarray(neural_states)
                    for column, (true_line, neural_line) in enumerate(lines):
                        true_line.set_data(times, true_array[:, column])
                        neural_line.set_data(times, neural_array[:, column])
                        axes.flat[column].relim()
                        axes.flat[column].autoscale_view(scalex=False)
                    plt.pause(0.001)

                wait = start_time + step_index * dt - time.perf_counter()
                if wait > 0:
                    time.sleep(wait)
                else:
                    late_steps += 1

        print(f"Done. {late_steps}/{len(times) - 1} steps missed their deadline.")
        plt.ioff()
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
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = Path("results/graphics") / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / "phase_portrait.png")

        plt.show()