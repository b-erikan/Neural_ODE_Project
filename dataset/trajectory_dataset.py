import h5py
from .trajectory import Trajectory
import numpy as np
from simulation.mujoco_simulator import MujocoSimulator

class TrajectoryDataset:

    def __init__(self, config):
        self.config = config
        self.num_trajectories = self.config["num_trajectories"]
        self.output_path = self.config["raw_data_path"]
        self.qpos_init_range = self.config["qpos_init_range"]
        self.qvel_init_range = self.config["qvel_init_range"]
        self.control_range = self.config["control_range"]

    def exists(self):
        try:
            with h5py.File(self.output_path, "r") as f:
                return True
        except OSError:
            return False

    def append_trajectory(self, trajectory):
        if len(trajectory.time) == 0:
            raise ValueError("Empty trajectory.")

        with h5py.File(self.output_path, "a") as f:

            trajectory_id = len([
                key for key in f.keys()
                if key.startswith("trajectory_")
            ])

            group_name = f"trajectory_{trajectory_id:03d}"
            traj_group = f.create_group(group_name)
            observations = traj_group.create_group("observations")
            observations.create_dataset("time", data=np.asarray(trajectory.time))
            observations.create_dataset("qpos", data=np.asarray(trajectory.qpos))
            observations.create_dataset("qvel", data=np.asarray(trajectory.qvel))
            actions = traj_group.create_group("actions")
            actions.create_dataset("ctrl", data=np.asarray(trajectory.ctrl))
            f.attrs["num_trajectories"] = trajectory_id + 1

            print(f"Appended {group_name}")

    def load_trajectory(self, trajectory_id):

        with h5py.File(self.output_path, "r") as f:
            group_name = f"trajectory_{trajectory_id:03d}"
            if group_name not in f:
                raise IndexError(f"Trajectory {trajectory_id} does not exist.")

            group = f[group_name]
            trajectory = Trajectory()
            time = group["observations"]["time"][:]
            qpos = group["observations"]["qpos"][:]
            qvel = group["observations"]["qvel"][:]
            ctrl = group["actions"]["ctrl"][:]

            for i in range(len(time)):
                trajectory.append_sample(
                    time[i],
                    qpos[i],
                    qvel[i],
                    ctrl[i]
                )

            return trajectory
    def generate(self, simulator: MujocoSimulator):
        # Create a new HDF5 file to store the dataset
        for i in range(self.num_trajectories):
            qpos = np.random.uniform(
                low=[
                    self.qpos_init_range["cart_position"][0],
                    self.qpos_init_range["pendulum_angle"][0],
                ],
                high=[
                    self.qpos_init_range["cart_position"][1],
                    self.qpos_init_range["pendulum_angle"][1],
                ],
            )
            qvel = np.random.uniform(
                low=[
                    self.qvel_init_range["cart_velocity"][0],
                    self.qvel_init_range["pendulum_angular_velocity"][0],
                ],
                high=[
                    self.qvel_init_range["cart_velocity"][1],
                    self.qvel_init_range["pendulum_angular_velocity"][1],
                ],
            )
            control_sequence = np.random.uniform(
                low=[self.control_range["cart_force"][0]],
                high=[self.control_range["cart_force"][1]],
                size=(self.config["raw_trajectory_size"], simulator.model.nu)
            )
            trajectory = simulator.generate_trajectory_from_initial(
                trajectory_length_samples=self.config["raw_trajectory_size"],
                initial_state=(qpos, qvel),
                control_sequence=control_sequence
            )
            self.append_trajectory(trajectory)
