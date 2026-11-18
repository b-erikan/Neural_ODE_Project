import h5py
from utils.trajectory import Trajectory
from .trajectory_dataset import TrajectoryDataset
import numpy as np

class NeuralODEDataset:

    def __init__(self, config):
        self.config = config
        self.input_path = self.config["raw_data_path"]
        self.output_path = self.config["processed_data_path"]
        self.window_size=self.config["window_size"]
        self.stride=self.config["stride"]
        self.num_trajectories=self.config["num_trajectories"]

    def exists(self):
        try:
            with h5py.File(self.output_path, "r") as f:
                return True
        except OSError:
            return False

    def load_subtrajectories(self, trajectory_id: int):

        subtrajectories = []

        with h5py.File(self.output_path, "r") as f:

            num_subtrajectories = f.attrs["num_subtrajectories"]

            for sub_id in range(num_subtrajectories):

                group_name = f"subtrajectory_{sub_id:03d}"
                group = f[group_name]

                #Scan through all subtrajectories, return the ones that belong to the given trajectory ID.
                source_id = group.attrs["source_trajectory_id"]

                if source_id != trajectory_id:
                    continue

                trajectory = Trajectory()
                time = group["observations"]["time"][:]
                qpos = group["observations"]["qpos"][:]
                qvel = group["observations"]["qvel"][:]
                ctrl = group["actions"]["ctrl"][:]

                for i in range(len(time)):
                    trajectory.append_sample(time[i],qpos[i],qvel[i],ctrl[i])

                subtrajectories.append(trajectory)

        if len(subtrajectories) == 0:
            raise IndexError(f"No subtrajectories found for trajectory {trajectory_id}.")

        return subtrajectories

    def generate(self):

        subtrajectory_id = 0

        with h5py.File(self.input_path, "r") as raw_file, h5py.File(self.output_path, "w") as processed_file:

            num_trajectories = raw_file.attrs["num_trajectories"]

            processed_file.attrs["window_size"] = self.window_size
            processed_file.attrs["stride"] = self.stride
            processed_file.attrs["num_source_trajectories"] = num_trajectories

            for traj_id in range(num_trajectories):

                group_name = f"trajectory_{traj_id:03d}"

                raw_group = raw_file[group_name]

                # Load original trajectory into memory once.
                time = raw_group["observations"]["time"][:]
                qpos = raw_group["observations"]["qpos"][:]
                qvel = raw_group["observations"]["qvel"][:]
                ctrl = raw_group["actions"]["ctrl"][:]

                trajectory_length = len(time)

                if trajectory_length < self.window_size:
                    continue

                # Sliding window. Every subtrajectory is basically another valid trajectory where first sample of the trajectory is the inidial condition.
                for start_idx in range(
                    0,
                    trajectory_length - self.window_size + 1,
                    self.stride
                ):

                    end_idx = start_idx + self.window_size

                    subtrajectory_name = (
                        f"subtrajectory_{subtrajectory_id:03d}"
                    )

                    sub_group = processed_file.create_group(subtrajectory_name)
                    observations = sub_group.create_group("observations")
                    observations.create_dataset("time", data=time[start_idx:end_idx])
                    observations.create_dataset("qpos", data=qpos[start_idx:end_idx])
                    observations.create_dataset("qvel", data=qvel[start_idx:end_idx])
                    actions = sub_group.create_group("actions")
                    actions.create_dataset("ctrl", data=ctrl[start_idx:end_idx])

                    # Metadata. Store the complete trajectory from which this subtrajectory is generated.
                    # We keep index info and main trajectory ID here in case we need it later for visual matching etc. 
                    # Currently this info is not needed, but stored anyway.
                    
                    sub_group.attrs["source_trajectory_id"] = traj_id
                    sub_group.attrs["start_idx"] = start_idx
                    sub_group.attrs["end_idx"] = end_idx

                    subtrajectory_id += 1

            processed_file.attrs["num_subtrajectories"] = subtrajectory_id

        print(f"Generated {subtrajectory_id} subtrajectories.")
        

        

