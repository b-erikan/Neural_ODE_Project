import h5py
from .trajectory import Trajectory
from .trajectory_dataset import TrajectoryDataset
import numpy as np

class NeuralODEDataset:

    def __init__(self, config):
        self.config = config
        self.input_path = self.config["raw_data_path"]
        self.output_path = self.config["processed_data_path"]

    def exists(self):
        try:
            with h5py.File(self.output_path, "r") as f:
                return True
        except OSError:
            return False
        
    def load_trajectory(self, raw_dataset : TrajectoryDataset):
        if not raw_dataset.exists():
            raise ValueError("Raw dataset could not be parsed.")
        
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


    def generate(self):
        trajectory = Trajectory()
        

        

