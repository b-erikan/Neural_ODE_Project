import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class TorchDataset(Dataset):

    def __init__(self,config, source_trajectory_ids):

        self.config = config
        self.input_path = self.config["processed_data_path"]
        self.source_trajectory_ids = set(source_trajectory_ids)

        self.samples = []
        self.t = None

        self._load_data()

    def _load_data(self):

        with h5py.File(self.input_path, "r") as f:
            for subtrajectory_id in f.keys():
                group = f[subtrajectory_id]
                source_id = group.attrs["source_trajectory_id"]

                #Only load subtrajectories belonging to selected source trajectories
                if source_id not in self.source_trajectory_ids:
                    continue

                time = group["observations/time"][:]
                qpos = group["observations/qpos"][:]
                qvel = group["observations/qvel"][:]
                #State vector: [qpos, qvel]
                x = np.concatenate((qpos, qvel), axis=1)

                #Every trajectory starts from t=0.
                time = time - time[0]

                # Time vector is shared for all samples so we can just get one. No need to execute this block every time.
                if self.t is None:
                    self.t = time
                elif not np.allclose(self.t, time):
                    #Make sure time vector is the same for all samples.
                    raise ValueError(
                        f"Inconsistent time vector in {subtrajectory_id}"
                    )
                x0 = x[0]
                self.samples.append((torch.tensor(x0, dtype=torch.float32),torch.tensor(x, dtype=torch.float32)))

        if len(self.samples) == 0:
            raise ValueError("No samples in the processed dataset!")
        self.t = torch.tensor(self.t, dtype=torch.float32) 

    def __len__(self):
        return len(self.samples) 

    def __getitem__(self, index):
        x0, x = self.samples[index]
        return {
            "x0": x0,
            "x": x
        }
