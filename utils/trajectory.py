import numpy as np

class Trajectory:
    def __init__(self):
        self.time = []
        self.qpos = []
        self.qvel = []
        self.ctrl = []

    def append_sample(self, time, qpos, qvel, ctrl):
        self.time.append(time)
        self.qpos.append(np.array(qpos).copy())
        self.qvel.append(np.array(qvel).copy())
        self.ctrl.append(np.array(ctrl).copy())
        
    def clear(self):
        self.time = []
        self.qpos = []
        self.qvel = []
        self.ctrl = []

    def num_samples(self):
        return self.time.__len__()
