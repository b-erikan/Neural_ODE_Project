import mujoco
from dataset.trajectory import Trajectory
import numpy as np

class MujocoSimulator:

    def __init__(self, config):
        self.config = config
        xml_path = self.config["xml_path"]
        self.model, self.data = self.load_model(xml_path)
        self.set_dt(self.config["dt"])

    def load_model(self, xml_path=None):
        if xml_path is None:
            xml_path = self.config["xml_path"]
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)
        return model, data

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def step(self, ctrl=None):

        if ctrl is not None:
            self.data.ctrl[:] = ctrl

        mujoco.mj_step(self.model, self.data)

        return self.data.qpos.copy(), self.data.qvel.copy()

    def get_state(self):
        return self.data.qpos.copy(), self.data.qvel.copy()

    def set_state(self, qpos, qvel):
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        mujoco.mj_forward(self.model, self.data)

    def set_dt(self, dt):
        self.model.opt.timestep = dt

    def generate_trajectory_from_initial(self, trajectory_length_samples, initial_state=None, control_sequence=None):
        self.reset()
        #Returns a trajectory equal to to the control sequence. Last sample is left out.
        if initial_state is None:
            raise ValueError("Initial state must be provided.")

        if control_sequence is None:
            control_sequence = np.zeros((trajectory_length_samples, self.model.nu))
        
        self.set_state(initial_state[0], initial_state[1])
        trajectory = Trajectory()
        for ctrl in control_sequence:
            qpos, qvel = self.step(ctrl)
            trajectory.append_sample(self.data.time, qpos.copy(), qvel.copy(), ctrl)

        return trajectory

