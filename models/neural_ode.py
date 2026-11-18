import torch
import torch.nn as nn
from torchdiffeq import odeint


class NeuralODE(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_dim = config["hidden_dim"]
        self.state_dim = config["state_dim"] 
        self.input_dim = self.state_dim + 1 #2 inputs for wrapping pendulum angle sin/cos
        self.method = config["method"]
        self.loss_function = nn.MSELoss()

        #TODO: Currently 3 HL, 64 Neurons each. Will tune later.
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.state_dim)
        )

    def forward(self, t, x):

        x_encoded = torch.stack(
            (
                x[..., 0],
                torch.sin(x[..., 1]),
                torch.cos(x[..., 1]),
                x[..., 2],
                x[..., 3]
            ),
            dim=-1
        )

        return self.net(x_encoded)


    def integrate(self, x_init, t):
        return odeint(self, x_init, t, method=self.method)