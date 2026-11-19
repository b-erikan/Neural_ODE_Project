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
        #Initialize std values as 1. We'll give them as inputs later. 
        self.register_buffer("state_std", torch.ones(self.state_dim))
        self.register_buffer("deriv_std", torch.ones(self.state_dim))

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

    def set_normalization_stats(self, state_std, deriv_std):
        state_std = torch.as_tensor(state_std,dtype=self.state_std.dtype,device=self.state_std.device)
        deriv_std = torch.as_tensor(deriv_std,dtype=self.deriv_std.dtype,device=self.deriv_std.device)
        self.state_std.copy_(state_std)
        self.deriv_std.copy_(deriv_std)

    def forward(self, t, x):

        x_encoded = torch.stack(
            (
                #UPDATE: Normalized input. 
                x[..., 0] / self.state_std[0],
                torch.sin(x[..., 1]),
                torch.cos(x[..., 1]),
                x[..., 2] / self.state_std[2],
                x[..., 3] / self.state_std[3]
            ),
            dim=-1
        )

        return self.net(x_encoded) * self.deriv_std #Return the de-normalized version to the integrator.


    def integrate(self, x_init, t):
        return odeint(self, x_init, t, method=self.method)