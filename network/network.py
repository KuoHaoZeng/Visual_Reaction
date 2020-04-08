import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as tdist
from network.mobilenet import mobilenet_v2
import numpy as np

class NetForecaster(nn.Module):
    def __init__(self, batch_size=8, ND=3, hidden_size=256, input_size=300, input_num_frame=3):
        super(NetForecaster, self).__init__()
        # some hypeparameters
        self.batch_size = batch_size
        self.ND = ND
        self.input_size = input_size
        self.input_num_frame = input_num_frame
        self.hidden_size = hidden_size
        self.tanh = nn.Tanh()
        self.representation = None

        # allocate model parameters
        self.backbone = mobilenet_v2(num_classes=hidden_size)
        self.backbone_pva = nn.Sequential(
            nn.Linear(3 * ND + 2, 128),
            nn.Tanh(),
            nn.Linear(128, 256),
            nn.Tanh(),
            nn.Linear(256, hidden_size),
            nn.Tanh(),
        )
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.final_p = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.Tanh(),
            nn.Linear(hidden_size // 4, ND),
        )
        self.final_v = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.Tanh(),
            nn.Linear(hidden_size // 4, ND),
        )
        self.final_a = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.Tanh(),
            nn.Linear(hidden_size // 4, ND),
        )

    def forward(self, x, pos, vel, acc, agent_angle_y, agent_angle_xz):
        # resort the three frames along batch dimension
        x = x.view(self.batch_size, self.input_num_frame * 3, self.input_size, self.input_size)

        # do forward through CNN
        x = self.tanh(self.backbone(x))

        # do forward through FC on agent's position
        agent_angle_y = agent_angle_y.unsqueeze(1)
        agent_angle_xz = agent_angle_xz.unsqueeze(1)
        inp_pva = torch.cat([pos, vel, acc, agent_angle_y, agent_angle_xz], dim=1)
        out_pva = self.backbone_pva(inp_pva)

        # concat image_rep with pos_rep
        x = torch.cat((x, out_pva), dim=1)

        # do one more FC
        output = self.tanh(self.fc1(x))
        self.representation = output.clone()
        output_p = self.final_p(output).view(self.batch_size, self.ND)
        output_v = self.final_v(output).view(self.batch_size, self.ND)
        output_a = self.final_a(output).view(self.batch_size, self.ND)
        return output_p, output_v, output_a, self.representation

    def forecast(self, pos, v, a, t):
        output = []
        for _ in range(t):
            pos = pos + v
            v = v + a
            output.append(pos)
        output = torch.cat(output, 0)
        return output

class NetPolicy(nn.Module):
    def __init__(self, batch_size=8, n_dims=3, hidden_size=256, AD=41):
        super(NetPolicy, self).__init__()
        # some hypeparameters
        self.batch_size = batch_size
        self.n_dims = n_dims
        self.hidden_size = hidden_size
        self.AD = AD
        self.Tanh = nn.Tanh()
        self.representation = None

        # allocate model parameters
        self.backbone = nn.Sequential(
                nn.Linear(3 * n_dims, 64),
                nn.Tanh(),
                nn.Linear(64, 128),
                nn.Tanh(),
                nn.Linear(128, hidden_size),
                nn.Tanh(),
            )
        self.embed = nn.Linear(hidden_size * 2, hidden_size)
        self.final_v = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.Tanh(),
            nn.Linear(hidden_size // 4, 1),
        )
        self.final_a = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.Tanh(),
            nn.Linear(hidden_size // 4, n_dims),
        )
        self.final_a2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.Tanh(),
            nn.Linear(hidden_size // 4, n_dims),
            nn.Softplus(),
        )

    def forward(self, x, r):
        self.representation = self.backbone(x)
        if r.shape[0] != len(self.representation):
            r = r.repeat(len(self.representation), 1)
        self.representation = torch.cat((r, self.representation), dim=1)
        self.representation = self.Tanh(self.embed(self.representation))
        action_dist = self.final_a(self.representation)
        action_dist_2 = self.final_a2(self.representation) + 0.001
        value = self.final_v(self.representation)
        return value, action_dist, action_dist_2

class DroneModel(nn.Module):
    def __init__(self, batch_size=8, n_dims=1, hidden_size=0, frequency=0.02, cuda=False, gpu_id=0):
        super(DroneModel, self).__init__()
        # some hypeparameters
        self.batch_size = batch_size
        self.n_dims = n_dims
        self.vel = torch.zeros(batch_size, n_dims)
        self.frequency = frequency
        self.cuda_enable = cuda
        if self.cuda_enable:
            self.vel = self.vel.cuda(gpu_id)
        self.gpu_id = gpu_id

    def forward(self, x):
        x = x.view(self.batch_size, -1, self.n_dims)
        if self.vel.shape[0] != self.batch_size:
            self.vel = self.vel.repeat(self.batch_size, 1)
        self.vel += x[:, 1, :]
        s = (x[:, 0, :] + self.vel * self.frequency)
        return s

    def reset_vel(self):
        self.vel = torch.zeros(self.batch_size, self.n_dims)
        if self.cuda_enable:
            self.vel = self.vel.cuda(self.gpu_id)
