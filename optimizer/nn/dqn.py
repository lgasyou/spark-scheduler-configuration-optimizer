import torch
from torch import nn
from torch.nn import functional as F

from optimizer.nn.noisylinear import NoisyLinear


class DQN(nn.Module):

    def __init__(self, args, action_space):
        super().__init__()
        self.atoms = args.atoms
        self.action_space = action_space

        self.convs = nn.Sequential(nn.Conv2d(args.history_length, 16, [8, 4], stride=1), nn.ReLU(),
                                   nn.Conv2d(16, 32, [5, 1], stride=1), nn.ReLU(),
                                   nn.Conv2d(32, 64, [2, 1], stride=1), nn.ReLU(),
                                   nn.Conv2d(64, 128, [2, 1], stride=1), nn.ReLU())
        self.conv_output_size = 896

        self.fc_h_v = NoisyLinear(self.conv_output_size, args.hidden_size, std_init=args.noisy_std)
        self.fc_h_a = NoisyLinear(self.conv_output_size, args.hidden_size, std_init=args.noisy_std)
        self.fc_z_v = NoisyLinear(args.hidden_size, self.atoms, std_init=args.noisy_std)
        self.fc_z_a = NoisyLinear(args.hidden_size, action_space * self.atoms, std_init=args.noisy_std)

    def forward(self, x, log=False) -> torch.Tensor:
        x = self.convs(x)
        x = x.view(-1, self.conv_output_size)
        v = self.fc_z_v(F.relu(self.fc_h_v(x)))  # Value stream
        a = self.fc_z_a(F.relu(self.fc_h_a(x)))  # Advantage stream
        v, a = v.view(-1, 1, self.atoms), a.view(-1, self.action_space, self.atoms)
        q = v + a - a.mean(1, keepdim=True)  # Combine streams
        if log:  # Use log softmax for numerical stability
            q = F.log_softmax(q, dim=2)  # Log probabilities with action over second dimension
        else:
            q = F.softmax(q, dim=2)  # Probabilities with action over second dimension
        return q

    def reset_noise(self):
        for name, module in self.named_children():
            if 'fc' in name:
                module.reset_noise()
