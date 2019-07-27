import logging
import os
import random

import torch
from torch import optim

from optimizer.environment import AbstractEnv
from optimizer.hyperparameters import CUDA_DEVICES
from optimizer.nn import DQN
from optimizer.replaymemory import ReplayMemory
from optimizer.util import fileutil


class Agent(object):
    """
    The DRL agent of this project.
    """

    SAVE_FILENAME = './results/losses.txt'

    def __init__(self, args, env: AbstractEnv):
        self.logger = logging.getLogger(__name__)
        self.action_space = env.action_space()
        self.atoms = args.atoms
        self.V_min = args.V_min
        self.V_max = args.V_max
        self.support = torch.linspace(args.V_min, args.V_max, self.atoms).to(device=args.device)  # Support (range) of z
        self.delta_z = (args.V_max - args.V_min) / (self.atoms - 1)
        self.batch_size = args.batch_size
        self.n = args.multi_step
        self.discount = args.discount

        self.online_net = DQN(args, self.action_space).to(device=args.device)
        self.online_net = torch.nn.DataParallel(self.online_net, CUDA_DEVICES)
        if args.model and os.path.isfile(args.model):
            # Always load tensors onto CPU by default, will shift to GPU if necessary
            self.online_net.load_state_dict(torch.load(args.model, map_location='cpu'))
            self.logger.info('Agent model %s loaded.' % args.model)
        self.online_net.train()

        self.target_net = DQN(args, self.action_space).to(device=args.device)
        self.target_net = torch.nn.DataParallel(self.target_net, CUDA_DEVICES)
        self.update_target_net()
        self.target_net.train()
        for param in self.target_net.parameters():
            param.requires_grad = False

        self.optimiser = optim.Adam(self.online_net.parameters(), lr=args.lr, eps=args.adam_eps)

    # Resets noisy weights in all linear layers (of online net only)
    def reset_noise(self):
        self.online_net.module.reset_noise()

    # Acts based on single state (no batch)
    def act(self, state):
        with torch.no_grad():
            return (self.online_net(state.unsqueeze(0)) * self.support).sum(2).argmax(1).item()

    # Acts with an ε-greedy policy (used for evaluation only)
    # High ε can reduce evaluation scores drastically
    def act_e_greedy(self, state, epsilon=0.001) -> int:
        return random.randrange(self.action_space) if random.random() < epsilon else self.act(state)

    def learn(self, mem: ReplayMemory, time: int):
        # Sample transitions
        idxs, states, actions, returns, next_states, nonterminals, weights = mem.sample(self.batch_size)

        # Calculate current state probabilities (online network noise already sampled)
        log_ps = self.online_net(states, log=True)  # Log probabilities log p(s_t, ·; θonline)
        log_ps_a = log_ps[range(self.batch_size), actions]  # log p(s_t, a_t; θonline)

        with torch.no_grad():
            # Calculate nth next state probabilities
            pns = self.online_net(next_states)  # Probabilities p(s_t+n, ·; θonline)
            dns = self.support.expand_as(pns) * pns  # Distribution d_t+n = (z, p(s_t+n, ·; θonline))
            # Perform argmax action selection using online network: argmax_a[(z, p(s_t+n, a; θonline))]
            argmax_indices_ns = dns.sum(2).argmax(1)
            self.target_net.module.reset_noise()  # Sample new target net noise
            pns = self.target_net(next_states)  # Probabilities p(s_t+n, ·; θtarget)
            # Double-Q probabilities p(s_t+n, argmax_a[(z, p(s_t+n, a; θonline))]; θtarget)
            pns_a = pns[range(self.batch_size), argmax_indices_ns]

            # Compute Tz (Bellman operator T applied to z)
            # Tz = R^n + (γ^n)z (accounting for terminal states)
            Tz = returns.unsqueeze(1) + nonterminals * (self.discount ** self.n) * self.support.unsqueeze(0)
            Tz = Tz.clamp(min=self.V_min, max=self.V_max)  # Clamp between supported values
            # Compute L2 projection of Tz onto fixed support z
            b = (Tz - self.V_min) / self.delta_z  # b = (Tz - Vmin) / Δz
            l, u = b.floor().to(torch.int64), b.ceil().to(torch.int64)
            # Fix disappearing probability mass when l = b = u (b is int)
            l[(u > 0) * (l == u)] -= 1
            u[(l < (self.atoms - 1)) * (l == u)] += 1

            # Distribute probability of Tz
            m = states.new_zeros(self.batch_size, self.atoms)
            offset = torch.linspace(0, ((self.batch_size - 1) * self.atoms), self.batch_size).unsqueeze(1).expand(
                self.batch_size, self.atoms).to(actions)
            m.view(-1).index_add_(0, (l + offset).view(-1),
                                  (pns_a * (u.float() - b)).view(-1))  # m_l = m_l + p(s_t+n, a*)(u - b)
            m.view(-1).index_add_(0, (u + offset).view(-1),
                                  (pns_a * (b - l.float())).view(-1))  # m_u = m_u + p(s_t+n, a*)(b - l)

        loss = -torch.sum(m * log_ps_a, 1)  # Cross-entropy loss (minimises DKL(m||p(s_t, a_t)))
        str_loss = ','.join([str(float(i)) for i in loss])
        self.logger.info('Learnt with losses %s.' % str_loss)
        self.log_loss(str_loss, time)

        self.online_net.zero_grad()
        (weights * loss).mean().backward()  # Backpropagate importance-weighted minibatch loss
        self.optimiser.step()

        mem.update_priorities(idxs, loss.detach().cpu().numpy())  # Update priorities of sampled transitions

    def update_target_net(self) -> None:
        self.target_net.load_state_dict(self.online_net.state_dict())

    # Save model parameters on current device (don't test move model between devices)
    def save(self, path) -> None:
        filename = os.path.join(path, 'model.pth')
        torch.save(self.online_net.state_dict(), filename)
        self.logger.info('Agent model %s saved.' % filename)

    # Evaluates Q-value based on single state (no batch)
    def evaluate_q(self, state) -> float:
        with torch.no_grad():
            return (self.online_net(state.unsqueeze(0)) * self.support).sum(2).max(1)[0].item()

    def train(self):
        self.online_net.train()

    def eval(self):
        self.online_net.eval()

    def log_loss(self, loss: str, time: int, filename: str = None):
        data = '%d,%s' % (time, loss)
        fileutil.log_into_file(data, filename or self.SAVE_FILENAME)
