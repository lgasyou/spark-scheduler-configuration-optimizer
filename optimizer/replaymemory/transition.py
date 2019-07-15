from collections import namedtuple

import torch

from optimizer.hyperparameters import STATE_SHAPE

Transition = namedtuple('Transition', ('timestep', 'state', 'action', 'reward', 'nonterminal'))
blank_trans = Transition(0, torch.zeros(*STATE_SHAPE, dtype=torch.float32), None, 0, False)
