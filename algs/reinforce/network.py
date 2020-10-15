import torch
import torch.nn.functional as F
from torch import nn


class ReinforceBaseModel(nn.Module):
    def __init__(self, env, hidden_out):
        super(ReinforceBaseModel, self).__init__()
        self.n_channels = env.observation_space.shape[0]
        self.n_outputs = env.action_space.n
        self.hidden_out = hidden_out

        # actor's layer
        self.action_head = nn.Linear(hidden_out, self.n_outputs)
        # critic's layer
        self.value_head = nn.Linear(hidden_out, 1)

    def forward(self, x):
        # actor: choses action to take from state s_t
        # by returning probability of each action
        action_prob = F.softmax(self.action_head(x), dim=-1)

        # critic: evaluates being in the state s_t
        state_values = self.value_head(x)

        # return values for both actor and critic as a tupel of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t
        return action_prob, state_values


class Model_2x2(nn.Module):
    def __init__(self, env):
        super(Model_2x2, self).__init__()
        self.n_inputs = env.observation_space.shape[0]
        self.n_outputs = env.action_space.n
        self.conv1 = torch.nn.Conv2d(self.n_inputs, 8, kernel_size=1, stride=1)
        self.conv2 = torch.nn.Conv2d(8, 16, kernel_size=1, stride=1)
        self.conv3 = torch.nn.Conv2d(16, 16, kernel_size=2, stride=1)
        self.affine1 = nn.Linear(1 * 1 * 16, 16)

        # actor's layer
        self.action_head = nn.Linear(self.affine1.out_features, self.n_outputs)

        # critic's layer
        self.value_head = nn.Linear(self.affine1.out_features, 1)

    def forward(self, x):
        """
        forward of both actor and critic
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.affine1(x))

        # actor: choses action to take from state s_t
        # by returning probability of each action
        action_prob = F.softmax(self.action_head(x), dim=-1)

        # critic: evaluates being in the state s_t
        state_values = self.value_head(x)

        # return values for both actor and critic as a tupel of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t
        return action_prob, state_values


class Model_10x10(nn.Module):
    def __init__(self, env):
        super(Model_10x10, self).__init__()
        self.n_inputs = env.observation_space.shape[0]
        self.n_outputs = env.action_space.n

        self.conv1 = torch.nn.Conv2d(self.n_inputs, 32, kernel_size=1, stride=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.conv4 = torch.nn.Conv2d(64, 128, kernel_size=5, stride=1)

        self.affine1 = nn.Linear(128 * 2 * 2, 512)
        # actor's layer
        self.action_head = nn.Linear(self.affine1.out_features, self.n_outputs)

        # critic's layer
        self.value_head = nn.Linear(self.affine1.out_features, 1)

    def forward(self, orig_x):
        x = F.relu(self.conv1(orig_x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.affine1(x))

        # actor: choses action to take from state s_t
        # by returning probability of each action
        action_prob = F.softmax(self.action_head(x), dim=-1)

        # critic: evaluates being in the state s_t
        state_values = self.value_head(x)

        # return values for both actor and critic as a tupel of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t
        return action_prob, state_values


def get_model_class(params):
    if params.size == '2x2':
        return Model_2x2
    if params.size == '10x10':
        return Model_10x10
