import torch
import torch.nn.functional as F
from torch import nn


class Model_2x2(nn.Module):
    def __init__(self, env):
        super(Model_2x2, self).__init__()
        in_channels = env.observation_space.shape[0]
        out_features = env.action_space.n
        self.conv1 = nn.Conv2d(in_channels, 8, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=1, stride=1)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=2, stride=1)
        self.affine1 = nn.Linear(1 * 1 * 16, 16)
        self.action_head = nn.Linear(self.affine1.out_features, out_features)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.affine1(x))
        return self.action_head(x)


class Model_10x10(nn.Module):
    def __init__(self, env):
        super(Model_10x10, self).__init__()
        in_channels = env.observation_space.shape[0]
        out_features = env.action_space.n
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=5, stride=1)

        self.affine1 = nn.Linear(128 * 2 * 2, 512)
        self.action_head = nn.Linear(self.affine1.out_features, out_features)

    def forward(self, orig_x):
        orig_x = torch.from_numpy(orig_x)

        x = F.relu(self.conv1(orig_x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.affine1(x))

        return self.action_head(x)


def get_model_class(params):
    if params.size == '2x2':
        return Model_2x2
    if params.size == '10x10':
        return Model_10x10
