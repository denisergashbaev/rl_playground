from collections import namedtuple

import torch
from stable_baselines3.common import utils as sb3_utils
import torch.nn as nn
import numpy as np
from torch import optim
from torch.autograd import Variable
from torch.distributions import Categorical

from algs.reinforce import reinforce_helper, network
from utils import helper

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


class Model(nn.Module):
    def __init__(self, env, params, device):
        super(Model, self).__init__()
        self.state_space = env.observation_space.shape[0]
        self.action_space = env.action_space.n

        self.model = network.get_model_class(params)(env)
        self.gamma = params.gamma

        self.device = device

        self.policy_history = Variable(torch.Tensor())
        self.reward_episode = []

        self.reward_history = []
        self.loss_history = []

    def forward(self, x):
        x = x.to(self.device)
        return self.model(x)


class Bob:
    def __init__(self, env, params, device):
        self.model = Model(env, params, device).to(device)
        self.optimizer = optim.Adam(self.model.model.parameters(), lr=params.learning_rate)
        self.params = params
        self.device = device
        self.env = env

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        actions, state_value = self.model(state)

        # create a categorical distribution over the list of probabilities of actions
        c = Categorical(actions)

        # and sample an action using the distribution
        action = c.sample()

        if len(self.model.policy_history) > 0:
            self.model.policy_history = torch.cat([self.model.policy_history, actions.cpu().detach().numpy().log_prob(action).reshape(1)])
        else:
            self.model.policy_history = actions.cpu().detach().numpy().log_prob(action).reshape(1)
        return action

    def update_policy(self):
        R = 0
        rewards = []

        for r in self.model.reward_episode[::-1]:
            R = r + self.model.gamma * R
            rewards.insert(0, R)

        rewards = torch.tensor(rewards).float()
        rewards = (rewards - rewards.mean()) / (rewards.std() + + np.finfo(np.float).eps)

        # calculate loss
        loss = (torch.sum(torch.mul(self.model.policy_history, rewards).mul(-1), -1))

        # update network weights
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # save and initialize episode history counters
        self.model.loss_history.append(loss.item())
        self.model.reward_history.append(np.sum(self.model.reward_episode))
        self.model.policy_history = Variable(torch.Tensor())
        self.model.reward_episode = []


def run(params):
    device, use_cuda = helper.get_pytorch_device()
    sb3_utils.set_random_seed(params.seed, using_cuda=use_cuda)
    env = helper.make_env(params)

    bob = Bob(env, params, device)

    ep_no = 0
    while ep_no < params.num_episodes:
        ep_no += 1

        # reset environment and episode reward
        state = env.reset()
        ep_reward = 0

        # for each episode, only run 10_000 steps so that we don't
        # infinite loop while learning
        for t in range(10_000):

            # select action from policy
            action = bob.select_action(state)
            # take the action
            state, reward, done, info = env.step(action.item())

            # update bob's reward
            bob.model.reward_episode.append(reward)

            ep_reward += reward
            if done:
                break

        # perform backprop
        bob.update_policy()

        if ep_no % 50 == 0:
            print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(ep_no, t, ep_reward))

if __name__ == '__main__':
    parser = reinforce_helper.parse_args()
    params = helper.get_parsed_params(parser)
    run(params)
