import argparse

import gym
import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3.common import utils as sb3_utils
from torch import nn
from torch import optim
from torch.distributions import Categorical

# Sources:
# overview https://www.endtoend.ai/blog/pytorch-pg-implementations/
# https://github.com/higgsfield/RL-Adventure-2/blob/master/1.actor-critic.ipynb
# https://github.com/pytorch/examples/blob/master/reinforcement_learning/actor_critic.py
parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
parser.add_argument('--gamma', type=float, default=.99, help='discount factor (default: 0.99)')
parser.add_argument('--batch_size', type=int, default=500, help='batch size')
parser.add_argument('--num_episodes', type=int, default=20000, help='number of episodes')
parser.add_argument('--normalize_rewards', action='store_true', help='with/without reward normalization')
parser.add_argument('--learning_rate', type=float, default=3e-2, help='learning rate')
parser.add_argument('--scaling_factor', type=float, default=1, help='learning rate')
parser.add_argument('--render', action='store_true', help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')

args = parser.parse_args()

print("pyTorch version:\t{}".format(torch.__version__))
print("Arguments used: {}".format(args))

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

env_name = "CartPole-v1"

env = gym.make(env_name)

env.seed(args.seed)
sb3_utils.set_random_seed(args.seed, using_cuda=use_cuda)


class Model(nn.Module):

    def __init__(self, env):
        super(Model, self).__init__()
        self.n_inputs = env.observation_space.shape[0]
        self.n_outputs = env.action_space.n
        hidden_size = 128
        # Define network
        self.affine1 = nn.Linear(self.n_inputs, hidden_size)

        # actor's layer
        self.action_head = nn.Linear(hidden_size, self.n_outputs)

        # critic's layer
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = torch.FloatTensor(x).to(device)

        """
        forward of both actor and critic
        """
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


model = Model(env).to(device)
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

ep_no = 0
total_rewards = []


def discount_rewards(rewards, gamma):
    G = []
    R = 0
    # sum the rewards (taking causality into account)
    for reward in rewards[::-1]:
        R = reward + gamma * R
        G.insert(0, R)
    G = torch.tensor(G).to(device)
    #if args.normalize_rewards:
    #    # normalizing the returns:
    G = (G - G.mean()) / (G.std() + np.finfo(np.float32).eps.item())
    return G


while ep_no < args.num_episodes:
    ep_no += 1
    batch_pred_vals, batch_log_probs, batch_returns = [], [], []

    while len(batch_pred_vals) < args.batch_size:
        # unroll the policy
        state = env.reset()
        rewards = []
        for time_steps in range(10_000):
            action_probs, pred_val = model(state)
            c = Categorical(action_probs)
            action = c.sample()
            s_tp1, reward, done, _ = env.step(action.item())

            batch_pred_vals.append(pred_val)
            batch_log_probs.append(c.log_prob(action))
            rewards.append(reward)
            state = s_tp1
            if done:
                break
        total_rewards.append(sum(rewards))

        batch_returns += discount_rewards(rewards, args.gamma)

    # log results
    if ep_no % args.log_interval == 0:
        last_no_of_episodes = 100
        avg_rewards = np.mean(total_rewards[-last_no_of_episodes:])
        # Print running average
        print("\rEp: {} Last episode length: {}, Average of last {} episodes: {:.2f}"
              .format(ep_no, time_steps, last_no_of_episodes, avg_rewards))

    batch_pred_vals = torch.stack(batch_pred_vals).squeeze()
    batch_returns = torch.stack(batch_returns)
    policy_loss = (-torch.stack(batch_log_probs) * (batch_returns - batch_pred_vals).detach()).mean()
    baseline_loss = F.smooth_l1_loss(batch_pred_vals, batch_returns, reduction='mean')
    loss = policy_loss + args.scaling_factor * baseline_loss

    # reset the gradients
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
