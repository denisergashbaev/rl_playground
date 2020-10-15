from collections import namedtuple

import torch
import torch.nn.functional as F
from stable_baselines3.common import utils as sb3_utils
from torch import optim
from torch.distributions import Categorical

from algs.reinforce import reinforce_helper, network
from utils import helper

# Adapted from source:
# https://github.com/pytorch/examples/blob/master/reinforcement_learning/actor_critic.py


SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


def run(params):
    device, use_cuda = helper.get_pytorch_device()
    sb3_utils.set_random_seed(params.seed, using_cuda=use_cuda)
    env = helper.make_env(params)
    writer = helper.get_summary_writer(__file__, params)

    m = network.get_model_class(params)

    class Model(m):
        def __init__(self, env):
            super(Model, self).__init__(env)
            # action & reward buffer
            self.saved_actions = []
            self.rewards = []

    model = Model(env).to(device)
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)

    ep_no = 0
    total_rewards = []

    while ep_no < params.num_episodes:
        ep_no += 1

        # reset environment and episode reward
        state = env.reset()
        ep_reward = 0

        # for each episode, only run 10_000 steps so that we don't
        # infinite loop while learning
        for t in range(10_000):

            # select action from policy
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
            probs, state_value = model(state)

            # create a categorical distribution over the list of probabilities of actions
            c = Categorical(probs)

            # and sample an action using the distribution
            action = c.sample()

            # save to action buffer
            model.saved_actions.append(SavedAction(c.log_prob(action), state_value))

            # take the action
            state, reward, done, info = env.step(action.item())

            model.rewards.append(reward)
            ep_reward += reward
            if done:
                break
        total_rewards.append(ep_reward)

        reinforce_helper.log_results(ep_no, total_rewards, info, t, params, '')
        helper.add_scalar(writer, 'episode_reward', ep_reward, ep_no, params)
        helper.add_scalar(writer, 'episode_covered_count', info['covered_count'], ep_no, params)
        helper.add_scalar(writer, 'episode_steps_count', info['steps_count'], ep_no, params)

        # perform backprop
        saved_actions = model.saved_actions
        policy_losses = []  # list to save actor (policy) loss
        value_losses = []  # list to save critic (value) loss

        returns = reinforce_helper.discount_rewards(model.rewards, params)

        for (log_prob, value), R in zip(saved_actions, returns):
            advantage = R - value.item()

            # calculate actor (policy) loss
            policy_losses.append(-log_prob * advantage)

            # calculate critic (value) loss using L1 smooth loss
            value_losses.append(F.smooth_l1_loss(value.squeeze(0), torch.tensor([R]).to(device)))

        # reset gradients
        optimizer.zero_grad()

        # sum up all the values of policy_losses and value_losses
        loss = torch.stack(policy_losses).mean() + params.scaling_factor * torch.stack(value_losses).mean()

        # perform backprop
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), 0.0001)
        optimizer.step()

        # reset rewards and action buffer
        del model.rewards[:]
        del model.saved_actions[:]
    helper.close_summary_writer(writer)


if __name__ == '__main__':
    parser = reinforce_helper.parse_args()
    params = helper.get_parsed_params(parser)
    run(params)