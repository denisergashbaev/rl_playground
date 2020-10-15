from collections import namedtuple

import torch
import torch.nn.functional as F
from stable_baselines3.common import utils as sb3_utils
from torch import optim
from torch.distributions import Categorical

from common.env import ColoringEnv
from algs.curriculum import network
from algs.curriculum.network import ModelAlice_2x2
from algs.reinforce import reinforce_helper
from utils import helper
import numpy as np

# Adapted from source:
# https://github.com/pytorch/examples/blob/master/reinforcement_learning/actor_critic.py


SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

class AliceRunner:
    def __init__(self, bob, params, env, device, writer):
        self.bob = bob
        self.params = params
        self.env = env
        self.device = device
        self.writer = writer
        self.model = ModelAlice_2x2(self.env).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.params.learning_rate)

    def run(self):
        ep_no = 0
        total_rewards = []

        while ep_no < self.params.num_episodes:
            ep_no += 1

            # reset environment and episode reward
            state = self.env.reset()
            ep_reward = 0

            # for each episode, only run 10_000 steps so that we don't
            # infinite loop while learning
            info = {'covered_count': 1, 'steps_count': 0}
            for t in range(10_000):

                # select action from policy
                state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
                probs, state_value = self.model(state)

                # create a categorical distribution over the list of probabilities of actions
                c = Categorical(probs)

                # and sample an action using the distribution
                action = c.sample()

                # save to action buffer
                self.model.saved_actions.append(SavedAction(c.log_prob(action), state_value))

                action_item = action.item()
                #action_item = np.random.randint(0, 2)
                done = False
                if ModelAlice_2x2.CONTINUE == action_item:
                    # take __RANDOM__ action
                    state, _, done, info = self.env.step(np.random.randint(0, 4))
                    reward = 0
                done |= ModelAlice_2x2.STOP == action_item

                if done:
                    alice_state = state
                    if torch.is_tensor(alice_state):
                       alice_state = alice_state.squeeze(0).cpu().numpy()
                    if params.render:
                        self.env.render()
                    bob_info = self.bob.run(alice_state, ep_no)
                    reward = int(bob_info['steps_count']) - np.count_nonzero(alice_state[ColoringEnv.channel_stitch])

                self.model.rewards.append(reward)
                ep_reward += reward
                if done:
                    break

            total_rewards.append(ep_reward)

            reinforce_helper.log_results(ep_no, total_rewards, info, t, self.params, 'Alice')
            helper.add_scalar(self.writer, 'alice_episode_reward', ep_reward, ep_no, self.params)
            helper.add_scalar(self.writer, 'alice_episode_covered_count', info['covered_count'], ep_no, self.params)
            helper.add_scalar(self.writer, 'alice_episode_steps_count', info['steps_count'], ep_no, self.params)

            # perform backprop
            saved_actions = self.model.saved_actions
            policy_losses = []  # list to save actor (policy) loss
            value_losses = []  # list to save critic (value) loss

            returns = reinforce_helper.discount_rewards(self.model.rewards, self.params)

            for (log_prob, value), R in zip(saved_actions, returns):
                advantage = R - value.item()

                # calculate actor (policy) loss
                policy_losses.append(-log_prob * advantage)

                # calculate critic (value) loss using L1 smooth loss
                value_losses.append(F.smooth_l1_loss(value.squeeze(0), torch.tensor([R]).to(self.device)))

            # reset gradients
            self.optimizer.zero_grad()

            # sum up all the values of policy_losses and value_losses
            loss = torch.stack(policy_losses).mean() + self.params.scaling_factor * torch.stack(value_losses).mean()

            # perform backprop
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.0001)
            self.optimizer.step()

            # reset rewards and action buffer
            del self.model.rewards[:]
            del self.model.saved_actions[:]


class BobRunner:
    def __init__(self, params, env, device, writer):
        self.params = params
        self.env = env
        self.device = device
        self.writer = writer

        class Model(network.get_model_class(self.params)):
            def __init__(self, env):
                super(Model, self).__init__(env)
                # action & reward buffer
                self.saved_actions = []
                self.rewards = []

        self.model = Model(env).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.params.learning_rate)
        self.total_rewards = []

    def run(self, alice_state, ep_no):
        # reset environment and episode reward
        state = self.env.reset(alice_state)

        ep_reward = 0

        # for each episode, only run 10_000 steps so that we don't
        # infinite loop while learning
        for t in range(10_000):
            # select action from policy
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            probs, state_value = self.model(state)

            # create a categorical distribution over the list of probabilities of actions
            c = Categorical(probs)

            # and sample an action using the distribution
            action = c.sample()

            # save to action buffer
            self.model.saved_actions.append(SavedAction(c.log_prob(action), state_value))

            # take the action
            state, reward, done, info = self.env.step(action.item())

            self.model.rewards.append(reward)
            ep_reward += reward
            if done:
                break
        if params.render:
            self.env.render()
        self.total_rewards.append(ep_reward)

        reinforce_helper.log_results(ep_no, self.total_rewards, info, t, self.params, 'Bob')
        helper.add_scalar(self.writer, 'bob_episode_reward', ep_reward, ep_no, self.params)
        helper.add_scalar(self.writer, 'bob_episode_covered_count', info['covered_count'], ep_no, self.params)
        helper.add_scalar(self.writer, 'bob_episode_steps_count', info['steps_count'], ep_no, self.params)

        # perform backprop
        saved_actions = self.model.saved_actions
        policy_losses = []  # list to save actor (policy) loss
        value_losses = []  # list to save critic (value) loss

        returns = reinforce_helper.discount_rewards(self.model.rewards, self.params)

        for (log_prob, value), R in zip(saved_actions, returns):
            advantage = R - value.item()

            # calculate actor (policy) loss
            policy_losses.append(-log_prob * advantage)

            # calculate critic (value) loss using L1 smooth loss
            value_losses.append(F.smooth_l1_loss(value.squeeze(0), torch.tensor([R]).to(self.device)))

        # reset gradients
        self.optimizer.zero_grad()

        # sum up all the values of policy_losses and value_losses
        loss = torch.stack(policy_losses).mean() + self.params.scaling_factor * torch.stack(value_losses).mean()

        # perform backprop
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), 0.0001)
        self.optimizer.step()

        # reset rewards and action buffer
        del self.model.rewards[:]
        del self.model.saved_actions[:]
        return info


def run(params):
    device, use_cuda = helper.get_pytorch_device()
    sb3_utils.set_random_seed(params.seed, using_cuda=use_cuda)

    writer = helper.get_summary_writer(__file__, params)

    bob_env = helper.make_env(params, 'Bob')
    bob = BobRunner(params, bob_env, device, writer)
    alice_env = helper.make_env(params, 'Alice')
    alice = AliceRunner(bob, params, alice_env, device, writer)
    alice.run()

    helper.close_summary_writer(writer)


if __name__ == '__main__':
    parser = reinforce_helper.parse_args()
    params = helper.get_parsed_params(parser)
    run(params)