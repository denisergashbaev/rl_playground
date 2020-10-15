from collections import namedtuple

import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3.common import utils as sb3_utils
from torch import optim
from torch.distributions import Categorical

from algs.curriculum import network
from algs.curriculum.network import ModelAlice_2x2
from algs.dqn.dqn_main_simple_oop import Bob
from algs.reinforce import reinforce_helper
from utils import helper

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


class Alice:
    def __init__(self, bob: Bob, params, writer, device):
        self.bob = bob
        self.params = params
        self.env = helper.make_env(params, 'Alice')
        self.writer = writer
        self.device = device
        self.model = network.get_model_class(params, True)(self.env).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.params.learning_rate)

    def run(self):
        ep_no = 0
        total_rewards = []

        other_t = 0
        total_t = 0
        while ep_no < self.params.num_episodes:
            ep_no += 1

            # reset environment and episode reward
            alice_state = self.env.reset()
            ep_reward = 0

            # for each episode, only run 10_000 steps so that we don't
            # infinite loop while learning
            alice_info = {'covered_count': 1, 'steps_count': 0}
            for t in range(10_000):
                total_t += 1
                # select action from policy
                x = torch.from_numpy(alice_state).float().unsqueeze(0).to(self.device)
                x2 = torch.Tensor([t, self.env.max_steps, t / self.env.max_steps]).unsqueeze(0).to(self.device)
                probs, state_value = self.model(x, x2)

                # create a categorical distribution over the list of probabilities of actions
                c = Categorical(probs)

                # and sample an action using the distribution
                action = c.sample()
                total_t_div = (total_t // 1000 + 1) * 5
                action = torch.Tensor([ModelAlice_2x2.STOP if total_t_div == t else ModelAlice_2x2.CONTINUE]).to(device)
                # save to action buffer
                self.model.saved_actions.append(SavedAction(c.log_prob(action), state_value))

                action_item = action.item()
                #action_item = np.random.randint(0, 2)
                alice_done = False
                if ModelAlice_2x2.CONTINUE == action_item:
                    # take __RANDOM__ action
                    if params.use_bob_for_alice:
                        a, _ = self.bob.select_action(alice_state, t)
                    else:
                        a = np.random.randint(0, 4)
                    alice_state, _, alice_done, alice_info = self.env.step(a)
                    reward = 0
                alice_done |= ModelAlice_2x2.STOP == action_item

                if alice_done:
                    bob_info = self.bob.run(alice_state=alice_state, alice_env=self.env)
                    # -1 because the first position is always marked (ie, color_on_visit)
                    #reward = bob_info['steps_count'] - (self.env.covered_count(alice_state) - int(self.env.color_on_visit))
                    #reward = -bob_info['init_uncovered_count'] / bob_info['steps_count']
                    if params.use_bob_for_alice:
                        bob_steps_count = bob_info['steps_count']
                        other_t = bob_info['init_uncovered_count'] - bob_info['total_count'] - bob_info['covered_count']
                        reward = bob_info['steps_count'] - alice_info['steps_count']
                        #reward = bob_info['init_uncovered_count'] - bob_info['steps_count']
                        #reward = bob_info['steps_count'] / bob_info['init_uncovered_count']
                    else:
                        reward = bob_info['init_uncovered_count'] - bob_info['steps_count']
                    if alice_info['covered_count'] == 1:
                        reward = -int(self.env.max_steps)
                    print('Alice reward: {}, alice steps_count: {}, alice covered_count: {}, '
                          'bob steps count: {} bob covered_count: {}'
                        .format(reward, alice_info['steps_count'], alice_info['covered_count'],
                                bob_info['steps_count'], bob_info['covered_count']))
                    print('total_t:', total_t, 'total_t_div:', total_t_div, 't:', t)
                    self.bob.run(alice_state=None)

                self.model.rewards.append(reward)
                ep_reward += reward
                if alice_done:
                    break

            total_rewards.append(ep_reward)
            reinforce_helper.log_results(ep_no, total_rewards, alice_info, t, self.params, 'Alice')
            helper.add_scalar(self.writer, 'alice_probs_0', probs.squeeze()[0].item(), ep_no, self.params)
            helper.add_scalar(self.writer, 'alice_episode_reward', ep_reward, ep_no, self.params)
            helper.add_scalar(self.writer, 'alice_episode_covered_count', alice_info['covered_count'], ep_no, self.params)
            helper.add_scalar(self.writer, 'alice_episode_steps_count', alice_info['steps_count'], ep_no, self.params)

            # Alice: only learn if bob (DQN) started learning
            if bob.is_start_learn():
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


if __name__ == '__main__':
    parser = reinforce_helper.parse_args()
    parser.add_argument("--target_update_rate", type=float, default=0.1)
    parser.add_argument("--replay_size", type=int, default=int(1e6))
    parser.add_argument("--start_train_ts", type=int, default=10000)
    parser.add_argument("--epsilon_start", type=float, default=1.0)
    parser.add_argument("--epsilon_end", type=float, default=0.01)
    parser.add_argument("--epsilon_decay", type=int, default=30000)
    parser.add_argument("--max_ts", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--target_network_update_f", type=int, default=10000)
    parser.add_argument("--max_grad_norm", type=float)
    parser.add_argument("--use_bob_for_alice", action="store_true")

    params = helper.get_parsed_params(parser)
    device, use_cuda = helper.get_pytorch_device()
    sb3_utils.set_random_seed(params.seed, using_cuda=use_cuda)
    writer = helper.get_summary_writer(__file__, params)
    bob = Bob(params, writer, device)
    alice = Alice(bob, params, writer, device)
    alice.run()

    helper.close_summary_writer(writer)