import math
import random

import numpy as np
import torch
from stable_baselines3.common import utils as sb3_utils
from torch import optim
from torch.nn import functional as F

from algs.dqn import network
from algs.dqn.replay_buffer import ReplayBuffer
from algs.reinforce import reinforce_helper
from utils import helper


class Bob:
    def __init__(self, params, writer, device):
        self.params = params
        self.writer = writer
        self.device = device
        self.env = helper.make_env(params, 'Bob')

        self.q = network.get_model_class(params)(self.env).to(self.device)
        self.q_hat = network.get_model_class(params)(self.env).to(self.device)
        self.q_hat.load_state_dict(self.q.state_dict())

        self.replay_buffer = ReplayBuffer(params.replay_size)
        self.optimizer = optim.Adam(self.q.parameters(), lr=params.learning_rate)
        self.t = 0
        self.ep_no = 0
        self.total_rewards = []
        self.ret_render = ''

    def get_epsilon(self, frame_idx):
        val = self.params.epsilon_end + (self.params.epsilon_start - self.params.epsilon_end) \
             * math.exp(-1.0 * frame_idx / self.params.epsilon_decay)
        return val

    def select_action(self, state, t):
        epsilon = self.get_epsilon(t)
        if random.random() < epsilon:
            a = random.randrange(self.env.action_space.n)
        else:
            val = self.q(np.expand_dims(state, axis=0))
            a = torch.argmax(val).item()
            # equivalent to q(...).max(1)[1].data[0] (selects max tensor with .max(1) and its index with ...[1])
        return a, epsilon

    def run(self, alice_state=None, alice_env=None):
        if self.params.render and alice_env is not None:
            self.ret_render = alice_env.render()
        state = self.env.reset(alice_state=alice_state)

        if self.params.render and alice_env is not None:
            self.ret_render += '\n' + self.env.render()

        for _ in range(1, self.params.max_ts + 1):
            ep_reward = 0
            done = False
            while not done:
                self.t += 1
                a, epsilon = self.select_action(state, self.t)
                s_tp1, r, done, info = self.env.step(a)
                self.replay_buffer.add(state, a, r, s_tp1, done)
                ep_reward += r
                state = s_tp1

                if done:
                    prefix = 'bob_target_task' if alice_state is None else 'bob'
                    helper.add_scalar(self.writer, '{}_episode_reward'.format(prefix), ep_reward, self.ep_no, self.params)
                    helper.add_scalar(self.writer, '{}_episode_steps_count'.format(prefix), info['steps_count'], self.ep_no, self.params)
                    helper.add_scalar(self.writer, '{}_episode_covered_count'.format(prefix), info['covered_count'], self.ep_no,
                                      self.params)
                    helper.add_scalar(self.writer, '{}_episode_reduced_ratio'.format(prefix), info['reduced_ratio'], self.ep_no, self.params)
                    helper.add_scalar(self.writer, '{}_init_uncovered_count'.format(prefix), info['init_uncovered_count'], self.ep_no, self.params)
                    helper.add_scalar(self.writer, '{}_epsilon'.format(prefix), epsilon, self.ep_no, self.params)

                    self.total_rewards.append(ep_reward)
                    ep_reward = 0
                    self.ep_no += 1

                    reinforce_helper.log_results(self.ep_no, self.total_rewards, info, self.t, self.params, 'Bob')
                    if self.ep_no % self.params.log_interval == 0:
                        # print('replaybuffer size:', len(replay_buffer))
                        out_str = "{} Timestep {}".format(prefix, self.t)
                        if len(self.total_rewards) > 0:
                            out_str += ",Reward:{}".format(self.total_rewards[-1])
                        out_str += ", done: {}".format(done)
                        out_str += ', steps_count {}'.format(info['steps_count'])
                        out_str += ', epsilon {}'.format(epsilon)
                        print(out_str)

                    if self.params.render:
                        self.ret_render += '\n' + self.env.render()
                        print(self.ret_render)
                        print('{} info:', prefix, info['steps'])
                        self.ret_render = ''
                    state = self.env.reset(alice_state=alice_state)

                self.learn()

        return info

    def is_start_learn(self):
        return len(self.replay_buffer) > self.params.start_train_ts

    def learn(self):
        # replay buffer reached minimum capacity
        if self.is_start_learn():
            obses_t, actions, rewards, obses_tp1, dones = self.replay_buffer.sample(self.params.batch_size)
            rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
            actions = torch.tensor(actions).unsqueeze(1).to(self.device)
            dones = torch.tensor(dones).unsqueeze(1).to(self.device)
            with torch.no_grad():
                # Compute the target Q values
                target_q = self.q_hat(obses_tp1)
                # Follow greedy policy: use the one with the highest value
                target_q, _ = target_q.max(dim=1)
                # Avoid potential broadcast issue
                target_q = target_q.reshape(-1, 1)
                # 1-step TD target
                target_q = rewards + ~dones * self.params.gamma * target_q

            # Get current Q estimates
            current_q = self.q(obses_t)

            # Retrieve the q-values for the actions from the replay buffer
            current_q = torch.gather(current_q, dim=1, index=actions.long())

            # Compute Huber loss (less sensitive to outliers)
            loss = F.smooth_l1_loss(current_q, target_q)

            # Optimize the policy
            self.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            torch.nn.utils.clip_grad_norm_(self.q.parameters(), self.params.max_grad_norm)
            self.optimizer.step()
            if self.t % self.params.target_network_update_f == 0:
                print('weights copied')
                sb3_utils.polyak_update(self.q.parameters(), self.q_hat.parameters(), 1.0)
