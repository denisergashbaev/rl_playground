import argparse
import math
import random
from operator import add

import numpy as np
import torch
from stable_baselines3.common import utils as sb3_utils
from torch import optim
from torch.nn import functional as F

import common.config as cfg
from algs.dqn import network
from algs.dqn.replay_buffer import ReplayBuffer
from common.env import ColoringEnv
from utils import helper

device, use_cuda = helper.get_pytorch_device()


def get_epsilon(epsilon_start, epsilon_final, epsilon_decay, frame_idx):
    return epsilon_final + (epsilon_start - epsilon_final) * math.exp(
        -1.0 * frame_idx / epsilon_decay)


def run(params):
    sb3_utils.set_random_seed(params.seed, using_cuda=use_cuda)
    writer = helper.get_summary_writer(__file__, params)
    env = helper.make_env(params, 'env')

    q = network.get_model_class(params)(env).to(device)
    q_hat = network.get_model_class(params)(env).to(device)
    q_hat.load_state_dict(q.state_dict())

    replay_buffer = ReplayBuffer(params.replay_size)
    #todo check optimizer
    opt = optim.Adam(q.parameters(), lr=params.learning_rate)

    all_rewards = []
    state = env.reset()
    episode_reward = [0]
    episode_no = 0
    for t in range(1, params.max_ts + 1):
        # order of terms important so that the call to 'next(eps)' does not decrease epsilon
        epsilon = get_epsilon(
            params.epsilon_start, params.epsilon_end, params.epsilon_decay, t
        )
        if random.random() < epsilon:
            a = random.randrange(env.action_space.n)
        else:
            val = q(np.expand_dims(state, axis=0))
            a = torch.argmax(val).item()
            # equivalent to q(...).max(1)[1].data[0] (selects max tensor with .max(1) and its index with ...[1])
        s_tp1, r, done, infos = env.step(a)
        episode_reward = list(map(add, episode_reward, [r]))
        replay_buffer.add(state, a, r, s_tp1, done)
        state = s_tp1
        if done:
            state = env.reset()
            all_rewards.append(episode_reward)
            episode_reward = [0]
            episode_no += 1

        # replay buffer reached minimum capacity
        if len(replay_buffer) > params.start_train_ts:
            obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(params.batch_size)
            rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)
            actions = torch.tensor(actions).unsqueeze(1).to(device)
            dones = torch.tensor(dones).unsqueeze(1).to(device)
            if True: 
                with torch.no_grad():
                    # Compute the target Q values
                    target_q = q_hat(obses_tp1)
                    # Follow greedy policy: use the one with the highest value
                    target_q, _ = target_q.max(dim=1)
                    # Avoid potential broadcast issue
                    target_q = target_q.reshape(-1, 1)
                    # 1-step TD target
                    target_q = rewards + ~dones * params.gamma * target_q

                # Get current Q estimates
                current_q = q(obses_t)

                # Retrieve the q-values for the actions from the replay buffer
                current_q = torch.gather(current_q, dim=1, index=actions.long())

                # Compute Huber loss (less sensitive to outliers)
                loss = F.smooth_l1_loss(current_q, target_q)

            else:
                val_tp1 = q(obses_tp1)
                val_t = q(obses_t)
                val_hat_tp1 = q_hat(obses_tp1)
                # .T to iterate over columns of the array: https://stackoverflow.com/a/10148855/256002

                r = torch.from_numpy(rewards).to(device)
                #if params.summed_q:
                #    head = heads[idx]
                #else:
                #    head = heads[mirrored_envs.use_for_decisions_idx]
                a = torch.argmax(val_tp1, dim=1)
                td_errors = r + ~dones * params.gamma * val_hat_tp1.gather(1, a.unsqueeze(1)).squeeze()
                q_vals = val_t.gather(1, actions).squeeze()
                #loss = (td_errors.detach() - q_vals).pow(2).mean()
                loss = F.smooth_l1_loss(q_vals, td_errors.detach())

            if done:
                writer.add_scalar("loss_idx", loss.data, episode_no)

            if done:
                writer.add_scalar("total_loss", loss.data, episode_no)

            # Optimize the policy
            opt.zero_grad()
            loss.backward()
            # Clip gradient norm
            torch.nn.utils.clip_grad_norm_(q.parameters(), params.max_grad_norm)
            opt.step()
            if t % params.target_network_update_f == 0:
                print('weights copied')
                sb3_utils.polyak_update(q.parameters(), q_hat.parameters(), 1.0)

        if done:
            for idx, ep_reward in enumerate(all_rewards[-1]):
                helper.add_scalar(writer, "episode_reward_idx{}".format(idx), ep_reward, episode_no, params)
            helper.add_scalar(writer, "steps_count", info['steps_count'], episode_no, params)

            if episode_no % params.log_every == 0:
                #print('replaybuffer size:', len(replay_buffer))
                out_str = "Timestep {}".format(t)
                if len(all_rewards) > 0:
                    out_str += ",Reward:{}".format(all_rewards[-1])
                out_str += ", done: {}".format(done)
                out_str += ', steps_count {}'.format(infos['steps_count'])
                out_str += ', epsilon {}'.format(epsilon)
                print(out_str)
    helper.close_summary_writer(writer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=str, default="2x2")
    parser.add_argument("--learning_rate", type=float, default=3e-2)
    parser.add_argument("--gamma", type=float, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--step_reward", default="1", type=float)
    parser.add_argument("--env_as_image", action="store_true")
    parser.add_argument("--depth_channel_first", action="store_true")
    parser.add_argument("--with_step_penalty", action="store_true")
    parser.add_argument("--with_revisit_penalty", action="store_true")
    parser.add_argument("--stay_inside", action="store_true")
    parser.add_argument("--with_color_reward", action="store_true")
    parser.add_argument("--total_reward", action="store_true")
    parser.add_argument("--covered_steps_ratio", action="store_true")
    parser.add_argument('--num_episodes', type=int, default=20000, help='number of episodes')
    parser.add_argument('--scaling_factor', type=float, default=1, help='learning rate')
    parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                        help='interval between training status logs (default: 10)')
    parser.add_argument("--log_tensorboard", action="store_true")
    parser.add_argument("--render", action="store_true")

    parser.add_argument("--target_update_rate", type=float, default=0.1)
    parser.add_argument("--replay_size", type=int, default=int(1e6))
    parser.add_argument("--start_train_ts", type=int, default=10000)
    parser.add_argument("--epsilon_start", type=float, default=1.0)
    parser.add_argument("--epsilon_end", type=float, default=0.01)
    parser.add_argument("--epsilon_decay", type=int, default=30000)
    parser.add_argument("--max_ts", type=int)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--target_network_update_f", type=int, default=10000)
    parser.add_argument("--max_grad_norm", type=float)


    parsed = parser.parse_args()
    print('arguments:', parsed)
    run(parsed)