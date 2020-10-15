import argparse
import numpy as np
import torch

eps = np.finfo(np.float32).eps.item()

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
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

    return parser


def log_results(ep_no, total_rewards, info, t, params, prefix):
    # log results
    if ep_no % params.log_interval == 0:
        last_no_of_episodes = 100
        avg_rewards = np.mean(total_rewards[-last_no_of_episodes:])
        # Print running average
        print("\rEp: {}, {}, Last episode length: {}, Average of last {} episodes: {:.2f}"
              .format(ep_no, prefix, t+1, last_no_of_episodes, avg_rewards))
        print('Last infos:', info)


def discount_rewards(rewards, params):
    # calculate the true value using rewards returned from the environment
    R = 0
    returns = []  # list to save the true values
    for r in rewards[::-1]:
        # calculate the discounted value
        R = r + params.gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns)
    #returns = (returns - returns.mean()) / (returns.std() + eps)
    return returns



