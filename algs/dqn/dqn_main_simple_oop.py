import argparse

from stable_baselines3.common import utils as sb3_utils

from algs.dqn.bob import Bob
from utils import helper

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=str, default="2x2")
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--target_update_rate", type=float, default=0.1)
    parser.add_argument("--replay_size", type=int, default=int(1e6))
    parser.add_argument("--start_train_ts", type=int, default=10000)
    parser.add_argument("--epsilon_start", type=float, default=1.0)
    parser.add_argument("--epsilon_end", type=float, default=0.01)
    parser.add_argument("--epsilon_decay", type=int, default=30000)
    parser.add_argument("--max_ts", type=int, default=1400000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gamma", type=float, default=1)
    parser.add_argument("--log_interval", type=int, default=10000)
    parser.add_argument("--target_network_update_f", type=int, default=10000)
    parser.add_argument("--log_tensorboard", action="store_true")

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--step_reward", default="1", type=float)
    parser.add_argument("--max_grad_norm", type=float)
    parser.add_argument("--env_as_image", action="store_true")
    parser.add_argument("--depth_channel_first", action="store_true")
    parser.add_argument("--with_step_penalty", action="store_true")
    parser.add_argument("--with_revisit_penalty", action="store_true")
    parser.add_argument("--stay_inside", action="store_true")
    parser.add_argument("--with_color_reward", action="store_true")
    parser.add_argument("--total_reward", action="store_true")
    parser.add_argument("--covered_steps_ratio", action="store_true")
    parser.add_argument("--load_model", type=str, default=None)
    parser.add_argument('--render', action='store_true')

    params = helper.get_parsed_params(parser)

    device, use_cuda = helper.get_pytorch_device()
    sb3_utils.set_random_seed(params.seed, using_cuda=use_cuda)
    writer = helper.get_summary_writer(__file__, params)
    bob = Bob(params, writer, device)
    bob.run(alice_state=None, alice_env=None)
    helper.close_summary_writer(writer)
