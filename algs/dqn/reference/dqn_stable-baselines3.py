from algs.dqn.reference.dqn_callback import SaveOnBestTrainingRewardCallback
import argparse

from algs.dqn.reference.network import ColoringCNN
from stable_baselines3 import DQN
from stable_baselines3.common import utils as sb3_utils
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.dqn import CnnPolicy
from utils import helper
from comet_ml import Experiment  # type: ignore


device, use_cuda = helper.get_pytorch_device()


def run(experiment: Experiment, params: argparse.Namespace):
    sb3_utils.set_random_seed(params.seed, using_cuda=use_cuda)
    env = helper.make_env(params, 'env')

    # Logs will be saved in log_dir/monitor.csv
    env = Monitor(env)

    with experiment.train():
        callback = SaveOnBestTrainingRewardCallback(experiment, check_freq=1000)
        # Deactivate all the DQN extensions to have the original version
        # In practice, it is recommend to have them activated
        model = DQN(CnnPolicy, env, learning_rate=params.learning_rate,
                    gamma=params.gamma, seed=params.seed, max_grad_norm=params.max_grad_norm,
                    verbose=1, device=device,
                    policy_kwargs={'features_extractor_class': ColoringCNN})
        model.learn(total_timesteps=params.max_ts, callback=callback)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--size", type=str, default="2x2")
    parser.add_argument("--learning_rate", type=float, default=3e-2)
    # default is set to 50_000 in stable_baselines3/dqn/dqn.py
    parser.add_argument("--learning_starts", type=int, default=None)
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

    params = helper.get_parsed_params(parser)
    experiment = Experiment()
    experiment.log_parameters(params)  # type: ignore
    run(experiment, params)
