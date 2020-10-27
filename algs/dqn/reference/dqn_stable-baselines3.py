import argparse
import os
from datetime import datetime

import numpy as np
from pathlib import Path
from stable_baselines3 import DQN
from stable_baselines3.common import utils as sb3_utils
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import load_results, Monitor
from stable_baselines3.common.results_plotter import ts2xy
from stable_baselines3.dqn import CnnPolicy
from torch.utils.tensorboard import SummaryWriter

from common.env import ColoringEnv
from utils import constants
from algs.dqn.reference.network import ColoringCNN

from utils import helper


device, use_cuda = helper.get_pytorch_device()


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """

    def __init__(self, check_freq: int, log_dir: str, writer: SummaryWriter, params: argparse.Namespace, verbose: int=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.writer = writer
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
            data_frame = load_results(self.log_dir)
            x, y = ts2xy(data_frame, 'timesteps')
            len_x = len(x)
            if len_x > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                mean_steps = np.mean(data_frame.l.values[-100:])
                if self.verbose > 0:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(
                        f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")
                    print("Last mean time steps: {}".format(mean_steps))
                helper.add_scalar(self.writer, "reward", mean_reward, len_x, params)
                helper.add_scalar(self.writer, "steps", mean_steps, len_x, params)
                self.writer.add_scalar("reward", mean_reward, global_step=len_x)
                self.writer.add_scalar("steps", mean_steps, global_step=len_x)
                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print(f"Saving new best model to {self.save_path}.zip")
                    self.model.save(self.save_path)

        return True


def run(params: argparse.Namespace):
    sb3_utils.set_random_seed(params.seed, using_cuda=use_cuda)
    env = helper.make_env(params, 'env')

    # results after ~1hr
    # step_penalty: 0.1, env_as_image: True, stay_inside=True, color_on_visit=True,
    # reward 52.65: with_step_penalty=True, with_revisit_penalty=False, with_color_reward=True, total_reward=False, covered_steps_ratio=False
    # reward 52.38: with_step_penalty=True, with_revisit_penalty=True, with_color_reward=True, total_reward=False, covered_steps_ratio=False
    # reward 52.65  with_step_penalty=False, with_revisit_penalty=True,  with_color_reward=False, total_reward=False, covered_steps_ratio=False
    # reward 200.0: with_step_penalty=False, with_revisit_penalty=False, with_color_reward=True, total_reward=False, covered_steps_ratio=False
    experiment_datetime = helper.get_experiment_datetime(__file__)
    log_dir = os.path.dirname(os.path.realpath(__file__)) + '/log_dir/' + experiment_datetime
    writer = helper.get_summary_writer(__file__, params)
    
    os.makedirs(log_dir, exist_ok=True)

    # Logs will be saved in log_dir/monitor.csv
    env = Monitor(env, log_dir)

    callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir, writer=writer, params=params)
    # Deactivate all the DQN extensions to have the original version
    # In practice, it is recommend to have them activated
    model = DQN(CnnPolicy, env, learning_rate=params.learning_rate,
                gamma=params.gamma, seed=params.seed, max_grad_norm=params.max_grad_norm,
                verbose=1, device=device,
                policy_kwargs={'features_extractor_class': ColoringCNN})
    model.learn(total_timesteps=params.max_ts, callback=callback)
    writer.flush()
    writer.close()


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

    params = helper.get_parsed_params(parser)
    run(params)
