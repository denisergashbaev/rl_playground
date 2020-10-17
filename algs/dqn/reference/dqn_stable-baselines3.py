import argparse
import os
from datetime import datetime

import numpy as np
import torch
from stable_baselines3 import DQN
from stable_baselines3.common import utils as sb3_utils
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import load_results, Monitor
from stable_baselines3.common.results_plotter import ts2xy
from stable_baselines3.dqn import CnnPolicy
from torch.utils.tensorboard import SummaryWriter

import common.config as cfg
from common.env import ColoringEnv
from utils import constants
from algs.dqn.reference.network import ColoringCNN

using_cuda = torch.cuda.is_available()
device = torch.device('cuda' if using_cuda else 'cpu')
print('dqn_main.py using device:', device)


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """

    def __init__(self, check_freq: int, log_dir: str, writer: SummaryWriter, verbose=1):
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


def run(params):
    sb3_utils.set_random_seed(params.seed, using_cuda=using_cuda)
    c = cfg.a2c_config_train_2x2 if params.size == '2x2' else cfg.a2c_config_train_10x10
    c.step_reward = -params.step_reward
    c.init()
    start_position = (1, 0) if params.size == '2x2' else (1, 3)

    env = ColoringEnv(c, 'env', start_position,
                        depth_channel_first=params.depth_channel_first,
                        with_step_penalty=params.with_step_penalty,
                        with_revisit_penalty=params.with_revisit_penalty,
                        stay_inside=params.stay_inside,
                        color_on_visit=params.color_on_visit,
                        with_color_reward=params.with_color_reward,
                        total_reward=params.total_reward,
                        covered_steps_ratio=params.covered_steps_ratio,
                        as_image=params.env_as_image)

    # results after ~1hr
    # step_penalty: 0.1, env_as_image: True, stay_inside=True, color_on_visit=True,
    # reward 52.65: with_step_penalty=True, with_revisit_penalty=False, with_color_reward=True, total_reward=False, covered_steps_ratio=False
    # reward 52.38: with_step_penalty=True, with_revisit_penalty=True, with_color_reward=True, total_reward=False, covered_steps_ratio=False
    # reward 52.65  with_step_penalty=False, with_revisit_penalty=True,  with_color_reward=False, total_reward=False, covered_steps_ratio=False
    # reward 200.0: with_step_penalty=False, with_revisit_penalty=False, with_color_reward=True, total_reward=False, covered_steps_ratio=False
    if params.load_model is not None:
        model = DQN.load(params.load_model, env=env)
        episode_rewards = [0.0]
        env = model.get_env()
        state = env.reset()
        done = False
        while not done:
            action, _ = model.predict(state, deterministic=True)
            state, reward, done, info = env.step(action)
            # Stats
            episode_rewards.append(reward)

        # Compute mean reward for the last 100 episodes
        print("Total reward:", sum(episode_rewards))
        # print(info)
        steps = info[0]['steps']
        s = [(s[0], s[1]) for s in steps]
        print('Steps ({}):\n{}'.format(len(s), str(s).strip('[]')))
    else:
        experiment_datetime = str(datetime.now()).replace(' ', '_').replace(':', '_')
        writer = SummaryWriter(constants.v2_dqn_sb3_dir + '/runs/' + experiment_datetime)
        log_dir = "{}/log_dir/{}".format(constants.v2_dqn_sb3_dir, experiment_datetime)
        os.makedirs(log_dir, exist_ok=True)

        # Logs will be saved in log_dir/monitor.csv
        env = Monitor(env, log_dir)

        callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir, writer=writer)
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
    parser.add_argument("--size", type=str)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--max_ts", type=int, default=1400000)
    parser.add_argument("--gamma", type=float, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--step_reward", type=float)
    parser.add_argument("--max_grad_norm", type=float)
    parser.add_argument("--env_as_image", action="store_true")
    parser.add_argument("--depth_channel_first", action="store_true")
    parser.add_argument("--with_step_penalty", action="store_true")
    parser.add_argument("--with_revisit_penalty", action="store_true")
    parser.add_argument("--stay_inside", action="store_true")
    parser.add_argument("--color_on_visit", action="store_true")
    parser.add_argument("--with_color_reward", action="store_true")
    parser.add_argument("--total_reward", action="store_true")
    parser.add_argument("--covered_steps_ratio", action="store_true")
    parser.add_argument("--load_model", type=str, default=None)
    parsed = parser.parse_args()
    print('arguments:', parsed)
    run(parsed)