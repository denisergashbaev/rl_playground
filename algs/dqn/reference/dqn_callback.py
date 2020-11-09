from typing import Any

import numpy as np
from comet_ml import Experiment  # type: ignore
from common import constants
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import load_results
from stable_baselines3.common.results_plotter import ts2xy


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).
    :param check_freq: (int)
    :param verbose: (int)
    """
    def __init__(self, experiment: Experiment, check_freq: int, verbose: int = 1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.experiment = experiment
        self.check_freq = check_freq
        self.best_mean_reward: float = np.NINF  # type: ignore
        self.verbose = verbose

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Retrieve training reward
            data_frame = load_results(constants.OUT_DIR)
            x: np.ndarray[Any]
            y: np.ndarray[Any]
            x, y = ts2xy(data_frame, 'timesteps')  # type: ignore
            len_x = len(x)
            if len_x > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                mean_steps = np.mean(data_frame.l.values[-100:])  # type: ignore
                if self.verbose > 0:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(
                        f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")
                    print("Last mean time steps: {}".format(mean_steps))
                self.experiment.log_metric('rewards', mean_reward, len_x)
                self.experiment.log_metric('steps', mean_steps, len_x)
                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:  # type: ignore
                    self.best_mean_reward = mean_reward  # type: ignore
                    # Example for saving best model
                    # if self.verbose > 0:
                    #     print(f"Saving new best model to {self.save_path}.zip")
                    # self.model.save(self.save_path)
        return True
