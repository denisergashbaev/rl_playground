import logging

import torch
import argparse

from common import config as cfg
from common.env import ColoringEnv
from typing import Tuple

log = logging.getLogger(__name__)


def get_parsed_params(parser: argparse.ArgumentParser) -> argparse.Namespace:
    params = parser.parse_args()
    print('\n============BEGIN PARAMETERS============')
    print('\n'.join(["{}: {}".format(key, getattr(params, key)) for key in sorted(params.__dict__)]))
    print('============END PARAMETERS============\n')
    return params


def get_pytorch_device() -> Tuple[torch.device, bool]:
    print("pyTorch version:\t{}".format(torch.__version__))  # type: ignore
    use_cuda: bool = torch.cuda.is_available()  # type: ignore
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Device:', device)
    return device, use_cuda


def make_env(params: argparse.Namespace, worker_id: str = 'env') -> ColoringEnv:
    c = cfg.a2c_config_train_2x2 if params.size == '2x2' else cfg.a2c_config_train_10x10
    c.step_reward = -params.step_reward
    c.init()
    start_position = (1, 0) if params.size == '2x2' else (1, 3)
    env = ColoringEnv(c, worker_id, start_position,
                      depth_channel_first=params.depth_channel_first,
                      with_step_penalty=params.with_step_penalty,
                      with_revisit_penalty=params.with_revisit_penalty,
                      stay_inside=params.stay_inside,
                      with_color_reward=params.with_color_reward,
                      total_reward=params.total_reward,
                      covered_steps_ratio=params.covered_steps_ratio,
                      as_image=params.env_as_image)
    return env
