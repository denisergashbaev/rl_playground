import io
import logging
from collections import OrderedDict

import gym
import numpy as np
from gym import spaces, Env
from gym.spaces import Box
from prettytable import PrettyTable
from random import randint


log = logging.getLogger(__name__)


class ColoringEnv(gym.Env):
    emb_pattern_empty = 0
    emb_pattern_to_color = 1
    agent_pos_code = 1
    agent_pos_empty = 0
    channel_pattern = 0
    channel_stitch = 1
    channel_agent = 2

    def __init__(self, c, worker_id, start_position, with_step_penalty, with_revisit_penalty,
                 stay_inside, with_color_reward, total_reward, covered_steps_ratio,
                 depth_channel_first=True, changing_start_positions=False, as_image=False, color_on_visit=True):
        Env.__init__(self)
        log.info('creating environment for files {}'.format(c.data_files))
        # needed in order to simulate gym environment
        self.reward_range = None
        self.metadata = {'render.modes': []}
        self.spec = None
        self.enabled = False
        self.observation_space = None

        self.c = c
        # First channel
        # 0 - blank cell
        # 1 - pattern cell
        # Second channel
        # 0 - not stitched
        # 1 - stitched
        # Third channel
        # 0 - no agent
        # 1 - agent
        self.with_step_penalty = with_step_penalty
        self.with_revisit_penalty = with_revisit_penalty
        self.stay_inside = stay_inside
        self.action_encodings = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}
        self.with_color_reward = with_color_reward
        self.total_reward = total_reward
        self.covered_steps_ratio = covered_steps_ratio
        self.inv_action_encodings = {v: k for k, v in self.action_encodings.items()}
        self.action_space = spaces.Discrete(len(self.action_encodings))
        self.layer_descriptions = OrderedDict([
            (ColoringEnv.channel_pattern, 'Pattern'),
            (ColoringEnv.channel_stitch, 'Completed pattern'),
            (ColoringEnv.channel_agent, 'Agent position'),
        ])
        self.worker_id = worker_id
        self.start_position = start_position
        self.env_reset_count = 0
        self.steps = []
        self.emb_pattern_layer = None
        self.emb_pattern_count = 0
        self.x_dim = None
        self.y_dim = None
        self.base_observation = None
        self.initial_observation = None
        self.max_steps = -1
        self.step_count = 0
        self.data_file = None
        self.done = False
        self.depth_channel_first = depth_channel_first
        self.changing_start_positions = changing_start_positions
        self.as_image = as_image
        self.color_on_visit = color_on_visit
        self.alice_state = None
        self.init_uncovered_count = 0
        self.reset()

    def reset(self, alice_state=None):
        self.data_file, x = self.c.load_file(self.env_reset_count)
        # TODO make sure all the environments are between [0, 1]
        assert x.min() == 0
        assert x.max() == 1
        self.alice_state = alice_state
        self.emb_pattern_layer = x
        del x
        self.nonzero_indices = [(i, j) for i, j in zip(*np.nonzero(self.emb_pattern_layer))]
        self.emb_pattern_count = np.count_nonzero(self.emb_pattern_layer)
        log.info('Worker {} loaded the file {} (reset count {})'
                 .format(self.worker_id, self.data_file, self.env_reset_count))
        self.env_reset_count += 1
        # TODO do not do that for embroideries > 10x10
        self.max_steps = np.prod(self.emb_pattern_layer.shape) * 2  # ie, 28 * 28 * 2
        #self.max_steps = 30
        self.step_count = 0

        self.x_dim, self.y_dim = self.emb_pattern_layer.shape

        original_position = self.nonzero_indices[randint(0, len(self.nonzero_indices) - 1)] \
            if self.changing_start_positions else self.start_position
        #print('new position:', original_position, 'value at the new position:', self.emb_pattern_layer[original_position])
        # no .clear() because we do not want to changed returned by reinforce data
        self.steps = []
        self.done = False

        # Base observation
        self.base_observation = np.zeros((len(self.layer_descriptions), self.x_dim, self.y_dim), dtype=int)
        self.base_observation[ColoringEnv.channel_pattern] = self.emb_pattern_layer.astype(float)
        self.base_observation[ColoringEnv.channel_stitch] = np.zeros_like(self.emb_pattern_layer, dtype=int)

        self.base_observation[ColoringEnv.channel_agent] = np.zeros_like(self.emb_pattern_layer, dtype=int)
        self.base_observation[ColoringEnv.channel_agent][original_position] = ColoringEnv.agent_pos_code
        self.initial_observation = np.copy(self.base_observation)

        curr_agent_position = self._get_agent_position()
        # Bob starts at the states where only the cells covered by Alice are uncolored,
        # the other cells (corresponding to the ColoringEnv.channel_pattern) are filled out
        if alice_state is not None:
            alice_state_channel = alice_state[ColoringEnv.channel_stitch].astype(np.copy(self.emb_pattern_layer).dtype)
            diff = self.base_observation[ColoringEnv.channel_pattern] - alice_state_channel
            diff[curr_agent_position] = ColoringEnv.emb_pattern_to_color
            # Alice has not completed the whole pattern
            if not (self.base_observation[ColoringEnv.channel_pattern] == diff).all():
                self.base_observation[ColoringEnv.channel_stitch] = diff
            #del diff, alice_state_channel
        #del alice_state
        if self.base_observation[ColoringEnv.channel_pattern][curr_agent_position] == ColoringEnv.emb_pattern_to_color:
            self.base_observation[ColoringEnv.channel_stitch][curr_agent_position] = ColoringEnv.emb_pattern_to_color
        self.init_uncovered_count = self.emb_pattern_count - self.covered_count(self.base_observation)
        #print('init_uncovered_count:', self.init_uncovered_count, 'self.base_observation[ColoringEnv.channel_stitch]:', self.base_observation[ColoringEnv.channel_stitch])

        if self.depth_channel_first:
            box_shape = (self.base_observation.shape[0], self.base_observation.shape[1], self.base_observation.shape[2])
        else:
            box_shape = (self.base_observation.shape[1], self.base_observation.shape[2], self.base_observation.shape[0])

        self.observation_space = Box(low=0, high=255, shape=(box_shape), dtype=np.uint8)

        return self._gen_state(self.base_observation)

    def covered_count(self, state):
        assert state.shape[0] == len(self.layer_descriptions)
        return np.count_nonzero(state[ColoringEnv.channel_stitch])

    def seed(self, s):
        pass

    def _get_agent_position(self):
        agent_position = tuple(np.argwhere(self.base_observation[ColoringEnv.channel_agent]
                                           == ColoringEnv.agent_pos_code)[0])
        return agent_position

    # gets the current position of the agent and colors cells around it more/less intense
    def _gen_state(self, obs):
        obs = np.copy(obs)
        if self.as_image:
            obs = obs * 255
        if not self.depth_channel_first:
            # https://machinelearningmastery.com/a-gentle-introduction-to-channels-first-and-channels-last-image-formats-for-deep-learning/
            # tensorflow expects the channel to be the last dimension
            # ie (28,28,3) instead of (3, 28, 28)
            obs = np.moveaxis(obs, 0, -1)
        return obs

    def get_rewards(self):
        return [s[2] for s in self.steps]

    def get_rewards_sum(self):
        return sum(self.get_rewards())

    def get_stitches(self):
        return [s[0] for s in self.steps if 's' == self.action_encodings[s[1]]]

    def get_positions_and_actions(self):
        return [(s[0], self.action_encodings[s[1]]) for s in self.steps]

    # See #layer_descriptions
    def step(self, a_t_num):
        if self.done:
            raise RuntimeError('Calling step when environment is done')
        self.step_count += 1
        a_t = self.action_encodings[a_t_num]

        x_lower_boundary = 0
        y_lower_boundary = 0
        x_upper_boundary = self.x_dim - 1
        y_upper_boundary = self.y_dim - 1

        s_tp1 = np.copy(self.base_observation)
        r_tp1 = None

        def move_agent(step):
            old_agent_position = self._get_agent_position()
            new_agent_position = tuple(np.array(old_agent_position) + step)
            # agent does not leave the the frames OR the pattern cells
            if not (x_lower_boundary <= new_agent_position[0] <= x_upper_boundary and
                    y_lower_boundary <= new_agent_position[1] <= y_upper_boundary and
                    (not self.stay_inside or s_tp1[ColoringEnv.channel_pattern][new_agent_position] != 0)):
                new_agent_position = old_agent_position

            # move the agent to the new position
            s_tp1[ColoringEnv.channel_agent][old_agent_position] = ColoringEnv.agent_pos_empty
            s_tp1[ColoringEnv.channel_agent][new_agent_position] = ColoringEnv.agent_pos_code
            ret_val = 0
            if self.with_step_penalty:
                ret_val += self.c.step_reward
            if self.with_revisit_penalty:
                if s_tp1[ColoringEnv.channel_stitch][new_agent_position] == ColoringEnv.emb_pattern_to_color:
                    ret_val += self.c.step_reward
            if s_tp1[ColoringEnv.channel_pattern][new_agent_position] == ColoringEnv.emb_pattern_to_color:
                if self.with_color_reward and s_tp1[ColoringEnv.channel_stitch][new_agent_position] != ColoringEnv.emb_pattern_to_color:
                    ret_val -= self.c.step_reward
                    #ret_val = 1
                s_tp1[ColoringEnv.channel_stitch][new_agent_position] = ColoringEnv.emb_pattern_to_color
            return ret_val
        curr_agent_position = self._get_agent_position()
        if a_t == 'u':
            r_tp1 = move_agent((-1, 0))
        elif a_t == 'd':
            r_tp1 = move_agent((+1, 0))
        elif a_t == 'l':
            r_tp1 = move_agent((0, -1))
        elif a_t == 'r':
            r_tp1 = move_agent((0, +1))

        #print('step a_t', a_t, 'state:', self.base_observation, 's_tp1', s_tp1)

        # done stitching if all color cells are stitched
        covered_count = self.covered_count(s_tp1)
        if self.step_count >= self.max_steps or covered_count == self.emb_pattern_count:
            self.done = True
        self.base_observation = s_tp1

        if self.done:
            if self.total_reward:
                r_tp1 += self.c.step_reward * self.step_count
            if self.covered_steps_ratio:
                r_tp1 += covered_count / self.step_count

        if r_tp1 is None:
            raise RuntimeError('Return cannot be None')

        self.steps.append([curr_agent_position, a_t, r_tp1, self.done])
        assert self.step_count == len(self.steps)

        ret_s_tp1 = self._gen_state(s_tp1)
        reduced_ratio = self.init_uncovered_count / self.step_count
        assert not self.done or 0 < reduced_ratio <= 1
        infos = {
            'episode': None,
            'steps': self.steps,
            'steps_count': self.step_count,
            'covered_count': covered_count,
            'total_count': self.emb_pattern_count,
            'init_uncovered_count': self.init_uncovered_count,
            'reduced_ratio': reduced_ratio
        }
        return ret_s_tp1, r_tp1, self.done, infos

    def render(self):
        x = PrettyTable(['{}, {} (Layer {})'.format(self.worker_id, v, k) for k, v in self.layer_descriptions.items()])
        row = []
        for r in self.base_observation:
            # https://stackoverflow.com/a/42046765/256002
            bio = io.BytesIO()
            np.savetxt(bio, r, fmt='%d')
            mystr = bio.getvalue().decode('latin1').rstrip('\n')
            row.append(mystr)
        x.add_row(row)
        return x.get_string()

    def __str__(self):
        return '<{}>'.format(type(self).__name__)


