import unittest
import common.config as cfg
import numpy as np
from common.env import ColoringEnv


class TestColoringEnv(unittest.TestCase):

    def setUp(self) -> None:
        self.envs = {}
        self.c = cfg.a2c_config_train_2x2
        self.c.step_reward = -0.1
        self.c.init()
        self.start_position = (1, 0)
        depth_channel_first = True
        env_as_image = False
        self.envs['env0'] = ColoringEnv(self.c, 'with_step_penalty=True', self.start_position,
                                        depth_channel_first=depth_channel_first,
                                        with_step_penalty=True,
                                        with_revisit_penalty=False,
                                        stay_inside=True,
                                        with_color_reward=False, total_reward=False,
                                        covered_steps_ratio=False,
                                        as_image=env_as_image)

        self.envs['env_all'] = ColoringEnv(self.c, 'all', self.start_position, depth_channel_first=depth_channel_first,
                                           with_step_penalty=True,
                                           with_revisit_penalty=True,
                                           stay_inside=True,
                                           with_color_reward=True, total_reward=False,
                                           covered_steps_ratio=False,
                                           as_image=env_as_image)

    @staticmethod
    def agent_position(state, coords):
        _, *dim = state.shape
        agent_layer = np.full(dim, ColoringEnv.agent_pos_empty)
        agent_layer[coords[0]][coords[1]] = ColoringEnv.agent_pos_code
        return np.all(state[ColoringEnv.channel_agent] == agent_layer)

    @staticmethod
    def stitch_layer(state, *kwargs):
        _, *dim = state.shape
        stitch_layer = np.full(dim, ColoringEnv.emb_pattern_empty)
        for coords in kwargs:
            stitch_layer[coords[0]][coords[1]] = ColoringEnv.emb_pattern_to_color
        return np.all(state[ColoringEnv.channel_stitch] == stitch_layer)

    def test_init(self):
        for env in self.envs.values():
            state = env.reset()
            self.assertTrue(TestColoringEnv.agent_position(state, self.start_position))
            _, *dim = state.shape
            pattern_layer = np.full(dim, ColoringEnv.emb_pattern_to_color)
            pattern_layer[0][1] = ColoringEnv.emb_pattern_empty
            self.assertTrue(np.all(state[ColoringEnv.channel_pattern] == pattern_layer))
            self.assertTrue(TestColoringEnv.stitch_layer(state, (1, 0)))

    def test_env0(self):
        env = self.envs['env0']
        _ = env.reset()
        # down (from position (1, 0))
        s, r, d, _ = env.step(env.inv_action_encodings['d'])
        self.assertEqual(r, self.c.step_reward)
        self.assertTrue(TestColoringEnv.agent_position(s, self.start_position))
        self.assertTrue(TestColoringEnv.stitch_layer(s, (1, 0)))
        self.assertFalse(d)
        # right (from position (1, 0))
        s, r, d, _ = env.step(env.inv_action_encodings['r'])
        self.assertEqual(r, self.c.step_reward)
        self.assertTrue(TestColoringEnv.agent_position(s, (1, 1)))
        self.assertTrue(TestColoringEnv.stitch_layer(s, (1, 0), (1, 1)))
        self.assertFalse(d)
        # right (from position (1, 1))
        s, r, d, _ = env.step(env.inv_action_encodings['r'])
        self.assertEqual(r, self.c.step_reward)
        self.assertTrue(TestColoringEnv.agent_position(s, (1, 1)))
        self.assertTrue(TestColoringEnv.stitch_layer(s, (1, 0), (1, 1)))
        self.assertFalse(d)
        # left (from position (1, 1))
        s, r, d, _ = env.step(env.inv_action_encodings['l'])
        self.assertEqual(r, self.c.step_reward)
        self.assertTrue(TestColoringEnv.agent_position(s, (1, 0)))
        self.assertTrue(TestColoringEnv.stitch_layer(s, (1, 0), (1, 1)))
        self.assertFalse(d)
        # up (from position (1, 1))
        s, r, d, _ = env.step(env.inv_action_encodings['u'])
        self.assertEqual(r, self.c.step_reward)
        self.assertTrue(TestColoringEnv.agent_position(s, (0, 0)))
        self.assertTrue(TestColoringEnv.stitch_layer(s, (1, 0), (1, 1), (0, 0)))
        self.assertTrue(d)

    def test_env_all(self):
        env = self.envs['env_all']
        _ = env.reset()
        # down (from position (1, 0))
        s, r, d, _ = env.step(env.inv_action_encodings['d'])
        self.assertEqual(r, self.c.step_reward + self.c.step_reward)
        self.assertTrue(TestColoringEnv.agent_position(s, self.start_position))
        self.assertTrue(TestColoringEnv.stitch_layer(s, (1, 0)))
        self.assertFalse(d)
        # right (from position (1, 0))
        s, r, d, _ = env.step(env.inv_action_encodings['r'])
        self.assertEqual(r, self.c.step_reward + abs(self.c.step_reward))
        self.assertTrue(TestColoringEnv.agent_position(s, (1, 1)))
        self.assertTrue(TestColoringEnv.stitch_layer(s, (1, 0), (1, 1)))
        self.assertFalse(d)
        # right (from position (1, 1))
        s, r, d, _ = env.step(env.inv_action_encodings['r'])
        self.assertEqual(r, self.c.step_reward + self.c.step_reward)
        self.assertTrue(TestColoringEnv.agent_position(s, (1, 1)))
        self.assertTrue(TestColoringEnv.stitch_layer(s, (1, 0), (1, 1)))
        self.assertFalse(d)
        # left (from position (1, 1))
        s, r, d, _ = env.step(env.inv_action_encodings['l'])
        self.assertEqual(r, self.c.step_reward + self.c.step_reward)
        self.assertTrue(TestColoringEnv.agent_position(s, (1, 0)))
        self.assertTrue(TestColoringEnv.stitch_layer(s, (1, 0), (1, 1)))
        self.assertFalse(d)
        # up (from position (1, 1))
        s, r, d, _ = env.step(env.inv_action_encodings['u'])
        self.assertEqual(r, self.c.step_reward + abs(self.c.step_reward))
        self.assertTrue(TestColoringEnv.agent_position(s, (0, 0)))
        self.assertTrue(TestColoringEnv.stitch_layer(s, (1, 0), (1, 1), (0, 0)))
        self.assertTrue(d)


if __name__ == '__main__':
    unittest.main()