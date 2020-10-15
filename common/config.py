import os
import numpy as np
from utils import constants, helper
import copy
import datetime
import multiprocessing


class Config:
    def __init__(self, data_dir, data_files, avg_reward_data_files, step_reward, fast_fail, load_dir, discount_factor,
                 num_workers, with_global_net, update_batch_size, max_global_steps,
                 keep_last_agent_moves, prefer_nearby_cells, merged_network, total_loss,
                 test, debug, out_dir):
        self.timestamp = None
        self.data_dir = data_dir
        self.data_files = data_files
        self.loaded_data_files = []
        self.step_reward = step_reward
        self.fast_fail = fast_fail
        # directory from which the checkpoints can be loaded
        self.load_dir = load_dir
        self.discount_factor = discount_factor
        self.num_workers = num_workers
        self.with_global_net = with_global_net
        self.update_batch_size = update_batch_size
        self.max_global_steps = max_global_steps
        self.keep_last_agent_moves = keep_last_agent_moves
        self.prefer_nearby_cells = prefer_nearby_cells
        self.merged_network = merged_network
        self.total_loss = total_loss
        self.test = test
        self.debug = debug
        self.out_dir = out_dir
        self.hostname = os.environ['HOME']

    @staticmethod
    def create_for_test(alg, load_dir, data_dir, run_on):
        settings = helper.load_settings(alg, load_dir)
        settings['out_dir'] = alg
        settings['test'] = True
        settings['load_dir'] = load_dir

        for exclude in ['timestamp', 'hostname', 'hostname', 'loaded_data_files',  'valid_eval_data_files']:
            try:
                del settings[exclude]
            except KeyError:
                pass
        #print('ONLY USING 1 WORKER FOR SPEED')
        settings['num_workers'] = 1
        settings['data_dir'] = data_dir
        settings['data_files'] = run_on

        c = Config(**settings)
        return c

    def init(self):
        assert(not self.total_loss or self.merged_network)
        assert(self.num_workers == 1 or self.with_global_net)
        # reinitialize because i do deepcopy
        self.loaded_data_files = []
        if 'all' in self.data_files:
            directory = os.fsencode(self.data_dir)
            for file in os.listdir(directory):
                fname = os.fsdecode(file)
                if fname.endswith('.npy'):
                    self.loaded_data_files.append((fname, np.load(os.path.join(self.data_dir, fname))))
        else:
            for fname in self.data_files:
                self.loaded_data_files.append((fname, np.load(os.path.join(self.data_dir, '{}.npy'.format(fname)))))
        print('Going to use these files:', [a[0] for a in self.loaded_data_files])

    def create_dirs(self):
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d-%H:%M:%S.%f")
        o_dir = self.get_out_dir()
        for d in [o_dir, self.get_checkpoints_save_dir(), self.get_logs_dir(), self.get_stats_dir()]:
            if not os.path.exists(d):
                os.makedirs(d)
        settings = vars(self)
        np.save(os.path.join(o_dir, 'settings.npy'), settings)
        with open(os.path.join(o_dir, 'settings.txt'), 'w') as f:
            f.write(self.get_settings())

    def get_settings(self):
        s = []
        for k, v in vars(self).items():
            if k == 'loaded_data_files':
                v = [val[0] for val in v]
            s.append('{}: {}'.format(k, v))
        return '\n'.join(s)

    def get_out_dir(self):
        s = [self.timestamp, 'files={}'.format(','.join(self.data_files))]
        if self.test:
            s.append('test')
        return os.path.join(self.out_dir, '_'.join(s))

    def get_stats_dir(self):
        return os.path.join(self.get_out_dir(), 'stats')

    def get_checkpoints_save_dir(self):
        return os.path.join(self.get_out_dir(), 'checkpoints')

    def get_checkpoints_load_dir(self):
        return os.path.join(self.out_dir, self.load_dir, 'checkpoints')

    def get_logs_dir(self):
        return os.path.join(self.get_out_dir(), 'logs')

    def load_file(self, env_idx):
        fname, file = self.loaded_data_files[env_idx % len(self.loaded_data_files)]
        return fname, np.copy(file)


# training
dqn_config_train_2x2 = Config(
    data_dir=constants.data_dir,
    data_files=['2x2'],  # '5_15.npy',  # 2x2.npy, 7_17.npy, 0_13.npy
    avg_reward_data_files=None,
    step_reward=-0.01,  # 0.01,  # -0.1, -0.5, -1
    fast_fail=True,
    load_dir=False,
    discount_factor=0.99,
    num_workers=min(multiprocessing.cpu_count(), 4) - 1,
    with_global_net=True,
    update_batch_size=20,
    max_global_steps=100e3,
    keep_last_agent_moves=0,
    prefer_nearby_cells=True,
    merged_network=True,
    total_loss=False,
    # False or folder name
    test=False,
    debug=False,
    out_dir=constants.dqn_dir
)

dqn_config_train_0 = copy.deepcopy(dqn_config_train_2x2)
dqn_config_train_0.data_dir = constants.mnist_28x28_train_dir
dqn_config_train_0.data_files = ['0_13']

a3c_config_train_2x2 = copy.deepcopy(dqn_config_train_2x2)
a3c_config_train_2x2.out_dir = constants.a3c_dir

a2c_config_train_2x2 = copy.deepcopy(a3c_config_train_2x2)
a2c_config_train_2x2.data_dir = constants.two_by_two_files_dir
a2c_config_train_2x2.data_files = ['2x2']
a2c_config_train_2x2.avg_reward_data_files = ['0_88', '2_26', '4_4', '6_84', '9_83']  #['0_88', '1_9', '2_26', '3_73', '4_4', '5_13', '6_84', '7_17', '8_78', '9_83']
a2c_config_train_2x2.max_global_steps = 25e3
a2c_config_train_2x2.keep_last_agent_moves = 0
a2c_config_train_2x2.prefer_nearby_cells = False
a2c_config_train_2x2.merged_network = False
a2c_config_train_2x2.total_loss = False
a2c_config_train_2x2.step_reward = -0.1

a2c_config_train_10x10 = copy.deepcopy(a3c_config_train_2x2)
a2c_config_train_10x10.data_dir = constants.ten_by_ten_files_dir
a2c_config_train_10x10.num_workers = 32
a2c_config_train_10x10.data_files = ['0_13']
a2c_config_train_10x10.avg_reward_data_files = ['0_13']  #['0_88', '1_9', '2_26', '3_73', '4_4', '5_13', '6_84', '7_17', '8_78', '9_83']
a2c_config_train_10x10.valid_eval_data_files = ['0_13']
a2c_config_train_10x10.max_global_steps = 25e3
a2c_config_train_10x10.keep_last_agent_moves = 0
a2c_config_train_10x10.prefer_nearby_cells = False
a2c_config_train_10x10.merged_network = False
a2c_config_train_10x10.total_loss = False
a2c_config_train_10x10.step_reward = -0.1
