from os import path

this_dir = path.join(path.dirname(path.realpath(__file__)))
data_dir = path.join(this_dir, '..', 'data')
mnist = 'mnist'
two_by_two = '2x2'
ten_by_ten = '10x10'
mnist_files_dir = path.join(data_dir, mnist)
two_by_two_files_dir = path.join(data_dir, two_by_two)
ten_by_ten_files_dir = path.join(data_dir, ten_by_ten)
mnist_28x28_train_dir = path.join(mnist_files_dir, '28x28_train')
mnist_28x28_test_dir = path.join(mnist_files_dir, '28x28_test')
gui_dir = path.join(this_dir, '..', 'gui')

out_dir = path.join(this_dir, '..', 'out')

dqn_dir = path.join(out_dir, 'dqn/')
ddqn_dir = path.join(out_dir, 'ddqn/')
a3c_dir = path.join(out_dir, 'a3c/')
pytorch_dir = path.join(out_dir, 'pytorch/')
stable_baselines = path.join(out_dir, 'stable_baselines/')

ep_stats_file_name = 'ep_stats'
