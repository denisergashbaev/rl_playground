from os.path import join

""" Constants representing directories wher the raw input files are stored
or where output of Mapper/Reducer is written to"""

CODE_ROOT_DIR = join('algs')
OUT_DIR = join(CODE_ROOT_DIR, 'out')
OUT_SB3_MONITOR = join(OUT_DIR, 'stable_baselines3_monitor')
