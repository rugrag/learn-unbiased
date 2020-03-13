# -*- coding: utf-8 -*-
import argparse
import os

parser = argparse.ArgumentParser()


parser.add_argument('--var',                 default='0.020',                           help='0.020')
parser.add_argument('--num_epochs',          default=501,     type=int,                 help='num epochs')
parser.add_argument('--batch_size',          default=128,     type=int,                 help='mini-batch size')
parser.add_argument('--lr_C',                default=0.0001,  type=float,               help='learning rate')
parser.add_argument('--num_iter_MI',         default=40,      type=int,                 help='iterations mine update ')
parser.add_argument('--dim_z',               default=64,      type=int,                 help='dimension bottleneck')
parser.add_argument('--dim_c',               default=8,       type=int,                 help='number of bins for c label')
parser.add_argument('--lmb',                 default=1.0,     type=float,               help='value for lambda')
parser.add_argument('--run',                 default=1,       type=int,                 help='run ID')


def get_config(exp_dir): #imdb/experiments
    config = parser.parse_args()
    config.folder_name = 'col_' + config.var + '_lmb_' + str(float(config.lmb)) + '_iter_MI_' + str(config.num_iter_MI)
    exp_dir = os.path.join(exp_dir, config.var, config.folder_name)

    config.summary_dir    = os.path.join(exp_dir, "summary")
    config.checkpoint_dir = os.path.join(exp_dir, "checkpoint")

    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    if not os.path.exists(config.summary_dir):
        os.makedirs(config.summary_dir)

    if not os.path.exists(config.checkpoint_dir):
        os.makedirs(config.checkpoint_dir)

    config.summary_dir    = os.path.join(exp_dir, "summary", str(config.run))
    config.checkpoint_dir = os.path.join(exp_dir, "checkpoint", str(config.run)+"/")

    if not os.path.exists(config.summary_dir):
        os.makedirs(config.summary_dir)

    if not os.path.exists(config.checkpoint_dir):
        os.makedirs(config.checkpoint_dir)

    # write in a log file all model parameters
    with open(os.path.join(exp_dir, 'config_file.txt'), 'w') as f:
        f.write('num_epochs: {:d}\n'.format(config.num_epochs))
        f.write('batch_size: {:d}\n'.format(config.batch_size))
        f.write('lr_C: {:.8f}\n'.format(config.lr_C))
        f.write('num_iter_MI: {:d}\n'.format(config.num_iter_MI))
        f.write('dim_z: {:d}\n'.format(config.dim_z))
        f.write('lmb: {:.8f}\n'.format(config.lmb))
        f.write('run: {:d}\n'.format(config.run))

    return config
