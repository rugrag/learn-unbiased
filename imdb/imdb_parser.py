# -*- coding: utf-8 -*-
import argparse
import os

parser = argparse.ArgumentParser()



parser.add_argument('--exp_name',                   default='eb1',                                  help='base, eb1, eb2')
parser.add_argument('--tr_samples_eb1',             default=2000,         type=int,                   help='number of samples for eb1. use -1 for all')
parser.add_argument('--tr_samples_eb2',             default=-1,         type=int,                   help='number of samples for eb1. use -1 for all')
parser.add_argument('--num_epochs',                 default=10,          type=int,                   help='num epochs')
parser.add_argument('--batch_size',                 default=24,         type=int,                   help='mini-batch size')
parser.add_argument('--lr_C',                       default=0.00001,    type=float,                 help='learning rate feature extractor optimizer')
parser.add_argument('--lr_M',                       default=0.1,        type=float,                 help='learning rate MI estimator')
parser.add_argument('--num_iter_MI',                default=20,         type=int,                   help='iterations mine update')
parser.add_argument('--num_iter_accumulation',      default=10,         type=int,                   help='number of accumulated batches')
parser.add_argument('--dim_z',                      default=128,        type=int,                   help='bottleneck dimension')
parser.add_argument('--lmb',                        default=0.0,        type=float,                 help='lambda hyperparameter')
parser.add_argument('--run',                        default=1,          type=int,                   help='run ID')



def get_config(exp_dir): #imdb/experiments

    config = parser.parse_args()
    config.folder_name = config.exp_name + '_lmb_' + str(float(config.lmb)) + '_iter_MI_' + str(config.num_iter_MI) + '_iter_acc_'+ str(config.num_iter_accumulation) + '_lr_M_' + str(config.lr_M)
    exp_dir = os.path.join(exp_dir, config.exp_name, config.folder_name)

    config.summary_dir    = os.path.join(exp_dir, "summary")
    config.checkpoint_dir = os.path.join(exp_dir, "checkpoint")

    # create dirs for summaries and checkpoints
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
        f.write('num_iter_acc: {:d}\n'.format(config.num_iter_accumulation))
        f.write('dim_z: {:d}\n'.format(config.dim_z))
        f.write('lmb: {:.8f}\n'.format(config.lmb))
        f.write('run: {:d}\n'.format(config.run))


    return config
