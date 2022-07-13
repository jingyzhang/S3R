#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/10/27 9:25 PM
# @Author  : Jingyang.Zhang
'''
CISR for cross-site continual learning
'''
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import random
import sys
import argparse
import torch
import numpy as np
from random import shuffle
from collections import OrderedDict
import agents
import dataloaders.base
from utils import metric
from torch.utils.data import DataLoader

torch.manual_seed(123)
torch.cuda.manual_seed(123)
torch.cuda.manual_seed_all(123)
np.random.seed(123)
random.seed(123)
torch.manual_seed(123)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.set_default_tensor_type('torch.FloatTensor')


def get_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpuid', nargs="+", type=int, default=[0])
    parser.add_argument('--model_type', type=str, default='seg_regression')
    parser.add_argument('--model_name', type=str, default='Unet_regression')
    parser.add_argument('--agent_type', type=str, default='regularization_CISR')
    parser.add_argument('--agent_name', type=str, default='Reg_CISR')
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--dataroot', type=str, default='data')
    parser.add_argument('--dataset', type=str, default='MultiSite')
    parser.add_argument('--exp_dir', type=str, default='./exp_2022/debug')
    parser.add_argument('--alpha', type=float, default=0.001, help='Weight for the importance for backward of loss_regression.')
    parser.add_argument('--alpha1', type=float, default=0.05, help='Weight for the importance for backward of loss_embedding.')
    parser.add_argument('--beta', type=float, default=1.0, help='Weight for loss_regression.')
    parser.add_argument('--beta1', type=float, default=1.0, help='Weight for loss_embedding.')
    parser.add_argument('--reg_coef', nargs="+", type=float, default=[10000.], help="The weight for selective regularization.")

    parser.add_argument('--epoches', type=int, default=200, help="#Epoches for training")
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--reset_optimizer', dest='reset_optimizer', default=False, action='store_true', help='Whether to reset optimizer.')
    parser.add_argument('--lr', type=float, default=5e-4, help="Learning rate")
    parser.add_argument('--momentum', type=float, default=0, help='Used for SGD.. rather than Adam')
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--print_freq', type=float, default=100, help="Print the log at every x iteration")
    parser.add_argument('--model_weights', type=str, default=None, help="The path to the file for the model weights (*.pth).")
    parser.add_argument('--offline_training', dest='offline_training', default=False, action='store_true',
                        help="Non-incremental learning by make all data available in one batch. For measuring the upperbound performance.")

    args = parser.parse_args(argv)
    return args


def run(args):
    agent_config = {'lr': args.lr, 'momentum': args.momentum, 'weight_decay': args.weight_decay,
                    'reset_optimizer':args.reset_optimizer,
                    'model_type':args.model_type, 'model_name': args.model_name, 'model_weights':args.model_weights,
                    'alpha':args.alpha,
                    'alpha1':args.alpha1,
                    'uc_threshold':args.uc_threshold,
                    'beta':args.beta,
                    'beta1':args.beta1,
                    'save_best':args.save_best,
                    'epoches':args.epoches,
                    'exp_dir':args.exp_dir,
                    'optimizer':args.optimizer,
                    'print_freq':args.print_freq, 'gpuid': args.gpuid,
                    'reg_coef':args.reg_coef}

    print(agent_config)
    agent = agents.__dict__[args.agent_type].__dict__[args.agent_name](agent_config)
    # print(agent.model)
    print('#parameter of model:',agent.count_parameter())

    site_names = ['ISBI', 'ISBI_1.5', 'I2CVB', 'UCL', 'BIDMC', 'HK']
    print('Site order:', site_names)

    if args.offline_training:
        train_loader, val_loader = dataloaders.base.__dict__[args.dataset](args.dataroot, site_names, args.batch_size)

        # training with minibatch
        agent.learn_batch('joint', train_loader, val_loader)

    else:
        train_loaders, val_loaders = dataloaders.base.__dict__[args.dataset](args.dataroot, site_names, args.batch_size)

        # continual learning with minibatch on each site
        for i, site_name in enumerate(site_names):
            print('\r\r======================',site_name,'=======================')
            train_loader = train_loaders[site_name]
            val_loader = val_loaders[site_name]

            agent.learn_batch(site_name, train_loader, val_loaders)





if __name__ == '__main__':
    args = get_args(sys.argv[1:])
    run(args)