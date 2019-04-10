#!/usr/bin/env python
# encoding: utf-8


"""
@version: 0.0
@author: hailang
@Email: seahailang@gmail.com
@software: PyCharm
@file: config.py
@time: 2019/3/26 11:01
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mode',default='train')
parser.add_argument('--epochs',default=20)
parser.add_argument('--batch_size',default=256)

args = parser.parse_args()

VOCAB_SIZE = 8000

class Config(object):
    def __init__(self):
        self.num_vocab = 8000
        self.vocab_size = 128
        self.num_pos = 24
        self.pos_size = 128
        self.position_size = 64
        self.d_model = 256
        self.num_units = 256

        self.num_target = 49
        self.target_size = 4

        self.mode = args.mode

        
        self.train_epochs = args.epochs
        self.batch_size = args.batch_size
        self.val_steps = 21632//self.batch_size + 1
        self.steps_each_epoch = 172991//self.batch_size + 1

        self.infer_steps = 9949//self.batch_size + 1

        self.optimizer = 'adam'
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.learning_rate_decay = False
        self.learning_rate = 1e-3
        self.decay_steps = 5000
        self.decay_rate = 0.5

        self.ckpt_path = './ckpt/'
        self.ckpt_name = 'baidu'

if __name__ == '__main__':
    pass