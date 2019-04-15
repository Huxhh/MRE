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
import math
parser = argparse.ArgumentParser()
parser.add_argument('--mode',default='train')
parser.add_argument('--epochs',default=100,type=int)
parser.add_argument('--batch_size',default=256,type=int)

args = parser.parse_args()

VOCAB_SIZE = 8000

class Config(object):
    def __init__(self):
        self.is_char = True
        self.num_vocab = 8000
        self.vocab_size = 128
        self.num_pos = 25
        self.pos_size = 128
        self.position_size = 64
        self.d_model = 256
        self.num_units = 256

        self.num_target = 49
        self.target_size = 4

        self.mode = args.mode

        
        self.train_epochs = args.epochs
        self.batch_size = args.batch_size
        self.val_steps = math.ceil(21639/self.batch_size)
        self.steps_each_epoch = math.ceil(173108/self.batch_size)

        self.infer_steps = math.ceil(9949/self.batch_size)

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