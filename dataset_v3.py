#!/usr/bin/env python
# encoding: utf-8


"""
@version: 0.0
@author: hailang
@Email: seahailang@gmail.com
@software: PyCharm
@file: dataset_v3.py
@time: 2019/4/11 10:54
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import json
from config import Config

my_conf =Config()
VOCAB_SIZE = my_conf.num_vocab
num_target = my_conf.num_target

id2char = [s.strip() for s in open('./char_map.txt',encoding='utf-8')]
char2id = {s:i for i,s in enumerate(id2char) if i<VOCAB_SIZE}

id2word = [s.strip() for s in open('./word_map.txt',encoding='utf-8')]
word2id = {s:i for i,s in enumerate(id2word) if i<VOCAB_SIZE}

id2pos = [s.strip() for s in open('./pos_map.txt',encoding='utf-8')]
pos2id = {s:i for i,s in enumerate(id2pos)}

id2spo = [s.strip() for s in open('./relation_map.txt',encoding='utf-8')]
spo2id = {s:i for i,s in enumerate(id2spo)}

OUTPUT_TYPES = (tf.int32,tf.int32,tf.int32,tf.int32,tf.int32)
OUTPUT_SHAPES = ([None,None],[None,None],[None],[],[None,None,num_target*4])

def decode_train(line,is_char):
    if is_char:
        w2i = char2id
    else:
        w2i = word2id
    line = line.decode('utf-8')
    item = json.loads(line)
    pos_list = item['pos_list']
    relations = item['relations']
    text = []
    pos = []
    length = []
    target = []
    for pos_tag in pos_list:
        text.append(w2i.get(pos_tag['word'], 1))
        pos.append(pos2id.get(pos_tag['pos']))
        target.append([0] * num_target * 4)
        # target.append([0]*2)
    length.append(len(text))
    for relation in relations:
        idx = spo2id.get(relation['predicate'])
        # idx = 0
        target[relation['subject_begin']][idx] = 1
        target[relation['subject_end']][idx + num_target] = 1
        target[relation['object_begin']][idx + 2 * num_target] = 1
        target[relation['object_end']][idx + 3 * num_target] = 1
    return np.array(text).astype(np.int32),\
           np.array(pos).astype(np.int32),\
           np.array(length).astype(np.int32),\
           np.array(target).astype(np.int32)

def decode_test(line,is_char):
    if is_char:
        w2i = char2id
    else:
        w2i = word2id
    line = line.decode('utf-8')
    item = json.loads(line)
    pos_list = item['pos_list']
    text = []
    pos = []
    length = []
    for pos_tag in pos_list:
        text.append(w2i.get(pos_tag['word'], 1))
        pos.append(pos2id.get(pos_tag['pos']))
    length.append(len(text))
    return np.array(text).astype(np.int32),\
           np.array(pos).astype(np.int32),\
           np.array(length).astype(np.int32)


def make_train_dataset(files,batch_size=128,is_char=True,shuffle=True):
    if not isinstance(files,list):
        files = [files]
    dataset = tf.data.TextLineDataset(files)

    decode_fn = lambda line:decode_train(line,is_char)
    map_fn = lambda line:tf.py_func(decode_fn,inp=[line],Tout=[tf.int32,tf.int32,tf.int32,tf.int32])
    if shuffle:
        dataset = dataset.shuffle(1000)
    dataset = dataset.map(map_fn,num_parallel_calls=32)
    dataset = dataset.padded_batch(batch_size=batch_size,padded_shapes=([-1],[-1],[1],[-1,num_target * 4]))
    dataset = dataset.repeat()
    return dataset

def make_test_dataset(files,batch_size=128,is_char=True):
    if not isinstance(files,list):
        files = [files]
    dataset = tf.data.TextLineDataset(files)

    decode_fn = lambda line:decode_test(line,is_char)
    map_fn = lambda line:tf.py_func(decode_fn,inp=[line],Tout=[tf.int32,tf.int32,tf.int32])
    dataset = dataset.map(map_fn)
    dataset = dataset.padded_batch(batch_size=batch_size,padded_shapes=([-1],[-1],[1]))
    dataset = dataset.repeat()
    return dataset

if __name__ == '__main__':
    dataset = make_train_dataset('./data/train_data_char.json',256)
    iterator = dataset.make_one_shot_iterator()
    a = iterator.get_next()
    sess = tf.Session()
    for i in range(1000000):
        array = sess.run(a)
        print(i)
        print(array[0].shape)
        print(array[1].shape)
        print(array[2].shape)