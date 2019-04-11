#!/usr/bin/env python
# encoding: utf-8


"""
@version: 0.0
@author: hailang
@Email: seahailang@gmail.com
@software: PyCharm
@file: dataset_v2.py.py
@time: 2019/4/11 9:52
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

OUTPUT_TYPES = (tf.int32,tf.int32,tf.int32,tf.int32)
OUTPUT_SHAPES = ([None,None],[None,None],[None],[None,None,num_target*4])

class train_generator(object):
    def __init__(self,data_path,batch_size=128,is_char=True,num_threads=8):
        train_files = data_path

        if is_char:
            w2i = char2id
        else:
            w2i = word2id
        self.T = []
        self.P = []
        self.L = []
        self.Y = []
        if not isinstance(train_files,list):
            train_files = [train_files]
        for filename in train_files:
            with open(filename,encoding='utf-8') as file:
                for line in file.readlines():
                    item = json.loads(line)
                    pos_list = item['pos_list']
                    relations = item['relations']
                    text = []
                    pos = []
                    target = []

                    for pos_tag in pos_list:
                        text.append(w2i.get(pos_tag['word'], 1))
                        pos.append(pos2id.get(pos_tag['pos']))
                        target.append([0] * num_target * 4)
                        # target.append([0]*2)
                    for relation in relations:
                        idx = spo2id.get(relation['predicate'])
                        # idx = 0
                        target[relation['subject_begin']][idx] = 1
                        target[relation['subject_end']][idx + num_target] = 1
                        target[relation['object_begin']][idx + 2 * num_target] = 1
                        target[relation['object_end']][idx + 3 * num_target] = 1
                    self.T.append(text)
                    self.P.append(pos)
                    self.Y.append(target)
                    self.L.append(len(text))

        self.steps = len(self.L)
        self.batch_size = batch_size
        self.num_threads = num_threads
    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            idxs = list(range(self.steps))
            np.random.shuffle(idxs)
            T = []
            P = []
            L = []
            Y = []
            for i in idxs:
                T.append(self.T[i].copy())
                P.append(self.P[i].copy())
                L.append(self.L[i])
                Y.append(self.Y[i].copy())


                if len(T) == self.batch_size or i == idxs[-1]:
                    max_length = max(L)
                    if i == idxs[-1]:
                        res = self.batch_size-len(T)
                        T.extend([[] for _ in range(res)])
                        P.extend([[] for _ in range(res)])
                        Y.extend([[] for _ in range(res)])
                        L.extend([0 for _ in range(res)])

                    for j in range(len(T)):
                        if L[j] < max_length:
                            T[j].extend([0]*(max_length-L[j]))
                            P[j].extend([0]*(max_length-L[j]))
                            Y[j].extend([[0]*num_target*4]*(max_length-L[j]))


                    yield np.array(T).astype(np.int32),\
                          np.array(P).astype(np.int32),\
                          np.array(L).astype(np.int32),\
                          np.array(Y).astype(np.int32)
                    T,P,L,Y = [],[],[],[]

    def __call__(self):
        return self.__iter__()





class infer_generator(object):
    def __init__(self,data_path,batch_size=128,is_char=True):
        train_files = data_path

        if is_char:
            w2i = char2id
        else:
            w2i = word2id
        T = []
        P = []
        L = []
        if not isinstance(train_files,list):
            train_files = [train_files]
        for filename in train_files:
            with open(filename,encoding='utf-8') as file:
                for line in file:
                    item = json.loads(line)
                    pos_list = item['pos_list']
                    relations = item['relations']
                    text = []
                    pos = []
                    target = []

                    for pos_tag in pos_list:
                        text.append(w2i.get(pos_tag['word'], 1))
                        pos.append(pos2id.get(pos_tag['pos']))
                        target.append([0] * num_target * 4)
                        # target.append([0]*2)
                    for relation in relations:
                        idx = spo2id.get(relation['predicate'])
                        # idx = 0
                        target[relation['subject_begin']][idx] = 1
                        target[relation['subject_end']][idx + num_target] = 1
                        target[relation['object_begin']][idx + 2 * num_target] = 1
                        target[relation['object_end']][idx + 3 * num_target] = 1
                    T.append(text)
                    P.append(pos)
                    L.append(len(text))
        self.T = T
        self.P = P
        self.L = L
        self.steps = len(L)
        self.batch_size = batch_size
    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            idxs = list(range(self.steps))
            T = []
            P = []
            L = []
            for i in idxs:
                T.append(self.T[i].copy())
                P.append(self.P[i].copy())
                L.append(self.L[i])
                if len(T) == self.batch_size or i == idxs[-1]:
                    max_length = max(L)
                    if i == idxs[-1]:
                        res = self.batch_size-len(T)
                        T.extend([[] for _ in range(res)])
                        P.extend([[] for _ in range(res)])
                        L.extend([0 for _ in range(res)])

                    for j in range(len(T)):
                        if L[j] < max_length:
                            T[j].extend([0]*(max_length-L[j]))
                            P[j].extend([0]*(max_length-L[j]))

                    yield np.array(T).astype(np.int32),\
                          np.array(P).astype(np.int32),\
                          np.array(L).astype(np.int32)
                    T,P,L = [],[],[]
    def __call__(self):
        return self.__iter__()

def make_train_dataset(files,batch_size=128,is_char=True):
    tg = train_generator(files,batch_size,is_char)
    train_set = tf.data.Dataset.from_generator(tg,output_shapes=OUTPUT_SHAPES,output_types=OUTPUT_TYPES)
    return train_set
def make_test_dataset(files,batch_size,is_char=True):
    tg = infer_generator(files,batch_size,is_char)
    infer_set = tf.data.Dataset.from_generator(tg,output_shapes=OUTPUT_SHAPES[-1],output_types=OUTPUT_TYPES[-1])
    return infer_set






if __name__ == '__main__':
    tg = train_generator(data_path='./data/dev_data_char.json',batch_size = 256)
    train_set = tf.data.Dataset.from_generator(tg,output_shapes=OUTPUT_SHAPES,output_types=OUTPUT_TYPES)
    iterator = train_set.make_one_shot_iterator()
    a = iterator.get_next()
    sess = tf.Session()
    for i in range(1000000):
        print(i)
        sess.run(a)
    sess.close()
        