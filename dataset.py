#!/usr/bin/env python
# encoding: utf-8


"""
@version: 0.0
@author: hailang
@Email: seahailang@gmail.com
@software: PyCharm
@file: dataset.py
@time: 2019/3/26 9:34
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
# OUTPUT_SHAPES = ([None,None],[None,None],[None],[],[None,None,2])

class train_generator(object):
    def __init__(self,data_path,batch_size=128,is_char=True):
        data = []
        with open(data_path,encoding='utf-8') as file:
            for line in file:
                data.append(json.loads(line))
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data)//self.batch_size
        if len(self.data)%self.batch_size != 0:
            self.steps += 1
        if is_char:
            self.word2id = char2id
        else:
            self.word2id = word2id
    def __len__(self):
        return self.steps

    def __iter__(self):
        w2i = self.word2id
        while True:
            idxs = list(range(len(self.data)))
            np.random.shuffle(idxs)

            T, P, L, Y = [], [], [], []

            for i in idxs:
                item = self.data[i]
                pos_list = item['pos_list']
                relations = item['relations']
                text = []
                pos = []
                target = []

                for pos_tag in pos_list:
                    text.append(w2i.get(pos_tag['word'],1))
                    pos.append(pos2id.get(pos_tag['pos']))
                    target.append([0]*num_target*4)
                    # target.append([0]*2)
                for relation in relations:
                    idx = spo2id.get(relation['predicate'])
                    # idx = 0
                    target[relation['subject_begin']][idx] = 1
                    target[relation['subject_end']][idx+num_target] = 1
                    target[relation['object_begin']][idx+2*num_target] = 1
                    target[relation['object_end']][idx+3*num_target] = 1
                    # target[relation['subject_begin']][0] = 1
                    # target[relation['subject_end']][1] = 1


                T.append(text)
                P.append(pos)
                Y.append(target)
                L.append(len(text))
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
                            # Y[j].extend([[0]*2]*(max_length-L[j]))

                    yield np.array(T).astype(np.int32),\
                          np.array(P).astype(np.int32),\
                          np.array(L).astype(np.int32),\
                          np.array(max_length).astype(np.int32),\
                          np.array(Y).astype(np.int32)
                    T,P,L,Y = [],[],[],[]

    def __call__(self):
        return self.__iter__()


class infer_generator(train_generator):
    def __int__(self,):





if __name__ == '__main__':
    g_f = train_generator('./data/train_data_char.json')

    g = g_f()
    for i in range(1600):
        s = next(g)
        print(s[-1].shape)
        print(i)
    # # for g in g_f():
    # #     pass
    # #
    # # # def g_f():
    # # #     while True:
    # # #         yield 1,1,1,1,1
    # # # g = g_f()
    # data = tf.data.Dataset.from_generator(g_f,output_types=OUTPUT_TYPES,output_shapes=OUTPUT_SHAPES)
    # it = data.make_one_shot_iterator()
    # sess = tf.Session()
    # a = it.get_next()

    # i = 0
    # while True:
    #     i += 1
    #     sess.run(a)
    #     if i%100 == 0:
    #         print(i)
