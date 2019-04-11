#!/usr/bin/env python
# encoding: utf-8


"""
@version: 0.0
@author: hailang
@Email: seahailang@gmail.com
@software: PyCharm
@file: model.py
@time: 2019/3/25 14:09
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import json
from sklearn.metrics import f1_score

import dataset_v3 as ds
from config import Config

class Model(object):
    def __init__(self,config,train_set=None,val_set=None,infer_set =None):

        self.mode = config.mode

        self.num_vocab = config.num_vocab
        self.vocab_size = config.vocab_size

        self.num_pos = config.num_pos
        self.pos_size = config.pos_size

        self.num_target = config.num_target
        self.target_size = config.target_size

        self.d_model = config.d_model
        self.num_units = config.num_units

        self.batch_size = config.batch_size

        self.steps_each_epoch = config.steps_each_epoch
        self.train_epochs = config.train_epochs
        self.val_steps = config.val_steps
        self.infer_steps = config.infer_steps

        self.global_step = tf.train.get_or_create_global_step()

        if config.learning_rate_decay:
            self.learning_rate = tf.train.exponential_decay(config.learning_rate,
                                                            global_step=self.global_step,
                                                            decay_steps=config.decay_steps,
                                                            decay_rate=config.decay_rate)
        else:
            self.learning_rate = config.learning_rate
        if config.optimizer == 'adam':
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                    beta1=config.beta1,
                                                    beta2=config.beta2,
                                                    epsilon=config.epsilon)
        elif config.optimizer == 'adagrad':
            self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                                       initial_accumulator_value=config.accumulator_value)
        else:
            self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)

        self.gpu_config = tf.ConfigProto()
        self.gpu_config.gpu_options.allow_growth = True

        if self.mode == 'train':
            types = train_set.output_types
            shapes = train_set.output_shapes
            # classes = train_set.output_classes
            self.train_iterator = train_set.make_one_shot_iterator().string_handle()
            self.val_iterator = val_set.make_one_shot_iterator().string_handle()

            self.handle_holder = tf.placeholder(dtype=tf.string,
                                                shape=None,
                                                name='input_handle_holder')

            self.iterator = tf.data.Iterator.from_string_handle(self.handle_holder,
                                                                output_types=types,
                                                                output_shapes=shapes)
        else:
            self.iterator = infer_set.make_one_shot_iterator()

        self.logit = None
        self.loss = None
        self.run_ops = []
        self.target = None

        self.ckpt_name = config.ckpt_name
        self.ckpt_path = config.ckpt_path

    def build_graph(self):
        if self.mode == 'train' or self.mode == 'val':
            T,P,L,self.target = self.iterator.get_next()
        else:
            T, P, L = self.iterator.get_next()
        random_mask_T = tf.cast(tf.greater(tf.random_uniform(shape= tf.shape(T)),0.2),tf.int32)

        # input dropout
        if self.mode == 'train':
            T = tf.multiply(T,random_mask_T)
        random_mask_P = tf.cast(tf.greater(tf.random_uniform(shape=tf.shape(P)),0.2),tf.int32)
        if self.mode == 'train':
            P = tf.multiply(P,random_mask_P)

        L = tf.squeeze(L,axis=-1)
        max_length = tf.shape(T)[1]
        batch_size = tf.shape(T)[0]


        self.T = T
        self.P = P
        self.L = L
        self.max_length = max_length

        word_embedding = tf.get_variable('word_embedding',
            shape=[self.num_vocab,self.vocab_size],initializer=tf.truncated_normal_initializer())
        pos_embedding = tf.get_variable('pos_embedding',
            shape = [self.num_pos,self.pos_size],initializer=tf.truncated_normal_initializer())

        text = tf.nn.embedding_lookup(word_embedding,T)
        pos = tf.nn.embedding_lookup(pos_embedding,P)
        features = tf.concat([text,pos],axis=-1)

        features = text
        for i in range(1):

            f_cell = tf.nn.rnn_cell.GRUCell(num_units=self.num_units//2)
            b_cell = tf.nn.rnn_cell.GRUCell(num_units=self.num_units//2)



            f_state = f_cell.zero_state(batch_size,dtype=tf.float32)
            b_state = b_cell.zero_state(batch_size,dtype=tf.float32)

            outputs,_ = tf.nn.bidirectional_dynamic_rnn(cell_fw=f_cell,
                                                        cell_bw=b_cell,
                                                        inputs=features,
                                                        initial_state_fw=f_state,
                                                        initial_state_bw=b_state,
                                                        dtype=tf.float32,
                                                        sequence_length=L)

            features = tf.concat(outputs,axis=-1)


        # mask =tf.cast(tf.greater(T,0),tf.float32)
        mask = tf.sequence_mask(L,dtype=tf.float32)
        mask = tf.expand_dims(mask,axis=-1)

        x = features

        for i in range(3):
            x1 = tf.layers.conv1d(x,kernel_size=2,filters=self.d_model//4,padding='same',activation=tf.nn.relu)
            x2 = tf.layers.conv1d(x,kernel_size=3,filters=self.d_model//4,padding='same',activation=tf.nn.relu)
            x3 = tf.layers.conv1d(x,kernel_size=4,filters=self.d_model//4,padding='same',activation=tf.nn.relu)
            x4 = tf.layers.conv1d(x,kernel_size=5,filters=self.d_model//4,padding='same',activation=tf.nn.relu)
            _x = tf.concat([x1,x2,x3,x4],axis=-1)
            x = x+_x
        
        x_max = tf.reduce_max(x + (mask-1)*1e10,axis=1,keep_dims=True)

        x_max = tf.tile(x_max,[1,max_length,1])
        x = tf.concat([x,x_max],axis=-1)

        # output dropout
        if self.mode == 'train':
            x = tf.nn.dropout(x,0.5)
        x = tf.layers.conv1d(x,kernel_size=1,filters=self.num_target*self.target_size,padding='same',activation=None)
        # logit = x + (mask-1)*1e10
        logit = x
        self.logit = logit

    def compute_loss(self):
        assert self.logit != None,'must build graph before compute loss'
        assert self.target != None, 'must compute loss in train mode or val mode'
        logit = self.logit
        label = tf.cast(self.target,tf.float32)
        # label = tf.one_hot(label,depth=self.target_size)
        # label = tf.reshape(label,[-1,self.target_size])
        # logit = tf.reshape(logit,[-1,self.target_size])
        mask = tf.sequence_mask(self.L,dtype=tf.float32)
        losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=label,logits=logit)*tf.expand_dims(mask,-1)
        loss = tf.reduce_sum(losses)/tf.reduce_sum(mask)
        self.loss = loss
        return self.loss

    def train(self):
        assert self.loss != None,'must compute loss before train'
        var_list = tf.global_variables()
        train_variables = tf.trainable_variables()
        global_step = self.global_step
        run_ops = {'step':global_step,'loss':self.loss}
        grads_and_vars = self.optimizer.compute_gradients(self.loss,train_variables)
        run_ops['train_ops'] = self.optimizer.apply_gradients(grads_and_vars,global_step=global_step)
        saver = tf.train.Saver(var_list, max_to_keep=3, filename=self.ckpt_name)
        initializer = tf.global_variables_initializer()
        with tf.Session(config = self.gpu_config) as sess:
            sess.run(initializer)
            ckpt = tf.train.latest_checkpoint(self.ckpt_path, self.ckpt_name)
            if ckpt:
                saver.restore(sess,ckpt)
                print('restore model from %s'%ckpt)
            train_handle = sess.run(self.train_iterator)
            for e in range(self.train_epochs):
                result={}
                for i in range(self.steps_each_epoch):
                    result = sess.run(run_ops,feed_dict={self.handle_holder:train_handle})
                    if i%100 == 0:
                        print('%d:\t%f'%(result['step'],result['loss']))
                saver.save(sess,self.ckpt_path,
                           global_step=result['step'],
                           latest_filename=self.ckpt_name)

                print('epoch %d'%e)






    def train_val(self,val_fn=None):
        assert self.loss != None, 'must compute loss before train'

        var_list = tf.global_variables()
        train_variables = tf.trainable_variables()
        global_step = self.global_step
        run_ops = {'step': global_step, 'loss': self.loss}
        grads_and_vars = self.optimizer.compute_gradients(self.loss, train_variables)
        run_ops['train_ops'] = self.optimizer.apply_gradients(grads_and_vars, global_step=global_step)
        saver = tf.train.Saver(var_list, max_to_keep=5, filename=self.ckpt_name)
        initializer = tf.global_variables_initializer()
        with tf.Session(config=self.gpu_config) as sess:
            sess.run(initializer)
            ckpt = tf.train.latest_checkpoint(self.ckpt_path, self.ckpt_name)
            if ckpt:
                saver.restore(sess, ckpt)
            train_handle = sess.run(self.train_iterator)
            val_handle = sess.run(self.val_iterator)
            best_score = 0
            for e in range(self.train_epochs):
                result = {}
                for i in range(self.steps_each_epoch):
                # for i in range(10):
                    result = sess.run(run_ops, feed_dict={self.handle_holder: train_handle})
                    if result['step'] %100 == 0:
                        print('%d:\t%f' % (result['step'], result['loss']))
                logit_list = []
                label_list = []
                loss_list = []
                length_list = []

                for j in range(self.val_steps):
                # for j in range(10):
                    logit_array,label_array,length_array,loss_ =\
                     sess.run([self.logit,self.target,self.L,self.loss],
                        feed_dict={self.handle_holder:val_handle})
                    logit_list.extend(logit_array)
                    label_list.extend(label_array)
                    length_list.extend(length_array)
                    loss_list.append(loss_)

                print('epoch %d'%e)
                # score = val_fn(label_list,logit_list)
                score = 1/np.mean(loss_list)
                print(1/score)
                if score>best_score:
                    best_score = score
                    saver.save(sess, self.ckpt_path,
                               global_step=result['step'],
                               latest_filename=self.ckpt_name)
                    out_file = './reslut_dev_epoch_%d.json'%e
                    decode_fn(logit_list,
                        length_list,
                        './data/dev_data_char.json',
                        out_file,
                        num_target=self.num_target
                        )



    def infer(self):

        var_list = tf.global_variables()
        saver = tf.train.Saver(var_list, max_to_keep=5, filename=self.ckpt_name)
        logit_tensor = self.logit
        length_tensor = self.L

        out_file = open('out.json','w')
        with tf.Session(config=self.gpu_config) as sess:
            ckpt = tf.train.latest_checkpoint(self.ckpt_path, self.ckpt_name)
            saver.restore(sess, ckpt)
            logit_list = []
            length_list =[]
            for _ in range(self.infer_steps):
            # for _ in range(10):
                logit_array,length_array = sess.run([logit_tensor,length_tensor])
                logit_list.extend(logit_array)
                length_list.extend(length_array)
            decode_fn(logit_list,
                length_list,
                './data/test_data_char.json',
                './result.json',
                num_target=self.num_target)



def decode_fn(logit_array,length_array,data_file,out_file,num_target=49):
    with open('./relation_map.txt',encoding='utf-8') as file:
        relations = [line.strip() for line in file]
    file = open(data_file,encoding='utf-8')
    data = [json.loads(line) for line in file]
    result_file = open(out_file,'w')
    for logit,length,item in zip(logit_array,length_array,data):
        spo = {}
        for tgt in range(num_target):
            stack1 = []
            stack2 = []
            for i in range(length):
                if logit[i][tgt] >= 0.5:
                    stack1.append(i)
                if logit[i][tgt+num_target]>=0.5 and len(stack1)>0:
                    if not tgt in spo:
                        spo[tgt] = {'subject': [], 'object': []}
                    spo[tgt]['subject'].append((stack1.pop(),i))
                if logit[i][tgt+2*num_target]>=0.5:
                    stack2.append(i)
                if logit[i][tgt+3*num_target]>=0.5 and len(stack2)>0:
                    if not tgt in spo:
                        spo[tgt] = {'subject': [], 'object': []}
                    spo[tgt]['object'].append((stack2.pop(),i))
        pos_list = item['pos_list']
        text = item['text']

        spo_list = []
        for r,t in spo.items():
            object_list = t['object']
            subject_list = t['subject']
            for obj in object_list:
                # print(obj)
                for sbj in subject_list:
                    # print(sbj)
                    spo_list.append({'predicate':relations[int(r)]})
                    spo_list[-1]['object'] = ''.join([pos_list[i]['word'] for i in range(obj[0],obj[1]+1)])
                    spo_list[-1]['subject'] = ''.join([pos_list[i]['word'] for i in range(sbj[0],sbj[1]+1)])
                    spo_list[-1]['object_type'] = ''
                    spo_list[-1]['subject_type'] = ''

        result_file.write(json.dumps({'text':text,'spo_list':spo_list},ensure_ascii=False))
        result_file.write('\n')
    result_file.close()





if __name__ == '__main__':

    conf = Config()
    if conf.mode == 'train':
        train_set = ds.make_train_dataset('./data/train_data_char.json',batch_size=conf.batch_size)
        val_set = ds.make_train_dataset('./data/dev_data_char.json',batch_size=conf.batch_size,shuffle=False)

        model = Model(conf,train_set,val_set)
        model.build_graph()
        model.compute_loss()
        model.train_val()
    else:
        infer_set = ds.make_test_dataset('./data/test_data_char.json',batch_size=conf.batch_size)
        model = Model(conf,infer_set=infer_set)
        model.build_graph()
        model.infer()