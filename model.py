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

import dataset as ds
from config import Config

class Model(object):
    def __init__(self,config,train_set,val_set=None,infer_set =None):

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
        self.infer_steps = config.inter_steps

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

        types = train_set.output_types
        shapes = train_set.output_shapes
        # classes = train_set.output_classes
        self.train_iterator = train_set.make_one_shot_iterator().string_handle()
        self.val_iterator = val_set.make_one_shot_iterator().string_handle()
        if infer_set:
            self.infer_iterator = infer_set.make_one_shot_iterator().string_handle()
        else:
            self.infer_iterator = self.val_iterator
        self.handle_holder = tf.placeholder(dtype=tf.string,
                                            shape=None,
                                            name='input_handle_holder')

        self.iterator = tf.data.Iterator.from_string_handle(self.handle_holder,
                                                            output_types=types,
                                                            output_shapes=shapes)
        self.logit = None
        self.loss = None
        self.run_ops = []
        self.target = None

        self.ckpt_name = config.ckpt_name
        self.ckpt_path = config.ckpt_path

    def build_graph(self):
        if self.mode == 'train' or self.mode == 'val':
            T,P,L,max_length,self.target = self.iterator.get_next()
        else:
            T, P, L, max_length = self.iterator.get_next()

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
        # features = text

        f_cell = tf.nn.rnn_cell.GRUCell(num_units=self.num_units//2)
        b_cell = tf.nn.rnn_cell.GRUCell(num_units=self.num_units//2)



        f_state = f_cell.zero_state(self.batch_size,dtype=tf.float32)
        b_state = b_cell.zero_state(self.batch_size,dtype=tf.float32)

        outputs,_ = tf.nn.bidirectional_dynamic_rnn(cell_fw=f_cell,
                                                    cell_bw=b_cell,
                                                    inputs=features,
                                                    initial_state_fw=f_state,
                                                    initial_state_bw=b_state,
                                                    dtype=tf.float32,
                                                    sequence_length=L)


        features = tf.concat(outputs,axis=-1)


        mask =tf.cast(tf.greater(T,0),tf.float32)
        mask = tf.expand_dims(mask,axis=-1)

        x = features
        
        

        

        x = tf.layers.conv1d(x,kernel_size = 3,filters=self.d_model,padding='same',activation=tf.nn.relu)
        x = tf.layers.conv1d(x,kernel_size=3,filters=self.d_model,padding='same',activation=tf.nn.relu)
        x_max = tf.reduce_max(x + (mask-1)*1e10,axis=1,keep_dims=True)
        x_max = tf.tile(x_max,[1,max_length,1])
        x = tf.concat([x,x_max],axis=-1)
        x = tf.layers.conv1d(x,kernel_size=1,filters=self.num_target*self.target_size,padding='same',activation=None)
        # x = tf.layers.conv1d(x,kernel_size=1,filters=2,padding='same',activation=None)

        x = tf.multiply(x,mask)
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
        mask = tf.cast(tf.greater(self.T, 0), tf.float32)
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
                    result = sess.run(run_ops, feed_dict={self.handle_holder: train_handle})
                    if result['step'] %1 == 0:
                        print('%d:\t%f' % (result['step'], result['loss']))
                logit_list = []
                label_list = []
                loss_list = []

                for j in range(self.val_steps):
                    logit_array,label_array,loss_ = sess.run([self.logit,self.target,self.loss],
                                                       feed_dict={self.handle_holder:val_handle})
                    logit_list.extend(logit_array)
                    label_list.extend(label_array)
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

    def infer(self):
        assert self.infer_iterator != None
        var_list = tf.global_variables()
        saver = tf.train.Saver(var_list, max_to_keep=5, filename=self.ckpt_name)
        logit_tensor = self.logit

        out_file = open('out.json','w')
        with tf.Session(config=self.gpu_config) as sess:
            ckpt = tf.train.latest_checkpoint(self.ckpt_path, self.ckpt_name)
            saver.restore(sess, ckpt)
            infer_handle = sess.run(self.infer_iterator)
            for _ in range(self.infer_steps):
                logit_array = sess.run(logit_tensor,feed_dict={self.handle_holder:infer_handle})

                for logit in logit_array:
                    spo = {}
                    for tgt in self.num_target:
                        stack1 = []
                        stack2 = []
                        for i in range(len(logit)):
                            if logit[i][tgt] > 0.5:
                                stack1.append(i)
                            if logit[i][tgt+self.num_target]>0.5 and len(stack1)>0:
                                if not tgt in spo:
                                    spo[tgt] = {'subject': [], 'object': []}
                                spo[tgt]['subject'].append((stack1.pop(),i))
                            if logit[i][tgt+2*self.num_target]>0.5:
                                stack2.append(i)
                            if logit[i][tgt+3*self.num_target]>0.5 and len(stack2)>0:
                                if not tgt in spo:
                                    spo[tgt] = {'subject': [], 'object': []}
                                spo[tgt]['object'].append((stack2.pop(),i))
                        out_file.write(json.dumps(spo))
                        out_file.write('\n')









    def train_val_infer(self):
        pass

def dev_fn(labels,logits):
    TP = 0
    FP = 0

    for label_array,logit_array in zip(labels,logits):
        pred_array = np.argmax(logit_array,axis=-1)
        for i in range(self.num_target):
            for p,l in zip(label_array[:,i],pred_array[:,i]):
                pass
                





if __name__ == '__main__':

    conf = Config()
    tg = ds.train_generator(data_path='./data/train_data_char.json',batch_size = conf.batch_size)
    vg = ds.train_generator(data_path='./data/dev_data_char.json',batch_size=conf.batch_size)

    train_set = tf.data.Dataset.from_generator(tg,output_shapes=ds.OUTPUT_SHAPES,output_types=ds.OUTPUT_TYPES)
    val_set = tf.data.Dataset.from_generator(vg,output_shapes=ds.OUTPUT_SHAPES,output_types=ds.OUTPUT_TYPES)

    model = Model(conf,train_set,val_set)
    model.build_graph()
    model.compute_loss()
    model.train_val()
