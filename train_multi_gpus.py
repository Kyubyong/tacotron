# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/tacotron
'''

from __future__ import print_function

import os

import librosa
from tqdm import tqdm

from data_load import get_batch
from hyperparams import Hyperparams as hp
from modules import *
from networks import encode, decode1, decode2
import numpy as np
from prepro import *
import tensorflow as tf
from utils import shift_by_one


class Graph:
    def __init__(self, is_training=True):
        self.graph = tf.Graph()
        
        with self.graph.as_default():
            if is_training:
                self.x, self.y, self.z, self.num_batch = get_batch()
                self.decoder_inputs = shift_by_one(self.y)
                
                # Make sure that batch size was multiplied by # gpus.
                # Now we split the mini-batch data by # gpus.
                self.x = tf.split(self.x, hp.num_gpus, 0)
                self.y = tf.split(self.y, hp.num_gpus, 0)
                self.z = tf.split(self.z, hp.num_gpus, 0)
                self.decoder_inputs = tf.split(self.decoder_inputs, hp.num_gpus, 0)
                
                # Sequence lengths for masking
                self.x_lengths = tf.to_int32(tf.reduce_sum(tf.sign(tf.abs(self.x)), -1)) # (N,)
                self.x_masks = tf.to_float(tf.expand_dims(tf.sign(tf.abs(self.x)), -1)) # (N, T, 1)
                # optimizer
                self.optimizer = tf.train.AdamOptimizer(learning_rate=hp.lr)
            
                self.losses, self.grads_and_vars_list = [], []
                for i in range(hp.num_gpus):
                    with tf.variable_scope('net', reuse=bool(i)):
                        with tf.device('/gpu:{}'.format(i)):
                            with tf.name_scope('gpu_{}'.format(i)):
                                # Encoder
                                self.memory = encode(self.x[i], is_training=is_training) # (N, T, E)
                                
                                # Decoder 
                                self.outputs1 = decode1(self.decoder_inputs[i], 
                                                         self.memory,
                                                         is_training=is_training) # (N, T', hp.n_mels*hp.r)
                                self.outputs2 = decode2(self.outputs1, is_training=is_training) # (N, T', (1+hp.n_fft//2)*hp.r)
              
                                # Loss
                                if hp.loss_type=="l1": # L1 loss
                                    self.loss1 = tf.abs(self.outputs1 - self.y[i])
                                    self.loss2 = tf.abs(self.outputs2 - self.z[i])
                                else: # L2 loss
                                    self.loss1 = tf.squared_difference(self.outputs1, self.y[i])
                                    self.loss2 = tf.squared_difference(self.outputs2, self.z[i])
                                    
                                # Target masking
                                if hp.target_zeros_masking:
                                    self.loss1 *= tf.to_float(tf.not_equal(self.y[i], 0.))
                                    self.loss2 *= tf.to_float(tf.not_equal(self.z[i], 0.))
                                
                                self.loss1 = tf.reduce_mean(self.loss1)
                                self.loss2 = tf.reduce_mean(self.loss2)
                                self.loss = self.loss1 + self.loss2   
                                
                                self.losses.append(self.loss)
                                self.grads_and_vars = self.optimizer.compute_gradients(self.loss) 
                                self.grads_and_vars_list.append(self.grads_and_vars)    
                
                with tf.device('/cpu:0'):
                    # Aggregate losses, then calculate average loss.
                    self.mean_loss = tf.add_n(self.losses) / len(self.losses)
                     
                    #Aggregate gradients, then calculate average gradients.
                    self.mean_grads_and_vars = []
                    for grads_and_vars in zip(*self.grads_and_vars_list):
                        grads = []
                        for grad, var in grads_and_vars:
                            if grad is not None:
                                grads.append(tf.expand_dims(grad, 0))
                        mean_grad = tf.reduce_mean(tf.concat(grads, 0), 0) #()
                        self.mean_grads_and_vars.append((mean_grad, var))
                 
                # Training Scheme
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                self.train_op = self.optimizer.apply_gradients(self.mean_grads_and_vars, self.global_step)
                 
                # Summmary 
                tf.summary.scalar('mean_loss', self.mean_loss)
                self.merged = tf.summary.merge_all()
                
            else: # Evaluation
                self.x = tf.placeholder(tf.int32, shape=(None, None))
                self.y = tf.placeholder(tf.float32, shape=(None, None, hp.n_mels*hp.r))
                self.decoder_inputs = shift_by_one(self.y)
                with tf.variable_scope('net'):
                    # Encoder
                    self.memory = encode(self.x, is_training=is_training) # (N, T, E)
                     
                    # Decoder
                    self.outputs1 = decode1(self.decoder_inputs, self.memory, is_training=is_training) # (N, T', hp.n_mels*hp.r)
                    self.outputs2 = decode2(self.outputs1, is_training=is_training) # (N, T', (1+hp.n_fft//2)*hp.r)
         
def main():   
    g = Graph(); print("Training Graph loaded")
    
    with g.graph.as_default():
        # Load vocabulary 
        char2idx, idx2char = load_vocab()
         
        # Training 
        sv = tf.train.Supervisor(logdir=hp.logdir,
                                 save_model_secs=600)
        with sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            for epoch in range(1, hp.num_epochs+1): 
                if sv.should_stop(): break
                for step in tqdm(range(g.num_batch), total=g.num_batch, ncols=70, leave=False, unit='b'):
                    sess.run(g.train_op)
                 
                # Write checkpoint files at every epoch
                gs = sess.run(g.global_step) 
                sv.saver.save(sess, hp.logdir + '/model_epoch_%02d_gs_%d' % (epoch, gs))

if __name__ == '__main__':
    main()
    print("Done")
            
