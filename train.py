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
from prepro import load_vocab
import tensorflow as tf
from utils import shift_by_one


class Graph:
    # Load vocabulary 
    char2idx, idx2char = load_vocab()
    
    def __init__(self, is_training=True):
        self.graph = tf.Graph()
        
        with self.graph.as_default():
            if is_training:
                self.x, self.y, self.z, self.num_batch = get_batch()
            else: # Evaluation
                self.x = tf.placeholder(tf.int32, shape=(None, None))
                self.y = tf.placeholder(tf.float32, shape=(None, None, hp.n_mels*hp.r))

            self.decoder_inputs = shift_by_one(self.y)
            
            with tf.variable_scope("net"):
                # Encoder
                self.memory = encode(self.x, is_training=is_training) # (N, T, E)
                
                # Decoder 
                self.outputs1 = decode1(self.decoder_inputs, 
                                         self.memory,
                                         is_training=is_training) # (N, T', hp.n_mels*hp.r)
                self.outputs2 = decode2(self.outputs1, is_training=is_training) # (N, T', (1+hp.n_fft//2)*hp.r)
             
            if is_training:  
                # Loss
                if hp.loss_type=="l1": # L1 loss
                    self.loss1 = tf.abs(self.outputs1 - self.y)
                    self.loss2 = tf.abs(self.outputs2 - self.z)
                else: # L2 loss
                    self.loss1 = tf.squared_difference(self.outputs1, self.y)
                    self.loss2 = tf.squared_difference(self.outputs2, self.z)
                
                # Target masking
                if hp.target_zeros_masking:
                    self.loss1 *= tf.to_float(tf.not_equal(self.y, 0.))
                    self.loss2 *= tf.to_float(tf.not_equal(self.z, 0.))
                
                self.mean_loss1 = tf.reduce_mean(self.loss1)
                self.mean_loss2 = tf.reduce_mean(self.loss2)
                self.mean_loss = self.mean_loss1 + self.mean_loss2 
                
                # Logging  
                ## histograms
                self.expected1_h = tf.reduce_mean(tf.reduce_mean(self.y, -1), 0)
                self.got1_h = tf.reduce_mean(tf.reduce_mean(self.outputs1, -1),0)
                
                self.expected2_h = tf.reduce_mean(tf.reduce_mean(self.z, -1), 0)
                self.got2_h = tf.reduce_mean(tf.reduce_mean(self.outputs2, -1),0)
                
                ## images
                self.expected1_i = tf.expand_dims(tf.reduce_mean(self.y[:1], -1, keep_dims=True), 1)
                self.got1_i = tf.expand_dims(tf.reduce_mean(self.outputs1[:1], -1, keep_dims=True), 1)
                
                self.expected2_i = tf.expand_dims(tf.reduce_mean(self.z[:1], -1, keep_dims=True), 1)
                self.got2_i = tf.expand_dims(tf.reduce_mean(self.outputs2[:1], -1, keep_dims=True), 1)
                                                
                # Training Scheme
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                self.optimizer = tf.train.AdamOptimizer(learning_rate=hp.lr)
                self.train_op = self.optimizer.minimize(self.mean_loss, global_step=self.global_step)
                   
                # Summmary 
                tf.summary.scalar('mean_loss1', self.mean_loss1)
                tf.summary.scalar('mean_loss2', self.mean_loss2)
                tf.summary.scalar('mean_loss', self.mean_loss)
                
                tf.summary.histogram('expected_values1', self.expected1_h)
                tf.summary.histogram('gotten_values1', self.got1_h)
                tf.summary.histogram('expected_values2', self.expected2_h)
                tf.summary.histogram('gotten values2', self.got2_h)
                                
                tf.summary.image("expected_values1", self.expected1_i*255)
                tf.summary.image("gotten_values1", self.got1_i*255)
                tf.summary.image("expected_values2", self.expected2_i*255)
                tf.summary.image("gotten_values2", self.got2_i*255)
                
                self.merged = tf.summary.merge_all()
         
def main():   
    g = Graph(); print("Training Graph loaded")
    
    with g.graph.as_default():
        # Load vocabulary 
        char2idx, idx2char = load_vocab()
        
        # Training 
        sv = tf.train.Supervisor(logdir=hp.logdir,
                                 save_model_secs=0)
        with sv.managed_session() as sess:
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
