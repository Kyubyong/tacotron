# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong
'''

from __future__ import print_function
from hyperparams import Hyperparams as hp
import tensorflow as tf
from modules import *
from data import load_vocab

def _encode(x):
    '''
    Args:
      x: A 2d tensor with shape of [N, T], dtype of int32.
    '''
    with tf.variable_scope('encoder'):
        # Load vocabulary 
        char2idx, idx2char = load_vocab()
          
        # Character Embedding
        inputs = embed(x, len(char2idx), hp.embed_size) # (N, T, 256)  

        # Encoder pre-net
        enc = prenet(inputs) # (N, T, 128)
          
        # Encoder CBHG 
        ## Conv1D bank 
        enc = conv1d_banks(enc, K=hp.encoder_num_banks) # (N, T, 2048=K * hp.embed_size // 2)
          
        ### Max pooling
        enc = tf.layers.max_pooling1d(enc, 2, 1, padding="same")  # (N, T, 2048)
          
        ### Conv1D projections
        enc = conv1d(enc, 3, hp.embed_size, variable_scope="conv1d_1") # (N, T, 256)
        enc = conv1d(enc, 3, hp.embed_size, bn=False, act=False, variable_scope="conv1d_2") # (N, T, 256)
        enc += inputs # (N, T, 256) # residual connections
          
        ### Highway Nets
        for i in range(hp.num_highwaynet_blocks):
            with tf.variable_scope('num_{}'.format(i)):
                enc = highwaynet(enc) # # (N, T, 256)
          
        ### Bidirectional GRU
        memory = gru(enc, hp.embed_size//2, True) # (N, T, 128*2)
    
    return memory
        
def _decode1(decoder_inputs, memory):
    '''
    Args:
      decoder_inputs: A 3d tensor with shape of [N, T', C'], where C'=hp.n_mels*hp.r, 
        dtype of float32. Shifted melspectrogram of sound files.
      memory: A 3d tensor with shape of [N, T, C], where C=hp.embed_size.
      
    Returns
      Predicted melspectrogram.
    '''
    with tf.variable_scope('decoder1'):
        # Decoder pre-net
        dec = prenet(decoder_inputs) # (N, T', 128)
          
        # Decoder RNNs
        dec_ = attention_decoder(dec, hp.embed_size, memory, variable_scope="attention_decoder1") # (N, T', 256)
        dec = dec_ + attention_decoder(dec_, hp.embed_size, memory, variable_scope="attention_decoder2") # (N, T', 256) # residual connections
          
        # Outputs => (N, T', hp.n_mels*hp.r)
        out_dim = decoder_inputs.get_shape().as_list()[-1]
        outputs = tf.contrib.layers.fully_connected(dec, out_dim) 
    
    return outputs

def _decode2(inputs):
    '''
    Args:
      inputs: A 3d tensor with shape of [N, T', C'], where C'=hp.n_mels*hp.r, 
        dtype of float32. Predicted melspectrogram of sound files.
      
    Returns
      Predicted magnitude spectrogram.
    '''
    with tf.variable_scope('decoder2'):
        # Decoder Post-processing net = CBHG
        ## Conv1D bank
        dec = conv1d_banks(inputs, K=hp.decoder_num_banks)
         
        ## Max pooling
        dec = tf.layers.max_pooling1d(dec, 2, 1, padding="same") # (N, T, embed_size*K)
         
        ## Conv1D projections
        dec = conv1d(dec, 3, hp.embed_size, variable_scope="conv1d_1") # (N, T, embed_size)
        dec = conv1d(dec, 3, hp.embed_size, bn=False, act=None, variable_scope="conv1d_2") # (N, T, embed_size)
         
        ## Highway Nets
        for i in range(4):
            with tf.variable_scope('num_{}'.format(i)):
                dec = highwaynet(dec)
         
        ## Bidirectional GRU    
        dec = gru(dec, hp.embed_size//2, True) # (N, T, 256)  
        
        # Outputs
        out_dim = (1+hp.n_fft/2)*hp.r
        outputs = tf.contrib.layers.fully_connected(dec, out_dim)
    
    return outputs
