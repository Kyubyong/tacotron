# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong
'''

from __future__ import print_function
import tensorflow as tf
from hyperparams import Hyperparams as HP

def embed(inputs, vocab_size, embed_size, variable_scope="embed"):
    '''
    inputs = tf.expand_dims(tf.range(5), 0) => (1, 5)
    _embed(inputs, 5, 10) => (1, 5, 10)
    '''
    with tf.variable_scope(variable_scope):
        lookup_table = tf.get_variable('lookup_table', 
                                       dtype=tf.float32, 
                                       shape=[vocab_size, embed_size],
                                       initializer=tf.truncated_normal_initializer())
    return tf.nn.embedding_lookup(lookup_table, inputs)
 
def batch_normalize(inputs, is_training=True, variable_scope="bn"):
    '''Applies batch normalization.
    Args:
      inputs: A tensor.
      is_training: A boolean.
    
    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    with tf.variable_scope(variable_scope):
        outputs = tf.contrib.layers.batch_norm(inputs=inputs, 
                                                 center=True, 
                                                 scale=True, 
                                                 activation_fn=None, 
                                                 updates_collections=None,
                                                 is_training=is_training)
        
    return outputs

def conv1d(inputs, size, num_outputs, is_training=True, bn=True, act=True, use_bias=False, variable_scope="conv1d"):
    '''Applies 1d convolution.
    Args:
      inputs: A 3d tensor with shape of [N, T, C]
      size: An int. Kernel width.
      num_outputs: An int. Output dimension.
      is_training: A boolean. This is passed to an argument of `batch_normalize`.
      bn: A boolean. If True, `batch_normalize` is applied.
      act: A boolean. If True, `ReLU` is applied.
      use_bias: A boolean. If True, units for bias are added.
      
    Returns
      A 3d tensor with shape of [N, T, num_outputs].  
    '''
    params = {"inputs":inputs, "filters":num_outputs, "kernel_size":size,
            "dilation_rate":1, "padding":"SAME", "activation":None, 
            "kernel_initializer":tf.contrib.layers.xavier_initializer(),
            "use_bias":use_bias}
    with tf.variable_scope(variable_scope):                     
        outputs = tf.layers.conv1d(**params)
        if bn:
            outputs = batch_normalize(outputs, is_training=is_training)
        if act:
            outputs = tf.nn.relu(outputs)
    return outputs

def conv1d_banks(inputs, K=16, is_training=True, variable_scope="conv1d_banks"):
    '''Applies a series of conv1d separately.
    Args:
      inputs: A 3d tensor with shape of [N, T, C]
      K: An int. The size of conv1d banks. That is, 
        The `inputs` are convolved with K filters: 1, 2, ..., K.
      is_training: A boolean. This is passed to an argument of `batch_normalize`.
    
    Returns:
      A 3d tensor with shape of [N, T, K*Hp.embed_size//2].
    '''
    with tf.variable_scope(variable_scope):
        outputs = conv1d(inputs, 1, HP.embed_size//2, is_training=is_training) # k=1
        for k in range(2, K+1): # k = 2...K
            with tf.variable_scope("num_{}".format(k)):
                output = conv1d(inputs, k, HP.embed_size//2, is_training=is_training)
                outputs = tf.concat((outputs, output), -1)
    return outputs # (N, T, 128*K)

def gru(inputs, num_units, bidirection=False, variable_scope="gru"):
    '''Applies a GRU.
    Args:
      inputs: A 3d tensor with shape of [N, T, C].
      num_units: An int. The number of hidden units.
      bidirection: A boolean. If True, bidirectional results are concatenated.
    
    Returns:
      If bidirection is True, a 3d tensor with shape of [N, T, 2*num_units],
      otherwise [N, T, num_units].
    '''
    with tf.variable_scope(variable_scope):
        cell = tf.contrib.rnn.GRUCell(num_units)  
        if bidirection: 
            cell_bw = tf.contrib.rnn.GRUCell(num_units)
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell, cell_bw, inputs, dtype=tf.float32)
            return tf.concat(outputs, 2)  
        else:
            outputs, _ = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)
            return outputs

def attention_decoder(inputs, num_units, memory, variable_scope="attention_decoder"):
    '''Applies a GRU to `inputs`, while attending `memory`.
    Args:
      inputs: A 3d tensor with shape of [N, T', C']. Decoder inputs.
      num_units: An int. Attention size.
      memory: A 3d tensor with shape of [N, T, C]. Outputs of encoder network.
    Returns:
      A 3d tensor with shape of [N, T, num_units].    
    '''
    with tf.variable_scope(variable_scope):
        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units, memory)
        decoder_cell = tf.contrib.rnn.GRUCell(num_units)
        cell_with_attetion = tf.contrib.seq2seq.DynamicAttentionWrapper(decoder_cell, attention_mechanism, num_units)
        outputs, _ = tf.nn.dynamic_rnn(cell_with_attetion, inputs, dtype=tf.float32) #( 1, 6, 16)
    return outputs

def prenet(inputs, variable_scope="prenet"):
    '''Prenet for Encoder and Decoder.
    Args:
      inputs: A 3D tensor of shape [N, T, HP.embed_size].
    
    Returns:
      A 3D tensor of shape [N, T, HP.embed_size//2].
    '''
    with tf.variable_scope(variable_scope):
        outputs = tf.contrib.layers.fully_connected(inputs, HP.embed_size, tf.nn.relu)
        outputs = tf.nn.dropout(outputs, .5)
        outputs = tf.contrib.layers.fully_connected(outputs, HP.embed_size//2, tf.nn.relu)
        outputs = tf.nn.dropout(outputs, .5) 
    return outputs # (N, T, 128)
    
def highwaynet(inputs, variable_scope="highwaynet"):
    '''Refer to https://arxiv.org/abs/1505.00387'''
    with tf.variable_scope(variable_scope):
        H = tf.contrib.layers.fully_connected(inputs, HP.embed_size, activation_fn=tf.nn.relu)
        T = tf.contrib.layers.fully_connected(inputs, HP.embed_size, activation_fn=tf.nn.sigmoid)
        C = 1. - T
        outputs = H * T + inputs * C
    return outputs
  
    