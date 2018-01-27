# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/tacotron
'''

from __future__ import print_function

from hyperparams import Hyperparams as hp
from modules import *
import tensorflow as tf


def encoder(inputs, is_training=True, scope="encoder", reuse=None):
    '''
    Args:
      inputs: A 2d tensor with shape of [N, T_x, E], with dtype of int32. Encoder inputs.
      is_training: Whether or not the layer is in training mode.
      scope: Optional scope for `variable_scope`
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
    
    Returns:
      A collection of Hidden vectors. So-called memory. Has the shape of (N, T_x, E).
    '''
    with tf.variable_scope(scope, reuse=reuse): 
        # Encoder pre-net
        prenet_out = prenet(inputs, is_training=is_training) # (N, T_x, E/2)
        
        # Encoder CBHG 
        ## Conv1D banks
        enc = conv1d_banks(prenet_out, K=hp.encoder_num_banks, is_training=is_training) # (N, T_x, K*E/2)
        
        ## Max pooling
        enc = tf.layers.max_pooling1d(enc, pool_size=2, strides=1, padding="same")  # (N, T_x, K*E/2)
          
        ## Conv1D projections
        enc = conv1d(enc, filters=hp.embed_size//2, size=3, scope="conv1d_1") # (N, T_x, E/2)
        enc = bn(enc, is_training=is_training, activation_fn=tf.nn.relu, scope="conv1d_1")

        enc = conv1d(enc, filters=hp.embed_size // 2, size=3, scope="conv1d_2")  # (N, T_x, E/2)
        enc = bn(enc, is_training=is_training, scope="conv1d_2")

        enc += prenet_out # (N, T_x, E/2) # residual connections
          
        ## Highway Nets
        for i in range(hp.num_highwaynet_blocks):
            enc = highwaynet(enc, num_units=hp.embed_size//2, 
                                 scope='highwaynet_{}'.format(i)) # (N, T_x, E/2)

        ## Bidirectional GRU
        memory = gru(enc, num_units=hp.embed_size//2, bidirection=True) # (N, T_x, E)
    
    return memory
        
def decoder1(inputs, memory, is_training=True, scope="decoder1", reuse=None):
    '''
    Args:
      inputs: A 3d tensor with shape of [N, T_y/r, n_mels(*r)]. Shifted log melspectrogram of sound files.
      memory: A 3d tensor with shape of [N, T_x, E].
      is_training: Whether or not the layer is in training mode.
      scope: Optional scope for `variable_scope`
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
        
    Returns
      Predicted log melspectrogram tensor with shape of [N, T_y/r, n_mels*r].
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Decoder pre-net
        inputs = prenet(inputs, is_training=is_training)  # (N, T_y/r, E/2)

        # Attention RNN
        dec, state = attention_decoder(inputs, memory, num_units=hp.embed_size) # (N, T_y/r, E)

        ## for attention monitoring
        alignments = tf.transpose(state.alignment_history.stack(),[1,2,0])

        # Decoder RNNs
        dec += gru(dec, hp.embed_size, bidirection=False, scope="decoder_gru1") # (N, T_y/r, E)
        dec += gru(dec, hp.embed_size, bidirection=False, scope="decoder_gru2") # (N, T_y/r, E)
          
        # Outputs => (N, T_y/r, n_mels*r)
        mel_hats = tf.layers.dense(dec, hp.n_mels*hp.r)
    
    return mel_hats, alignments

def decoder2(inputs, is_training=True, scope="decoder2", reuse=None):
    '''Decoder Post-processing net = CBHG
    Args:
      inputs: A 3d tensor with shape of [N, T_y/r, n_mels*r]. Log magnitude spectrogram of sound files.
        It is recovered to its original shape.
      is_training: Whether or not the layer is in training mode.  
      scope: Optional scope for `variable_scope`
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
        
    Returns
      Predicted linear spectrogram tensor with shape of [N, T_y, 1+n_fft//2].
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Restore shape -> (N, Ty, n_mels)
        inputs = tf.reshape(inputs, [tf.shape(inputs)[0], -1, hp.n_mels])

        # Conv1D bank
        dec = conv1d_banks(inputs, K=hp.decoder_num_banks, is_training=is_training) # (N, T_y, E*K/2)
         
        # Max pooling
        dec = tf.layers.max_pooling1d(dec, pool_size=2, strides=1, padding="same") # (N, T_y, E*K/2)

        ## Conv1D projections
        dec = conv1d(dec, filters=hp.embed_size // 2, size=3, scope="conv1d_1")  # (N, T_x, E/2)
        dec = bn(dec, is_training=is_training, activation_fn=tf.nn.relu, scope="conv1d_1")

        dec = conv1d(dec, filters=hp.n_mels, size=3, scope="conv1d_2")  # (N, T_x, E/2)
        dec = bn(dec, is_training=is_training, scope="conv1d_2")

        # Extra affine transformation for dimensionality sync
        dec = tf.layers.dense(dec, hp.embed_size//2) # (N, T_y, E/2)
         
        # Highway Nets
        for i in range(4):
            dec = highwaynet(dec, num_units=hp.embed_size//2, 
                                 scope='highwaynet_{}'.format(i)) # (N, T_y, E/2)
         
        # Bidirectional GRU    
        dec = gru(dec, hp.embed_size//2, bidirection=True) # (N, T_y, E)
        
        # Outputs => (N, T_y, 1+n_fft//2)
        outputs = tf.layers.dense(dec, 1+hp.n_fft//2)

    return outputs
