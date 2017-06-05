# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/tacotron
'''

from __future__ import print_function
import codecs
import copy
import os

import tensorflow as tf
import numpy as np

import librosa
from scipy.io.wavfile import write

from hyperparams import Hyperparams as hp
from prepro import *
if hp.num_gpus == 1: from train import Graph
else: from train_multi_gpus import Graph
from utils import *

def eval(): 
    # Load graph
    g = Graph(is_training=False)
    print("Graph loaded")
    
    # Load data
    X = load_eval_data() # texts
    char2idx, idx2char = load_vocab()
             
    with g.graph.as_default():    
        sv = tf.train.Supervisor()
        with sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            # Restore parameters
            sv.saver.restore(sess, tf.train.latest_checkpoint(hp.logdir))
            print("Restored!")
             
            # Get model
            mname = open(hp.logdir + '/checkpoint', 'r').read().split('"')[1] # model name
               
            timesteps = 100 # Adjust this number as you want
            outputs_shifted = np.zeros((hp.batch_size, timesteps, hp.n_mels*hp.r), np.int32)
            outputs = np.zeros((hp.batch_size, timesteps, hp.n_mels*hp.r), np.float32)   # hp.n_mels*hp.r  
            for j in range(timesteps):
                # predict next frames
                _outputs = sess.run(g.outputs1, {g.x: X, g.decoder_inputs: outputs_shifted})
                # update character sequence
                if j < timesteps - 1:
                    outputs_shifted[:, j + 1] = _outputs[:, j, :]
                outputs[:, j, :] = _outputs[:, j, :]
              
            outputs2 = sess.run(g.outputs2, {g.outputs1: outputs})
            spectrograms = restore_shape(outputs2, hp.r)
             
     
    # Generate wav files
    if not os.path.exists('samples'): os.mkdir('samples') 
    with codecs.open('samples/text.txt', 'w', 'utf-8') as fout:
        for i, (x, s) in enumerate(zip(X, spectrograms)):
            # write text
            fout.write(str(i) + "\t" + "".join(idx2char[idx] for idx in np.fromstring(x, np.int32) if idx != 0) + "\n")
             
            # generate wav files
            audio = spectrogram2wav(np.power(np.e, s)**hp.power)
            write("samples/{}_{}.wav".format(mname, i), hp.sr, audio)     
                                          
if __name__ == '__main__':
    eval()
    print("Done")
    
    
