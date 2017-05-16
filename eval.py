# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong
'''

from __future__ import print_function
from hyperparams import Hyperparams as hp
import codecs
import tensorflow as tf
import numpy as np
from data import *
from train import Graph
import copy
from scipy.io.wavfile import write

def spectrogram2wav(spectrogram):
    X_best = copy.deepcopy(spectrogram)
    for i in range(hp.n_iter):
        X_t = invert_spectrogram(X_best)
        est = librosa.stft(X_t, hp.n_fft, hp.hop_length).T
        phase = est / np.maximum(1e-8, np.abs(est))
        X_best = spectrogram * phase[:len(spectrogram)]
    X_t = invert_spectrogram(X_best)
    
    return np.real(X_t)

def invert_spectrogram(spectrogram):
    return librosa.istft(spectrogram.T, hp.hop_length, win_length=hp.win_length, window="hann")

def restore_shape(arry):
    '''Restore and adjust the shape and content of `inputs` according to r.
    Args:
      inputs: A 3d array with shape of [N, T, C]
    
    Returns:
      A 3d tensor with shape of [T+num_paddings//r, C*r]
    '''
    N, T, C = arry.shape
    return arry.reshape((N, -1, C//hp.r))

def eval(): 
    # Load graph
    g = Graph(is_training=False)
    
    with g.graph.as_default():    
        sv = tf.train.Supervisor()
        with sv.managed_session() as sess:
            # Restore parameters
            sv.saver.restore(sess, tf.train.latest_checkpoint(hp.logdir))
            print("Restored!")
            mname = open('asset/train/checkpoint', 'r').read().split('"')[1] # model name
            
            # Load data
            texts, sound_files = load_eval_data()
            char2idx, idx2char = load_vocab()
             
            timesteps = 155
            outputs_shifted = np.zeros((hp.batch_size, timesteps, hp.n_mels*hp.r), np.int32)
            outputs = np.zeros((hp.batch_size, timesteps, hp.n_mels*hp.r), np.float32)   # hp.n_mels*hp.r  
            for j in range(timesteps):
                # predict next frames
                _outputs = sess.run(g.outputs1, {g.decoder_inputs: outputs_shifted})
                # update character sequence
                if j < timesteps - 1:
                    outputs_shifted[:, j + 1] = _outputs[:, j, :]
                outputs[:, j, :] = _outputs[:, j, :]
            
            outputs2, loss, x = sess.run([g.outputs2, g.loss, g.x], {g.outputs1: outputs})
            t1 = "".join(idx2char[tt] for tt in x[0])
            t2 = "".join(idx2char[tt] for tt in x[1])
            print("="*100, t1, "|", t2)
            spectrograms = restore_shape(outputs2)
            print("loss=", loss)
    
    for i, (t, f) in enumerate(zip(texts, spectrograms)):
        audio = spectrogram2wav(f)
        t_ = np.fromstring(t, np.int32)
        write("samples/{}.wav".format(i), hp.sr, audio)                                       
if __name__ == '__main__':
    eval()
    print("Done")
    
    
