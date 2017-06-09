# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/tacotron
'''
from __future__ import print_function
import tensorflow as tf
import numpy as np
import librosa

import copy
from hyperparams import Hyperparams as hp

def get_spectrograms(sound_file): 
    '''Extracts melspectrogram and log magnitude from given `sound_file`.
    Args:
      sound_file: A string. Full path of a sound file.

    Returns:
      Transposed S: A 2d array. A transposed melspectrogram with shape of (T, n_mels)
      Transposed magnitude: A 2d array.Has shape of (T, 1+hp.n_fft//2)
    '''
    # Loading sound file
    y, sr = librosa.load(sound_file, sr=None) # or set sr to hp.sr.
    
    # stft. D: (1+n_fft//2, T)
    D = librosa.stft(y=y,
                     n_fft=hp.n_fft, 
                     hop_length=hp.hop_length, 
                     win_length=hp.win_length) 
    
    # magnitude spectrogram
    magnitude = np.abs(D) #(1+n_fft/2, T)
    
    # power spectrogram
    power = magnitude**2 #(1+n_fft/2, T) 
    
    # mel spectrogram
    S = librosa.feature.melspectrogram(S=power, n_mels=hp.n_mels) #(n_mels, T)

    return np.transpose(S.astype(np.float32)), np.transpose(magnitude.astype(np.float32)) # (T, n_mels), (T, 1+n_fft/2)
            
def shift_by_one(inputs):
    '''Shifts the content of `inputs` to the right by one 
      so that it becomes the decoder inputs.
      
    Args:
      inputs: A 3d tensor with shape of [N, T, C]
    
    Returns:
      A 3d tensor with the same shape and dtype as `inputs`.
    '''
    return tf.concat((tf.zeros_like(inputs[:, :1, :]), inputs[:, :-1, :]), 1)

def reduce_frames(arry, r):
    '''Reduces and adjust the shape and content of `arry` according to r.
    
    Args:
      arry: A 2d array with shape of [T, C]
      r: Reduction factor
     
    Returns:
      A 2d array with shape of [-1, C*r]
    '''
    T, C = arry.shape
    num_paddings = hp.r - (T % r) if T % r != 0 else 0
     
    padded = np.pad(arry, [[0, num_paddings], [0, 0]], 'constant')
    output = np.reshape(padded, (-1, C*r))
    return output

def spectrogram2wav(spectrogram):
    '''
    spectrogram: [t, f], i.e. [t, nfft // 2 + 1]
    '''
    spectrogram = spectrogram.T  # [f, t]
    X_best = copy.deepcopy(spectrogram)  # [f, t]
    for i in range(hp.n_iter):
        X_t = invert_spectrogram(X_best)
        est = librosa.stft(X_t, hp.n_fft, hp.hop_length, win_length=hp.win_length)  # [f, t]
        phase = est / np.maximum(1e-8, np.abs(est))  # [f, t]
        X_best = spectrogram * phase  # [f, t]
    X_t = invert_spectrogram(X_best)

    return np.real(X_t)

def invert_spectrogram(spectrogram):
    '''
    spectrogram: [f, t]
    '''
    return librosa.istft(spectrogram, hp.hop_length, win_length=hp.win_length, window="hann")


def restore_shape(arry, r):
    '''Restore and adjust the shape and content of `inputs` according to r.
    Args:
      arry: A 3d array with shape of [N, T, C]
      r: Reduction factor
      
    Returns:
      A 3d tensor with shape of [-1, C*r]
    '''
    N, T, C = arry.shape
    return arry.reshape((N, -1, C//r))
