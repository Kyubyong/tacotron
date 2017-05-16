# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong
'''

import numpy as np
import librosa
from hyperparams import Hyperparams as hp
import glob
import re
import os

def load_vocab():
    vocab = "E abcdefghijklmnopqrstuvwxyz'" # E: Empty, S: end of Sentence
    char2idx = {char:idx for idx, char in enumerate(vocab)}
    idx2char = {idx:char for idx, char in enumerate(vocab)}
    return char2idx, idx2char    
 
def create_train_data():
    # Load vocabulary
    char2idx, idx2char = load_vocab() 
      
    texts, sound_files = [], []
    with open(hp.text_file, 'r') as fin:
        while True:
            line = fin.readline()
            if not line: break
             
            sound_fname, text, duration = line.strip().split("\t")
            sound_file = hp.sound_fpath + sound_fname + ".wav"
            text = re.sub(r"[^ a-z']", "", text.strip().lower())
             
            if (len(text) <= hp.max_len) and (1. < float(duration) <= hp.max_duration):
                texts.append(np.array([char2idx[char] for char in text], np.int32).tostring())
                sound_files.append(sound_file)
             
    return texts, sound_files
     
def load_train_data():
    texts, sound_files = create_train_data()
    return texts[:-hp.batch_size], sound_files[:-hp.batch_size]
 
def load_eval_data():
    texts, sound_files = create_train_data()
    return texts[-hp.batch_size:], sound_files[-hp.batch_size:]
     
if __name__ == '__main__':
    # Load vocabulary
    char2idx, idx2char = load_vocab() 
    texts, sound_files = load_eval_data()
    for i in range(len(texts)):
#         print(np.fromstring(texts[i], np.int32))
        print("".join(idx2char[idx] for idx in np.fromstring(texts[i], np.int32)))
        print(sound_files[i])
#     for i in range(len(texts)):
#         print(sound_files[i])
#         print(texts[i])
#         print(np.fromstring(texts[i], int))
    print("Done")
 

