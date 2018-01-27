# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/tacotron
'''

from __future__ import print_function

from hyperparams import Hyperparams as hp
import numpy as np
from data_load import load_data
import tensorflow as tf
from train import Graph
from utils import load_spectrograms


def eval(): 
    # Load graph
    g = Graph(mode="eval"); print("Evaluation Graph loaded")

    # Load data
    fpaths, text_lengths, texts = load_data(mode="eval")

    # Parse
    text = np.fromstring(texts[0], np.int32) # (None,)
    fname, mel, mag = load_spectrograms(fpaths[0])

    x = np.expand_dims(text, 0) # (1, None)
    y = np.expand_dims(mel, 0) # (1, None, n_mels*r)
    z = np.expand_dims(mag, 0) # (1, None, n_mfccs)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint(hp.logdir)); print("Restored!")

        writer = tf.summary.FileWriter(hp.logdir, sess.graph)

        # Feed Forward
        ## mel
        y_hat = np.zeros((1, y.shape[1], y.shape[2]), np.float32)  # hp.n_mels*hp.r
        for j in range(y.shape[1]):
            _y_hat = sess.run(g.y_hat, {g.x: x, g.y: y_hat})
            y_hat[:, j, :] = _y_hat[:, j, :]

        ## mag
        merged, gs = sess.run([g.merged, g.global_step], {g.x:x, g.y:y, g.y_hat: y_hat, g.z: z})
        writer.add_summary(merged, global_step=gs)
        writer.close()

if __name__ == '__main__':
    eval()
    print("Done")
    
    
