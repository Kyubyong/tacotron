# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/tacotron
'''
from __future__ import print_function
from functools import wraps
import threading
from tensorflow.python.platform import tf_logging as logging
import tensorflow as tf
import numpy as np
import librosa
import os
from tqdm import tqdm
from hyperparams import Hyperparams as hp
from data import *
from networks import encode, decode1, decode2
from modules import *

# Borrowed from the `sugartensor` code.
# https://github.com/buriburisuri/sugartensor/blob/master/sugartensor/sg_queue.py
def producer_func(func):
    r"""Decorates a function `func` as sg_producer_func.

    Args:
      func: A function to decorate.
    """
    @wraps(func)
    def wrapper(sources, dtypes, capacity, num_threads):
        r"""Manages arguments of `tf.sg_.

        Args:
          **kwargs:
            source: A source queue list to enqueue
            dtypes: Data types of each tensor
            capacity: Queue capacity. Default is 32.
            num_threads: Number of threads. Default is 1.
        """
        # enqueue function
        def enqueue_func(sess, op):
            # read data from source queue
            data = func(sess.run(sources))
            # create feeder dict
            feed_dict = {}
            for ph, col in zip(placeholders, data):
                feed_dict[ph] = col
            # run session
            sess.run(op, feed_dict=feed_dict)

        # create place holder list
        placeholders = []
        for dtype in dtypes:
            placeholders.append(tf.placeholder(dtype=dtype))

        # create FIFO queue
        queue = tf.FIFOQueue(capacity, dtypes=dtypes)

        # enqueue operation
        enqueue_op = queue.enqueue(placeholders)

        # create queue runner
        runner = _FuncQueueRunner(enqueue_func, queue, [enqueue_op] * num_threads)

        # register to global collection
        tf.train.add_queue_runner(runner)

        # return de-queue operation
        return queue.dequeue()

    return wrapper


class _FuncQueueRunner(tf.train.QueueRunner):

    def __init__(self, func, queue=None, enqueue_ops=None, close_op=None,
                 cancel_op=None, queue_closed_exception_types=None,
                 queue_runner_def=None):
        # save ad-hoc function
        self.func = func
        # call super()
        super(_FuncQueueRunner, self).__init__(queue, enqueue_ops, close_op, cancel_op,
                                               queue_closed_exception_types, queue_runner_def)

    # pylint: disable=broad-except
    def _run(self, sess, enqueue_op, coord=None):

        if coord:
            coord.register_thread(threading.current_thread())
        decremented = False
        try:
            while True:
                if coord and coord.should_stop():
                    break
                try:
                    self.func(sess, enqueue_op)  # call enqueue function
                except self._queue_closed_exception_types:  # pylint: disable=catching-non-exception
                    # This exception indicates that a queue was closed.
                    with self._lock:
                        self._runs_per_session[sess] -= 1
                        decremented = True
                        if self._runs_per_session[sess] == 0:
                            try:
                                sess.run(self._close_op)
                            except Exception as e:
                                # Intentionally ignore errors from close_op.
                                logging.vlog(1, "Ignored exception: %s", str(e))
                        return
        except Exception as e:
            # This catches all other exceptions.
            if coord:
                coord.request_stop(e)
            else:
                logging.error("Exception in QueueRunner: %s", str(e))
                with self._lock:
                    self._exceptions_raised.append(e)
                raise
        finally:
            # Make sure we account for all terminations: normal or errors.
            if not decremented:
                with self._lock:
                    self._runs_per_session[sess] -= 1
                    
def get_spectrograms(sound_file): 
    '''Extracts melspectrogram and magnitude from given `sound_file`.
    Args:
      sound_file: A string. Full path of a sound file.

    Returns:
      Transposed S: A 2d array. A transposed melspectrogram with shape of (t, n_mels)
      Transposed magnitude: A 2d array. A transposed magnitude spectrogram 
        with shape of (t, 1+hp.n_fft/2)
    '''
    # Loading sound file
    y, sr = librosa.load(sound_file, sr=hp.sr)
    
    # stft. D: (1+n_fft//2, t)
    D = librosa.stft(y=y,
                     n_fft=hp.n_fft, 
                     hop_length=hp.hop_length, 
                     win_length=hp.win_length) 
    
    # power magnitude spectrogram
    magnitude = np.abs(D)**hp.power #(1+n_fft/2, t)
    
    # mel spectrogram
    S = librosa.feature.melspectrogram(S=magnitude, n_mels=hp.n_mels) #(n_mels, t)

    return np.transpose(S.astype(np.float32)), np.transpose(magnitude.astype(np.float32)) # (t, n_mels), (t, 1+n_fft/2)

def get_batch(is_training=True):
    print("is training of get_batch", is_training)
    with tf.device('cpu:0'):
        # Load data
        if is_training:
            texts, sound_files = load_train_data() # byte, string
        else:
            texts, sound_files = load_eval_data() # byte, string
        
        # calc total batch count
        num_batch = len(texts) // hp.batch_size
         
        # Convert to tensor
        texts = tf.convert_to_tensor(texts)
        sound_files = tf.convert_to_tensor(sound_files)
         
        # Create Queues
        text, sound_file = tf.train.slice_input_producer([texts, sound_files], 
                                                         shuffle=is_training)
        

        @producer_func
        def get_text_and_spectrograms(_sources):
            _text, _sound_file = _sources
            
            # Processing
            _text = np.fromstring(_text, np.int32) # byte to int
            _spectrogram, _magnitude = get_spectrograms(_sound_file)
             
            _spectrogram = reduce_frames(_spectrogram)
            _magnitude = reduce_frames(_magnitude)
    
            return _text, _spectrogram, _magnitude
            
        # Decode sound file
        x, y, z = get_text_and_spectrograms(sources=[text, sound_file], 
                                            dtypes=[tf.int32, tf.float32, tf.float32],
                                            capacity=128,
                                            num_threads=32)
        # create batch queues
        x, y, z= tf.train.batch([x, y, z],
                                shapes=[(None,), (None, hp.n_mels*hp.r), (None, (1+hp.n_fft//2)*hp.r)],
                                num_threads=32,
                                batch_size=hp.batch_size, 
                                capacity=hp.batch_size*32,   
                                dynamic_pad=True)
    return x, y, z, num_batch

            
def shift_by_one(inputs):
    '''Shifts the content of `inputs` to the right by one so that it becomes the decoder inputs.
    Args:
      inputs: A 3d tensor with shape of [N, T, C]
    
    Returns:
      A 3d tensor with the same shape and dtype as `inputs`.
    '''
    return tf.concat((tf.zeros_like(inputs[:, :1, :]), inputs[:, :-1, :]), 1)

def reduce_frames(arry):
    '''Reduces and adjust the shape and content of `arry` according to r.
    Args:
      arry: A 2d array with shape of [T, C]
     
    Returns:
      A 2d array with shape of [T+num_paddings//r, C*r]
    '''
    T, C = arry.shape
    num_paddings = hp.r - (T % hp.r) if T % hp.r != 0 else 0
     
    padded = np.pad(arry, [[0, num_paddings], [0, 0]], 'constant')
    output = np.reshape(padded, (-1, C*hp.r))
    return output
                     
class Graph:
    def __init__(self, is_training=True):
        self.graph = tf.Graph()
        
        with self.graph.as_default():
            self.x, self.y, self.z, self.num_batch = get_batch(is_training=is_training)
            self.decoder_inputs = shift_by_one(self.y)
             
            # Encoder
            self.memory = encode(self.x, is_training=is_training)
             
            # Decoder
            self.outputs1 = decode1(self.decoder_inputs, self.memory)#, hp.n_mels, 1+hp.n_fft/2)
            self.outputs2 = decode2(self.outputs1, is_training=is_training)#, hp.n_mels, 1+hp.n_fft/2)
             
            # L1 loss
            self.loss = tf.reduce_mean(tf.abs(self.outputs1 - self.y)) +\
                        tf.reduce_mean(tf.abs(self.outputs2 - self.z))
             
            if is_training:      
                # Training Scheme
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                self.optimizer = tf.train.AdamOptimizer(learning_rate=hp.lr)
                self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)
                   
                # Summmary 
                tf.summary.scalar('loss', self.loss)
                self.merged = tf.summary.merge_all()
         
def main():   
    g = Graph(is_training=True); print("Training Graph loaded")
    
    print("num_batch =", g.num_batch)
    with g.graph.as_default():
        # Load vocabulary 
        char2idx, idx2char = load_vocab()
         
        sv = tf.train.Supervisor(logdir=hp.logdir,
                                 save_model_secs=0)
        with sv.managed_session() as sess:
            # Training
            for epoch in range(1, hp.num_epochs+1): 
                if sv.should_stop(): break
                for step in tqdm(range(g.num_batch), total=g.num_batch, ncols=70, leave=False, unit='b'):
                    sess.run(g.train_op)
                     
                # Write checkpoint files at every epoch
                loss, gs = sess.run([g.loss, g.global_step])  
                print("After epoch %02d, the training loss is %.2f" % (epoch, loss))
                sv.saver.save(sess, hp.logdir + '/model_epoch_%02d_gs_%d_loss_%.2f' % (epoch, gs, loss))

if __name__ == '__main__':
    main()
    print("Done")
            