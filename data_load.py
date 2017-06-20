# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/tacotron
'''

from functools import wraps
import threading

from tensorflow.python.platform import tf_logging as logging

from hyperparams import Hyperparams as hp
import numpy as np
from prepro import *
import tensorflow as tf
from utils import *


# Adapted from the `sugartensor` code.
# https://github.com/buriburisuri/sugartensor/blob/master/sugartensor/sg_queue.py
def producer_func(func):
    r"""Decorates a function `func` as producer_func.

    Args:
      func: A function to decorate.
    """
    @wraps(func)
    def wrapper(inputs, dtypes, capacity, num_threads):
        r"""
        Args:
            inputs: A inputs queue list to enqueue
            dtypes: Data types of each tensor
            capacity: Queue capacity. Default is 32.
            num_threads: Number of threads. Default is 1.
        """
        # enqueue function
        def enqueue_func(sess, op):
            # read data from source queue
            data = func(sess.run(inputs))
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
                    
def get_batch():
    """Loads training data and put them in queues"""
    with tf.device('/cpu:0'):
        # Load data
        texts, sound_files = load_train_data() # byte, string
        
        # calc total batch count
        num_batch = len(texts) // hp.batch_size
         
        # Convert to tensor
        texts = tf.convert_to_tensor(texts)
        sound_files = tf.convert_to_tensor(sound_files)
         
        # Create Queues
        text, sound_file = tf.train.slice_input_producer([texts, sound_files], shuffle=True)

        @producer_func
        def get_text_and_spectrograms(_inputs):
            '''From `_inputs`, which has been fetched from slice queues,
               makes text, spectrogram, and magnitude,
               then enqueue them again. 
            '''
            _text, _sound_file = _inputs
            
            # Processing
            _text = np.fromstring(_text, np.int32) # byte to int
            _spectrogram, _magnitude = get_spectrograms(_sound_file)
             
            _spectrogram = reduce_frames(_spectrogram, hp.win_length//hp.hop_length, hp.r)
            _magnitude = reduce_frames(_magnitude, hp.win_length//hp.hop_length, hp.r)
    
            return _text, _spectrogram, _magnitude
            
        # Decode sound file
        x, y, z = get_text_and_spectrograms(inputs=[text, sound_file], 
                                            dtypes=[tf.int32, tf.float32, tf.float32],
                                            capacity=128,
                                            num_threads=32)
        
        # create batch queues
        x, y, z = tf.train.batch([x, y, z],
                                shapes=[(None,), (None, hp.n_mels*hp.r), (None, (1+hp.n_fft//2)*hp.r)],
                                num_threads=32,
                                batch_size=hp.batch_size, 
                                capacity=hp.batch_size*32,   
                                dynamic_pad=True)
        
        if hp.use_log_magnitude:
            z = tf.log(z+1e-10)
            
    return x, y, z, num_batch
