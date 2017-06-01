# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/tacotron
'''

class Hyperparams:
    '''Hyper parameters'''
    # data
    text_file = '../WEB/text.csv'
    sound_fpath = '../WEB'
    max_len = 150 # maximum length of text
    max_duration = 5.0 # maximum duration of a sound file. seconds.
    
    # signal processing
    sr = 22050 # Sampling rate. Paper => 24000
    n_fft = 2048 # fft points (samples)
    frame_shift = 0.0125 # seconds
    frame_length = 0.05 # seconds
    hop_length = int(sr*frame_shift) # samples  This is dependent on the frame_shift.
    win_length = int(sr*frame_length) # samples This is dependent on the frame_length.
    n_mels = 80 # Number of Mel banks to generate
    power = 1.2 # Exponent for the magnitude melspectrogram
    n_iter = 30 # Number of inversion iterations 
    
    # model
    embed_size = 256 # alias = E
    encoder_num_banks = 16
    decoder_num_banks = 8
    num_highwaynet_blocks = 4
    r = 5 # Reduction factor. Paper => 2, 3, 5
    
    # training scheme
    lr = 0.001 # Paper => Exponential decay
    logdir = "logdir"
    batch_size = 32
    num_epochs = 200 # Paper => 2M global steps!
    loss_type = "l1" # Or you can test "l2"
    
    # etc
    target_zeros_masking = False # If True, we mask zero padding on the target, 
                                 # so exclude them from the loss calculation.     
