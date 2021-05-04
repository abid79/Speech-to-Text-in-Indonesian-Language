# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import os

from functools import partial
import csv
import numpy as np
import pandas
import tensorflow as tf
tf.enable_eager_execution()
tf.compat.v1.enable_eager_execution
from tensorflow.python.ops import gen_audio_ops as contrib_audio
import matplotlib.pyplot as plt
from util.config import Config
from util.text import text_to_char_array
from util.flags import FLAGS
from util.spectrogram_augmentations import augment_freq_time_mask, augment_dropout, augment_pitch_and_tempo, augment_speed_up, augment_sparse_warp
from util.audio import read_frames_from_file, vad_split, DEFAULT_FORMAT

wav_filename = '/home/abid/DeepSpeech-0.6.1/data/train_new/001.wav'
samples = tf.io.read_file(wav_filename)

decoded = contrib_audio.decode_wav(samples, desired_channels=1)
print(decoded)

feature_win_len = 32
feature_win_step = 16
audio_sample_rate = 16000

audio_windows_samples = audio_sample_rate * (feature_win_len / 1000)
print(audio_windows_samples)
audio_step_samples = audio_sample_rate * (feature_win_step / 1000)
print(audio_step_samples)

#def samples_to_spectrogram(samples, sample_rate, train_phase=False):
spectrogram = contrib_audio.audio_spectrogram(decoded.audio,
                                                window_size=audio_windows_samples,
                                                stride=audio_step_samples,
                                                magnitude_squared=True)
print(spectrogram)

n_input = 257
spectrogram = tf.reshape(spectrogram, [-1, n_input])
#print(spectrogram)
#features, features_len = spectrogram, tf.shape(input=spectrogram)[0]
##features = tf.expand_dims(features, 0)
#print(features)
#features_len = tf.expand_dims(features_len, 0)
#print(features_len)
#spectrogram = tf.abs(spectrogram)
features, features_len = spectrogram, tf.shape(input=spectrogram)[0]
print(features)
print(features_len)
y=features.shape
print(y)
#array = spectrograms.numpy().astype(np.float)[0]
#print(array)
fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(20,4))
cax = ax.matshow(features.numpy().T, aspect='auto', origin='lower')
#cax = ax.matshow(np.transpose(spectrogram), interpolation='nearest', aspect='auto', cmap=plt.cm.afmhot, origin='lower')
fig.colorbar(cax)
plt.show()

with open('data_ciri.csv', 'w', newline='') as test1:
    writer = csv.writer(test1, delimiter=',')
    writer.writerows([features])