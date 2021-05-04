import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy import signal
from scipy.io import wavfile
import numpy as np
import csv
import os, sys
import glob
import pickle
test = np.array([])

# Open a file
path = '/home/abid/DepSpeech-0.6.1/data/test_new/'
files = os.listdir(path)


with open('data_test.csv', 'w', newline='') as test1:
    writer = csv.writer(test1, delimiter=',')
    for filename in sorted(glob.glob(os.path.join(path, '*.wav'), recursive = True)):
        sizes = os.path.getsize(filename)
        print(filename)
        print(sizes)
        test = (filename, sizes)
        print(test)


    #with open('tests.csv', 'w', newline='') as test1:
        # writer = csv.writer(test1, delimiter=',')
        writer.writerows([test])