##

""" Import libraries """

import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
from utils.scanData import scan_audiofile
import utils.spect as spect
import numpy as np
import pandas as pd
import pickle
import os

##

""" Define helpful functions """




##

""" Set up file paths and load """

# Store path to file
file = "20150824-210000_multichannel_wav_SAMBAYu.wav"
DATA_PATH = os.path.join("C:/Users/mgoldwater/Desktop/WHOI Storage/SAMBAY",
                         "acoustic data",
                         "multichannel_wav_1h",
                         file)

# Directory where results should be stored
WRITE_PATH = "C:/Users/mgoldwater/Desktop/WHOI Storage/scan_results"

##

""" Load model and training data mean and process data """

# Load model
f = open("C:/Users/mgoldwater/ssf_goldwater/models/mu_1595010328", "rb")
mu = pickle.load(f)
f.close()

model = tf.keras.models.load_model("C:/Users/mgoldwater/ssf_goldwater/models/1595010328.h5",
                                   custom_objects={'KerasLayer': hub.KerasLayer})

scan_audiofile(DATA_PATH, WRITE_PATH, 2, model, mu, "calls", step_size=1000)

##

""" Load data frame and audio to visualize """

calls_df = pd.read_csv(os.path.join(WRITE_PATH, "calls.csv"))
Fs_original, samples_norm = spect.get_and_normalize_sound(DATA_PATH)

##

""" Choose a row and calculate spectrogram"""

row = 2

samples = samples_norm[calls_df.loc[row].StartSample:calls_df.loc[row].EndSample, calls_df.loc[row].Channel - 1]
_, _, Zxx, _ = spect.my_stft(samples, Fs_original, window_N=31, window_overlap=5, NFFT=2 ** 8)
Zxx = Zxx[round(Zxx.shape[0] / 2):, :]
spectro = 10 * np.log10(np.abs(Zxx) ** 2)

plt.imshow(spectro)

##