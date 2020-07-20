##

""" Import libraries """

import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
from utils.scanData import scan_audiofile
import utils.spect as spect
import numpy as np
import pandas as pd
import os

##

""" Define helpful functions """


def plot_grid(X):

    num_per_row = 6
    rows = np.ceil(len(X) / num_per_row)

    plt.figure(figsize=(5, 5))
    for i, image in enumerate(X):
        plt.subplot(rows, num_per_row, i + 1)
        plt.axis("off")
        plt.imshow(image)

    plt.subplots_adjust(hspace=0.1, wspace=0.1)
    plt.show()


##

""" Set up file paths and load """

# Store path to file
file = "20150827-060000_multichannel_wav_SAMBAYu.wav"
DATA_PATH = os.path.join("C:/Users/mgoldwater/Desktop/WHOI Storage/SAMBAY",
                         "acoustic data",
                         "multichannel_wav_1h")
FILE_PATH = os.path.join(DATA_PATH, file)

# Get list of audio file names
audio_files = os.listdir(DATA_PATH)
print(audio_files)

# Directory where results should be stored
WRITE_PATH = "C:/Users/mgoldwater/Desktop/WHOI Storage/scan_results"

##

""" Load model and training data mean and process data """

model = tf.keras.models.load_model("C:/Users/mgoldwater/ssf_goldwater/models/1595113438.h5",
                                   custom_objects={'KerasLayer': hub.KerasLayer})

for channel in range(1, 7):
    scan_audiofile(DATA_PATH, WRITE_PATH, channel, model, "geq2_modes_calls", batch_size=500, step_size=2536)

##

""" Load data frame and audio to visualize """

calls_df = pd.read_csv(os.path.join(WRITE_PATH, "calls.csv"))
Fs_original, samples_norm = spect.get_and_normalize_sound(DATA_PATH)

##

""" Choose a row and calculate spectrogram"""

X = []

for row in range(len(calls_df)):
    samples = samples_norm[calls_df.loc[row].StartSample:calls_df.loc[row].EndSample, calls_df.loc[row].Channel - 1]
    _, _, Zxx, _ = spect.my_stft(samples, Fs_original, window_N=31, window_overlap=5, NFFT=2 ** 8)
    Zxx = Zxx[round(Zxx.shape[0] / 2):, :]
    spectro = 10 * np.log10(np.abs(Zxx) ** 2)
    X.append(spectro)

##

plot_grid(X)

##