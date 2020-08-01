##

""" Import libraries """
from utils.scanData import scan_audiofile
import matplotlib.pyplot as plt
import utils.spect as spect
import pickle
import numpy as np
import pandas as pd
import os

##

""" Define helpful functions """


def plot_grid(X):

    num_per_row = 6
    rows = np.ceil(len(X) / num_per_row)

    plt.ion()
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
file = "20150825-170000_multichannel_wav_SAMBAYu.wav"
DATA_PATH = os.path.join("C:/Users/mgoldwater/Desktop/WHOI Storage/SAMBAY",
                         "acoustic data",
                         "multichannel_wav_1h")
FILE_PATH = os.path.join(DATA_PATH, file)

# Get list of audio file names
audio_files = os.listdir(DATA_PATH)

# Directory where results should be stored
WRITE_PATH = "C:/Users/mgoldwater/Desktop/WHOI Storage/scan_results"

csv_name = file.split(".")[0]

##

""" Load model and training data mean and process data """

for channel in range(1, 7):
    scan_audiofile(FILE_PATH, WRITE_PATH, channel, csv_name, batch_size=500)

##

""" Load data frame and audio to visualize """

calls_df = pd.read_csv(os.path.join(WRITE_PATH, csv_name + ".csv"))
Fs_original, samples_norm = spect.get_and_normalize_sound(FILE_PATH)

##

""" Choose a row and calculate spectrogram"""

X = []

for row in range(len(calls_df)):
    samples = samples_norm[calls_df.loc[row].StartSample:calls_df.loc[row].EndSample, calls_df.loc[row].Channel - 1]
    _, _, Zxx, _ = spect.my_stft(samples, Fs_original, window_N=31, window_overlap=5, NFFT=2 ** 8)
    Zxx = Zxx[round(Zxx.shape[0] / 2):, :]
    spectro = 10 * np.log10(np.abs(Zxx) ** 2)
    X.append(spectro)

X = np.asarray(X)

##

plot_grid(X[0:33])
#plt.imshow(X[68])

##

""" Collect false positives to add to training set """

# Get all the CSVs which contain hard negative mined data
CSVs = os.listdir(WRITE_PATH)

X = []
y = []

for CSV in CSVs:

    # Get CSV with false positives and corresponding audio
    calls_df = pd.read_csv(os.path.join(WRITE_PATH, CSV))
    file = CSV.split('.')[0] + '.wav'
    Fs_original, samples_norm = spect.get_and_normalize_sound(os.path.join(DATA_PATH, file))

    for row in range(len(calls_df)):

        # Calculate spectrogram
        samples = samples_norm[calls_df.loc[row].StartSample:calls_df.loc[row].EndSample, calls_df.loc[row].Channel - 1]
        _, _, Zxx, _ = spect.my_stft(samples, Fs_original, window_N=31, window_overlap=5, NFFT=2 ** 8)
        Zxx = Zxx[round(Zxx.shape[0] / 2):, :]
        spectro = 10 * np.log10(np.abs(Zxx) ** 2)

        # Only take false positives
        lbl = calls_df.loc[row].Labels
        if lbl != 1:
            X.append(spectro)
            y.append(lbl)

X = np.asarray(X)
y = np.asarray(y)

##

""" Save off the hard negatives """

f = open("C:/Users/mgoldwater/Desktop/WHOI Storage/scan_results/hard_negatives", "wb")
pickle.dump(X, f)
pickle.dump(y, f)
f.close()

##
