import os
import matplotlib.pyplot as plt
import numpy as np
import utils.spect as spect
import pickle

# Constants
DT = 0.511
NUM_CHANNELS = 6
WINDOW_N = 31

# For when the program is running
running = True

# Count saved samples
count = 0

# Where audio data is located
DATA_PATH = os.path.join("C:/Users/mgoldwater/Desktop/WHOI Storage/SAMBAY",
                         "acoustic data",
                         "multichannel_wav_1h")

# Where to write noise spectrograms
WRITE_PATH = os.path.join("C:/Users/mgoldwater/Desktop/WHOI Storage/data/noise/noise_data_5")

# Get list of audio file names
audio_files = os.listdir(DATA_PATH)

# Array for spectrograms
X = []

while running:

    # Get random file and load sound
    file_name = audio_files[np.random.choice(len(audio_files), 1, replace=True)[0]]
    Fs, samples_norm = spect.get_and_normalize_sound(os.path.join(DATA_PATH, file_name))

    # Get random start time  and channel and get audio slice
    starti = np.random.choice(len(samples_norm) - round(DT*Fs), 1, replace=True)[0]
    stopi = starti + round(DT*Fs)
    channel = np.random.choice(NUM_CHANNELS, 1, replace=True)[0]
    samples = samples_norm[starti:stopi, channel]

    # Create spectrogram
    _, _, Zxx, fs = spect.my_stft(samples, Fs, WINDOW_N, window_overlap=2, NFFT=2 ** 9)
    Zxx = Zxx[round(Zxx.shape[0] / 2):, :]
    spectro = 10 * np.log10(np.abs(Zxx) ** 2)
    plt.imshow(spectro)
    plt.show()

    # Decide to save the image or not
    processed = False
    while not processed:
        keep = input("Keep?:")
        if keep == "1":
            X.append(spectro)
            count += 1
            processed = True
        elif keep == "0":
            processed = True
        elif keep == "exit":
            running = False
            processed = True
        else:
            print("Please enter a valid command")

    print("{} samples saved".format(count))

f = open(WRITE_PATH, "wb")
pickle.dump(X, f)
f.close()



