import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import stft, decimate
from scipy.signal.windows import hamming
import numpy as np
import pandas as pd
import os


# ------------------ Important constants ------------------------ #

MAX_INT_16 = 2 ** 15 - 1  # Maximum integer represented with 8 bits
NFFT = 2048  # fft size for spectrogram computation
SECONDS_PER_HOUR = 3600  # Number of second in an hour
SECONDS_PER_MINUTE = 60  # Number of seconds in a minute
DECIMATE_FACTOR = 4  # Factor used to decimate the signal

# ------------------ Important constants ------------------------ #


def range_to_indices(start_t, stop_t, fs):

    """
        Returns a cropped section of a sound file between two
        particular timestamps (seconds).

        :param start_t: starting time in seconds
        :param stop_t: ending time in seconds
        :param fs: sample rate

        :return start_index: beginning index to crop at
        :return stop_index: end index to crop at
    """

    return int(round(start_t * fs)), int(round(stop_t * fs))


def get_and_normalize_sound(file_path, max_int=MAX_INT_16):

    """
        Gets desired sound file and normalizes it to largest possible integer.

        :param file_path: path to desired file
        :param max_int: integer to use in normalization

        :return sample_rate: sample rate of the sound file
        :return samples: sound samples from file
    """

    sample_rate, samples = wavfile.read(file_path)

    return sample_rate, np.divide(samples, max_int)


def parse_begin_date_and_time(row):

    """

    Parses date/time information of a row in the data frame created from AllCalls_multichannel_2.txt
    to put it the format of the corresponding sound file in SAMBAY/..../multichannel_wave_1h.

    :param row: row of a pandas dataframe from AllCalls_multichannel_2.txt to analyze
    :return filename: datetime string in format 'yyyymmdd-hh0000_multichannel_wav_SAMBAYu.wav'

    """

    date_and_time = row.BeginDateTime.split()
    date, time = date_and_time[0], date_and_time[1]

    # Parse out the file to get sound from
    time = time.split(':')

    if len(time[0]) == 1:
        time_end = '0' + time[0] + '0000'
    else:
        time_end = time[0] + '0000'

    if len(date) == 8:
        date = date.replace("/", "0")
    else:
        date = date.replace("/", "0", 1).replace("/", '')

    file_name = date + '-' + time_end + '_multichannel_wav_SAMBAYu.wav'

    # Get start and end times
    time_start = float(time[1])*SECONDS_PER_MINUTE + float(time[2])
    time_stop = time_start + float(row.DeltaTime)

    return time_start, time_stop, file_name


def my_stft(samples, fs, window_N):

    """
    Creates spectrogram from the provided data.

    :param samples: signal
    :param fs: samples rate
    :param window_N: length of window

    :return time: timeseries
    :return freq: frequency array
    :return Zxx: stft
    :return fs: new sampling frequency after signal has been decimated

    """

    # Mean center the data
    samples = samples - np.mean(samples)
    s = decimate(samples, DECIMATE_FACTOR)
    fs = fs / DECIMATE_FACTOR

    # Get length of signal
    Ns = len(s)

    # Calculate time and frequency axes
    time = np.linspace(0, Ns-1, Ns) / fs
    freq = np.linspace(0, NFFT-1, NFFT) * (fs/NFFT)

    # Create custom window and calculate stft
    window = hamming(window_N)
    _, _, Zxx = stft(s, nfft=NFFT, return_onesided=False, window=window, nperseg=window_N, noverlap=window_N-1)

    # Get power spectrum
    Zxx = abs(Zxx) ** 2

    return time, freq, Zxx, fs


if __name__ == "__main__":

    DATA_PATH = "C:/Users/mgoldwater/Desktop/WHOI Storage/SAMBAY"

    # Read in .txt file which describes the data as a pandas dataframe
    df = pd.read_csv(os.path.join(DATA_PATH, 'metadata/AllCalls_multichannel_2.txt'), sep='\t')
    df.columns = ['Selection',
                  'View',
                  'Channel',
                  'BeginTimes',
                  'EndTimes',
                  'LowFreq',
                  'HighFreq',
                  'BeginDateTime',
                  'DeltaFreq',
                  'DeltaTime',
                  'CenterFreq',
                  'PeakFreq',
                  'BasicCat',
                  'Quality',
                  'Localization']

    # Loop indices to plot a spectrogram for each row in the dataframe
    for call in [1129 - 7]:

        # Get current row and the name of the corresponding file
        curr_row = df.loc[call]
        start_time, stop_time, corresponding_file = parse_begin_date_and_time(curr_row)

        # Get and normalize the data from the file
        fs, samples_norm = get_and_normalize_sound(os.path.join(DATA_PATH,
                                                                "acoustic data/multichannel_wav_1h",
                                                                corresponding_file))

        # Get the start and end time of the call, and add a 0.5 [s] buffer on either side.
        # Then, convert these times to indices in the vector
        starti, stopi = range_to_indices(start_time - 0.1, stop_time + 0.1, fs)

        # Get the channel, subtracting one for indexing purposes
        channel = curr_row.Channel - 1

        # Crop the sound vector
        samples = samples_norm[starti:stopi, channel]

        # Calculate the stft
        window_N = 31
        time, freq, Zxx, fs = my_stft(samples, fs, window_N)
        spectro = np.abs(Zxx) ** 2

        # Plot the figure
        plt.figure()
        # plt.imshow(10 * np.log10(spectro), aspect='auto')
        # plt.ylim(spectro.shape[0], spectro.shape[0] / 2)
        plt.pcolormesh(time, freq, 10 * np.log10(Zxx))
        plt.ylim(0, fs / 2)
        plt.show()
