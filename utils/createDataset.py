import utils.spect as spect
import pandas as pd
import numpy as np
import os
import pickle


def image_name_to_audio_file_name(file):

    """
    Converts the name of the image file stored in 'C:/Users/mgoldwater/Desktop/WHOI Storage/data/dispersion'
    and 'C:/Users/mgoldwater/Desktop/WHOI Storage/data/no_dispersion' and converts it to the name of the
    corresponding audio (.wav) file.

    :param file: Name of image file for a given row in labels dataframe
    :return audio_file_name: Name of the corresponding .wav file
    """

    audio_file_name = file.split("_selec_")[0] + ".wav"
    return audio_file_name


if __name__ == "__main__":

    # Relevant paths
    LABELS_PATH = os.path.join("C:/Users/mgoldwater/Desktop/WHOI Storage/data", "labels_with_quality_and_channel.csv")
    DATA_PATH = os.path.join("C:/Users/mgoldwater/Desktop/WHOI Storage/SAMBAY",
                             "acoustic data",
                             "multichannel_wav_1h")

    # Arrays to store images and labels
    X = []
    y = []

    # Read in the labels
    labels = pd.read_csv(LABELS_PATH)

    # Filter out only second and first highest quality recordings
    labels = labels.query("Quality == '1' or Quality == '2'").reset_index()

    # Get number of rows
    rows = len(labels)

    # Iterate through the examples
    for row in range(rows):

        # Get current row
        curr_row = labels.loc[row]

        # Get relevant variables
        file = curr_row.File
        label = curr_row.Label
        window_N = curr_row.WindowNumber
        buffer = curr_row.Buffer
        startTime = curr_row.StartTime
        endTime = curr_row.EndTime
        timeShift = curr_row.TimeShift
        channel = curr_row.Channel - 1

        # Parse name of audio file
        wav_name = image_name_to_audio_file_name(file)

        # Produce the spectrogram
        fs, samples_norm = spect.get_and_normalize_sound(os.path.join(DATA_PATH, wav_name))
        starti, stopi = spect.range_to_indices(startTime - buffer + timeShift, endTime + buffer + timeShift, fs)
        samples = samples_norm[starti:stopi, channel]
        _, _, Zxx, fs = spect.my_stft(samples, fs, window_N)
        Zxx = Zxx[round(Zxx.shape[0] / 2):, :]
        spectro = 10 * np.log10(np.abs(Zxx) ** 2)

        # Store spectrogram and label
        X.append(spectro)
        y.append(label)

        print("| example: {}/{}".format(row + 1, rows))

    f = open("C:/Users/mgoldwater/Desktop/WHOI Storage/data/spectrogram_data_qual_1_2", "wb")
    pickle.dump(X, f)
    pickle.dump(y, f)
    f.close()
