import tensorflow as tf
import tensorflow_hub as hub
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import utils.spect as spect
import pandas as pd
import numpy as np
import time
import random
import os

model = tf.keras.models.load_model("C:/Users/mgoldwater/ssf_goldwater/models/no_hard_negatives_130_L2_1596399856.h5",
                                    custom_objects={'KerasLayer': hub.KerasLayer})

# Figure for plotting spectrograms
fig = plt.figure()


def remap(x, oldMin, oldMax, newMin, newMax):

    """
    Remaps a value from one range to another.

    :param x: Value to be remapped
    :param oldMin: Minimum of old range
    :param oldMax: Maximym of old range
    :param newMin: Minimum of new range
    :param newMax: Maximum of new range
    :return: Remapped value (x)
    """

    old_range = oldMax - oldMin
    new_range = newMax - newMin
    x = (((x - oldMin) * new_range) / old_range) + newMin
    return x


def scan_audiofile(data_path, write_path, channel, log_name='results.csv', batch_size=50, batches=None, save_spects=False):

    """
    Applies a pre-trained Tensorflow/Keras model to an audio file to sort spectrograms of a shifting window
    into the following classes:

        0) At least 2 modes present
        1) Less than two modes present
        2) No call

    Spectrograms labeled in the "1" class are logged in a CSV file.


    :param data_path: Path to folder where audio data is
    :param write_path: Path to where the output CSV is to be written
    :param channel: Channel to scan
    :param log_name: Text string for the output CSV file
    :param batch_size: Size of group of spectrograms to apply to model to simultaneously
    :param batches: Number of batches to process (processes the whole file if 'None')
    :param step_size: Number of samples to shift in creating each spectrogram
    :param save_spects: If True, the function will save off reference images for dispersively-classified spectrograms
    :return: None
    """

    # Create file to store discovered dispersive curves
    calls_CSV_path = os.path.join(write_path, log_name + ".csv")
    columns = ['File', 'StartSample', 'EndSample', 'Channel', 'Prob', 'DecFactor', 'StepSize']
    if os.path.exists(calls_CSV_path):
        calls_df = pd.read_csv(calls_CSV_path, index_col=[0])
        row_count = len(calls_df)
    else:
        calls_df = pd.DataFrame(columns=columns)
        row_count = 0

    # Get audio file name
    file = data_path  # data_path.split("\\")[-1]

    # Load the sound file
    Fs_original, samples_norm = spect.get_and_normalize_sound(data_path)

    # Append channel dimension if none already
    if len(samples_norm.shape) == 1:
        samples_norm = samples_norm[:, np.newaxis]

    # Calculate appropriate decimation factor
    if Fs_original % 1000 == 0:
        decimate_factor = Fs_original // 1000
    else:
        decimate_factor = int(Fs_original / 1000)

    # Start at beginning of file
    start_sample = 0
    step_size = round(Fs_original * 0.634)

    # Process all the batches
    if batches is None:
        batches = int(len(samples_norm - round(Fs_original * 0.634))/(batch_size*step_size))
    for batch in range(batches):

        # Store images, indices, and softmax output collected in batch
        X = []
        indices = []

        for example in range(batch_size):
            # Start and end sample of a given spectrogram
            end_sample = start_sample + round(Fs_original * 0.634)

            # Calculate spectrogram
            samples = samples_norm[start_sample:end_sample, channel - 1]
            times, freq, Zxx, _ = spect.my_stft(samples, Fs_original, window_N=31, window_overlap=5, NFFT=2 ** 8, DECIMATE_FACTOR=decimate_factor)
            Zxx = Zxx[round(Zxx.shape[0] / 2):, :]
            spectro = 10 * np.log10(np.abs(Zxx) ** 2)

            # Cut off trailing end of spectrogram if decimate factor was rounded
            length = spectro.shape[1]
            if length != 128:
                spectro = spectro[:, :length - (length - 128)]

            X.append(spectro)
            indices.append((start_sample, end_sample))

            # Iterate start_sample
            start_sample += step_size

        # Process batch with model
        X = np.asarray(X)[:, :, :, np.newaxis]
        indices = np.asarray(indices)

        # rescale
        for i in range(len(X)):
            X[i] = X[i] = remap(X[i], np.min(X[i]), np.max(X[i]), 0, 1)

        # Apply the model
        result_batch = model.predict(X)
        y_pred = np.argmax(result_batch, axis=1)
        probs = tf.keras.activations.softmax(tf.constant(result_batch))

        # Get indices and spectrograms for dispersive calls with at least two modes
        disp_indices = indices[y_pred == 1]
        disp_X = X[y_pred == 1]

        # Get indices for the probabilities of dispersive calls
        probs = probs.numpy()
        probs = probs[y_pred == 1]

        # Populate df and save reference spectrograms
        for sample in range(len(disp_indices)):
            row = [file, disp_indices[sample][0], disp_indices[sample][1], channel, probs[sample, 1], decimate_factor, step_size]
            calls_df.loc[row_count] = row

            if save_spects:
                # Clear the figure
                fig.clf()
                # get current axes
                ax = plt.gca()
                # Set ticks to be at pixel edge rather than center
                ax.set_xticks(np.linspace(-0.5, disp_X[sample].shape[1] - 0.5, 9))
                ax.set_yticks(np.linspace(disp_X[sample].shape[0] - 0.5, -0.5, 9))
                # Relabel the ticks
                ax.set_xticklabels(np.round(np.linspace(times[0], times[-1], 9), 3))
                ax.set_yticklabels(np.round(np.linspace(freq[0], freq[-1], 9), 2))
                # Plot image
                plt.imshow(disp_X[sample, :, :, 0], aspect='auto')
                # Make sure labels are visible
                plt.subplots_adjust(left=0.15, bottom=0.1)
                plt.xlabel("Time [s]")
                plt.ylabel("Frequency [Hz]")

                fig.savefig(os.path.join(write_path, "sample_{}".format(row_count + 1)))

            row_count += 1

        print("File: {} -- Channel: {} -- Batch: {}/{}".format(file, channel, batch + 1, batches))

    # Save as CSV
    calls_df.to_csv(calls_CSV_path)
