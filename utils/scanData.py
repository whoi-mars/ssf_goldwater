import tensorflow as tf
import tensorflow_hub as hub
import utils.spect as spect
import pandas as pd
import numpy as np
import os


def scan_audiofile(data_path, write_path, channel, model, mu, log_name, batch_size=50, batches=None, step_size=6000):

    # Create file to store discovered dispersive curves
    calls_CSV_path = os.path.join(write_path, log_name + ".csv")
    columns = ['File', 'StartSample', 'EndSample', 'Channel']
    row_count = 0
    if os.path.exists(calls_CSV_path):
        calls_df = pd.read_csv(calls_CSV_path)
    else:
        calls_df = pd.DataFrame(columns=columns)

    # Get audio file name
    file = data_path.split('/')[-1]

    # Load the sound file
    Fs_original, samples_norm = spect.get_and_normalize_sound(data_path)
    start_sample = 0

    # Process all the batches
    if batches is None:
        batches = int(len(samples_norm - round(Fs_original * 0.634))/(batch_size*step_size))
    for batch in range(batches):
        # Store images collected in batch
        X = []
        indices = []
        for example in range(batch_size):
            # Start and end sample of a given spectrogram
            end_sample = start_sample + round(Fs_original * 0.634)

            # Calculate spectrogram
            samples = samples_norm[start_sample:end_sample, channel - 1]
            _, _, Zxx, _ = spect.my_stft(samples, Fs_original, window_N=31, window_overlap=5, NFFT=2 ** 8)
            Zxx = Zxx[round(Zxx.shape[0] / 2):, :]
            spectro = 10 * np.log10(np.abs(Zxx) ** 2)

            X.append(spectro)
            indices.append((start_sample, end_sample))

            # Iterate start_sample
            start_sample += step_size

        # Process batch with model
        X = np.asarray(X)[:, :, :, np.newaxis]
        indices = np.asarray(indices)

        for i in range(len(X)):
            # rescale
            X[i] = (X[i] - np.min(X[i])) / (np.max(X[i]) - np.min(X[i]))

        # Mean center using mean from training data
        X = X - mu

        # Apply the model
        result_batch = model.predict(X)
        y_pred = np.argmax(result_batch, axis=1)

        # Get indices for dispersive calls with at least two modes
        disp_indices = indices[y_pred == 1]

        # Populate df
        for indices in disp_indices:
            row = [file, indices[0], indices[1], channel]
            calls_df.loc[row_count] = row
            row_count += 1

        print("Batch: {}/{}".format(batch + 1, batches))

    # Save as CSV
    calls_df.to_csv(calls_CSV_path, index=False)
