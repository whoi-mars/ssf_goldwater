from tensorflow.keras.preprocessing.image import apply_affine_transform
import matplotlib.pyplot as plt
import numpy as np
import pickle


def augment_spectrograms(X):

    """
    Augment spectrogram images by zooming and shifting horizontally.

    :param X: Array of images to augment (dimensions: [N, H, W, 1]
    :return X_aug: Array of same size as X with augmented images
    """

    X_aug = np.zeros(X.shape)

    for i in range(len(X)):

        # Augmentation parameters
        width_shift = np.random.uniform(0, 55)

        # Shift
        X_aug[i] = apply_affine_transform(X[i], ty=-width_shift, fill_mode='wrap')

    return X_aug


if __name__ == "__main__":

    # Code for testing the augment_spectrograms function #

    f1 = open("C:/Users/mgoldwater/Desktop/WHOI Storage/data/train", "rb")
    X_train = pickle.load(f1)
    y_train = pickle.load(f1)
    f1.close()

    f2 = open("C:/Users/mgoldwater/Desktop/WHOI Storage/data/validation", "rb")
    X_val = pickle.load(f2)
    y_val = pickle.load(f2)
    f2.close()

    # Add channel dimension
    X_train = X_train[:, :, :, np.newaxis]
    X_val = X_val[:, :, :, np.newaxis]

    X = augment_spectrograms(X_train)

    fig, axes = plt.subplots(3, 3, figsize=(6, 6))
    axes = axes.flatten()
    i = 0
    for img, ax in zip(X[0:9], axes):
        ax.imshow(np.squeeze(img), aspect='auto')
        i += 1
    plt.tight_layout()

    fig1, axes1 = plt.subplots(3, 3, figsize=(6, 6))
    axes1 = axes1.flatten()
    i = 0
    for img, ax in zip(X_train[0:9], axes1):
        ax.imshow(np.squeeze(img), aspect='auto')
        i += 1
    plt.tight_layout()
    plt.show()
