##
""" Import libraries """

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

# Disable GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

##
""" Import the labeled data set"""

f = open("C:/Users/mgoldwater/Desktop/WHOI Storage/data/spectrogram_data_qual_1_2", "rb")
X = pickle.load(f)
y = pickle.load(f)
f.close()

X = np.asarray(X)
y = np.asarray(y)

# Add "batch" dimension
X = X[:, :, :, np.newaxis]

# Percent training data
train_perc = 0.8

# Create training data
X_train = X[:round(len(X)*train_perc)]
y_train = y[:round(len(X)*train_perc)]


# Create test data
X_test = X[round(len(X)*train_perc):]
y_test = y[round(len(X)*train_perc):]

class_names = ['other', 'disp_2']

num_train = len(X_train)
num_test = len(X_test)

##
""" Define model parameters """

BATCH_SIZE = 32  # Number of training examples to process before updating model parameters
IMG_SHAPE = 150  # Data consists of images 1024 X 632 pixels

##
""" Generate image data set """

# Create image generator objects
train_image_generator = ImageDataGenerator()
test_image_generator = ImageDataGenerator()

# Use object to create data set
train_data_gen = train_image_generator.flow(X_train, y_train, batch_size=BATCH_SIZE, shuffle=True)
test_data_gen = test_image_generator.flow(X_test, y_test, batch_size=BATCH_SIZE, shuffle=False)


##
""" Define a function to view some images """


def plotImages(images_arr, labels_arr, dim=3):
    fig, axes = plt.subplots(dim, dim, figsize=(6, 6))
    axes = axes.flatten()
    i = 0
    for img, ax in zip(images_arr, axes):
        ax.imshow(np.squeeze(img), aspect='auto')
        ax.set_xlabel(class_names[labels_arr[i]])
        i += 1
    plt.tight_layout()
    plt.show()


##
""" Plot sample images """

# Grab a batch of images
sample_images, sample_labels = next(train_data_gen)

# Plot 9 images
plotImages(sample_images[:9], sample_labels[:9])


##
""" Define the model """

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(2)
])

##
""" Compule the model """

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

##
""" Model summary """

model.summary()

##
""" Train the model """

EPOCHS = 10
history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=int(np.ceil(num_train / float(BATCH_SIZE))),
    epochs=EPOCHS
)

##