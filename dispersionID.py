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

# TODO: Do this when generating the data
X = np.asarray(X)
y = np.asarray(y)

# Add "batch" dimension
X = X[:, :, :, np.newaxis]

# Normalize each image by its square sum
square_sum = np.sum(np.sum(X ** 2, axis=1), axis=1)[:, np.newaxis, np.newaxis]
X = X / square_sum

##
""" Function to shuffle data and labels in unison"""


def shuffle_in_unison(X, y):
    rng_state = np.random.get_state()
    np.random.shuffle(X)
    np.random.set_state(rng_state)
    np.random.shuffle(y)
    return X, y


##
""" Split data """

# Shuffle
X, y = shuffle_in_unison(X, y)

# Percent training data
train_perc = 0.8

# Create training data
X_train = X[:round(len(X)*train_perc)]
y_train = y[:round(len(X)*train_perc)]


# Create test data
X_val = X[round(len(X)*train_perc):]
y_val = y[round(len(X)*train_perc):]

class_names = ['other', 'disp_2']

num_train = len(X_train)
num_val = len(X_val)

##
""" Define model parameters """

BATCH_SIZE = 32  # Number of training examples to process before updating model parameters
IMG_SHAPE = 128  # Data consists of images 1024 X 632 pixels

##
""" Generate image data set """

# Create image generator objects
train_image_generator = ImageDataGenerator()
val_image_generator = ImageDataGenerator()

# Use object to create data set
train_data_gen = train_image_generator.flow(X_train, y_train, batch_size=BATCH_SIZE, shuffle=True)
val_data_gen = val_image_generator.flow(X_val, y_val, batch_size=BATCH_SIZE, shuffle=False)


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
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SHAPE, IMG_SHAPE, 1)),
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

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

##
""" Model summary """

model.summary()

##
""" Train the model """

EPOCHS = 5
history = model.fit(
    train_data_gen,
    steps_per_epoch=int(np.ceil(num_train / float(BATCH_SIZE))),
    epochs=EPOCHS,
    validation_data=val_data_gen
)

##
""" Visualize the results of training """

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label="Training Accuracy")
plt.plot(epochs_range, val_acc, label="Validation Accuracy")
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label="Training Loss")
plt.plot(epochs_range, val_loss, label="Validation Loss")
plt.legend(loc='lower right')
plt.title('Training and Validation Loss')
plt.show()

##
