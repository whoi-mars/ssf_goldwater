##
""" Import libraries """

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import KFold

from sklearn.metrics import precision_score

import matplotlib.pyplot as plt
import numpy as np
import pickle

##
""" Import the labeled data set"""

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

##
""" Preprocess data """

# TODO: Vectorize
for i in range(len(X_train)):
    # rescale
    X_train[i] = (X_train[i] - np.min(X_train[i])) / (np.max(X_train[i]) - np.min(X_train[i]))

# TODO: Vectorize
for i in range(len(X_val)):
    # rescale
    X_val[i] = (X_val[i] - np.min(X_val[i])) / (np.max(X_val[i]) - np.min(X_val[i]))

# Concatenate train and validation data because we're using K-fold cross validation
X = np.concatenate((X_train, X_val), axis=0)
y = np.concatenate((y_train, y_val), axis=0)

# Define per-fold score containers and histories
acc_per_fold = []
loss_per_fold = []
history_per_fold = []
precision_per_fold = []

# Define array of class names
class_names = ['other', 'disp_2']


##
""" Define model parameters """

BATCH_SIZE = 32  # Number of training examples to process before updating model parameters
IMG_SHAPE = 128  # Data consists of images 1024 X 632 pixels
NUM_FOLDS = 5


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

# Plot 9 images
plotImages(X[:9], y[:9], dim=3)


##
""" Train and Evaluate K-Fold Model """

fold_no = 1
kf = KFold(n_splits=NUM_FOLDS, shuffle=True)
for train, val in kf.split(X):

    # Define the model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(8, (3, 3), activation='relu', input_shape=(IMG_SHAPE, IMG_SHAPE, 1)),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(2)
    ])

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    print("--------------------------------------------------------------------------------")
    print("Training for fold {} ...".format(fold_no))

    # Train the model
    EPOCHS = 30
    history = model.fit(
                X[train], y[train],
                batch_size=BATCH_SIZE,
                steps_per_epoch=int(np.ceil(len(train) / float(BATCH_SIZE))),
                epochs=EPOCHS,
                validation_data=(X[val], y[val])
    )

    # calculate fold loss/accuracy
    loss_acc = model.evaluate(X[val], y[val], verbose=0)

    # calculate fold precision
    y_pred = model.predict(X[val])
    y_pred = np.argmax(y_pred, axis=1)
    prec = precision_score(y[val], y_pred, average='binary')

    # Report and store fold metrics
    print("Fold {} --> val_loss: {} - val_accuracy: {} - val_precision: {}".format(fold_no, loss_acc[0], loss_acc[1], prec))
    acc_per_fold.append(loss_acc[1] * 100)
    loss_per_fold.append(loss_acc[0])
    history_per_fold.append(history)
    precision_per_fold.append(prec)

    # Iterate fold number
    fold_no += 1

# Provide average scores
print("--------------------------------------------------------------------------------")
print("Scores per fold")
for i in range(len(acc_per_fold)):
    print("Fold {} --> loss: {} - accuracy: {} - precision: {}".format(i + 1, loss_per_fold[i], acc_per_fold[i], precision_per_fold[i]))
print("--------------------------------------------------------------------------------")
print("Average scores for all folds")
print("loss: {} - accuracy: {} - precision: {}".format(np.mean(loss_per_fold), np.mean(acc_per_fold), np.mean(precision_per_fold)))
print("--------------------------------------------------------------------------------")


##
""" Visualize the results of training """

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, '-o', label="Training Accuracy")
plt.plot(epochs_range, val_acc, '-o', label="Validation Accuracy")
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, '-o', label="Training Loss")
plt.plot(epochs_range, val_loss, '-o', label="Validation Loss")
plt.legend(loc='lower right')
plt.title('Training and Validation Loss')
plt.show()

##
