##
""" Import libraries """
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import precision_score, recall_score

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

for i in range(len(X_train)):
    X_train[i] = (X_train[i] - np.min(X_train[i])) / (np.max(X_train[i]) - np.min(X_train[i]))

for i in range(len(X_val)):
    X_val[i] = (X_val[i] - np.min(X_val[i])) / (np.max(X_val[i]) - np.min(X_val[i]))

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
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(2)
])

##
""" Compule the model """

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

##
""" Model summary """

model.summary()

##
""" Train the model """

EPOCHS = 30
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

y_pred = model.predict(X_val)
y_pred = np.argmax(y_pred, axis=1)

print(precision_score(y_val, y_pred, average='binary'))

##
