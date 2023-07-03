# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools
from keras.utils.np_utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau

# Set a random seed for reproducibility
np.random.seed(2)

# Set the style and context for plotting using Seaborn
sns.set(style='white', context='notebook', palette='deep')

# Load the training and test datasets
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Separate the target variable (label) from the training dataset
Y_train = train["label"]

# Drop the 'label' column from the training dataset
X_train = train.drop(labels=["label"], axis=1)

# Normalize the pixel values in the datasets
X_train = X_train / 255.0
test = test / 255.0

# Reshape the training and test datasets to 3 dimensions (height = 28px, width = 28px, channel = 1)
X_train = X_train.values.reshape(-1, 28, 28, 1)
test = test.values.reshape(-1, 28, 28, 1)

# Encode the target variable (label) using one-hot encoding
Y_train = to_categorical(Y_train, num_classes=10)

# Set the random seed for reproducibility
random_seed = 2

# Split the training dataset into training and validation sets
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=random_seed)

# Build the CNN model (LeNet architecture)
model_mod = Sequential()
model_mod.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same', activation='relu', input_shape=(28, 28, 1)))
model_mod.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same', activation='relu'))
model_mod.add(MaxPool2D(pool_size=(2, 2)))
model_mod.add(Dropout(0.25))
model_mod.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu'))
model_mod.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu'))
model_mod.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
model_mod.add(Dropout(0.25))
model_mod.add(Flatten())
model_mod.add(Dense(256, activation="relu"))
model_mod.add(Dropout(0.5))
model_mod.add(Dense(10, activation="softmax"))

# Define the optimizer for the model
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

# Compile the model
model_mod.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

# Define a learning rate reduction callback
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=0.5, min_lr=0.00001)

# Define the number of epochs and batch size for training
epochs = 25
batch_size = 86

# Perform data augmentation using ImageDataGenerator
datagen = ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=False,
        vertical_flip=False)
datagen.fit(X_train)

# Fit the model to the training data with data augmentation
history_gen = model_mod.fit_generator(
    datagen.flow(X_train, Y_train, batch_size=batch_size),
    epochs=epochs,
    validation_data=(X_val, Y_val),
    verbose=2,
    steps_per_epoch=X_train.shape[0] // batch_size,
    callbacks=[learning_rate_reduction]
)
# Save model
model_mod.save('model.h5')