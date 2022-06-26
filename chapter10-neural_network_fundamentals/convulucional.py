# USAGE
# python keras_animals.py --output output/keras_animals.png
import argparse
import os

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import plaidml.keras
from plaidml.keras import backend as K

import matplotlib.pyplot as plt
import numpy as np
from imutils import paths
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from tensorflow.keras import backend as K
from tensorflow.keras import Input
from tensorflow.keras.layers import (Activation, Conv2D, Dense, Dropout,
                                     Flatten, MaxPooling2D)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD

from pyimagesearch.datasets import SimpleDatasetLoader
from pyimagesearch.preprocessing import (ImageToArrayPreprocessor,
                                         SimplePreprocessor)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-e", "--epoc", required=True,
    help="epocs")
args = vars(ap.parse_args())

# grab the list of images that we'll be describing
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))

# initialize the image preprocessors
sp = SimplePreprocessor(32, 32)
iap = ImageToArrayPreprocessor()

# load the dataset from disk then scale the raw pixel intensities
# to the range [0, 1]
sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.astype("float") / 255.0

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.25, random_state=42)

# convert the labels from integers to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)

width = 32
height = 32
depth = 3
input_shape = (width, height, 3)

if K.image_data_format() == "channels_first":
    input_shape = (3, height, width)
model = Sequential (
    [
        Input(shape = input_shape),
        Conv2D(32, (3, 3), input_shape=input_shape, activation="relu"),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), input_shape=input_shape, activation="relu"),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(64, activation="relu"),
        Dropout(0.5),
        Dense(32, activation="relu"),
        Dense(3),
        Activation("softmax")
    ]
)
opt = SGD(lr=0.005)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

epoc = int(args["epoc"])

# train the network
print("[INFO] training network...")
H = model.fit(trainX, trainY, validation_data=(testX, testY),
	batch_size=32, epochs=epoc, verbose=1)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=lb.classes_))

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, epoc), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, epoc), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, epoc), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, epoc), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()


