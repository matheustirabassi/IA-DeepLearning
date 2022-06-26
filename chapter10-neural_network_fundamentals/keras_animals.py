# USAGE
# python keras_animals.py --output output/keras_animals.png
from sys import displayhook
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import cifar10
from pyimagesearch.preprocessing import SimplePreprocessor
from pyimagesearch.datasets import SimpleDatasetLoader
from imutils import paths
import argparse
import numpy as np
import math
import matplotlib.pyplot as plt
from IPython.display import clear_output
from matplotlib.pyplot import figure

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,
	help="path to the output loss/accuracy plot")
args = vars(ap.parse_args())

imagePaths = list(paths.list_images('..\\datasets\\animals'))
#imagePaths = list(paths.list_images('..\\datasets\\TOMATES_LADOS_CORES'))

# initialize the image preprocessor, load the dataset from disk,
# and reshape the data matrix
sp = SimplePreprocessor(32, 32)
sdl = SimpleDatasetLoader(preprocessors=[sp])
(data, labels) = sdl.load(imagePaths, verbose=500)
print('Shape {}'.format(data.shape))
print('Shape[0] {}'.format(data.shape[0]))
data = data.reshape((data.shape[0], 3072))

print(data[0])

# convert the labels from integers to vectors
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

# Muda o tratamento em normalizar com norma 1 ou aplicar média zero e desvio padrão 1
normalizar = True

if (normalizar):
    # Aqui os valores dos canais de cores ficam entre zero e um
    #dataNormal = normalize(data)
    dataNormal = data.astype("float") / 255.0
    print(dataNormal[0])
    # partition the data into training and testing splits using 75% of
    # the data for training and the remaining 25% for testing
    (trainX, testX, trainY, testY) = train_test_split(dataNormal, labels,
	    test_size=0.25, random_state=42)
else:
    # Aqui os valores dos canais de cores ficam com média zero e variância um, valores positivos e negativos
    # partition the data into training and testing splits using 75% of
    # the data for training and the remaining 25% for testing
    (trainX, testX, trainY, testY) = train_test_split(data, labels,
        	test_size=0.25, random_state=42)
    print(trainX[0])
    # Os dados de treinamento são ajustados para média zero e desvio padrão 1
    scaler = StandardScaler()
    scaler.fit(trainX)
    trainX = scaler.transform(trainX)
    print(trainX[0])
    # Tecnicamente, não sabemos o que virá nos testes, 
    # por isso não podemos incluí-los no StandardScaler para realizar o fit()
    # que vai calcular a média e o desvio padrão
    testX = scaler.transform(testX)
    print(testX[0])
    
# define the 3072-1024-512-3 architecture using Keras
model = Sequential()
model.add(Dense(1024, input_shape=(3072,), activation="relu"))
model.add(Dense(512, activation="relu"))
model.add(Dense(3, activation="softmax"))

# train the model using SGD
print("[INFO] training network...")
sgd = SGD(0.01)
model.compile(loss="categorical_crossentropy", optimizer=sgd,
	metrics=["accuracy"])
H = model.fit(trainX, trainY, validation_data=(testX, testY),
	epochs=100, batch_size=32)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=lb.classes_))

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 100), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, 100), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["output"])


