# USAGE
# python keras_animals.py --output output/keras_animals.png
import argparse
from sys import displayhook

import matplotlib.pyplot as plt
import numpy as np
from imutils import paths
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (LabelBinarizer, LabelEncoder,
                                   StandardScaler, normalize)
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD

from pyimagesearch.datasets import SimpleDatasetLoader
from pyimagesearch.preprocessing import SimplePreprocessor

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,
	help="path to the output loss/accuracy plot")
args = vars(ap.parse_args())

imagePaths = list(paths.list_images('./datasets/tomates/TOMATES_LADOS_CORES'))
''' O caminho das imagens a serem usadas para treinar a rede neural '''

# initialize the image preprocessor, load the dataset from disk,
# and reshape the data matrix
width = 8
''' a largura para converter a imagem '''

height = 8
''' a altura para converter a imagem '''

sp = SimplePreprocessor(width, height)
sdl = SimpleDatasetLoader(preprocessors=[sp])
(data, labels) = sdl.load(imagePaths, verbose=500)
print('Shape {}'.format(data.shape))
print('Shape[0] {}'.format(data.shape[0]))

area = width * height
''' A área total da imagem '''

size = 3 * area
''' O tamanho total da entrada (tamanho da imagem) * 3(os três canais de cores do RGB) '''

data = data.reshape((data.shape[0], size))

print(data[0])

# convert the labels from integers to vectors
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

# Muda o tratamento em normalizar com norma 1 ou aplicar média zero e desvio padrão 1
normalizar = True

if (normalizar):
    # Aqui os valores dos canais de cores ficam entre zero e um
    #dataNormal = normalize(data)
    rgb_size = 255.0
    ''' O tamanho total da faixa de cores suportadda pelo RGB '''
    dataNormal = data.astype("float") / rgb_size
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
model.add(Dense(area , input_shape=(size,), activation="relu"))

model.add(Dense(area/2, activation="relu"))

output_size = 3
''' A quantidade de classes para saída, a função de ativação deverá ser o softmax para conseguir definir qual será a classe de saída. '''

model.add(Dense(output_size, activation="softmax"))

# train the model using SGD
print("[INFO] training network...")
sgd = SGD(0.01)
model.compile(loss="categorical_crossentropy", optimizer=sgd,
	metrics=["accuracy"])
epoch_size = 100
H = model.fit(trainX, trainY, validation_data=(testX, testY),
	epochs=epoch_size, batch_size=32)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=lb.classes_))

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, epoch_size), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, epoch_size), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, epoch_size), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, epoch_size), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["output"])


