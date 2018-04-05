import numpy
import sys
import os.path
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras.models import load_model
from keras import backend as K
import cv2

K.set_image_dim_ordering('th')

# fix random seed for reproducibility
seed = 20
numpy.random.seed(seed)

'''
Carrega el dataset per entrenar el model y testejarlo
Output: 
	Entrenament:
		X_train: vector de imatges(matrius) -> matrius 28x28 pixels
		y_train: vector que conte els outputs (digits) desitjats.
	Testing:
		X_test: vector de imatges(matrius) -> matrius 28x28 pixels
		y_test: vector que conte els outputs (digits) desitjats.
'''

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Reshape de les matrius [samples][pixels][width][height] a [samples][pixels]
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')


# Normalitzar les entrades de 0-255 a 0-1
X_train = X_train / 255
X_test = X_test / 255
# One hot encode outputs -> Transformar les sortides a format [samples][0 to 9 booleans]
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

#Creacio del model
def creation_model():

	model = Sequential()
	model.add(Conv2D(30, (5, 5), input_shape=(1, 28, 28), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(15, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dense(50, activation='relu'))
	model.add(Dense(num_classes, activation='sigmoid'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

#Carrega d'un model ja existent
def load_existing_model(modelName):
	model = load_model("./model_mnist28x28.h5")
	return model


def main():
	#Construeix un nou model depenen de la variable d'entrada
	trainingMode = sys.argv[1]
	modelName = sys.argv[2]
	if len(sys.argv) != 3:
		sys.exit("El nombre de arguments es incorrecte -> training.py 'create'|'load'  'nomModel.h5'")
	if modelName == "":
		sys.exit("El nom del model no pot ser nul")
	if modelName != "" and trainingMode == "load" and not os.path.isfile(modelName + "." + "h5") :
		sys.exit("No existeix cap model amb el nom %s" (modelName))
	if trainingMode != "load" and trainingMode != "create":
		sys.exit("No s'ha introduit correctament el mode d'execucio -> 'create' or 'load'")
	if trainingMode == "create":
		model = creation_model()
	if trainingMode == "load":
		model = load_existing_model(modelName)
	# Entrenament del model
	modelToSave = "./" + modelName + ".h5"
	callback = keras.callbacks.ModelCheckpoint(modelToSave, monitor='val_loss', verbose=1, save_best_only=True,  mode='auto')
	model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, callbacks=[callback])
	# Avaluacio del model
	prediction = model.predict(X_test)
	i = 0
	while (i < len(X_test)):
		print("Prediccio[]: " + str(prediction[i]) + " | Caracter real[]: " + str(y_test[i]))
		i = i + 1
	scores = model.evaluate(X_test, y_test, verbose=0)
	print("Error: %.2f%%" % (100-scores[1]*100))

if __name__ == "__main__":
	main()
