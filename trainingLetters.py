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
from keras.layers import InputLayer, Conv2D, Dense, MaxPool2D, Dropout, Flatten, ZeroPadding2D, Lambda
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from os.path import join
from keras.models import load_model
from keras import backend as K
import cv2


# coding: utf-8

# ##  Imports

# In[1]:


import keras


# In[2]:


from keras.models import Sequential


# In[3]:


from keras.layers import InputLayer, Conv2D, Dense, MaxPool2D, Dropout, Flatten, ZeroPadding2D, Lambda


# In[4]:


from IPython.display import FileLink


# In[5]:


from keras.preprocessing.image import ImageDataGenerator


# In[6]:


from keras.initializers import RandomUniform


# In[7]:


from keras.optimizers import Adam, RMSprop


# In[8]:


from keras.activations import relu,softmax


# In[9]:
import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
from keras.layers.advanced_activations import LeakyReLU 
from keras.preprocessing.image import ImageDataGenerator


K.set_image_dim_ordering('tf')

# fix random seed for reproducibility
#7

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
'''
(X_train, y_train), (X_test, y_test) = mnist.load_data()

i = 0
while (i < 20):
	cv2.imshow("vertical", X_train[i])
	cv2.waitKey(10000)
	cv2.destroyAllWindows()
	i += 1
'''
'''
# Reshape de les matrius [samples][pixels][width][height] a [samples][pixels
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')
#Mostra imatge abans dentrar
itera=0
while(itera <9):
	z = 0
	while (z < 28):
	    w = 0
	    while (w < 28):
	        print X_train[itera][0][z][w],
	        w+=1
	    z+=1
	    print
	print
	itera+=1
print
'''

'''
itera=0
while(itera <9):
	z = 0
	while (z < 28):
	    w = 0
	    while (w < 28):
	        print X_test[itera][0][z][w],
	        w+=1
	    z+=1
	    print
	print
	itera+=1
print

'''
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0],28, 28,1)
X_test = X_test.reshape(X_test.shape[0],28, 28,1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train/=255
X_test/=255

#idg = ImageDataGenerator()

#path = "./data/nist/"
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.layers import Conv2D, MaxPooling2D, Convolution2D, Dropout, Dense, Flatten, LSTM
from keras.models import Sequential, save_model
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from scipy.io import loadmat
import time
import pickle
import argparse
import keras
import numpy as np
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
from keras.layers import InputLayer, Conv2D, Dense, MaxPool2D, Dropout, Flatten, ZeroPadding2D, Lambda
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from os.path import join
from keras.models import load_model
from keras import backend as K
import cv2


# coding: utf-8

# ##  Imports

# In[1]:


import keras


# In[2]:


from keras.models import Sequential


# In[3]:


from keras.layers import InputLayer, Conv2D, Dense, MaxPool2D, Dropout, Flatten, ZeroPadding2D, Lambda


# In[4]:


from IPython.display import FileLink


# In[5]:


from keras.preprocessing.image import ImageDataGenerator


# In[6]:


from keras.initializers import RandomUniform


# In[7]:


from keras.optimizers import Adam, RMSprop


# In[8]:


from keras.activations import relu,softmax


# In[9]:
import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
from keras.layers.advanced_activations import LeakyReLU 
from keras.preprocessing.image import ImageDataGenerator


K.set_image_dim_ordering('tf')



start = time.time()


def load_data(mat_file_path, width=28, height=28, max_=None, verbose=True):
    def rotate(img):
        # Used to rotate images (for some reason they are transposed on read-in)
        flipped = np.fliplr(img)
        return np.rot90(flipped)

    def display(img, threshold=0.5):
        # Debugging only
        render = ''
        for row in img:
            for col in row:
                if col > threshold:
                    render += '@'
                else:
                    render += '.'
            render += '\n'
        return render

    # Load convoluted list structure form loadmat
    mat = loadmat(mat_file_path)

    # Load char mapping
    mapping = {kv[0]: kv[1:][0] for kv in mat['dataset'][0][0][2]}
    pickle.dump(mapping, open('bin/mapping.p', 'wb'))

    # Load training data
    if max_ == None:
        max_ = len(mat['dataset'][0][0][0][0][0][0])
    training_images = mat['dataset'][0][0][0][0][0][0][:max_].reshape(
        max_, height, width, 1)
    training_labels = mat['dataset'][0][0][0][0][0][1][:max_]

    # Load testing data
    if max_ == None:
        max_ = len(mat['dataset'][0][0][1][0][0][0])
    else:
        max_ = int(max_ / 6)
    testing_images = mat['dataset'][0][0][1][0][0][0][:max_].reshape(
        max_, height, width, 1)
    testing_labels = mat['dataset'][0][0][1][0][0][1][:max_]

    # Reshape training data to be valid
    if verbose == True:
        _len = len(training_images)
    for i in range(len(training_images)):
        if verbose == True:
            print('%d/%d (%.2lf%%)' % (i + 1, _len, ((i + 1)/_len) * 100))
        training_images[i] = rotate(training_images[i])
    if verbose == True:
        print('')




    # Reshape testing data to be valid
    if verbose == True:
        _len = len(testing_images)
    for i in range(len(testing_images)):
        if verbose == True:
            print('%d/%d (%.2lf%%)' % (i + 1, _len, ((i + 1)/_len) * 100))
        testing_images[i] = rotate(testing_images[i])
    if verbose == True:
        print('')

        ##DEBUG
    '''
    i = 0
    while (i < 3000):
    	print testing_labels[i]
        cv2.imshow("vertical", testing_images[i])
        cv2.waitKey(10000)
        cv2.destroyAllWindows()
        i += 1
    '''
    # Convert type to float32
    training_images = training_images.astype('float32')
    testing_images = testing_images.astype('float32')
    '''
    # Normalize to prevent issues with model
    training_images /= 255
    testing_images /= 255
	'''
    nb_classes = len(mapping)
    print training_images.shape
    print "Shape testing_images: ", training_labels.shape
    print "Shape testing labels: ", testing_labels.shape
    print testing_labels
    testing_labels = testing_labels.reshape(testing_labels.shape[0])
    #testing_labels -= 1
    training_labels = training_labels.reshape(training_labels.shape[0])
    #training_labels -= 1
    print testing_labels
    print training_labels.shape
    print "Shape testing labels: ", testing_labels.shape
    i = 0
    while (i < 10):
    	print testing_labels[i]
        cv2.imshow("v", testing_images[i])
        cv2.waitKey(10000)
        cv2.destroyAllWindows()
        i += 1

    i = 0
    training_images_capital_letters = []
    training_labels_capital_letters = []
    print training_labels.shape[0]
    while(i < training_labels.shape[0]):
    	if (training_labels[i] >= 10 and training_labels[i] <= 35):
    		training_labels_capital_letters.append(training_labels[i])
    		training_images_capital_letters.append(training_images[i])

    	i +=1
    i = 0
    testing_images_capital_letters = []
    testing_labels_capital_letters = []
    print testing_labels.shape[0]
    while(i < testing_labels.shape[0]):
    	if (testing_labels[i] >= 10 and testing_labels[i] <= 35):
    		print "Entro al bucle amb el valor: ", testing_labels[i]
    		testing_labels_capital_letters.append(testing_labels[i])
    		testing_images_capital_letters.append(testing_images[i])
    	i +=1

    testing_labels_capital_lettersAux = np.asarray(testing_labels_capital_letters)
    testing_images_capital_lettersAux = np.asarray(testing_images_capital_letters)
    training_labels_capital_lettersAux = np.asarray(training_labels_capital_letters)
    training_images_capital_lettersAux = np.asarray(training_images_capital_letters)
    training_labels_capital_lettersAux-=10
    testing_labels_capital_lettersAux-=10

    print training_images_capital_lettersAux.shape
    print "Shape training labels: ", training_labels_capital_lettersAux.shape
    print "Shape testing labels: ", testing_labels_capital_lettersAux.shape
    print testing_labels_capital_lettersAux

    print testing_labels_capital_lettersAux
    print training_labels_capital_lettersAux.shape
    print "Shape testing labels: ", testing_labels_capital_lettersAux.shape


    i = 0
    while (i < 10):
    	print str(unichr(testing_labels_capital_lettersAux[i] + 65))
        cv2.imshow("vertical", testing_images_capital_lettersAux[i])
        cv2.waitKey(15000)
        cv2.destroyAllWindows()
        i += 1
    # Normalize to prevent issues with model
    testing_images_capital_lettersAux /= 255
    training_images_capital_lettersAux /= 255
    return ((training_images_capital_lettersAux, training_labels_capital_lettersAux), (testing_images_capital_lettersAux, testing_labels_capital_lettersAux), mapping, nb_classes)


def creation_model():

    #model agusti28x28
    model = Sequential()

    model.add(Conv2D(64, (3, 3), input_shape=(28,28,1), activation= 'relu'))
    model.add(Conv2D(64, (3, 3), activation= 'relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(128, (3, 3), activation= 'relu'))
    model.add(Conv2D(128, (3, 3), activation= 'relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())
    model.add(Dense(128,activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(64,activation="relu"))
    model.add(Dropout(0.5))

    model.add(Dense(26,activation="sigmoid"))

    #model.add(Activation('sigmoid'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    return model

'''

def creation_model():

    # Initialize data
    (x_train, y_train), (x_test, y_test), mapping, nb_classes = training_data
    input_shape = (height, width, 1)

    print(x_test.shape)

    # Hyperparameters
    nb_filters = 32  # number of convolutional filters to use
    pool_size = (2, 2)  # size of pooling area for max pooling
    kernel_size = (3, 3)  # convolution kernel size

    model = Sequential()
    model.add(Convolution2D(64,
                            kernel_size,
                            padding='valid',
                            input_shape=input_shape,
                            activation='relu'))
    model.add(MaxPooling2D(pool_size=pool_size))

    # model.add(Convolution2D(64,
    #                         kernel_size,
    #                         activation='relu'))

    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.25))
    model.add(Convolution2D(64,
                            kernel_size,
                            padding='valid',
                            input_shape=input_shape,
                            activation='relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.25))
    model.add(Flatten())

    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    if verbose == True:
        print(model.summary())
    return model
'''



'''

if __name__ == '__main__':

    bin_dir = os.path.dirname(os.path.realpath(__file__)) + '/bin'
    if not os.path.exists(bin_dir):
        os.makedirs(bin_dir)

    training_data = load_data(args.file, width=args.width,
                              height=args.height, max_=args.max, verbose=args.verbose)

    train(model, training_data, epochs=args.epochs)
'''

#Carrega d'un model ja existent
def load_existing_model(modelName):
	model = load_model("./model_emnist_28x28.h5")
	return model

def main():
    #Construeix un nou model depenen de la variable d'entrada
    trainingMode = sys.argv[1]
    modelName = sys.argv[2]
    path = sys.argv[3]
    if len(sys.argv) != 4:
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

    #CARREGAR DADES ENTRENAMENT:
    training_data = load_data(path, width=28, height=28, max_=None, verbose=False)
    (x_train, y_train), (x_test, y_test), mapping, nb_classes = training_data
    y_train = np_utils.to_categorical(y_train, 26)
    y_test = np_utils.to_categorical(y_test, 26)
    #gen = ImageDataGenerator(
    	#rotation_range=20,
        #width_shift_range=0.2,
        #height_shift_range=0.2,
        #horizontal_flip=True)
    #gen = ImageDataGenerator(rotation_range=8, width_shift_range=0.15, shear_range=0.3, height_shift_range=0.15, zoom_range=0.15)
    #gen =  ImageDataGenerator( zoom_range=0.3, width_shift_range=0.2, height_shift_range=0.2)
    gen = ImageDataGenerator(rotation_range=8, width_shift_range=0.08, shear_range=0.3, height_shift_range=0.08, zoom_range=0.08)
    test_gen = ImageDataGenerator()
    train_generator = gen.flow(x_train, y_train, batch_size=64)
    test_generator = test_gen.flow(x_test, y_test, batch_size=64)   
    # Entrenament del model
    modelToSave = "./" + modelName + ".h5"
    callback = keras.callbacks.ModelCheckpoint(modelToSave, monitor='val_loss', verbose=1, save_best_only=True,  mode='auto')
    #model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=20, callbacks=[callback])
    #history = model.fit_generator(batches, 200, epochs=5, validation_data=val_batches, validation_steps=val_batches.samples/val_batches.batch_size, verbose=2, callbacks=[callback])
    model.fit_generator(train_generator, steps_per_epoch=60000//64, epochs=5, validation_data=test_generator, validation_steps=10000//64,  callbacks=[callback])
    # Avaluacio del model
    #prediction = model.predict(history)
    prediction = model.predict(x_test)
    i = 0
    while (i < len(X_test)):
        print("Prediccio[]: " + str(prediction[i]) + " | Caracter real[]: " + str(y_test[i]))
        i = i + 1
    #scores = model.evaluate(X_test, y_test, verbose=0)
    scores = model.evaluate(x_test, y_test, verbose=0)
    print("Error: %.2f%%" % (100-scores[1]*100))

if __name__ == "__main__":
    main()
#batches = idg.flow_from_directory(join(path, "train"), target_size=(64,64), color_mode="grayscale")
#val_batches = idg.flow_from_directory(join(path, "val"), target_size=(64,64), color_mode="grayscale")

number_of_classes = 10

Y_train = np_utils.to_categorical(y_train, number_of_classes)
Y_test = np_utils.to_categorical(y_test, number_of_classes)
#gen = ImageDataGenerator(rotation_range=8, width_shift_range=0.08, shear_range=0.3, height_shift_range=0.08, zoom_range=0.08)
#gen =  ImageDataGenerator( zoom_range=0.3, width_shift_range=0.2, height_shift_range=0.2)
#gen = ImageDataGenerator()
#gen = ImageDataGenerator(rotation_range=8, width_shift_range=0.15, shear_range=0.3, height_shift_range=0.15, zoom_range=0.15)
gen = ImageDataGenerator()
test_gen = ImageDataGenerator()
train_generator = gen.flow(X_train, Y_train, batch_size=64)
test_generator = test_gen.flow(X_test, Y_test, batch_size=64)
#i = 0
#while (i < 20):
#	cv2.imshow("vertical", X_train[i])
#	cv2.waitKey(10000)
#	cv2.destroyAllWindows()
#	i += 1

# Normalitzar les entrades de 0-255 a 0-1
#X_train = X_train / 255
#X_test = X_test / 255

'''
itera=0
while(itera <9):
	z = 0
	while (z < 28):
	    w = 0
	    while (w < 28):
	        print X_test[itera][0][z][w],
	        w+=1
	    z+=1
	    print
	print
	itera+=1
print
'''

# One hot encode outputs -> Transformar les sortides a format [samples][0 to 9 booleans]
#y_train = np_utils.to_categorical(y_train)
#y_test = np_utils.to_categorical(y_test)
#num_classes = y_test.sha





#Creacio del model
def creation_model():

	#model agusti28x28
	model = Sequential()

	model.add(Conv2D(64, (3, 3), input_shape=(28,28,1), activation= 'relu'))
	model.add(Conv2D(64, (3, 3), activation= 'relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Conv2D(128, (3, 3), activation= 'relu'))
	model.add(Conv2D(128, (3, 3), activation= 'relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))

	model.add(Flatten())
	model.add(Dense(128,activation="relu"))
	model.add(Dropout(0.5))
	model.add(Dense(64,activation="relu"))
	model.add(Dropout(0.5))

	model.add(Dense(10,activation="sigmoid"))

	#model.add(Activation('sigmoid'))
	model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
	return model


	'''
	#model agustimarcel
	model = Sequential()

	model.add(Conv2D(32, (3, 3), input_shape=(28,28,1), activation= 'relu'))
	model.add(Conv2D(32, (3, 3), input_shape=(28,28,1), activation= 'relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Conv2D(64, (3, 3), input_shape=(28,28,1), activation= 'relu'))
	model.add(Conv2D(64, (3, 3), input_shape=(28,28,1), activation= 'relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))

	model.add(Flatten())
	model.add(Dense(128,activation="relu"))
	model.add(Dropout(0.5))
	model.add(Dense(64,activation="relu"))
	model.add(Dropout(0.5))

	model.add(Dense(10,activation="sigmoid"))

	#model.add(Activation('sigmoid'))
	model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
	return model
	'''
	'''
	#MODEL PROMETEDOR betamodel
	model = Sequential()
	model.add(Conv2D(30, (5, 5), input_shape=(28, 28, 1), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(15, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128,activation="relu"))
	model.add(Dropout(0.5))
	model.add(Dense(64,activation="relu"))
	model.add(Dropout(0.5))
	model.add(Dense(10, activation='sigmoid'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
	'''

	'''
	model = Sequential()
	model.add(InputLayer(input_shape=(64,64,1)))


	model.add(Lambda(function=lambda x: 255 - x))
	model.add(Conv2D(64,3,  data_format=batches.data_format, padding="same", activation="relu"))
	model.add(Conv2D(64,3,  data_format=batches.data_format, padding="same", activation="relu"))
	model.add(MaxPool2D())
	model.add(Conv2D(128,3,  data_format=batches.data_format, padding="same", activation="relu"))
	model.add(Conv2D(128,3,  data_format=batches.data_format, padding="same", activation="relu"))
	model.add(MaxPool2D())

	model.add(Flatten())
	model.add(Dense(128,activation="relu"))
	model.add(Dropout(0.5))
	model.add(Dense(64,activation="relu"))
	model.add(Dropout(0.5))


	#model.add(Dense(26,activation="softmax"))
	model.add(Dense(10,activation="softmax"))
	return model
	'''


'''


def main():
	#Construeix un nou model depenen de la variable d'entrada
	trainingMode = sys.argv[1]
	modelName = sys.argv[2]
	if len(sys.argv) != 4:
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
	#model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=20, callbacks=[callback])
	#history = model.fit_generator(batches, 200, epochs=5, validation_data=val_batches, validation_steps=val_batches.samples/val_batches.batch_size, verbose=2, callbacks=[callback])
	model.fit_generator(train_generator, steps_per_epoch=60000//64, epochs=5, validation_data=test_generator, validation_steps=10000//64,  callbacks=[callback])
	# Avaluacio del model
	#prediction = model.predict(history)
	prediction = model.predict(X_test)
	i = 0
	while (i < len(X_test)):
		print("Prediccio[]: " + str(prediction[i]) + " | Caracter real[]: " + str(y_test[i]))
		i = i + 1
	#scores = model.evaluate(X_test, y_test, verbose=0)
	scores = model.evaluate(X_test, Y_test, verbose=0)
	print("Error: %.2f%%" % (100-scores[1]*100))

if __name__ == "__main__":
	main()
'''