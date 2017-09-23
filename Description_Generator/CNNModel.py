import tensorflow as tf
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Flatten
from keras.layers import Dense, Dropout
from keras.models import load_model
from keras.callbacks import History, ModelCheckpoint
from keras.optimizers import RMSprop, SGD

class CNNModel(object):
    
    def  load_cnn_model():
        
        root_path = 'Data/PreTrainedModels/'
        pre_trained_model = root_path + 'imageModel.h5'
        pre_trained_weights = root_path + 'imageModelWeights.h5'

        # Creating CNN model
        imageModel = Sequential()
        imageModel.add(Conv2D(32, (5, 5), input_shape = (64, 64, 3), activation = 'relu', padding = 'same'))
        imageModel.add(MaxPooling2D(pool_size = (2, 2)))

        imageModel.add(Conv2D(32, (5, 5), activation = 'relu', padding = 'same'))
        imageModel.add(MaxPooling2D(pool_size = (2, 2)))
        imageModel.add(Dropout(0.25))

        imageModel.add(Conv2D(64, (5, 5), activation = 'relu', padding = 'same'))
        imageModel.add(MaxPooling2D(pool_size = (2, 2)))

        imageModel.add(Conv2D(128, (5, 5), activation = 'relu', padding = 'same'))
        imageModel.add(MaxPooling2D(pool_size = (2, 2)))
        imageModel.add(Dropout(0.25))

        imageModel.add(Flatten())
        imageModel.add(Dense(units = 512, activation = 'relu', name = 'Diyath' ))
        imageModel.add(Dropout(0.5))
        imageModel.add(Dense(units = 3, activation = 'softmax'))
        
        imageModel.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

        # Load pre-trained CNN weights
        imageModel.load_weights(pre_trained_weights)
        print('Image Model has been loaded')
        
        return imageModel

