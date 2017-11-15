
import os,cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split

from keras import backend as K
K.set_image_dim_ordering('tf')

from keras.utils import np_utils

from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation,Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam,adadelta
import os.path
from keras.models import load_model

USE_SKLEARN_PREPROCESSING = False

from sklearn import preprocessing
from Dialogue_Manager.settings_manager import SettingsManager
from config import Configurations

class Hand_Gesture_Recognition_System:

    datasetpath = None
    img_rows = 128
    img_colms = 128
    num_channels = 3
    num_epoch = 6
    img_data_scaled = None
    SettingsController = None
    Configurations_Controller = None
    img_name = None

    def __init__(self, datasetpath=None):
        self.datasetpath = datasetpath
        self.SettingsController = SettingsManager()
        self.Configurations_Controller = Configurations()
        
    def image_to_feature_vector(image, size=(128,128)):
        # resize the image to a fixed size , then flttern the image into a list of raw pixels intensities
        return cv2.resize(image, size).flatten()

    def train_handGestureRecognition_CNN(self):
        if USE_SKLEARN_PREPROCESSING:
            img_data = self.img_data_scaled
    
        PATH = os.getcwd()

        # Define data path
        data_path = PATH + self.datasetpath
        data_dir_list = os.listdir(data_path)



        img_data_list=[]
        for dataset in data_dir_list:
            print(data_dir_list)
            img_list = os.listdir(data_path+'/'+dataset)
            print('Loaded the images of dataset-'+'{}\n'.format(dataset))
            for img in img_list:
                input_img = cv2.imread(data_path+'/'+dataset+'/'+img)

                input_img_flatten = self.image_to_feature_vector(input_img, (128, 128))

                img_data_list.append(input_img_flatten)

        img_data = np.array(img_data_list)
        img_data = img_data.astype('float')
        print('Image Data', img_data.shape)

        if self.num_channels == 1:
            if K.image_dim_ordering() == 'th':
                img_data=np.expand_dims(img_data, axis=1)
                print('Image Data BnW', img_data.shape)
            else:
                img_data=np.expand_dims(img_data, axis=4)
                print('Image Data BnW', img_data.shape)

        else:
            if K.image_dim_ordering() == 'th':
                img_data = np.rollaxis(img_data, 3, 1)
                print('Image Data RGB',img_data.shape)

        image_data_scaled=preprocessing.scale(img_data)
        print("Image Data Scaled" , image_data_scaled)

        if K.image_dim_ordering() == 'th':
            image_data_scaled = image_data_scaled.reshape(img_data.shape[0], self.num_channels, self.img_rows, self.img_colms)
            print('Image Data Scaled BnW', image_data_scaled.shape)
        else:
            image_data_scaled=image_data_scaled.reshape(img_data.shape[0], self.img_rows, self.img_colms, self.num_channels)
            print('Image Data Scaled RGB', image_data_scaled.shape)

        # Define classes

        num_classes=2

        num_samples = img_data.shape[0]
        labels = np.ones((num_samples,), dtype='int64')

        labels[0:113] = 0
        labels[113:] = 1

        names=['negative','positive']

        # convert class labels to on-hot encoding
        Y = np_utils.to_categorical(labels, num_classes)

        x,y=shuffle(img_data, Y, random_state=2)

        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)

        X_train = X_train.reshape(X_train.shape[0], self.img_colms, self.img_rows, -1)
        X_test = X_test.reshape(X_test.shape[0], self.img_colms, self.img_rows, -1)

        input_shape=(self.img_colms, self.img_rows, 1)

        # Defining the model
        input_shape = img_data[0].shape

        model = Sequential()

        model.add(Conv2D(32, (3,3),border_mode='same', input_shape=(128, 128, 3)))
        model.add(Activation('relu'))

        model.add(Conv2D(32,( 3, 3)))
        model.add(Activation('relu'))

        model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='th'))
        model.add(Dropout(0.5))

        model.add(Conv2D(64, (3, 3), dim_ordering='th'))
        model.add(Activation('relu'))

        model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='th'))
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(512)) # no of hidden layers
        model.add(Activation('relu'))
        model.add(Dropout(0.3))
        model.add(Dense(num_classes))
        model.add(Activation('softmax'))

        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy']) 

        # Viewing model_configuration

        model.summary()
        model.get_config()
        model.layers[0].get_config()
        model.layers[0].input_shape
        model.layers[0].output_shape
        model.layers[0].get_weights()
        np.shape(model.layers[0].get_weights()[0])
        model.layers[0].trainable

        # Training

        if os.path.isfile('Hand_Gesture_Recognition.h5') == False:
            hist = model.fit(X_train, y_train, batch_size=150, epochs=self.num_epoch, verbose=1, validation_data=(X_test, y_test))
            model.save('Hand_Gesture_Recognition.h5')
        else:
            hist=load_model('Hand_Gesture_Recognition.h5')

        # Evaluating the model

        score = model.evaluate(X_test, y_test, batch_size=150 , verbose=0) # batch_size=16
        print('Test Loss:', score[0])
        print('Test accuracy:', score[1])

        test_image = X_test[0:1]

        # print(model.predict(test_image))
        # print(model.predict_classes(test_image))
        y_test[0:1]

        return model

    # Testing a new image
    def Get_Reult_Of_HG(self, filepath):

        test_image = cv2.imread(filepath)
        test_image = cv2.resize(test_image, (128, 128))
        test_image = np.array(test_image)
        test_image = test_image.astype('float32')
        test_image /= 255
        print(test_image.shape)

        if self.num_channels == 1:
            if K.image_dim_ordering() == 'th':
                test_image = np.expand_dims(test_image, axis=1)
                print(test_image.shape)

            else:
                test_image = np.expand_dims(test_image, axis=4)
                print(test_image.shape)

        else:
            if K.image_dim_ordering() == 'th':
                test_image = np.rollaxis(test_image, 3, 1)
                print(test_image.shape)
            else:
                test_image = np.expand_dims(test_image, axis=0)
                print(test_image.shape)

        # Predicting the test image
        
        hgModel = self.load_hand_model()
        
        print(hgModel.predict(test_image))
        predict_class=hgModel.predict_classes(test_image)
        print(predict_class)

        result=''

        if predict_class == [1]:
            print('Result :', predict_class, '=> Positive Hand Gesture')
            result='Positive Hand Gesture'

        elif predict_class == [0]:
            print('Result :', predict_class, '=> Negative Hand Gesture')
            result='Negative Hand Gesture'

        return result
    
    def capture_HG_image(self):

        cam = cv2.VideoCapture(0)

        save = 'C:/Users/Nipuni/AnacondaProjects/Indoor_Object_Identification_System/IOIM/Original/Test'

        cv2.namedWindow("test")

        img_counter = 0

        while True:
            ret, frame = cam.read()
            cv2.imshow("test", frame)
            if not ret:
                break
            k = cv2.waitKey(1) % 256

            if k == 27:
                # ESC pressed
                print("Escape hit, closing...")
                break
            elif k == 32:
                # SPACE pressed
                self.img_name="hand_gesture_{}.png".format(img_counter)

                return_value, image = cam.read()

                cv2.imwrite(os.path.join(save, self.img_name), image)

                print("{} written!".format(self.img_name))

                img_counter += 1
            # Change environment to Outdoor
            if k == 101:
                self.Configurations_Controller.set_environment_mode_indoor()
                print("Environment Changing")
                k = 10

            if (k == 10 or
                    self.SettingsController.signal_recognition_engines_to_quit() or
                    self.SettingsController.signal_recognition_engines_to_quit_on_platform_change() or
                    self.SettingsController.signal_recognition_engines_to_quit_when_system_quits()):

                cv2.destroyAllWindows()
                cam.release()
                print("Indoor Object Detection System Exiting")
                break

        cam.release()

        cv2.destroyAllWindows()

        return self.img_name

    def load_hand_model(self):
        hist = load_model('Indoor_Object_Recognition_Engine/Hand_Gesture/Hand_Gesture_Recognition.h5')
        return hist

