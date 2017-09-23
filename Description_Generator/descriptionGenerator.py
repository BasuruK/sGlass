import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import h5py
from dataSetHandler import DataHandler

class Generator(object):
    
    def __init__(self, model, root_path, cnn_name='vgg19'):
        
        self.model = model
        self.root_path = root_path
        self.word_to_index_file = root_path + 'Captions/word_to_index.p'
        self.index_to_word_file = root_path + 'Captions/index_to_word.p'
        self.saved_image_features_file = root_path + 'Captions/' + cnn_name + '_extracted_features.h5'
        self.test_data_file = root_path + 'Captions/test_data.txt'
        self.images_path = root_path + 'Images/iaprtc12/'
        self.SENTENCE_BEGIN = None
        self.SENTENCE_END = None
        self.image_features = None
        self.maximum_token_length = None
        self.CNN_Name = cnn_name
        
        self.load_data()
        
        
    def load_data(self):
        
        saved_parameters = np.genfromtxt(self.root_path + 'saved_parameters.txt', delimiter=' ', dtype='str')
        saved_parameters = dict(zip(saved_parameters[:, 0], saved_parameters[:, 1]))
        self.SENTENCE_BEGIN = str(saved_parameters['SENTENCE_BEGIN:'])
        self.SENTENCE_END = str(saved_parameters['SENTENCE_END:'])
        self.image_features = int(saved_parameters['image_features:'])
        self.maximum_token_length = int(saved_parameters['maximum_caption_length:']) + 2
        
        self.test_data = pd.read_table(self.test_data_file, sep='*', encoding='ISO-8859-1')
        self.word_to_index = pickle.load(open(self.word_to_index_file, 'rb'))
        self.index_to_word = pickle.load(open(self.index_to_word_file, 'rb'))
        print(len(self.word_to_index))
        self.vocabulary_size = len(self.word_to_index)
        
        self.saved_image_features = h5py.File(self.saved_image_features_file)
        
    def show_caption(self, image_file=None):

        test_data = self.test_data

        if image_file == None:
            image_name = np.asarray(test_data.sample(1))[0][0]
            print(image_names)
        else:
            image_name = image_file
            
        features = self.get_image_features_to_evaluate(image_name)
        #features = self.saved_image_features[image_name]['image_features'][:]
        text = np.zeros((1, self.maximum_token_length, self.vocabulary_size))
        begin_sentence_token = self.word_to_index[self.SENTENCE_BEGIN]
        text[0, 0, begin_sentence_token] = 1
        image_features = np.zeros((1, self.maximum_token_length, self.image_features))
        image_features[0, 0, :] = features
    
        generated_caption = []
        for word_counter in range(self.maximum_token_length + 1):
            predictions = self.model.predict([text, image_features])
            word_id = np.argmax(predictions[0, word_counter, :])
            next_word_counter = word_counter + 1
            text[0, next_word_counter, word_id] = 1
            word = self.index_to_word[word_id]

            if word == self.SENTENCE_END:
                generated_caption.append('.')
                break
            else:
                generated_caption.append(word)
                
        return ' '.join(generated_caption)
        
    def get_image_features_to_evaluate(self, image_path):
        from keras.preprocessing import image
        from keras.models import Model
        from keras.applications.vgg19 import preprocess_input
        from keras.applications import VGG19
        from CNNModel import CNNModel

        if self.CNN_Name == 'basuru':
            cnn_model = CNNModel.load_cnn_model()
            model = Model(input=cnn_model.input, output=cnn_model.get_layer('Diyath').output)
            self.extracted_features = []
            img = image.load_img(image_path, target_size=(64, 64))
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)
            CNN_features = model.predict(img)
            self.extracted_features.append(np.squeeze(CNN_features))
            self.extracted_features = np.asarray(self.extracted_features)

            return self.extracted_features
        
        elif self.CNN_Name == 'vgg19':
            
            self.IMG_FEATS = 4096
            base_model = VGG19(weights='imagenet')
            model =  Model(input=base_model.input, output=base_model.get_layer('fc2').output)
            self.extracted_features = []
            img = image.load_img(image_path, target_size=(224, 224))
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)
            CNN_features = model.predict(img)
            self.extracted_features.append(np.squeeze(CNN_features))
            self.extracted_features = np.asarray(self.extracted_features)

            return self.extracted_features
        
        