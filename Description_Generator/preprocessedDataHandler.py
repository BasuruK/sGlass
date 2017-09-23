import h5py
import numpy as np
import pandas as pd
import pickle
import random
from .dataSetHandler import DataHandler

'''
This class handles preprocessed data of the DataHandler and provide into 
CaptionGenerator model

'''
class PreprocessDataHandler():
    
    def __init__(self, root_path, batch_size=64, cnn_name='vgg19'):
        
        self.root_path = root_path
        self.training_file = self.root_path + 'Captions/training_data.txt'
        self.testing_file = self.root_path + 'Captions/validation_data.txt'
        self.image_features_file = self.root_path + 'Captions/' + cnn_name + '_extracted_features.h5'
        self.training_data = None
        self.testing_data = None
        self.saved_features = None
        saved_parameters = np.genfromtxt(self.root_path + 'saved_parameters.txt', delimiter=' ', dtype='str')
        saved_parameters = dict(zip(saved_parameters[:, 0], saved_parameters[:, 1]))
        self.SENTENCE_BEGIN = str(saved_parameters['SENTENCE_BEGIN:'])
        self.SENTENCE_END = str(saved_parameters['SENTENCE_END:'])
        self.vocabulary_size = None
        self.word_to_index = None
        self.index_to_word = None
        self.batch_size = batch_size
        self.image_features = int(saved_parameters['image_features:'])
        self.maximum_token_length = int(saved_parameters['maximum_caption_length:']) + 2
        
        self.load_train_test_datasets()
        self.load_dictionaries()
        self.load_saved_image_features()
        
        
    def load_train_test_datasets(self):
        
        print('Loading training dataset....')
        temp_training_data = pd.read_table(self.training_file, delimiter='*', encoding='ISO-8859-1')
        temp_training_data = np.asarray(temp_training_data, dtype=str)
        self.training_data = temp_training_data
        
        print('Loading testing dataset....')
        temp_testing_data = pd.read_table(self.testing_file, delimiter='*', encoding='ISO-8859-1')
        temp_testing_data = np.asarray(temp_testing_data, dtype=str)
        self.testing_data = temp_testing_data
        
    def load_dictionaries(self):
        
        print('Loading dictionaries')
        self.word_to_index = pickle.load(open(self.root_path + 'Captions/word_to_index.p', 'rb'))
        self.index_to_word = pickle.load(open(self.root_path + 'Captions/index_to_word.p', 'rb'))
        self.vocabulary_size = len(self.word_to_index)
    
    def flow(self, mode):
        
        print('Loading data for the model...')

        if mode == 'train':
            data = self.training_data

        if mode == 'testing':
            data = self.testing_data
        
        image_names = data[:,0].tolist()
        empty_batch = self.make_empty_batch()
        captions_batch = empty_batch[0]
        images_batch = empty_batch[1]
        targets_batch = empty_batch[2]
        
        batch_counter = 0
        while True:
            for data_counter, image_name in enumerate(image_names):
                caption = data[data_counter,1]
                one_hot_caption = self.format_to_one_hot(caption)
                captions_batch[batch_counter, :, :] = one_hot_caption
                targets_batch[batch_counter, :, :]  = self.get_one_hot_target(
                                                            one_hot_caption)
                images_batch[batch_counter, :, :]   = self.get_image_features(
                                                            image_name)
                if batch_counter == self.batch_size - 1:
                    yield_dictionary = self.wrap_in_dictionary(captions_batch,
                                                                images_batch,
                                                                targets_batch)
                    yield yield_dictionary

                    empty_batch = self.make_empty_batch()
                    captions_batch = empty_batch[0]
                    images_batch = empty_batch[1]
                    targets_batch = empty_batch[2]
                    batch_counter = 0

                batch_counter = batch_counter + 1
     
                
    def load_saved_image_features(self):
        print('Load saved image features....')
        self.saved_features = h5py.File(self.image_features_file, 'r')
        
    def make_empty_batch(self):
        captions_batch = np.zeros((self.batch_size,self.maximum_token_length,
                                    self.vocabulary_size))
        images_batch = np.zeros((self.batch_size, self.maximum_token_length,
                                    self.image_features))
        targets_batch = np.zeros((self.batch_size,self.maximum_token_length,
                                    self.vocabulary_size))
        return captions_batch, images_batch , targets_batch

    def format_to_one_hot(self, caption):
        tokenized_caption = caption.split()
        tokenized_caption = [self.SENTENCE_BEGIN] + tokenized_caption + [self.SENTENCE_END]
        one_hot_caption = np.zeros((self.maximum_token_length,
                                    self.vocabulary_size))
        word_ids = [self.word_to_index[word] for word in tokenized_caption
                        if word in self.word_to_index]
        for sequence_arg, word_id in enumerate(word_ids):
            one_hot_caption[sequence_arg,word_id] = 1
        return one_hot_caption

    def get_image_features(self, image_name):
        image_feats = self.saved_features[image_name]\
                                            ['image_features'][:]
        image_input = np.zeros((self.maximum_token_length, self.image_features))
        image_input[0,:] =  image_feats
        return image_input

    def get_one_hot_target(self, one_hot_caption):
        one_hot_target = np.zeros_like(one_hot_caption)
        one_hot_target[:-1, :] = one_hot_caption[1:, :]
        return one_hot_target

    def wrap_in_dictionary(self, one_hot_caption,
                           image_features,
                           one_hot_target):

        return [{'text': one_hot_caption,
                'image': image_features},
                {'output': one_hot_target}]
        
    
        
    
        