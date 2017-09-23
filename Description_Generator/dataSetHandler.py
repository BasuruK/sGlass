import os
import pickle
from string import digits
import time
import h5py
import numpy as np
import pandas as pd
from collections import Counter
from itertools import chain

'''
This class handles all the dataset related activities for both images and captions
CreatedBy: A.R.M.D.N Ranasinghe
'''
class DataHandler(object):
    
    def __init__(self, root_path, caption_file, maximum_caption_length=30, separator='*', cnn_name='vgg19',                                                  allowed_word_frequency=2, image_dataset=None, extract_image_features=False):
        
        self.root_path = root_path
        self.caption_file = caption_file
        self.maximum_caption_length = maximum_caption_length
        self.separator = separator
        self.allowed_word_frequency = allowed_word_frequency
        self.image_dataset = image_dataset
        self.extract_image_features_status = extract_image_features
        #Split status for th data with training and validation
        self.split_data_status = True
        #Randomize status to retrieve data from captin dataset
        self.randomize = True
        self.SENTENCE_BEGIN = '<S>'
        self.SENTENCE_END = '<E>'
        self.captions = None
        self.images = None
        self.image_features = None
        self.word_to_index = None
        self.index_to_word = None
        self.dataset_word_frequency = None
        self.image_feature_files = None
        self.IMAGE_FEATURES = 4096
        self.CNN_Name = cnn_name
        
        
    # Process all data related activities
    def process_data(self):
        
        self.load_captions(self.caption_file)
        self.remove_long_captions_from_dataset()
        self.get_word_frequency()
        self.remove_infrequent_words()
        self.create_dictionaries()
        if self.extract_image_features_status == True:
            self.extract_image_features(image_directory=self.image_dataset)
            self.write_images_features()
        self.write_data()
        self.save_dictionaries()
        self.write_parameters()
        if self.split_data_status == True:
            self.split_data()
            
        
    # Load all captions from the caption dataset    
    def load_captions(self, caption_file):
        
        print('Loading Captions....')
        loaded_captions = pd.read_table(caption_file, sep=self.separator)
        loaded_captions = np.asarray(loaded_captions)
        #If randomize status is true, then load the captions by shuffling
        if self.randomize == True:
            np.random.shuffle(loaded_captions)
        self.images = loaded_captions[:, 0]
        self.captions = loaded_captions[:, 1]
        print(len(self.images), 'Captions has been loaded.')
        
    # Reconstruct loaded captions by removing unwanted characters   
    def reconstruct_loaded_captions(self, captions):
        
        unwanted_characters = "?¿'/*><:{}[\]|+;.," + digits
        unwanted_character_converter = str.maketrans('', '', unwanted_characters)
        quotes_converter = str.maketrans('', '', '"')
        reconstructed_captions = captions.strip().lower()
        reconstructed_captions = reconstructed_captions.translate(unwanted_character_converter)
        reconstructed_captions = reconstructed_captions.translate(quotes_converter)
        reconstructed_captions = reconstructed_captions.split(' ')
        
        return reconstructed_captions
    
    # Get each word's frequecy in the caption dataset
    def get_word_frequency(self):
        
        self.dataset_word_frequency = Counter(chain(*self.captions)).most_common()
    
    # Remove longer captions than specified limit from the loaded captions
    def remove_long_captions_from_dataset(self):
        
        print('Removing captions longer than', self.maximum_caption_length, '....')
        temp_captions = []
        temp_images = []
        for current_image, current_caption in enumerate(self.captions):
            reconstructed_caption = self.reconstruct_loaded_captions(current_caption)
            if (len(reconstructed_caption) <= self.maximum_caption_length):
                temp_images.append(self.images[current_image])
                temp_captions.append(reconstructed_caption)
        
        self.images = temp_images
        self.captions = temp_captions
        
    def remove_infrequent_words(self):
        #TODO Add option to remove captions that have a words not in vocabulary
        print('Removing words with a frequency less than',
                        self.allowed_word_frequency,'...')
        for frequency_counter, frequency_data in enumerate(self.dataset_word_frequency):
            frequency = frequency_data[1]
            if frequency <= self.allowed_word_frequency:
                allowed_word_frequency_arg = frequency_counter
                break

        if self.allowed_word_frequency != 0:
            self.dataset_word_frequency = np.asarray(
                        self.dataset_word_frequency[0:allowed_word_frequency_arg])
        else:
            self.dataset_word_frequency = np.asarray(self.dataset_word_frequency)

        current_vocabulary_size = self.dataset_word_frequency.shape[0]
        print('Vocabulary Size =',current_vocabulary_size)
    
    # Create dictionaries for easier training
    def create_dictionaries(self):
        
        words = self.dataset_word_frequency[:, 0]
        self.word_to_index = {self.SENTENCE_BEGIN:0, self.SENTENCE_END:1}
        self.word_to_index.update({word:word_id for word_id, word in enumerate(words, 2)})
        self.index_to_word = {word_id:word for word, word_id in self.word_to_index.items()}
        
        return self.word_to_index, self.index_to_word
    
    # Extract image features from image dataset using CNN
    def extract_image_features(self, image_directory):
        
        from keras.preprocessing import image
        from keras.models import Model
        from CNNModel import CNNModel
        from keras.applications.vgg19 import preprocess_input
        
        if self.CNN_Name == 'basuru': 

            self.IMAGE_FEATURES = 512
            cnn_model = CNNModel.load_cnn_model()
            model = Model(input=cnn_model.input, output=cnn_model.get_layer('Diyath').output)
            self.image_features = []
            self.image_feature_files = list(set(self.images))
            image_count = len(self.image_feature_files)
            for image_arg,image_file in enumerate(self.image_feature_files):

                image_path = image_directory + image_file
                if image_arg%100 == 0:

                    print('%.2f %% completed' % round(100*image_arg/image_count,2))
                img = image.load_img(image_path, target_size=(64, 64))
                img = image.img_to_array(img)
                img = np.expand_dims(img, axis=0)
                img = preprocess_input(img)
                CNN_features = model.predict(img)
                self.image_features.append(np.squeeze(CNN_features))
            self.image_features = np.asarray(self.image_features)
            
        elif self.CNN_Name == 'vgg19':

            from keras.applications import VGG19

            self.IMAGE_FEATURES = 4096
            cnn_model = VGG19(weights='imagenet')
            model =  Model(input=cnn_model.input, output=cnn_model.get_layer('fc2').output)
            self.image_features = []
            self.image_feature_files = list(set(self.images))
            image_count = len(self.image_feature_files)
            for image_arg,image_file in enumerate(self.image_feature_files):
                
                image_path = image_directory + image_file
                if image_arg%100 == 0:
                    
                    print('%.2f %% completed' % round(100*image_arg/image_count,2))
                img = image.load_img(image_path, target_size=(224, 224))
                img = image.img_to_array(img)
                img = np.expand_dims(img, axis=0)
                img = preprocess_input(img)
                CNN_features = model.predict(img)
                self.image_features.append(np.squeeze(CNN_features))
            self.image_features = np.asarray(self.image_features)

            
      
    # Preprocess image features to provide into CNN
    def preprocess_image(self, folder_name, file_name):
        
        from keras.preprocessing import image as processImage
        
        image = processImage.load_img(folder_name + '/' + file_name, target_size=(64, 64))
        test_image = processImage.img_to_array(image)
        test_image = np.expand_dims(test_image, axis=0)

        return test_image
    
    # Write extracted image features into a file
    def write_images_features(self):
        
        print('Writing extracted image features into a file')
        
        extracted_feature_file = h5py.File(self.root_path + 'Captions/' + self.CNN_Name + '_extracted_features.h5')
        number_of_features = len(self.image_feature_files)
        for image_counter, image_file in enumerate(self.image_feature_files):
            print(image_counter, image_file)
            file_id = extracted_feature_file.create_group(image_file)
            image_data = file_id.create_dataset('image_features',
                                        (self.IMAGE_FEATURES,), dtype='float32')
            image_data[:] = self.image_features[image_counter,:]

            if image_counter%100 == 0:
                print('Number of image processed:', image_counter)
                print('Number of image remaining:', number_of_features-image_counter)
                
        extracted_feature_file.close()
        
    # Write reconstrcuted caption data into a file    
    def write_data(self):
        
        edited_caption_file = open(self.root_path + 'Captions/Reconstructed_caption_data.txt','w',encoding="utf-8")
        edited_caption_file.write('image_names*caption\n')
        for image_counter, image_name in enumerate(self.images):
            #print(image_counter, image_name)
            caption = ' '.join(self.captions[image_counter])
            edited_caption_file.write('%s*%s\n' %(image_name, caption))
        edited_caption_file.close()
        
        
    def split_data(self, train_percentage=.80):

        reconstructed_caption_data = pd.read_table(self.root_path + 'Captions/Reconstructed_caption_data.txt',sep=self.separator)
        data_size = reconstructed_caption_data.shape[0]
        training_size = int(data_size*train_percentage)
        complete_training_data = reconstructed_caption_data[0:training_size]
        test_data = reconstructed_caption_data[training_size:]
        test_data.to_csv(self.root_path + 'Captions/test_data.txt',sep=self.separator,index=False)
        # splitting between validation and training 
        training_size = int(training_size*train_percentage)
        validation_data = complete_training_data[training_size:]
        training_data = complete_training_data[0:training_size]
        validation_data.to_csv(self.root_path + 'Captions/validation_data.txt',sep=self.separator,index=False)
        training_data.to_csv(self.root_path + 'Captions/training_data.txt',sep=self.separator,index=False)
        
    def write_parameters(self):
        
        parameter_file = open(self.root_path + 'saved_parameters.txt','w')
        parameter_file.write('SENTENCE_BEGIN: %s \n' % self.SENTENCE_BEGIN)
        parameter_file.write('SENTENCE_END: %s \n' % self.SENTENCE_END)
        parameter_file.write('image_features: %s \n' % self.IMAGE_FEATURES)
        parameter_file.write('maximum_caption_length: %s \n'
                        %self.maximum_caption_length)
        parameter_file.close()
        
    def save_dictionaries(self):
        pickle.dump(self.word_to_index, open(self.root_path + 'Captions/word_to_index.p', 'wb'))
        pickle.dump(self.index_to_word, open(self.root_path + 'Captions/index_to_word.p', 'wb'))
        
        
        
    
        
    