from keras.callbacks import ModelCheckpoint, History
from keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt
from descriptionGeneratorModel import DscGenModel
from dataSetHandler import DataHandler
from preprocessedDataHandler import PreprocessDataHandler
from models import DscGenModel
from descriptionGenerator import Generator

num_epochs = 200
batch_size = 64
root_path = 'Data/'
selected_CNN = 'vgg19'
captions_Dataset = root_path + 'Captions/IAPR_2012_captions.txt'
data_handler = DataHandler(root_path=root_path, caption_file=captions_Dataset,
                            maximum_caption_length=50,
                            allowed_word_frequency=2,
                            extract_image_features=True,
                            image_dataset=root_path + 'Images/iaprtc12/',
                            cnn_name=selected_CNN)

data_handler.process_data()

preProcessDataHandler = PreprocessDataHandler(root_path=root_path, batch_size=batch_size, cnn_name=selected_CNN)

num_training_samples =  preProcessDataHandler.training_data.shape[0]
num_testing_samples = preProcessDataHandler.testing_data.shape[0]
print('Number of training samples:', num_training_samples)
print('Number of testing samples:', num_testing_samples)

model = DscGenModel(max_token_length=preProcessDataHandler.maximum_token_length,
            vocabulary_size=preProcessDataHandler.vocabulary_size,
            rnn='lstm',
            num_image_features=preProcessDataHandler.image_features,
            hidden_size=512,
            embedding_size=512)

model.compile(loss='categorical_crossentropy',
              optimizer = 'rmsprop',
              metrics=['accuracy'])

print(model.summary())

history = History()

model_names = (root_path + 'PreTrainedModels/' + selected_CNN +
               '/descGenrator.{epoch:02d}-{val_loss:.2f}.hdf5')
model_checkpoint = ModelCheckpoint(model_names,
                                   monitor='val_loss',
                                   verbose=1,
                                   save_best_only=False,
                                   save_weights_only=False)

reduce_learning_rate = ReduceLROnPlateau(monitor='val_loss', factor=0.001,
                                         patience=4, verbose=1)

callbacks = [model_checkpoint, reduce_learning_rate, history]

model.fit_generator(generator=preProcessDataHandler.flow(mode='train'),
                    steps_per_epoch=int(num_training_samples / batch_size),
                    epochs=num_epochs,
                    verbose=1,
                    callbacks=callbacks,
                    validation_data=preProcessDataHandler.flow(mode='testing'),
                    validation_steps=int(num_testing_samples / batch_size))


def plot_graphs_on_data(history):
    
    # Plot Accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epocs')
    plt.legend(['Train Data', 'Test Data'], loc = 'upper left')
    plt.show()

    #Plot Loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epocs')
    plt.legend(['Train Data', 'Test Data'], loc = 'upper left')
    plt.show()
