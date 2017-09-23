from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense, Dropout
from keras.callbacks import History, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.preprocessing import image as img
import datetime
import os.path
import matplotlib.pyplot as plt
import numpy as np

# Initialize the CNN

classifier = Sequential()

# Step 1 - Convolution

classifier.add(Conv2D(32, (5, 5), input_shape = (64, 64, 3), activation = 'relu', padding='same'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 2 - Add more Convolution Layers making it Deep followed by a Pooling Layer

classifier.add(Conv2D(32, (5, 5), activation = 'relu', padding='same'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Dropout(0.25))

classifier.add(Conv2D(64, (5, 5), activation = 'relu', padding='same'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Conv2D(128, (5, 5), activation = 'relu', padding='same'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Dropout(0.25))

# Step 3 - Flattening

classifier.add(Flatten())

# Step 4 - Fully Connected Neural Network

# Hidden Layer 1 - Activation Function RELU
classifier.add(Dense(units = 512, activation = 'relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(units = 3, activation = 'softmax'))

# Compile the CNN
# Categorical Crossentropy - to classify between multiple classes of images
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Image Augmentation and Training Section

# Image Augmentation to prevent Overfitting (Applying random transformation on images to train set.ie.
# scaling, rotating and stretching)

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=0.3,
        width_shift_range=0.3,
        height_shift_range=0.3
        )

test_datagen = ImageDataGenerator(rescale=1./255)

# Load the training dataset folder
training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=105,
        class_mode='categorical')

# Load the test data set folder
test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=105,
        class_mode='categorical')

# Get the accuracy and loss data to plot the graph

history = History()
# checkpoint = ModelCheckpoint(filepath='models_backups/' + str(str(datetime.datetime.now().hour)), monitor='val_loss',
#                             verbose=0, mode='auto', period=1)

print(classifier.summary())

# Fit the clasifier on the CNN data
if os.path.isfile('my_model.h5') == False:
    classifier.fit_generator(
            training_set,
            steps_per_epoch=3000,
            epochs=5,
            validation_data=test_set,
            validation_steps=3000,
            callbacks=[history]
    )
    # Save the generated model to my_model.h5
    classifier.save('my_model.h5')
else:
    classifier = load_model('my_model.h5')


# Returns the labels for the classes according to the folder structre of clases
def get_labels_for_clases():
    # return ['car', 'cat', 'dog', 'shoe']
    return ['car', 'cat', 'dog']


# Run prediction for a single image
def predict_for_single_image(image):
    # label the images according the folder structure

    lables = get_labels_for_clases()
    out = classifier.predict_classes(image, verbose=0)
    return lables[out[0]]


# Run Prediction for image and give the output as percentages for each class similarities
def predict_probabilities_for_classes(classifier, image):
    labels = get_labels_for_clases()
    probabilities = classifier.predict(image)
    print(probabilities)
    # Expand two arrays to relevant class structure
    probabilities = [(format(x * 100, '.2f') + "%") for x in probabilities[0]]

    print(list(zip(labels, probabilities)))


# Plot the graphs
def plot_graphs_on_data(history):
    # Plot Accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epocs')
    plt.legend(['Train Data', 'Test Data'], loc='upper left')
    plt.show()

    # Plot Loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epocs')
    plt.legend(['Train Data', 'Test Data'], loc='upper left')
    plt.show()


# Preparedness image
def preprocess_image(folder_name, file_name):
    image = img.load_img(folder_name + '/' + file_name, target_size=(64, 64))
    test_image = img.img_to_array(image)
    test_image = np.expand_dims(test_image, axis=0)

    return test_image


# Run custom set testing
def custom_set_accuracy_test(input_folder_name):
    files_in_directory = os.listdir(input_folder_name + '/')
    match_count = 0
    fail_count = 0
    match_and_fail_count_list = []
    columns = 6
    i = 0
    plt.figure(figsize=(15, 15))

    # for each image in the directory run prediction and display that with the image
    for file_name in files_in_directory:

        test_image = preprocess_image(input_folder_name, file_name)

        prediction_for_image = predict_for_single_image(test_image)

        # Plot the images on a graph
        plt.subplot(len(files_in_directory) / columns + 1, columns, i + 1)

        if file_name.split(".")[0] == prediction_for_image:

            match_and_fail_count_list.append(file_name + " =======>" + " Match")
            match_count += 1

            # Plot Positive Images on the graph
            plt.title(file_name)
            plt.xlabel(prediction_for_image)
            plt.imshow(plt.imread(input_folder_name + '/' + file_name))
        else:

            match_and_fail_count_list.append(
                file_name + " =======>" + " Fail. " + "Predicted => " + prediction_for_image)
            fail_count += 1

            # Plot Positive Images on the graph
            plt.title(file_name)
            plt.xlabel(prediction_for_image)
            plt.imshow(plt.imread(input_folder_name + '/' + file_name))

        i += 1

    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.show()

    [print(x) for x in match_and_fail_count_list]  # Print each item in list

    custom_set_accuracy = (match_count / len(files_in_directory)) * 100
    print('Total Images : ', len(files_in_directory))
    print('Successes : ', match_count)
    print('Failures : ', fail_count)
    print("Custom Set accuracy = ", custom_set_accuracy)

# Draw the Graph for the predicted Results
# use this only after training.
# plot_graphs_on_data(history)

# image = img.load_img('custom_test/dog.1.jpg', target_size=(64, 64))
# test_image = img.img_to_array(image)
# test_image = np.expand_dims(test_image, axis=0)

# print(training_set.class_indices)
# predict_probabilities_for_classes(classifier, test_image)
custom_set_accuracy_test('custom_test')
