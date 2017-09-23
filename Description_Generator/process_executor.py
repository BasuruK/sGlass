from keras.models import load_model
from descriptionGenerator import Generator

root_path = 'Data/'
model_filename = root_path + 'PreTrainedModels/vgg19/descGenrator.30-2.19.hdf5'
model = load_model(model_filename)
evaluator = Generator(model, root_path, cnn_name='vgg19')

evaluator.show_caption(image_file='my.jpg')