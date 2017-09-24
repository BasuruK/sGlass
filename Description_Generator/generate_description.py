"""
This class is used to generate the description once an image
is given.
"""

from Description_Generator.descriptionGenerator import Generator

class DescriptionGenerator:

    importManager = None
    desc_generator = None

    def __init__(self, imports):

        self.importManager = imports

    def show_description(self, image_path):

        self.desc_generator = Generator(self.importManager.description_generator_model, 'Description_Generator/Data/', cnn_name='vgg19')

        generated_description = self.desc_generator.show_caption(image_file=image_path)

        return generated_description
