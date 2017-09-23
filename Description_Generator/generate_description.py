"""
This class is used to generate the description once an image
is given.
"""


class DescriptionGenerator:
    importManager = None
    model = None
    generator = None

    def __init__(self, import_manager):
        self.importManager = import_manager

        self.generator = self.importManager.Generator

    def main(self, image_path):
        self.generator.show_caption(image_file=image_path)
