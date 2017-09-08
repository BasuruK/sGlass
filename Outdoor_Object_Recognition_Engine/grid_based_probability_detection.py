"""
Author : Balasuriya B.K | IT14020254

Grid Based Probability Detection is a method proposed to address identifying multiple objects in a scene.
The algorithm applies a sliding window mechanism with the help of a powerful image classifier to filter
object locations.
The accuracy of the detection algorithm depends on the accuracy of the classification model used.
"""
class GBPD:
    IMPORT_MANAGER = None
    new_image = None
    classifier = None
    output_bounding_boxes = None
    window_size = None

    def __init__(self, imports, classifier, window_size=(256, 256)):
        self.IMPORT_MANAGER = imports
        self.classifier = classifier
        self.window_size = window_size
        print("Imports Loaded Successfully")

    def __del__(self):
        del self.IMPORT_MANAGER
        del self.new_image
        del self.classifier
        del self.output_bounding_boxes
        del self.window_size

    # Execute Main Functionality
    def main(self, image_stream):
        # save the image coordinates in a dictionary
        image_cordinates_grid = {}

        self.new_image = image_stream

        # if N x M window were to slide thought the image, identify the parameters for the loop condition
        slide_window_height_and_width = self.window_size

        # Convert the image in to an numpy array
        image_array = self.IMPORT_MANAGER.np.array(self.new_image)

        row_count_after_substraction = self.IMPORT_MANAGER.math.ceil(
            image_array.shape[0] / slide_window_height_and_width[0])
        col_count_after_substraction = self.IMPORT_MANAGER.math.ceil(
            image_array.shape[1] / slide_window_height_and_width[1])

        # Initialize x, y, w, h values
        x, y, w, h = 0, 0, slide_window_height_and_width[1], slide_window_height_and_width[0]
        count = 0
        print("Image will be transformed in to a", row_count_after_substraction, "x", col_count_after_substraction,
              "grid")
        for row in range(0, row_count_after_substraction):
            for col in range(0, col_count_after_substraction):
                # Change the X and Y locations respectively to add the sliding window effect.
                cropped_image = image_array[y: y + h, x: x + w]
                image_cordinates_grid[str(count) + '_rect'] = (x, y, w, h)
                cropped_image = self.prepossess_image(cropped_image)
                image_cordinates_grid[str(count) + '_prediction'] = self.predict_for_single_image(cropped_image)
                x += w
                count += 1
            x = 0
            y += h

        # Extract the relevant image cell to its corresponding prediction
        iteration_size = self.IMPORT_MANAGER.math.ceil(len(image_cordinates_grid) / 2)
        combined_grid = {}

        # Compare adjacent cells for the same prediction and combine them
        for i in range(0, iteration_size - 1):
            if image_cordinates_grid[str(i) + '_prediction'] == image_cordinates_grid[str(i + 1) + '_prediction']:

                # Join the regions
                cordinate_1 = image_cordinates_grid[str(i) + '_rect']
                cordinate_2 = image_cordinates_grid[str(i + 1) + '_rect']

                # Join in X axis
                if cordinate_1[1] == cordinate_2[1]:
                    combined_grid[str(i) + '_rect'] = (
                        cordinate_1[0], cordinate_1[1], cordinate_1[2] + cordinate_2[2], cordinate_1[3])

                # Join in Y axis
                if cordinate_1[0] == cordinate_2[0]:
                    print("Extreme Case of Y Join Found")
                    combined_grid[str(i) + '_rect'] = (
                        cordinate_1[0], cordinate_1[1], cordinate_1[2], cordinate_1[3] + cordinate_2[3])

        # Run Final Prediction for the Bounding Boxes found from GBPD algorithm
        self.output_bounding_boxes = [len(combined_grid)]
        for key, value in combined_grid.items():
            x, y, w, h = value
            prediction = self.prepossess_image(image_array[y: y + h, x: x + w])
            prediction = self.predict_for_single_image(prediction)
            self.output_bounding_boxes.append((prediction, self.IMPORT_MANAGER.np.array([x, y, w, h])))

        return self.output_bounding_boxes

    # Returns the labels for the classes according to the folder structure of classes
    @staticmethod
    def get_labels_for_classes():
        # return ['car', 'cat', 'dog', 'shoe']
        return ['car', 'cat', 'dog']

    # Run prediction for a single image
    def predict_for_single_image(self, image):
        # label the images according the folder structure
        labels = self.get_labels_for_classes()
        out = self.classifier.predict(image, verbose=0)[0]
        index = self.classifier.predict_classes(image, verbose=0)
        out = [(format(x * 100, '.2f') + "%") for x in out]
        out = ' '.join(list(zip(labels, out))[index[0]])
        return out

    # Pre-process image
    def prepossess_image(self, cropped_image_array):
        cropped_image = self.IMPORT_MANAGER.Image.fromarray(cropped_image_array)
        cropped_image = cropped_image.resize((64, 64))
        cropped_image = self.IMPORT_MANAGER.keras_preprocess.img_to_array(cropped_image)
        cropped_image = self.IMPORT_MANAGER.np.expand_dims(cropped_image, axis=0)
        return cropped_image
