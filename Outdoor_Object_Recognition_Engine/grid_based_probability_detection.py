class GBPD:

    IMPORT_MANAGER = None
    cat_image = None
    classifier = None

    def __init__(self, imports):
        self.IMPORT_MANAGER = imports
        print("Imports Loaded Successfully")

    # Execute Main Functionality
    def main(self):
        # save the image cordinates in a dictionary
        image_cordinates_grid = {}

        self.cat_image, self.classifier = self.load_image_and_classifier()

        image_dimention_width = self.cat_image.size[0]
        image_dimention_height = self.cat_image.size[1]

        # if 100 x 100 window were to slide thought the image, identify the parameters for the loop condition
        slide_window_height_and_width = (256, 256)

        # Convert the image in to an numpy array
        image_array = self.IMPORT_MANAGER.np.array(self.cat_image)

        row_count_after_substraction = self.IMPORT_MANAGER.math.ceil(image_array.shape[0] / slide_window_height_and_width[0])
        col_count_after_substraction = self.IMPORT_MANAGER.math.ceil(image_array.shape[1] / slide_window_height_and_width[1])

        # Initialize x, y, w, h values
        x, y, w, h = 0, 0, slide_window_height_and_width[1], slide_window_height_and_width[0]
        count = 0
        print("Image will be transformed in to a", row_count_after_substraction, "x", col_count_after_substraction,
              "grid")
        for row in range(0, row_count_after_substraction):
            for col in range(0, col_count_after_substraction):
                # Change the X and Y locations respectivly to add the sliding window effect.
                cropped_image = image_array[y: y + h, x: x + w]
                saving_image = cropped_image
                # Save the image
                self.IMPORT_MANAGER.imutils.imsave("Outdoor_Object_Recognition_Engine/custom_test/cutted/" + str(count) + ".jpg", cropped_image)
                image_cordinates_grid[str(count) + '_rect'] = (x, y, w, h)

                cropped_image = self.preprocess_image('Outdoor_Object_Recognition_Engine/custom_test/cutted/' + str(count) + ".jpg")
                image_cordinates_grid[str(count) + '_prediction'] = self.predict_for_single_image(cropped_image)

                # Delete existing image
                self.IMPORT_MANAGER.os.remove("Outdoor_Object_Recognition_Engine/custom_test/cutted/" + str(count) + ".jpg")
                # Lable image
                self.IMPORT_MANAGER.imutils.imsave(
                    "Outdoor_Object_Recognition_Engine/custom_test/cutted/" + str(count) + "_" + self.predict_for_single_image(cropped_image) + ".jpg",
                    saving_image)
                x += w
                count += 1
            x = 0
            y += h

        # Extract the relevent image cell to its corresponding prediction
        iteration_size = self.IMPORT_MANAGER.math.ceil(len(image_cordinates_grid) / 2)
        combined_grid = {}
        # Compare adjacent cells for the same prediction and combine them
        for i in range(0, iteration_size - 1):
            if image_cordinates_grid[str(i) + '_prediction'] == image_cordinates_grid[str(i + 1) + '_prediction']:
                # Join the regions
                cordinate_1 = image_cordinates_grid[str(i) + '_rect']
                cordinate_2 = image_cordinates_grid[str(i + 1) + '_rect']
                print("cordinate_1 : ", cordinate_1)
                print("cordinate_2 : ", cordinate_2)
                # Join in X axis

                if cordinate_1[1] == cordinate_2[1]:
                    combined_grid[str(i) + '_rect'] = (
                    cordinate_1[0], cordinate_1[1], cordinate_1[2] + cordinate_2[2], cordinate_1[3])
                # Join in Y axix

                if cordinate_1[0] == cordinate_2[0]:
                    print("came to else if")
                    combined_grid[str(i) + '_rect'] = (
                    cordinate_1[0], cordinate_1[1], cordinate_1[2], cordinate_1[3] + cordinate_2[3])

        # Clean Directory
        [self.IMPORT_MANAGER.os.remove("custom_test/cutted/" + x) for x in self.IMPORT_MANAGER.os.listdir("custom_test/cutted/")]

        # Run the 2nd wave predictions for the combined images
        for i, (key, value) in enumerate(combined_grid.items()):
            x, y, w, h = value
            image = image_array[y: y + h, x: x + w]
            path = "custom_test/cutted/" + str(i) + ".jpg"
            self.IMPORT_MANAGER.imutils.imsave(path, image)

            pred = self.predict_for_single_image(self.preprocess_image(path))
            self.IMPORT_MANAGER.os.remove(path)
            self.IMPORT_MANAGER.imutils.imsave("custom_test/cutted/" + str(i) + "_" + pred + ".jpg", image)

        fig, ax = self.IMPORT_MANAGER.plt.subplots(1)
        ax.imshow(image_array)

        for key, value in combined_grid.items():
            x, y, w, h = value
            rect = self.IMPORT_MANAGER.patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='g', facecolor='none')
            ax.add_patch(rect)

        self.IMPORT_MANAGER.plt.show()

    # Loads the image and the classifier
    def load_image_and_classifier(self):
        # Image structure = > (Height, Width, Color_Channel)
        cat_image = self.IMPORT_MANAGER.Image.open('Outdoor_Object_Recognition_Engine/custom_test/dog.6.jpg')
        # Load the Classifier
        classifier = self.IMPORT_MANAGER.load_model('Outdoor_Object_Recognition_Engine/my_model.h5')

        return cat_image, classifier

    # Returns the labels for the classes according to the folder structre of clases
    @staticmethod
    def get_labels_for_clases():
        # return ['car', 'cat', 'dog', 'shoe']
        return ['car', 'cat', 'dog']

    # Run prediction for a single image
    def predict_for_single_image(self, image):
        # label the images according the folder structure

        lables = self.get_labels_for_clases()
        out = self.classifier.predict_classes(image, verbose=0)
        return lables[out[0]]

    # Pre-process image
    def preprocess_image(self, file_path):
        cropped_image = self.IMPORT_MANAGER.keras_preprocess.load_img(file_path, target_size=(64, 64))
        cropped_image = self.IMPORT_MANAGER.keras_preprocess.img_to_array(cropped_image)
        cropped_image = self.IMPORT_MANAGER.np.expand_dims(cropped_image, axis=0)
        return cropped_image