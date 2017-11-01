"""
Author : Balasuriya B.K | IT14020254

Single Detection Mode class is a submodule which only uses the pre-trained CNN
to give out predictions to a single image.
"""
import cv2


class SingleDetection:

    IMPORT_MANAGER = None
    Image = None
    Prediction = None
    Frame = None
    CameraController = None
    CameraID = None
    Classifier = None

    def __init__(self, import_manager, camera_id, classifier):
        self.CameraID = camera_id
        self.IMPORT_MANAGER = import_manager
        self.Classifier = classifier
        self.CameraController = cv2.VideoCapture(self.CameraID)

        # Set the Resolution of the Camera to 1024 x 768
        self.CameraController.set(cv2.CAP_PROP_FRAME_HEIGHT, 1400)
        self.CameraController.set(cv2.CAP_PROP_FRAME_WIDTH, 1150)

    def __del__(self):
        del self.IMPORT_MANAGER
#        del self.Prediction
        del self.Frame
        del self.CameraController
        del self.CameraID

    def track_object(self):
        while True:
            _, self.Frame = self.CameraController.read()
            self.Frame = cv2.flip(self.Frame, 1)

            cv2.imshow("Frame", self.Frame)
            wait_key = cv2.waitKey(20) & 0xFF
            if wait_key == 10:
                # self.detect(self.Frame)
                cv2.destroyAllWindows()
                self.CameraController.release()
                print("System Exiting")
                quit(0)

            if wait_key == 108 or (self.check_command_queue() == 'wit_capture_image'):
                print("Capture")
                cv2.imwrite("Outdoor_Object_Recognition_Engine/edited.jpg", self.Frame)
                self.clear_command_queue()

                # print("Prediction : ", self.detect(self.Frame))
                cv2.destroyAllWindows()
                self.CameraController.release()
                return self.detect(self.Frame)

    def detect(self, unprocessed_image):
        self.Image = self.preprocessed_image(unprocessed_image)
        self.Prediction = self.predict_for_single_image(self.Image)

        return self.Prediction

    @staticmethod
    def get_labels_for_classes():
        # return ['car', 'cat', 'dog', 'shoe']
        return ['car', 'cat', 'dog']

    # Run prediction for a single image
    def predict_for_single_image(self, image):
        # label the images according the folder structure
        labels = self.get_labels_for_classes()
        out = self.Classifier.predict(image, verbose=0)[0]
        index = self.Classifier.predict_classes(image, verbose=0)
        out = [(format(x * 100, '.2f') + "%") for x in out]
        out = ' '.join(list(zip(labels, out))[index[0]])
        return out

    # Pre-process the image as for classifier requirements
    def preprocessed_image(self, image):
        processed_image = self.IMPORT_MANAGER.Image.fromarray(image)
        processed_image = processed_image.resize((64, 64))
        processed_image = self.IMPORT_MANAGER.keras_preprocess.img_to_array(processed_image)
        processed_image = self.IMPORT_MANAGER.np.expand_dims(processed_image, axis=0)

        return processed_image

    # Dialogue Manager Command Queue Methods
    @staticmethod
    def clear_command_queue():
        f = open("Dialogue_Manager/command_temp.txt", "w")
        f.write("")
        f.close()

    def check_command_queue(self):
        file_content = self.IMPORT_MANAGER.os.stat("Dialogue_Manager/command_temp.txt").st_size

        if file_content != 0:
            # Command is in the queue, read
            f = open("Dialogue_Manager/command_temp.txt", 'r')
            command = f.read()
            return command
        else:
            return False