"""
Author : Balasuriya B.K | IT14020254

This file contains the Hand Movement Tracking Engine, the engine works by taking a HSV map of the hand and then
applying color detection + skin detection algorithms to extract out the hand.

@DRAWBACKS :- The detection algorithm changes its values based on the skin color, only one type of skin color can
operate the system at a particular time, and only able to detect single hand.
"""
import cv2
import numpy as np
import os
from Dialogue_Manager.settings_manager import SettingsManager
from config import Configurations
from Dialogue_Manager.text_to_speech_processesor import speak_secondary


class TrackHand:

    frame = None
    blurValue = None
    threshold = None
    objectColor = None
    camera = None
    cameraController = None
    furthestPoint = None
    workerCount = 0
    SettingsController = None
    Configurations_Controller = None

    # Temporary Property
    objectHistogram = None

    def __init__(self, threshold=70, camera=0, blur_value=21):
        self.blurValue = blur_value
        self.threshold = threshold
        self.camera = camera
        self.blurValue = blur_value
        self.cameraController = cv2.VideoCapture(self.camera)
        self.SettingsController = SettingsManager()
        self.Configurations_Controller = Configurations()

        # Set the Resolution of the Camera to 1024 x 768
        self.cameraController.set(cv2.CAP_PROP_FRAME_HEIGHT, 1400)
        self.cameraController.set(cv2.CAP_PROP_FRAME_WIDTH, 1150)

        # Clear the command queue to remove any previous commands stored
        self.clear_command_queue()

        speak_secondary("Multiple Object Detection Mode.")

    def __del__(self):
        try:
            del self.frame
            del self.blurValue
            del self.threshold
            del self.objectColor
            del self.camera
            del self.cameraController
            del self.objectHistogram
            del self.furthestPoint
            del self.SettingsController
            del self.Configurations_Controller
        except AttributeError:
            pass

    def track_hand(self):

        while True:

            if self.objectColor is None:
                # # Checking if the color profile is already saved
                # if self.is_object_color_and_hsv_files_empty():
                #     self.objectColor, self.objectHistogram = self.get_hsv_of_hand()
                #     self.save_object_color_and_hsv_frame()
                # else:
                #     self.objectColor, self.objectHistogram = self.load_object_color_and_hsv_from_file()

                self.objectColor, self.objectHistogram = self.get_hsv_of_hand()

                if self.objectColor is None and self.objectHistogram is None:
                    break  # Platform change or quit issued

            elif self.objectColor is not None:

                _, self.frame = self.cameraController.read()
                self.frame = cv2.flip(self.frame, 1)

                hsv_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)

                object_segment = cv2.calcBackProject([hsv_frame], [0, 1], self.objectHistogram, [0, 180, 0, 256], 1)
                disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                cv2.filter2D(object_segment, -1, disc, object_segment)

                _, threshold_object_segment = cv2.threshold(object_segment, self.threshold, 255, cv2.THRESH_BINARY)

                threshold_object_segment = cv2.merge(
                    (threshold_object_segment, threshold_object_segment, threshold_object_segment))

                located_object = cv2.bitwise_and(self.frame, threshold_object_segment)
                located_object_gray = cv2.cvtColor(located_object, cv2.COLOR_BGR2GRAY)
                _, located_object_thresh = cv2.threshold(located_object_gray, self.threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                # located_object = cv2.GaussianBlur(located_object_thresh, (5, 5), 0, located_object_thresh)
                located_object = cv2.medianBlur(located_object_thresh, self.blurValue)

                # res = cv2.bitwise_and(self.frame, self.frame, mask=located_object)

                im2, contours, hierarchy = cv2.findContours(located_object, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                if self.get_max_contour(contours) is not None:

                    max_contours = contours[self.get_max_contour(contours)]
                    hull = cv2.convexHull(max_contours)
                    defects = self.get_defects(max_contours)
                    centroid = self.get_centroid(max_contours)
                    # cv2.drawContours(frame, [max_contours], 0, (255,150,0) ,3)
                    # cv2.drawContours(self.frame, [hull], 0, (255, 150, 0), 3)

                    if defects is not None and defects.shape[0] is not None:  # To ensure the camera accidentally does not pick up empty object
                        for i in range(defects.shape[0]):
                            s, e, f, d = defects[i, 0]
                            start = tuple(max_contours[s][0])
                            end = tuple(max_contours[e][0])
                            far = tuple(max_contours[f][0])

                            # cv2.arrowedLine(self.frame, far, start, [128, 255, 120], 2)
                            # cv2.circle(self.frame, far, 5, [0, 0, 255], -1)
                            cv2.circle(self.frame, centroid, 5, [128, 128, 128], -1)
                            cv2.putText(self.frame, "Center", centroid, cv2.FONT_HERSHEY_SIMPLEX, 0.50, (128, 128, 128), 1, cv2.LINE_AA)  # Mark the Center of the hull

                        if centroid is not None and defects is not None and len(defects) > 0:
                            self.furthestPoint = self.get_furthest_point(defects, max_contours, centroid)  # Get the furthest point from the detects

                            if self.furthestPoint is not None:
                                cv2.circle(self.frame, self.furthestPoint, 8, [0, 255, 0], -1)
                                # cv2.arrowedLine(self.frame, centroid, self.furthestPoint, [128, 255, 120], 2)
                                # cv2.putText(self.frame, "Furthest Point", self.furthestPoint, cv2.FONT_HERSHEY_SIMPLEX, 0.50, (51, 153, 255), 1, cv2.LINE_AA)
                                # print("Furthest Point", self.furthestPoint)

                # cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
                # cv2.resizeWindow('Frame', 1024, 768)
                cv2.imshow("Frame", self.frame)
                cv2.imshow("Thresh", located_object)

                waitkey = cv2.waitKey(20) & 0xFF

                # Disable Grid Output window DEV OPT
                if waitkey == 103:
                    self.Configurations_Controller.disable_gbpd_display()
                    print("Grid Display Output disabled")
                # Disable Finger Location Output window DEV OPT
                if waitkey == 102:
                    self.Configurations_Controller.disable_pointer_loc_display()
                    print("Finger Location Output Display disabled")
                # Change Platform mode to single detection
                if waitkey == 112:
                    self.Configurations_Controller.set_platform_mode_single_detection()
                    print("Platform Changing")
                    waitkey = 10
                # Change environment to indoor
                if waitkey == 101:
                    self.Configurations_Controller.set_environment_mode_indoor()
                    print("Environment Changing")
                    waitkey = 10

                # Platform change configurations
                if (waitkey == 10 or
                        self.SettingsController.signal_recognition_engines_to_quit() or
                        self.SettingsController.signal_recognition_engines_to_quit_on_platform_change() or
                        self.SettingsController.signal_recognition_engines_to_quit_when_system_quits()):

                    cv2.destroyAllWindows()
                    self.cameraController.release()
                    print("Multiple Object Detection System Exiting")
                    break

                # key == 13 works on windows, for linux change the code to cv2.waitKey(20) & 0xFF == 10
                if waitkey == 99 or (self.check_command_queue() == self.Configurations_Controller.capture_image):
                    # Return the frame and the furthest point
                    cv2.destroyAllWindows()
                    self.cameraController.release()
                    return self.get_pointing_point()

    # Returns an HSV color information captured of the hand
    def get_hsv_of_hand(self):
        while True:
            _, frame = self.cameraController.read()
            frame = cv2.flip(frame, 1)
            cv2.putText(frame, "Please put your hand in the box", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 0, 0),
                        1, cv2.LINE_AA)
            cv2.rectangle(frame, (300, 300), (500, 500), (255, 0, 255), 2)
            # cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
            # cv2.resizeWindow('frame', 1024, 768)
            cv2.imshow("frame", frame)

            # key == 13 works on windows, for linux change the code to cv2.waitKey(20) & 0xFF == 10
            waitkey = cv2.waitKey(20) & 0xFF
            # Platform change configurations
            if (waitkey == 10 or
                    self.SettingsController.signal_recognition_engines_to_quit() or
                    self.SettingsController.signal_recognition_engines_to_quit_on_platform_change() or
                    self.SettingsController.signal_recognition_engines_to_quit_when_system_quits()):

                cv2.destroyAllWindows()
                self.cameraController.release()
                print("Multiple Object Detection System Exiting")
                return None, None

            # C Key Pressed
            if waitkey == 99:
                object_color = frame[300:500, 300:500]
                cv2.destroyAllWindows()
                # Convert object color in to HSV range
                hsv_color = cv2.cvtColor(object_color, cv2.COLOR_BGR2HSV)
                # Calculate the Histogram
                object_histogram = cv2.calcHist([hsv_color], [0, 1], None, [12, 15], [0, 180, 0, 256])
                # Normalize the Histogram
                object_histogram = cv2.normalize(object_histogram, object_histogram, 0, 255, cv2.NORM_MINMAX)
                return object_color, object_histogram

    # Returns the Contour with the maximum area
    @staticmethod
    def get_max_contour(contours):
        # Returns the contour with maximum area
        length = len(contours)
        maxArea = -1
        ci = - 1
        if length > 0:
            for i in range(length):
                temp = contours[i]
                area = cv2.contourArea(temp)
                if area > maxArea:
                    maxArea = area
                    ci = i
            return ci
        else:
            return None

    # Returns Convexity Defects in a given Contour
    @staticmethod
    def get_defects(contour):
        hull = cv2.convexHull(contour, returnPoints=False)
        if hull is not None and len(hull) > 3 and len(contour) > 3:
            defects = cv2.convexityDefects(contour, hull)
            return defects
        else:
            return None

    # Returns the furthest Point in a given Convex Hull
    @staticmethod
    def get_furthest_point(defects, contour, centroid):
        s = defects[:, 0][:, 0]
        cx, cy = centroid

        x = np.array(contour[s][:, 0][:, 0], dtype=np.float)
        y = np.array(contour[s][:, 0][:, 1], dtype=np.float)

        xp = cv2.pow(cv2.subtract(x, cx), 2)
        yp = cv2.pow(cv2.subtract(y, cy), 2)
        dist = cv2.sqrt(cv2.add(xp, yp))

        dist_max_i = np.argmax(dist)

        if dist_max_i < len(s):
            farthest_defect = s[dist_max_i]
            farthest_point = tuple(contour[farthest_defect][0])
            return farthest_point
        else:
            return None

    # Returns the middle point of the contour
    @staticmethod
    def get_centroid(contour):
        moments = cv2.moments(contour)
        if moments['m00'] != 0:
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])
            return cx, cy
        else:
            return None

    # Returns the image and the location of the point of the furthest finger
    def get_pointing_point(self):
        frame_returned = self.frame
        finger_pointed = self.furthestPoint
        cv2.imwrite("Outdoor_Object_Recognition_Engine/edited.jpg", frame_returned)
        frame_returned = cv2.cvtColor(frame_returned, cv2.COLOR_BGR2RGB)
        print(frame_returned.shape)
        if frame_returned is None and finger_pointed is None:
            return None, None
        return frame_returned, finger_pointed

    # Dialogue Manager Command Queue Methods
    @staticmethod
    def clear_command_queue():
        f = open("Dialogue_Manager/command_temp.txt", "w")
        f.write("")
        f.close()

    @staticmethod
    def check_command_queue():
        file_content = os.stat("Dialogue_Manager/command_temp.txt").st_size

        if file_content != 0:
            # Command is in the queue, read
            f = open("Dialogue_Manager/command_temp.txt", 'r')
            command = f.read()
            return command
        else:
            return False

    # Save Object color and HSV Frames to a file
    def save_object_color_and_hsv_frame(self):
        """
        Save the Object color and HSV Frame to a txt file enabling single time detection for a single profile
        """
        try:

            object_color_file = open(os.getcwd() + "/Outdoor_Object_Recognition_Engine/object_color.txt", "r+")
            object_hsv_file = open(os.getcwd() + "/Outdoor_Object_Recognition_Engine/object_histogram.txt", "r+")

            object_color_file.write(str(self.objectColor))
            object_hsv_file.write(str(self.objectHistogram))

            print(self.objectColor)
            print(self.objectHistogram)

            object_color_file.close()
            object_hsv_file.close()

            return True

        except FileNotFoundError:
            print("Object Color or Object Histogram save files not found, will be created now")
            open("Outdoor_Object_Recognition_Engine/object_color.txt", "w+").close()
            open("Outdoor_Object_Recognition_Engine/object_histogram.txt", "w+").close()

    # Check weather Object color and HSV files are empty
    @staticmethod
    def is_object_color_and_hsv_files_empty():
        object_color_file = os.stat("Outdoor_Object_Recognition_Engine/object_color.txt").st_size
        object_hsv_file = os.stat("Outdoor_Object_Recognition_Engine/object_histogram.txt").st_size
        return object_color_file == 0 and object_hsv_file == 0

    # Load Color profile and HSV profile from the test file
    @staticmethod
    def load_object_color_and_hsv_from_file():
        object_color_file = open("Outdoor_Object_Recognition_Engine/object_color.txt", "r+")
        object_hsv_file = open("Outdoor_Object_Recognition_Engine/object_histogram.txt", "r+")

        object_c, object_hsv = object_color_file.read(), object_hsv_file.read()

        print(object_c)
        print(object_hsv)

        object_color_file.close()
        object_hsv_file.close()

        return object_c, object_hsv


# import os
#
# print(os.getcwd())
#
# object_color_file = open(os.getcwd() + "/object_color.txt", "r+")
#
# #object_hsv_file = open("Outdoor_Object_Recognition_Engine/object_histogram.txt", "r+")
#
# object_color_file.write("Sad")
# object_color_file.close()
