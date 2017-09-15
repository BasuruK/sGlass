"""
Author : Balasuriya B.K | IT14020254

This file contains the Hand Movement Tracking Engine, the engine works by taking a HSV map of the hand and then
applying color detection + skin detection algorithms to extract out the hand.

@DRAWBACKS :- The detection algorithm changes its values based on the skin color, only one type of skin color can
operate the system at a particular time, and only able to detect single hand.

TODO: OPTIMIZE TO IDENTIFY THE OPPOSITE SIDE OF THE HAND
"""

import cv2
import numpy as np


class TrackHand:

    frame = None
    blurValue = None
    threshold = None
    objectColor = None
    camera = None
    cameraController = None
    furthestPoint = None

    # Temporary Property
    objectHistogram = None

    def __init__(self, threshold=70, camera=0, blur_value=21):
        self.blurValue = blur_value
        self.threshold = threshold
        self.camera = camera
        self.blurValue = blur_value
        self.cameraController = cv2.VideoCapture(self.camera)

    def __del__(self):
        del self.frame
        del self.blurValue
        del self.threshold
        del self.objectColor
        del self.camera
        del self.cameraController
        del self.objectHistogram
        del self.furthestPoint

    def track_hand(self):

        while True:

            if self.objectColor is None:
                # Checking if the color profile is already saved
                self.objectColor, self.objectHistogram = self.get_hsv_of_hand()

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
                            cv2.circle(self.frame, self.furthestPoint, 5, [0, 255, 0], -1)
                            # cv2.arrowedLine(self.frame, centroid, self.furthestPoint, [128, 255, 120], 2)
                            # cv2.putText(self.frame, "Furthest Point", self.furthestPoint, cv2.FONT_HERSHEY_SIMPLEX, 0.50, (51, 153, 255), 1, cv2.LINE_AA)
                            # print("Furthest Point", self.furthestPoint)

                cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('Frame', 1024, 768)
                cv2.imshow("Frame", self.frame)
                cv2.imshow("Thresh", located_object)
                if cv2.waitKey(1) == 13:
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
            cv2.rectangle(frame, (200, 200), (300, 300), (255, 0, 255), 2)
            cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('frame', 1024, 768)
            cv2.imshow("frame", frame)

            if cv2.waitKey(1) == 13:
                object_color = frame[200:300, 200:300]
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
        frame_returned = cv2.cvtColor(frame_returned, cv2.COLOR_BGR2RGB)
        frame_returned = cv2.resize(frame_returned, (1024, 768), interpolation=cv2.INTER_AREA)  # Resize the Image
        return frame_returned, finger_pointed
