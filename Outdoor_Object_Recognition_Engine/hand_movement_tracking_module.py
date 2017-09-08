"""
Author : Balasuriya B.K | IT14020254

This file contains the Hand Movement Tracking Engine, the engine works by taking a HSV map of the hand and then
applying color detection + skin detection algorithms to extract out the hand.

@DRAWBACKS :- The detection algorithm changes its values based on the skin color, only one type of skin color can
operate the system at a particular time, and only able to detect single hand.

TODO: OPTIMIZE THE DETECTION ALGORITHMS, GET A CLEANED OUTPUT
"""

import cv2
import numpy as np


class TrackHand:

    frame = None
    blurValue = None
    threshold = None
    objectColor = None

    def __init__(self, threshold):
        self.blurValue = 5
        self.threshold = threshold

    def main(self):

        cap = cv2.VideoCapture(0)

        while True:
            _, self.frame = cap.read()

            frame = cv2.flip(self.frame, 1)

            self.objectColor = self.get_hsv_of_hand()
            # TODO wait till the upper method executes and try to put complex functions in to methods
            hsv_color = cv2.cvtColor(self.object_color, cv2.COLOR_BGR2HSV)

            object_histogram = cv2.calcHist([hsv_color], [0, 1], None, [12, 15], [0, 180, 0, 256])

            cv2.normalize(object_histogram, object_histogram, 0, 255, cv2.NORM_MINMAX)

            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            object_segment = cv2.calcBackProject([hsv_frame], [0, 1], object_histogram, [0, 180, 0, 256], 1)
            disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            cv2.filter2D(object_segment, -1, disc, object_segment)

            _, threshold_object_segment = cv2.threshold(object_segment, 70, 255, cv2.THRESH_BINARY)

            threshold_object_segment = cv2.merge(
                (threshold_object_segment, threshold_object_segment, threshold_object_segment))
            located_object = cv2.bitwise_and(frame, threshold_object_segment)

            located_object_gray = cv2.cvtColor(located_object, cv2.COLOR_BGR2GRAY)

            _, located_object_thresh = cv2.threshold(located_object_gray, 70, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            located_object = cv2.GaussianBlur(located_object_thresh, (5, 5), 0, located_object_thresh)
            # located_object = cv2.medianBlur(located_object_thresh, 5)

            res = cv2.bitwise_and(frame, frame, mask=located_object)

            im2, contours, hierarchy = cv2.findContours(located_object, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            if self.get_max_contour(contours) is not None:

                max_contours = contours[self.get_max_contour(contours)]
                hull = cv2.convexHull(max_contours)
                defects = self.get_defects(max_contours)
                centroid = self.get_centroid(max_contours)
                # cv2.drawContours(frame, [max_contours], 0, (255,150,0) ,3)
                cv2.drawContours(frame, [hull], 0, (255, 150, 0), 3)

                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    start = tuple(max_contours[s][0])
                    end = tuple(max_contours[e][0])
                    far = tuple(max_contours[f][0])
                    # cv2.line(frame,start,end,[0,255,0],2)
                    # cv2.circle(frame,far,5,[0,0,255],-1)

                if centroid is not None and defects is not None and len(defects) > 0:
                    furthest_point = self.get_furthest_point(defects, max_contours, centroid)

                    if furthest_point is not None:
                        cv2.circle(frame, furthest_point, 5, [0, 255, 0], -1)

            cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)

            cv2.resizeWindow('Frame', 1024, 768)

            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) == 13:
                break

        cv2.destroyAllWindows()
        cap.release()

    @staticmethod
    def get_hsv_of_hand(self):

        cap = cv2.VideoCapture(0)

        while True:
            _, frame = cap.read()

            frame = cv2.flip(frame, 1)

            cv2.putText(frame, "Please put your hand in the box", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 0, 0),
                        1, cv2.LINE_AA)
            cv2.rectangle(frame, (100, 100), (150, 150), (255, 0, 255), 2)
            cv2.imshow("frame", frame)

            if cv2.waitKey(1) == 13:
                object_color = frame[100:150, 100:150]
                cv2.destroyAllWindows()
                cap.release()
                return object_color

    @staticmethod
    def get_max_contour(self, contours):
        # Returns the contour with maximum area
        length = len(contours)
        maxArea = -1
        ci= - 1
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

    @staticmethod
    def get_defects(self, contour):
        hull = cv2.convexHull(contour, returnPoints=False)
        if hull is not None and len(hull) > 3 and len(contour) > 3:
            defects = cv2.convexityDefects(contour, hull)
            return defects
        else:
            return None

    @staticmethod
    def get_furthest_point(self, defects, contour, centroid):
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

    @staticmethod
    def get_centroid(self, contour):
        moments = cv2.moments(contour)
        if moments['m00'] != 0:
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])
            return cx, cy
        else:
            return None