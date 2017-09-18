"""
Author : Balasuriya B.K | IT14020254

This Class is responsible for identifying which area the user is pointing.
The Input for this glass is taken from the GBPD class, as the prediction and bounding boxes.
Returns the prediction and the bounding boxes which user is selecting as a 2D Array.
"""


class PointToFingerMapper:

    BoundingBoxes = None
    Finger_l = None

    # Return Value
    PredictionAndBoundingBox = None

    def __init__(self, bounding_box, finger_location):
        self.BoundingBoxes = bounding_box
        self.Finger_l = finger_location
        self.PredictionAndBoundingBox = [[]]
        print(self.BoundingBoxes)

    def __del__(self):
        del self.BoundingBoxes
        del self.Finger_l
        del self.PredictionAndBoundingBox

    def main(self):
        for predict, (x, y, w, h) in self.BoundingBoxes:
            """
            Equation:
                if a point P resides inside an rectangle then,
                    0 <= (P - P1) . (P2 - P1) <= |P2 - P1| ^ 2
                    and
                    0 <= (P - P1) . (P4 - P1) <= |P4 - P1| ^  2
            """

            # Calculate P1, P2, P3, P4 Rectangle corner points
            x1, y1 = x, y  # P1
            x2, y2 = x + w, y  # P2
            x3, y3 = x + w, y + h  # P3
            x4, y4 = x, y + h  # P4

            # Make P
            p = (self.Finger_l[0], self.Finger_l[1])

            # Calculate (P2 - P1)
            p2_subtract_p1 = (x2 - x1, y2 - y1)
            # Calculate (P4 - P1)
            p4_subtract_p1 = (x4 - x1, y4 - y1)

            # Calculate |P2 - P1| ^ 2
            p2_subtract_p1_magnitude_squared = p2_subtract_p1[0] ** 2 + p2_subtract_p1[1] ** 2
            # Calculate |P4 - P1| ^ 2
            p4_subtract_p1_magnitude_squared = p4_subtract_p1[0] ** 2 + p4_subtract_p1[1] ** 2

            # # Calculate (P - P1)
            point = (p[0] - x1, p[1] - y1)

            # Apply to the Equation
            if 0 <= point[0] * p2_subtract_p1[0] + point[1] * p2_subtract_p1[1] <= p2_subtract_p1_magnitude_squared:
                if 0 <= point[0] * p4_subtract_p1[0] + point[1] * p4_subtract_p1[1] <= p4_subtract_p1_magnitude_squared:
                    print("Finger Location : {}, is inside of: {}, pointing at a {}".format(p, (x, y, w, h), predict))
                    self.PredictionAndBoundingBox.append([predict, (x, y, w, h)])
                else:
                    print("Finger is not inside of {}".format((x, y, w, h)))
            else:
                print("Finger is not inside any of the cells")

        return self.PredictionAndBoundingBox
