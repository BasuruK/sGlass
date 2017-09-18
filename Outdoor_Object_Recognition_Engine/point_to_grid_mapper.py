"""
Author : Balasuriya B.K | IT14020254

This Class is responsible for identifying which area the user is pointing.
The Input for this glass is taken from the GBPD class, as the prediction and bounding boxes.
Returns the prediction and the bounding boxes which user is selecting.
"""


class PointToFingerMapper:

    BoundingBoxes = None
    Prediction = None
    Finger_l = None

    def __init__(self, bounding_box, finger_location):
        self.BoundingBoxes = bounding_box
        self.Finger_l = finger_location

    def __del__(self):
        del self.Prediction
        del self.BoundingBoxes

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
            x1, y1 = x, y
            x2, y2 = x + w, y
            x3, y3 = x + w, y + h
            x4, y4 = x, y + h

            # Add for P, P1, P2, P3, P4
            p = (self.Finger_l[0], self.Finger_l[1])
            p1 = (x1, y1)
            p2 = (x2, y2)
            p3 = (x3, y3)
            p4 = (x4, y4)

            # Calculate (P2 - P1)
            p2_subtract_p1 = (x2 - x1, y2 - y1)
            # Calculate (P4 - P1)
            p4_subtract_p1 = (x4 - x1, y4 - y1)

            # Calculate |P2 - P1| ^ 2
            p2_subtract_p1_magnitude_squared = p2_subtract_p1[0] ** 2 + p2_subtract_p1[1] ** 2
            # Calculate |P4 - P1| ^ 2
            p4_subtract_p1_magnitude_squared = p4_subtract_p1[0] ** 2 + p4_subtract_p1[1] ** 2

            # Apply to the Equation
            if 0 <= p[0] * p2_subtract_p1[0] + p[1] * p2_subtract_p1[1] <= p2_subtract_p1_magnitude_squared:
                if 0 <= p[0] * p4_subtract_p1[0] + p[1] * p4_subtract_p1[1] <= p4_subtract_p1_magnitude_squared:
                    print("Inside")
                else:
                    print("Outside")
            else:
                print("Outside")



