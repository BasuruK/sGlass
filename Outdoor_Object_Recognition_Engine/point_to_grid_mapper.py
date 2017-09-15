"""
TODO:
1. Make a Range from x, y, w, h values. ✔
2. Check weather the point is in the range calculated above. ✔
3. Change the function to get both Prediction string and the bounding box and return the prediction string
with the bounding box FIX THE RETURN VALUE
"""


class PointToFingerMapper:

    boundingBoxes = None
    Finger_l = None  # Finger Pointing Location
    prediction = None

    def __init__(self, bounding_box, pointer_location):
        self.boundingBoxes = bounding_box
        self.Finger_l = pointer_location
        finger_px, finger_py = self.Finger_l

        for prediction, (x, y, w, h) in self.boundingBoxes:
            x_plus_w = x + w
            y_plus_h = y + h

            # Find if the pointer is in the mentioned region
            if finger_px >= x and finger_px <= x_plus_w and finger_py >= y and finger_py <= y_plus_h:
                self.prediction = prediction
                self.boundingBoxes = (x, y, w, h)
                self.return_value()

    def __del__(self):
        del self.boundingBoxes
        del self.Finger_l

    def return_value(self):
        return self.prediction, self.boundingBoxes
