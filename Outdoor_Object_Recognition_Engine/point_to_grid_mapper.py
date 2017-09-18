"""
TODO:
1. Make a Range from x, y, w, h values. ✔
2. Check weather the point is in the range calculated above. ✔
3. use Analytic Geometry to find if a point is residing in a rectangle.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(1)
# Extract the Regions // Ignore the index 0

boundingBoxes = [[256, 0, 512, 256],
                [0, 0, 512, 256],
                [0, 256, 512, 256]]

# boundingBoxes = [[0, 0, 512, 256]]

for x, y, w, h in boundingBoxes:
    color = randomize_color()
    rect = patches.Rectangle((x, y), w, h, linewidth=3, edgecolor=color, facecolor=color, alpha=0.5)

    x1 = x
    y1 = y
    x2 = x + w
    y2 = y
    x3 = x + w
    y3 = y + h
    x4 = x
    y4 = y + h
    fingerx = 469
    fingery = 140

    p = (fingerx, fingery)
    p1 = (x1, y1)
    p2 = (x2, y2)
    p3 = (x3, y3)
    p4 = (x4, y4)

    p2_into_p1 = (x2 - x1, y2 - y1)
    p4_into_p1 = (x4 - x1, y4 - y1)

    p2_into_p1_magnitude_squared = p2_into_p1[0] ** 2 + p2_into_p1[1] ** 2
    p4_into_p1_magnitude_squared = p4_into_p1[0] ** 2 + p4_into_p1[1] ** 2

    point = (fingerx - x1, fingery - y1)

    if 0 <= point[0] * p2_into_p1[0] + point[1] * p2_into_p1[1] <= p2_into_p1_magnitude_squared:
        if 0 <= point[0] * p4_into_p1[0] + point[1] * p4_into_p1[1] <= p4_into_p1_magnitude_squared:
            print("Inside")
        else:
            print("Outside")
    else:
        print("Outside")

    pd = patches.Circle(p, 10)
    pd1 = patches.Circle(p1, 8)
    pd2 = patches.Circle(p2, 8)
    pd3 = patches.Circle(p3, 8)
    pd4 = patches.Circle(p4, 8)
    plt.scatter(fingerx, fingery)

    ax.add_patch(rect)
    ax.add_patch(pd)
    ax.add_patch(pd1)
    ax.add_patch(pd2)
    ax.add_patch(pd3)
    ax.add_patch(pd4)

    print("p1: {}, \np2: {}, \np3: {}, \np4: {}".format(p1, p2, p3, p4))

plt.show()

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



