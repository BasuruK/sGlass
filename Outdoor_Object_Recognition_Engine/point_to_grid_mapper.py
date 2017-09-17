"""
TODO:
1. Make a Range from x, y, w, h values. ✔
2. Check weather the point is in the range calculated above. ✔
3. use Analytic Geometry to find if a point is residing in a rectangle.
"""
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import skimage.io as imutils
import random


def randomize_color():
    cnames = {
        'aliceblue': '#F0F8FF', 'antiquewhite': '#FAEBD7', 'aqua': '#00FFFF', 'aquamarine': '#7FFFD4',
        'azure': '#F0FFFF'
        , 'beige': '#F5F5DC', 'bisque': '#FFE4C4', 'black': '#000000',
        'blanchedalmond': '#FFEBCD', 'blue': '#0000FF', 'blueviolet': '#8A2BE2', 'brown': '#A52A2A',
        'burlywood': '#DEB887',
        'cadetblue': '#5F9EA0', 'chartreuse': '#7FFF00', 'chocolate': '#D2691E', 'coral': '#FF7F50',
        'cornflowerblue': '#6495ED',
        'cornsilk': '#FFF8DC', 'crimson': '#DC143C', 'cyan': '#00FFFF', 'darkblue': '#00008B',
        'darkcyan': '#008B8B',
        'darkgoldenrod': '#B8860B', 'darkgray': '#A9A9A9', 'darkgreen': '#006400', 'darkkhaki': '#BDB76B',
        'darkmagenta': '#8B008B', 'darkolivegreen': '#556B2F', 'darkorange': '#FF8C00', 'darkorchid': '#9932CC',
        'darkred': '#8B0000', 'darksalmon': '#E9967A', 'darkseagreen': '#8FBC8F', 'darkslateblue': '#483D8B'}
    return random.sample(cnames.items(), 1)[0][1]


class PointToFingerMapper:

    boundingBoxes = [[0, 256, 512, 256]
                    , [512, 0, 512, 256]
                    , [0,0, 512, 256]
                    , [0,512, 512, 256]
                    , [512, 512, 512, 256]
                    , [256, 0, 512, 256]
                    , [256, 256, 512, 256]
                    , [512, 256, 512, 256]]

    Finger_l = (145, 249)  # Finger Pointing Location
    prediction = None
    returnBox = []

    def __init__(self, bounding_box, pointer_location):
        self.prediction = "bird"
        # self.boundingBoxes = bounding_box
        # self.Finger_l = pointer_location

    # def __del__(self):
    #     del self.boundingBoxes
    #     del self.Finger_l
    #     del self.prediction
    #     del self.returnBox

    def main(self):
        finger_px, finger_py = self.Finger_l

        for (x, y, w, h) in self.boundingBoxes:
            x_plus_w = x + w
            y_plus_h = y + h

            self.boundingBoxes = (x, y, w, h)
            self.returnBox.append([x, y, w, h])
            break

        print(self.returnBox)
        print("Finger loc :", self.Finger_l)
        return self.prediction, self.returnBox




# Initiate Finger to GBPD Mapper
Pointer_To_Location = PointToFingerMapper(1, 1)
prediction_for_selected_box, selected_box = Pointer_To_Location.main()

image = imutils.imread("edited.jpg")

fig, ax = plt.subplots(1)
# Extract the Regions // Ignore the index 0
ax.imshow(image)

for x, y, w, h in selected_box:
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
    fingerx = 145
    fingery = 249

    p = (fingerx, fingery)
    p1 = (x1, y1)
    p2 = (x2, y2)
    p3 = (x3, y3)
    p4 = (x4, y4)


    # pd1 = patches.Circle(p1, 15)
    # pd2 = patches.Circle(p2, 15)
    # pd3 = patches.Circle(p3, 15)
    # pd4 = patches.Circle(p4, 15)
    #
    # ax.add_patch(rect)
    # ax.add_patch(pd1)
    # ax.add_patch(pd2)
    # ax.add_patch(pd3)
    # ax.add_patch(pd4)

    print("p1: {}, \np2: {}, \np3: {}, \np4: {}".format(p1, p2, p3, p4))


plt.show()



