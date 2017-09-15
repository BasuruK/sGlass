import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

'''
Finger Location (217, 245)

cat 100.00% [256   0 512 256]
cat 100.00% [512 256 512 256]
cat 100.00% [  0 256 512 256]
cat 100.00% [512   0 512 256]
cat 100.00% [256 256 512 256]
car 100.00% [256 512 512 256]
cat 100.00% [512 512 512 256]
cat 100.00% [  0 512 512 256]
cat 100.00% [  0   0 512 256]

TODO:
1. Make a Range from x, y, w, h values ✔
2. Check weather the point is in the range calculated above. 
'''

bboxes = [[256, 0, 512, 256], [512, 256, 512, 256], [0,256, 512, 256], [512,0,512,256], [256,256,512,256],[256,512,512,256],[512,512,512,256], [0,512,512,256], [0,0,512,256]]
finger_l = (217, 245)

for x, y, w, h in bboxes:
    # print("x {}, y {}, w {}, h {}".format(x, y, w, h))
    # plt.scatter(x + w, y + h)

    print("x:{} to x + w:{},  y:{} to y + h:{}".format(x, x+w, y, y+w))
    print(finger_l)

    # Find if the pointer is in the mentioned region

    plt.scatter(x, y)
    plt.scatter(w+x, h+y)
    plt.scatter(finger_l[0], finger_l[1])
    break

# plt.scatter(X, Y)
# plt.scatter(W, H, marker="x", s=100)
# plt.scatter(finger_location[0], finger_location[1])
plt.show()
