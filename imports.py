import time
start_time = time.time()

print("Initializing the Application")

#Load all the imports
import tensorflow as tf
from keras.preprocessing import image as keras_preprocess
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import math
import skimage.io as imutils

print("All Modules Loaded within: ", time.time() - start_time)
print("Application Started")
