import time
start_time = time.time()

print("Initializing the Application")
# Load all the imports
from keras.preprocessing import image as keras_preprocess
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from PIL import Image
import math
import skimage.io as imutils
from keras.models import load_model

print("All Modules Loaded within: ", time.time() - start_time)
print("Application Started")

def get_keras_preprocess():
    return keras_preprocess