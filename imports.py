import time
start_time = time.time()

print("Initializing the Application")
# Use this section to load all the imports

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

# Use this section to import Machine Learning and AI Models

# Load Outdoor Object Recognition Model
print("Loading AI Models")
outdoor_objects_classifier = load_model('Outdoor_Object_Recognition_Engine/my_model.h5')

print("All Modules Loaded within: ", time.time() - start_time)
print("Application Started")
