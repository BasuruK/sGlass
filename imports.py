"""
Manage this file with utmost care, as this file contains all the imports used throughout the application.
Load all the AI models and imports in the sections mentioned below and do not change already added variables
or anything other unless it belongs to your component.

When implementing your module, it is advised to follow OOP approach but not necessarily.
import the imports.py (this file) at the beginning of the file and put an alias as IMPORT_MANAGER and use this to access
all the imports and models located in this file (imports.py)
"""


import time
start_time = time.time()

print("Initializing the Application")
# Use this section to load all the imports

import tensorflow as tf
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
