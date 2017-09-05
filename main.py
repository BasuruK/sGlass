import imports as IMPORT_MANAGER

from Outdoor_Object_Recognition_Engine.grid_based_probability_detection import GBPD

import time
# Initiate Outdoor Object Recognition Module
Grid_based_probability_detection = GBPD(IMPORT_MANAGER, IMPORT_MANAGER.outdoor_objects_classifier)

start_time = time.time()
image_stream = IMPORT_MANAGER.Image.open('Outdoor_Object_Recognition_Engine/custom_test/dog.6.jpg')
Grid_based_probability_detection.main(image_stream=image_stream)

print("GBPD algorithm: ", time.time() - start_time)
