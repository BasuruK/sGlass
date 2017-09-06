import imports as IMPORT_MANAGER
import time
from Outdoor_Object_Recognition_Engine.grid_based_probability_detection import GBPD
# Disable Tensorflow log information
IMPORT_MANAGER.tf.logging.set_verbosity(IMPORT_MANAGER.tf.logging.DEBUG)

# Initiate Outdoor Object Recognition Module
Grid_based_probability_detection = GBPD(IMPORT_MANAGER, IMPORT_MANAGER.outdoor_objects_classifier)

start_time = time.time()

image_stream = IMPORT_MANAGER.Image.open('Outdoor_Object_Recognition_Engine/custom_test/dog.6.jpg')
image_coordinates_with_predictions = Grid_based_probability_detection.main(image_stream=image_stream)

print("GBPD algorithm: ", time.time() - start_time)


print(image_coordinates_with_predictions)
