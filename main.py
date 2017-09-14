#import imports as IMPORT_MANAGER
import time
#from Outdoor_Object_Recognition_Engine.grid_based_probability_detection import GBPD
from Outdoor_Object_Recognition_Engine.hand_movement_tracking_module import TrackHand
import cv2
import matplotlib.pyplot as plt
from PIL import Image
'''
TODO: Attach the hand recognition module to main
TODO: Methodology
1. Run the Hand tracking Module. ✔
2. When the user issues the command take a snap. ✔
3. Take the exact coordinates of the users pointing finger. ✔
4. Send the snap to GBPD for classification.
5. Identify the location of the bounding box corresponding to the location of the finger point.
'''
# Initiate Outdoor Object Recognition Module & Hand Tracking Module
# Grid_based_probability_detection = GBPD(IMPORT_MANAGER, IMPORT_MANAGER.outdoor_objects_classifier, (256, 256))
Hand_Tracker = TrackHand(threshold=70, camera=0, blur_value=21)
print("Going in to Hand tracking class")
# Track the hand
captured_frame, finger_location = Hand_Tracker.track_hand()
print("Came out side the Hand tracking class")
print("Finger Location", finger_location)
captured_frame = Image.fromarray(captured_frame)
plt.imshow(captured_frame)
plt.plot()
'''
# Calculate the time for GBPD execution time
start_time = time.time()

image_stream = IMPORT_MANAGER.Image.open('Outdoor_Object_Recognition_Engine/custom_test/dog.6.jpg')
image_coordinates_with_predictions = Grid_based_probability_detection.main(image_stream=image_stream)

print("GBPD algorithm Execution Time: ", time.time() - start_time)

# Show the image with bounding boxes

fig, ax = IMPORT_MANAGER.plt.subplots(1)
ax.imshow(image_stream)

# Extract the Regions // Ignore the index 0
for prediction, image_coordinates in image_coordinates_with_predictions[1:]:
    print(prediction, image_coordinates)
    x, y, w, h = image_coordinates
    color = IMPORT_MANAGER.randomize_color()
    rect = IMPORT_MANAGER.patches.Rectangle((x, y), w, h, linewidth=3, edgecolor=color, facecolor='none')
    IMPORT_MANAGER.plt.text(x, y, prediction, color=color)
    ax.add_patch(rect)

IMPORT_MANAGER.plt.show()

'''
