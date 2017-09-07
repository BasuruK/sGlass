import imports as IMPORT_MANAGER
import time
from Outdoor_Object_Recognition_Engine.grid_based_probability_detection import GBPD

# Initiate Outdoor Object Recognition Module
Grid_based_probability_detection = GBPD(IMPORT_MANAGER, IMPORT_MANAGER.outdoor_objects_classifier, (256, 256))

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