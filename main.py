import imports as IMPORT_MANAGER
from Outdoor_Object_Recognition_Engine.grid_based_probability_detection import GBPD
from Outdoor_Object_Recognition_Engine.hand_movement_tracking_module import TrackHand
from Outdoor_Object_Recognition_Engine.point_to_grid_mapper import PointToFingerMapper


# Initiate Outdoor Object Recognition Module & Hand Tracking Module
# Grid_based_probability_detection = GBPD(IMPORT_MANAGER, IMPORT_MANAGER.outdoor_objects_classifier, (256, 256))
Hand_Tracker = TrackHand(threshold=70, camera=0, blur_value=21)
Grid_Based_Probability_Detection = GBPD(imports=IMPORT_MANAGER, classifier=IMPORT_MANAGER.outdoor_objects_classifier,
                                        window_size=(256, 256))
# Track the hand
captured_frame, finger_location = Hand_Tracker.track_hand()
print("Finger Location", finger_location)

# Calculate the time for GBPD execution time
start_time = IMPORT_MANAGER.time.time()
# image_stream = IMPORT_MANAGER.Image.open('Outdoor_Object_Recognition_Engine/custom_test/dog.6.jpg')
captured_frame = IMPORT_MANAGER.Image.fromarray(captured_frame, mode="RGB")
image_coordinates_with_predictions = Grid_Based_Probability_Detection.main(captured_frame)
print("GBPD algorithm Execution Time: ", IMPORT_MANAGER.time.time() - start_time)

fig, ax = IMPORT_MANAGER.plt.subplots(1)
ax.imshow(captured_frame)

# Extract the Regions // Ignore the index 0
for prediction, image_coordinates in image_coordinates_with_predictions[1:]:
    print(prediction, image_coordinates)
    x, y, w, h = image_coordinates
    color = IMPORT_MANAGER.randomize_color()
    rect = IMPORT_MANAGER.patches.Rectangle((x, y), w, h, linewidth=3, edgecolor=color, facecolor='none', )
    IMPORT_MANAGER.plt.text(x, y, prediction, color=color)
    ax.add_patch(rect)

IMPORT_MANAGER.plt.show()

# Initiate Finger to GBPD Mapper
Pointer_To_Location = PointToFingerMapper(image_coordinates_with_predictions[1:], finger_location)
prediction_and_selected_box = Pointer_To_Location.main()

print("Predicted : {}".format(prediction_and_selected_box))

read_image = IMPORT_MANAGER.imutils.imread("Outdoor_Object_Recognition_Engine/edited.jpg")

fig, ax = IMPORT_MANAGER.plt.subplots(1)
ax.imshow(read_image)

for prediction, image_coordinates in prediction_and_selected_box:
    print(prediction, image_coordinates)
    x, y, w, h = image_coordinates
    color = IMPORT_MANAGER.randomize_color()
    rect = IMPORT_MANAGER.patches.Rectangle((x, y), w, h, linewidth=3, edgecolor=color, facecolor=color, alpha=0.5)
    IMPORT_MANAGER.plt.text(x, y, prediction, color=color)
    ax.add_patch(rect)

IMPORT_MANAGER.plt.show()

