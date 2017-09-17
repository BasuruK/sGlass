import imports as IMPORT_MANAGER
from Outdoor_Object_Recognition_Engine.grid_based_probability_detection import GBPD
from Outdoor_Object_Recognition_Engine.hand_movement_tracking_module import TrackHand
#from Outdoor_Object_Recognition_Engine.point_to_grid_mapper import PointToFingerMapper

'''
TODO: Methodology
1. Run the Hand tracking Module. ✔
2. When the user issues the command take a snap. ✔
3. Take the exact coordinates of the users pointing finger. ✔
4. Send the snap to GBPD for classification. ✔
5. Identify the location of the bounding box corresponding to the location of the finger point.
(Optional) 6. Extract the regions from GBPD to sent to the Description Generator
'''
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
captured_frame = IMPORT_MANAGER.Image.fromarray(captured_frame)
image_coordinates_with_predictions = Grid_Based_Probability_Detection.main(captured_frame)
print("GBPD algorithm Execution Time: ", IMPORT_MANAGER.time.time() - start_time)

fig, ax = IMPORT_MANAGER.plt.subplots(1)
ax.imshow(captured_frame)

# Extract the Regions // Ignore the index 0
for prediction, image_coordinates in image_coordinates_with_predictions[1:]:
    print(prediction, image_coordinates)
    x, y, w, h = image_coordinates
    color = IMPORT_MANAGER.randomize_color()
    rect = IMPORT_MANAGER.patches.Rectangle((x, y), w, h, linewidth=3, edgecolor=color, facecolor='none')
    IMPORT_MANAGER.plt.text(x, y, prediction, color=color)
    ax.add_patch(rect)


IMPORT_MANAGER.plt.show()

# Initiate Finger to GBPD Mapper
#Pointer_To_Location = PointToFingerMapper(image_coordinates_with_predictions[1:], finger_location)
#prediction_for_selected_box, selected_box = Pointer_To_Location.main()

#print("Predicted : {}, box {}".format(prediction_for_selected_box, selected_box))

fig, ax = IMPORT_MANAGER.plt.subplots(1)
# Extract the Regions // Ignore the index 0
ax.imshow(captured_frame)



# for x, y, w, h in selected_box:
#     #x, y, w, h = selected_box
#     color = IMPORT_MANAGER.randomize_color()
#     rect = IMPORT_MANAGER.patches.Rectangle((x, y), w, h, linewidth=3, edgecolor=color, facecolor=color, alpha=0.5)
#     circlesx = IMPORT_MANAGER.patches.Circle((x, y), 10)
#     circlesy = IMPORT_MANAGER.patches.Circle((x+w, y+h), 10)
#     circlesxw = IMPORT_MANAGER.patches.Circle((x+w, 0), 10)
#     circlesyh = IMPORT_MANAGER.patches.Circle((0, y+h), 10)
#     #IMPORT_MANAGER.plt.text(x, y, prediction, color=color)
#     ax.add_patch(rect)
#     ax.add_patch(circlesx)
#     ax.add_patch(circlesy)
#     ax.add_patch(circlesxw)
#     ax.add_patch(circlesyh)
#
# IMPORT_MANAGER.plt.show()


