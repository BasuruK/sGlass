"""
Main.py handles the main functions of the Application.
NO UNAUTHORIZED EDITS ARE ALLOWED

TODO: 1. Change the Outdoor object recognition module to recognize one object at a time.
2. Hide the Hand Track and GBPD algorithm, add keypress event to activate them when required.
3. Integrate Indoor Object Recognition Module to work with voice commands as well as shortcut key.
4. Put the program in a infinite loop with a break condition to make sure the program runs every time.
"""

import imports as IMPORT_MANAGER
from config import Configurations as ConfigManager
from Outdoor_Object_Recognition_Engine.grid_based_probability_detection import GBPD
from Outdoor_Object_Recognition_Engine.hand_movement_tracking_module import TrackHand
from Outdoor_Object_Recognition_Engine.point_to_grid_mapper import PointToFingerMapper
from Outdoor_Object_Recognition_Engine.single_detection_mode import SingleDetection
from Description_Generator.generate_description import DescriptionGenerator
from Dialogue_Manager.text_to_speech_processesor import TextToSpeech

# Initiate Dialogue Manager
Text_To_Speech = TextToSpeech(IMPORT_MANAGER)
# Get Configurations Handler
Configurations = ConfigManager()

while True:
    # Initiate Checking for Indoor or Outdoor Mode
    if Configurations.is_indoor_mode():
        print("Indoor Mode")

    if Configurations.is_outdoor_mode():

        """
        ##############################
        REGION MULTIPLE OBJECT DETECTION
        ##############################
        """
        # Check if its Multiple Object Detection Platform
        if Configurations.is_multiple_object_detection_mode():
            # Initiate Outdoor Object Recognition Module & Hand Tracking Module
            Hand_Tracker = TrackHand(threshold=70, camera=0, blur_value=21)
            Grid_Based_Probability_Detection = GBPD(imports=IMPORT_MANAGER, classifier=IMPORT_MANAGER.outdoor_objects_classifier,
                                                    window_size=(256, 256))
            # Track the hand
            captured_frame, finger_location = Hand_Tracker.track_hand()
            print("Finger Location", finger_location)

            # Calculate the time for GBPD execution time
            start_time = IMPORT_MANAGER.time.time()

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
            if len(prediction_and_selected_box) is 0:
                print("The Object user is pointing cannot be identified")
                Text_To_Speech.speak("The Object user is pointing cannot be identified")

            read_image = IMPORT_MANAGER.imutils.imread("Outdoor_Object_Recognition_Engine/edited.jpg")

            fig, ax = IMPORT_MANAGER.plt.subplots(1)
            ax.imshow(read_image)

            for prediction, image_coordinates in prediction_and_selected_box:
                print(prediction, image_coordinates)
                string = "You pointed at a "
                Text_To_Speech.predict_speech(sentence=string, prediction=prediction)
                x, y, w, h = image_coordinates
                color = IMPORT_MANAGER.randomize_color()
                rect = IMPORT_MANAGER.patches.Rectangle((x, y), w, h, linewidth=3, edgecolor=color, facecolor=color, alpha=0.5)
                IMPORT_MANAGER.plt.text(x, y, prediction, color=color)
                ax.add_patch(rect)

            IMPORT_MANAGER.plt.show()
        """
        ##############################
        END REGION MULTIPLE OBJECT DETECTION
        ##############################
        """

        """
        ##############################
        REGION SINGLE OBJECT DETECTION
        ##############################
        """
        # Check if its Single Object Detection Platform
        if Configurations.is_single_object_detection_mode():
            print("Single Detection")

            ftry = SingleDetection(IMPORT_MANAGER, camera_id=0, classifier=IMPORT_MANAGER.outdoor_objects_classifier)
            prediction = ftry.track_object()

            print("Prediction :", prediction)
            Text_To_Speech.single_object_speech(prediction=prediction)

        """
        ##############################
        END REGION SINGLE OBJECT DETECTION
        ##############################
        """

# Initiate Description Generator Module

# Generate the description for the identified object
Description_Generator = DescriptionGenerator(imports=IMPORT_MANAGER)
generated_description = Description_Generator.show_description(image_path='Outdoor_Object_Recognition_Engine/edited.jpg')
print(generated_description)
Text_To_Speech.speak(generated_description)


