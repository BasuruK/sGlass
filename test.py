from Indoor_Object_Recognition_Engine.Hand_Gesture.Hand_Gesture_Recognition_System import Hand_Gesture_Recognition_System
from Indoor_Object_Recognition_Engine.IOIM.Indoor_Object_Recognition_System import Indoor_Object_Recognition_System
import os

Hand_Gesture = Hand_Gesture_Recognition_System()
Indoor_Object = Indoor_Object_Recognition_System()

hand_gesture_capture = Hand_Gesture.capture_HG_image()
hand_prediction = Hand_Gesture.Get_Reult_Of_HG("Indoor_Object_Recognition_Engine/IOIM/Original/Test/hand_gesture_0.png")

if hand_prediction == "Positive Hand Gesture":
    indoor_object = Indoor_Object.capture_IO_image()

    path = os.path.abspath(indoor_object)

    prediction = Indoor_Object.predict_object_class(path="Indoor_Object_Recognition_Engine/IOIM/Indoor Objects/Test/indoor_object_0.png")

    if prediction == [0]:
        print("This is a Bottle")
    elif prediction == [1]:
        print("This is a Mug")

else:
    print("Hand Gesture is not suitable for tracking the object")
