"""
Author : Balasuriya B.K | IT14020254

Unauthorized edits are not allowed.
This file is acting as the settings file for the program which corresponds to Outdoor and Indoor Object Recognition
Engine.
"""
import threading
from Dialogue_Manager.text_to_speech_processesor import speak


class Configurations(object):

    _instance = None
    _lock = threading.Lock()

    # Use a Singleton Object to initiate Thread safe Configurations Instance
    def __new__(cls, *args, **kwargs):
        """
        Singleton Configuration Object
        :return: Single object of Configuration class
        """
        with Configurations._lock:
            if Configurations._instance is None:
                Configurations._instance = super(Configurations, cls).__new__(cls)
        return Configurations._instance

    """
    ENVIRONMENT MODE:-
        1 => Indoor Object Detection (default)
        2 => Outdoor Object Detection
    """
    ENVIRONMENT_MODE = 2

    def get_current_environment_mode_name(self):
        """
        :return: current ENVIRONMENT_NAME
        """
        if self.is_indoor_mode():
            return "Indoor Mode"
        elif self.is_outdoor_mode():
            speak("Outdoor Mode")

    def get_environment_mode(self):
        """
        :return: current ENVIRONMENT_MODE
        """
        return self.ENVIRONMENT_MODE

    def set_environment_mode(self):
        """
        Changes the environment to next one
        :return: next ENVIRONMENT_MODE
        """
        if self.is_indoor_mode():
            self.set_environment_mode_outdoor()
        elif self.is_outdoor_mode():
            self.set_environment_mode_indoor()

    def set_environment_mode_indoor(self):
        """
        Set mode to Indoor Detection
        """
        self.ENVIRONMENT_MODE = 1

    def set_environment_mode_outdoor(self):
        """
        Set mode to Outdoor Detection
        """
        self.ENVIRONMENT_MODE = 2

    def is_indoor_mode(self):
        """
        :return: True if Indoor Mode
        """
        return self.ENVIRONMENT_MODE == 1

    def is_outdoor_mode(self):
        """
        :return: True if Outdoor Mode
        """
        return self.ENVIRONMENT_MODE == 2

    """
     PLATFORM_MODE:-
        1 => Single Object Detection Mode (default)
        2 => Multiple Object Detection Mode
        
    Used in Outdoor Object detection only
    """
    PLATFORM_MODE = 1

    def get_platform_mode(self):
        """
        :return: current PLATFORM_MODE
        """
        return self.PLATFORM_MODE

    def set_platform_mode(self, plt_mode):
        """
        Set the platform mode to the given mode
        :param plt_mode: platform mode [single/multiple]
        :return:
        """
        self.PLATFORM_MODE = plt_mode

    def set_platform_mode_single_detection(self):
        """
        Set Platform for Single detection
        """
        self.PLATFORM_MODE = 1

    def set_platform_mode_multiple_detection(self):
        """
        Set Platform for Multiple detection
        """
        self.PLATFORM_MODE = 2

    def is_single_object_detection_mode(self):
        """
        :return: True if Single Detection Mode
        """
        return self.PLATFORM_MODE == 1

    def is_multiple_object_detection_mode(self):
        """
        :return: True if Multiple Detection Mode
        """
        return self.PLATFORM_MODE == 2

    """
    DESCRIPTION_GENERATOR:-
        1 => Enable (default)
        0 => Disable
    """
    DESCRIPTION_GENERATOR = 1

    def get_description_generator(self):
        """
        :return: current DESCRIPTION_GENERATOR value
        """
        return self.DESCRIPTION_GENERATOR

    def enable_description_generator(self):
        """
        Enable Description Generation
        """
        self.DESCRIPTION_GENERATOR = 1

    def disable_description_generator(self):
        """
        Disable Description Generation
        """
        self.DESCRIPTION_GENERATOR = 0

    def is_description_generator_enabled(self):
        """
        :return: True if Enabled
        """
        return self.DESCRIPTION_GENERATOR == 1

    def is_description_generator_disabled(self):
        """
        :return: True if Disabled
        """
        return self.DESCRIPTION_GENERATOR == 0
