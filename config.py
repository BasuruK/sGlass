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
    PLATFORM_MODE = 2

    def get_platform_mode(self):
        """
        :return: current PLATFORM_MODE
        """
        return self.PLATFORM_MODE

    def set_platform_mode(self):
        """
        Set the platform mode to the next mode
        """
        if self.is_single_object_detection_mode():
            self.set_platform_mode_multiple_detection()
        elif self.is_multiple_object_detection_mode():
            self.set_platform_mode_single_detection()

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

    """
    GBPD_OUT_DISPLAY:-
        1 => Enabled
        0 => Disabled
    """
    GBPD_OUT_DISPLAY = 1

    def enable_gbpd_display(self):
        """
        Enable GBPD output Display
        """
        self.GBPD_OUT_DISPLAY = 1

    def disable_gbpd_display(self):
        """
        Disable GBPD output Display
        """
        self.GBPD_OUT_DISPLAY = 0

    def is_gbpd_enabled(self):
        """
        :return: True if Enabled
        """
        return self.GBPD_OUT_DISPLAY == 1

    def is_gbpd_disabled(self):
        """
        :return: True if Disabled
        """
        return self.GBPD_OUT_DISPLAY == 0

    """
    POINTER_LOC_DISPLAY:-
        1 => Enabled
        0 => Disabled
    """
    POINTER_LOC_DISPLAY = 1

    def enable_pointer_loc_display(self):
        """
        Enable GBPD output Display
        """
        self.POINTER_LOC_DISPLAY = 1

    def disable_pointer_loc_display(self):
        """
        Disable GBPD output Display
        """
        self.POINTER_LOC_DISPLAY = 0

    def is_pointer_loc_enabled(self):
        """
        :return: True if Enabled
        """
        return self.POINTER_LOC_DISPLAY == 1

    def is_pointer_loc_disabled(self):
        """
        :return: True if Disabled
        """
        return self.POINTER_LOC_DISPLAY == 0

    """
    Available User Commands are Listed below. The commands are sent as a reference point to other classes
    """
    change_environment = "wit_cng_env"

    change_environment_indoor = "wit_cng_env_in"

    change_environment_outdoor = "wit_cng_env_out"

    change_platform = "wit_cng_plt"

    change_platform_single_detection = "wit_cng_plt_sig"

    change_platform_multiple_detection = "wit_cng_plt_mul"

    enable_description = "wit_enb_desc"

    disable_description = "wit_dis_desc"

    current_environment = "wit_cur_env"

    enable_gbpd_display_output = "wit_gbpd_dis_enb"

    disable_gbpd_display_output = "wit_gbpd_dis_dis"

    enable_finger_loc_display_output = "wit_fin_loc_enb"

    disable_finger_loc_display_output = "wit_fin_loc_dis"

    quit = "wit_quit"
