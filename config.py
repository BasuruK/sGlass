"""
Author : Balasuriya B.K | IT14020254

Unauthorized edits are not allowed.
This file is acting as the settings file for the program which corresponds to Outdoor and Indoor Object Recognition
Engine.
"""


class Configurations:

    """
    ENVIRONMENT MODE:-
        1 => Indoor Object Detection (default)
        2 => Outdoor Object Detection
    """
    ENVIRONMENT_MODE = 2

    def get_environment_mode(self):
        return self.ENVIRONMENT_MODE

    def set_environment_mode(self, env_mode):
        self.ENVIRONMENT_MODE = env_mode

    def set_environment_mode_indoor(self):
        self.ENVIRONMENT_MODE = 1

    def set_environment_mode_outdoor(self):
        self.ENVIRONMENT_MODE = 2

    def is_indoor_mode(self):
        return self.ENVIRONMENT_MODE == 1

    def is_outdoor_mode(self):
        return self.ENVIRONMENT_MODE == 2

    """
     PLATFORM_MODE:-
        1 => Single Object Detection Mode (default)
        2 => Multiple Object Detection Mode
    """
    PLATFORM_MODE = 1

    def get_platform_mode(self):
        return self.PLATFORM_MODE

    def set_platform_mode(self, plt_mode):
        self.PLATFORM_MODE = plt_mode

    def set_platform_mode_single_detection(self):
        self.PLATFORM_MODE = 1

    def set_platform_mode_multiple_detection(self):
        self.PLATFORM_MODE = 2

    def is_single_object_detection_mode(self):
        return self.PLATFORM_MODE == 1

    def is_multiple_object_detection_mode(self):
        return self.PLATFORM_MODE == 0

    """
    HAND_TRACKER:-
        1 => On
        0 => Off
        
        * If Hand Tracker is on The GBPD is on by default
    """
    HAND_TRACKER = 1

    def get_hand_tracker(self):
        return self.HAND_TRACKER

    def enable_hand_tracker(self):
        self.HAND_TRACKER = 1

    def disable_hand_tracker(self):
        self.HAND_TRACKER = 0

    def is_hand_tracker_enabled(self):
        return self.HAND_TRACKER == 1

    def is_hand_tracker_disabled(self):
        return self.HAND_TRACKER == 0

    """
    DESCRIPTION_GENERATOR:-
        1 => Enable (default)
        0 => Disable
    """
    DESCRIPTION_GENERATOR = 1

    def get_description_generator(self):
        return self.DESCRIPTION_GENERATOR

    def enable_description_generator(self):
        self.DESCRIPTION_GENERATOR = 1

    def disable_description_generator(self):
        self.DESCRIPTION_GENERATOR = 0

    def is_description_generator_enabled(self):
        return self.DESCRIPTION_GENERATOR == 1

    def is_description_generator_disabled(self):
        return self.DESCRIPTION_GENERATOR == 0
