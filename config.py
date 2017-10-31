"""
Author : Balasuriya B.K | IT14020254

Unauthorized edits are not allowed.
This file is acting as the settings file for the program which corresponds to Outdoor and Indoor Object Recognition Engine.
"""


class Configurations:

    """
    ENVIRONMENT MODE:-
        1 => Indoor Object Detection (default)
        2 => Outdoor Object Detection
    """
    ENVIRONMENT_MODE = 2

    """
     PLATFORM_MODE:-
        1 => Single Object Detection Mode (default)
        2 => Multiple Object Detection Mode
    """
    PLATFORM_MODE = 1

    """
    HAND_TRACKER:-
        1 => On
        0 => Off
        
        * If Hand Tracker is on The GBPD is on by default
    """
    HAND_TRACKER = 1
