"""
Author : Balasuriya B.K | IT14020254

Unauthorized edits are not allowed.
This file is acting as the settings file for the program which corresponds to Outdoor and Indoor Object Recognition Engine.
"""


class Configurations:

    """
     __PLATFORM_MODE:-
        1 => Single Object Detection Mode (default)
        2 => Multiple Object Detection Mode
    """
    __PLATFORM_MODE = 1

    """
    __HAND_TRACKER:-
        1 => On
        0 => Off
        
        * If Hand Tracker is on The GBPD is on by default
    """
    __HAND_TRACKER = 1
