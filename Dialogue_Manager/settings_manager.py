"""
Author : Balasuriya B.K | IT14020254

This file is responsible for changing settings of the application. the command queue will be updated according to the
command the application receives
"""
import os
import threading
from config import Configurations as ConfigManager


class SettingsManager:

    CommandQueueFile = None
    Configurations = None

    # Singleton Variables
    _lock = threading.Lock()
    _instance = None

    # Use a Singleton Object to initiate Thread safe SettingsManager Instance
    def __new__(cls, *args, **kwargs):
        """
        Singleton Configuration Object
        :return: Single object of SettingsManager class
        """
        with SettingsManager._lock:
            if SettingsManager._instance is None:
                SettingsManager._instance = super(SettingsManager, cls).__new__(cls)
        return SettingsManager._instance

    def __init__(self):
        self.Configurations = ConfigManager()

    def __del__(self):
        # Close file pointer and queue
        self.CommandQueueFile.close()
        del self.CommandQueueFile
        del self.Configurations

    # Change Settings according to the commands present in Command Queue
    def queue_manager(self):
        """
        Runs the normal commands:- Platform changes, environment changes
        """
        if not self.is_command_queue_empty():
            self.CommandQueueFile = open("Dialogue_Manager/command_temp.txt", "r+")
            # Read the contents
            command = self.CommandQueueFile.read()
            # Close the file pointer and clear
            self.CommandQueueFile.close()

            # Change the current environment
            if command == self.Configurations.change_environment:
                self.Configurations.set_environment_mode()

            elif command == self.Configurations.change_environment_indoor:
                self.Configurations.set_environment_mode_indoor()

            elif command == self.Configurations.change_environment_outdoor:
                self.Configurations.set_environment_mode_outdoor()

            elif command == self.Configurations.change_platform:
                self.Configurations.set_platform_mode()

            elif command == self.Configurations.change_platform_single_detection:
                self.Configurations.set_platform_mode_single_detection()

            elif command == self.Configurations.change_platform_multiple_detection:
                self.Configurations.set_platform_mode_multiple_detection()

            elif command == self.Configurations.quit:
                self.clear_command_queue()
                exit(0)

            self.clear_command_queue()

    def immediate_queue(self):
        """
        Handle Immediate Queries:- Queries that are used to control application specific commands used for
        testing purposes in the development mode

        Does not use the class CommandQueueFile variable as it raises a conflict with the main Queue
        """
        # Run only for Immediate Queries
        CommandQueueFile = open("Dialogue_Manager/command_temp.txt", "r+")
        # Read the contents
        command = CommandQueueFile.read()
        # Close the file pointer and clear
        CommandQueueFile.close()

        if command == self.Configurations.enable_description:
            self.Configurations.enable_description_generator()
            self.clear_command_queue()

        elif command == self.Configurations.disable_description:
            self.Configurations.disable_description_generator()
            self.clear_command_queue()

        elif command == self.Configurations.enable_gbpd_display_output:
            self.Configurations.enable_gbpd_display()
            self.clear_command_queue()

        elif command == self.Configurations.disable_gbpd_display_output:
            self.Configurations.disable_gbpd_display()
            self.clear_command_queue()

        elif command == self.Configurations.enable_finger_loc_display_output:
            self.Configurations.enable_pointer_loc_display()
            self.clear_command_queue()

        elif command == self.Configurations.disable_finger_loc_display_output:
            self.Configurations.disable_pointer_loc_display()
            self.clear_command_queue()

        elif command == self.Configurations.current_environment:
            self.Configurations.get_current_environment_mode_name()

        elif command == self.Configurations.enable_description_generation:
            self.Configurations.enable_description_generator()

        elif command == self.Configurations.disable_description_generation:
            self.Configurations.disable_description_generator()

        elif command == self.Configurations.clear_hand_tracking_profiles:
            self.clear_hand_tracking_profiles()

        elif command == self.Configurations.quit:
            self.Configurations.set_to_shutdown_state()
            exit(0)

    @staticmethod
    def check_command_queue():
        file_content = os.stat("Dialogue_Manager/command_temp.txt").st_size

        if file_content != 0:
            # Command is in the queue, read
            f = open("Dialogue_Manager/command_temp.txt", 'r')
            command = f.read()
            return command
        else:
            return False

    # Clear Command Queue
    @staticmethod
    def clear_command_queue():
        open("Dialogue_Manager/command_temp.txt", 'w').close()

    # Push commands to Queue
    def push_to_queue(self, command):
        # Clear any existing commands
        self.clear_command_queue()
        # Add new Command
        if self.is_command_queue_empty():
            self.CommandQueueFile = open("Dialogue_Manager/command_temp.txt", "r+")
            self.CommandQueueFile.write(command)
            self.CommandQueueFile.close()

    # Check whether the file is empty
    @staticmethod
    def is_command_queue_empty():
        file_content = os.stat("Dialogue_Manager/command_temp.txt").st_size
        return file_content == 0

    # Check whether command queue possesses a command to change environment
    def is_environment_change_command_issued(self):
        self.CommandQueueFile = open("Dialogue_Manager/command_temp.txt", "r+")
        output = self.CommandQueueFile.read()
        self.CommandQueueFile.close()

        if output == "wit_cng_env":
            return "wit_cng_env"
        elif output == "wit_cng_env_in":
            return "wit_cng_env_in"
        elif output == "wit_cng_env_out":
            return "wit_cng_env_out"

    # Issue a signal based on the environment change command in queue
    def signal_recognition_engines_to_quit(self):
        self.CommandQueueFile = open("Dialogue_Manager/command_temp.txt", "r+")
        command = self.CommandQueueFile.read()
        self.CommandQueueFile.close()
        change = self.is_environment_change_command_issued()
        self.Configurations.set_to_reset_state()
        return command == change

    # Check weather command queue possesses a command to change platform
    def is_platform_change_command_issued(self):
        self.CommandQueueFile = open("Dialogue_Manager/command_temp.txt", "r+")
        output = self.CommandQueueFile.read()
        self.CommandQueueFile.close()

        if output == "wit_cng_plt":
            return "wit_cng_plt"
        elif output == "wit_cng_plt_sig":
            return "wit_cng_plt_sig"
        elif output == "wit_cng_plt_mul":
            return "wit_cng_plt_mul"

    def signal_recognition_engines_to_quit_on_platform_change(self):
        self.CommandQueueFile = open("Dialogue_Manager/command_temp.txt", "r+")
        command = self.CommandQueueFile.read()
        self.CommandQueueFile.close()
        self.Configurations.set_to_reset_state()
        change = self.is_platform_change_command_issued()

        return command == change

    # Issue a signal when quit command is issued
    def signal_recognition_engines_to_quit_when_system_quits(self):
        self.CommandQueueFile = open("Dialogue_Manager/command_temp.txt", "r+")
        command = self.CommandQueueFile.read()
        self.CommandQueueFile.close()
        quit_command = "wit_quit"
        self.Configurations.set_to_reset_state()

        return command == quit_command

    # Clear Object and HSV hand profiles
    def clear_hand_tracking_profiles(self):
        open("Outdoor_Object_Recognition_Engine/object_color.h5", "w+").close()
        open("Outdoor_Object_Recognition_Engine/object_histogram.h5", "w+").close()
        self.Configurations.set_to_reset_state()

    def reset_recognition_engine_when_hand_track_data_clears(self):
        if self.Configurations.is_in_reset_state():
            return True
        else:
            return False

