"""
This file is responsible for changing settings of the application. the command queue will be updated according to the
command the application receives
"""
import os
import threading
from config import Configurations as ConfigManager


class SettingsManager:

    CommandQueueFile = None
    Configurations = None

    def __init__(self):
        self.Configurations = ConfigManager()

    def __del__(self):
        # Close file pointer and queue
        self.CommandQueueFile.close()
        del self.CommandQueueFile
        del self.Configurations

    # Change Settings according to the commands present in Command Queue
    def queue_manager(self):
        if not self.is_command_queue_empty():
            self.CommandQueueFile = open("Dialogue_Manager/command_temp.txt", "r+")
            # Read the contents
            command = self.CommandQueueFile.read()
            # Close the file pointer and clear
            self.CommandQueueFile.close()

            # Change the current environment
            if command == "wit_cng_env":
                self.Configurations.set_environment_mode()
            elif command == "wit_cng_env_in":
                self.Configurations.set_environment_mode_indoor()
            elif command == "wit_cng_env_out":
                self.Configurations.set_environment_mode_outdoor()
            elif command == "wit_cng_plt_sig":
                self.Configurations.set_platform_mode_single_detection()
            elif command == "wit_cng_plt_mul":
                self.Configurations.set_platform_mode_multiple_detection()
            elif command == "wit_enb_hndtrc":
                self.Configurations.enable_hand_tracker()
            elif command == "wit_dis_hndtrc":
                self.Configurations.disable_hand_tracker()
            elif command == "wit_enb_desc":
                self.Configurations.enable_description_generator()
            elif command == "wit_dis_desc":
                self.Configurations.disable_description_generator()
            elif command == "wit_cur_env":
                self.Configurations.get_current_environment_mode_name()
            elif command == "wit_quit":
                print("Exitting")
                os._exit(0)

            self.clear_command_queue()

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
        change = self.is_environment_change_command_issued()
        return command == change

    # Issue a signal when quit command is issued
    def signal_recognition_engines_to_quit_when_system_quits(self):
        self.CommandQueueFile = open("Dialogue_Manager/command_temp.txt", "r+")
        command = self.CommandQueueFile.read()
        quit_command = "wit_quit"
        return command == quit_command
