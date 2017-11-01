"""
This file is responsible for changing settings of the application. the command queue will be updated according to the
command the application receives
"""
import os
from config import Configurations as ConfigManager


class SettingsManager:

    CommandQueueFile = None
    Configurations = None

    def __init__(self):
        self.CommandQueueFile = open("Dialogue_Manager/command_temp.txt", "w")
        self.Configurations = ConfigManager()

    def __del__(self):
        # Clear and Close file pointer and queue
        self.clear_command_queue()
        self.CommandQueueFile.close()

        del self.CommandQueueFile
        del self.Configurations

    # Change Settings according to the commands present in Command Queue
    def queue_manager(self):
        if not self.is_command_queue_empty():
            command = self.CommandQueueFile.read()
            self.clear_command_queue()

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

    # Clear Command Queue
    def clear_command_queue(self):
        self.CommandQueueFile.write("")

    # Push commands to Queue
    def push_to_queue(self, command):
        if self.is_command_queue_empty():
            self.CommandQueueFile.write(command)
        else:
            print("Queue is not Empty, Automatic Clearing triggered")
            self.clear_command_queue()

    # Check weather the file is empty
    @staticmethod
    def is_command_queue_empty():
        file_content = os.stat("Dialogue_Manager/command_temp.txt").st_size
        return file_content == 0
