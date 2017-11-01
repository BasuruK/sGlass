"""
This file is responsible for changing settings of the application. the command queue will be updated according to the
command the application receives
"""
import os


class SettingsManager:

    CommandQueueFile = None

    def __init__(self):
        self.CommandQueueFile = open("Dialogue_Manager/command_temp.txt", "w")

    def __del__(self):
        del self.CommandQueueFile

    def clear_command_queue(self):
        self.CommandQueueFile.write("")
        self.CommandQueueFile.close()
