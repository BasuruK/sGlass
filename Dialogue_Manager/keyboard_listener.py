"""
Author : Balasuriya B.K | IT14020254

This file is responsible for handling keyboard events. This file contains a class which acts as a listener
that listen to keyboard inputs

Future work:
    change this file to support hardware button when implemented in to the wearable device
"""
from pynput import keyboard
from Dialogue_Manager.speech_recognition import speech_coordinator_worker
import threading

lock = threading.Lock()


class KeyboardListener:

    @staticmethod
    def on_pressed(key):
        try:
            if key.char == 'l':
                speech_listener = threading.Thread(target=speech_coordinator_worker(), name="coordinator")
                speech_listener.daemon = True
                speech_listener.start()
        except AttributeError:
            # print("special key {0} pressed".format(key))
            pass

    def turn_on_listener(self):
        with keyboard.Listener(on_press=self.on_pressed) as listener:
            listener.join()


# Start to listen to Keyboard key presses, Thread Runnable method
def listen_to_keypress():
    lock.acquire()
    print("Keyboard Listener Started")

    keyboard_listen = KeyboardListener()
    keyboard_listen.turn_on_listener()

    print("Keyboard Listener Terminated")
    lock.release()
    print("Number of threads active: ",threading.active_count())
