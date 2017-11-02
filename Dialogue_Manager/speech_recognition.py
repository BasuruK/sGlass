"""
This file is responsible for speech recognition
uses Google Speech API with Google DialogFlow API to convert speech to text and execute commands based on the text
"""

import threading
import os
import json
from google.cloud import speech
from google.cloud.speech import enums
from google.cloud.speech import types
import apiai
from Dialogue_Manager.audio_recorder import RecordAudio
from Dialogue_Manager.text_to_speech_processesor import error, speak
from Dialogue_Manager.settings_manager import SettingsManager

lock = threading.Lock()


class RecognizeSpeech:

    file_name = None
    num_seconds = None
    recordAudio = None
    apiai_devAccessToken = None
    apiai_cliAccessToken = None
    speech_to_text_result = None

    def __init__(self, filename, num_seconds):
        # Create an object from RecordAudio
        self.file_name = filename
        self.num_seconds = num_seconds
        self.recordAudio = RecordAudio()
        self.apiai_devAccessToken = "3d8efcd4dfb9463f9c6374b39c08fefb"
        self.apiai_cliAccessToken = "1935fb3412a84fdfae962c4e7ca47583"

        # export google authentication to environment
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "Dialogue_Manager/Dialogue_Manager_59010a9be645.json"

    def __del__(self):
        del self.file_name
        del self.num_seconds
        del self.recordAudio
        del self.apiai_devAccessToken
        del self.apiai_cliAccessToken
        del self.speech_to_text_result

    def recognize(self):
        # Connect to Google Speech API by using Service Account access token
        # Instantiates a client
        client = speech.SpeechClient()

        # Start recording audio
        self.recordAudio.record_audio(self.num_seconds, self.file_name)

        # Read audio file created
        audio_loaded = self.recordAudio.read_audio(self.file_name)

        # Convert speech to text using Google Speech api
        audio = types.RecognitionAudio(content=audio_loaded)

        config = types.RecognitionConfig(
            encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code='en-US')

        # Detects speech in the audio file
        response = client.recognize(config, audio)

        for result in response.results:
            self.speech_to_text_result = result.alternatives[0].transcript
            print(result.alternatives[0].transcript)

        # Get function string using Google DialogFlow API
        dialog_flow = apiai.ApiAI(self.apiai_cliAccessToken)
        string_request = dialog_flow.text_request()
        string_request.query = self.speech_to_text_result

        response = string_request.getresponse()
        return response.read()


# This function is responsible for coordinating activities between the hand track module and main app
def speech_coordinator_worker():
    lock.acquire()
    settings_controller = SettingsManager()
    print("Locked Worker")
    import time
    start_time = time.time()
    print(threading.current_thread().getName() + "Started in the background, will start to read commands")
    try:
        audio_rec = RecognizeSpeech("speech_input.wav", 4)
        command = audio_rec.recognize()
        json_message = json.loads(command.decode('utf-8'))
        result = json_message['result']

        # Extract out the reply message and action
        reply_message = result["fulfillment"]["messages"][0]['speech']
        action = result['action']

        print(reply_message)
        # Output reply message to the user
        speak(reply_message)

        # Push the action command to the command queue
        print(action)
        settings_controller.push_to_queue(action)

        print("Unlocking Worker")
    except KeyError:
        error("Command Not Recognized")
        print("Command Not Recognized")

    lock.release()
    print("Speech Engine Time {}".format(time.time() - start_time))
