"""
TODO: Use GOOGLE SPEECH API and API.AI which is yield much faster responses
"""

import threading
import os
from google.cloud import speech
from google.cloud.speech import enums
from google.cloud.speech import types
import apiai
from Dialogue_Manager.audio_recorder import RecordAudio
from Dialogue_Manager.text_to_speech_processesor import error

lock = threading.Lock()


class RecognizeSpeech:

    file_name = None
    num_seconds = None
    recordAudio = None
    devAccessToken = None
    cliAccessToken = None

    def __init__(self, filename, num_seconds):
        # Create an object from RecordAudio
        self.file_name = filename
        self.num_seconds = num_seconds
        self.recordAudio = RecordAudio()
        self.apiai_devAccessToken = "3d8efcd4dfb9463f9c6374b39c08fefb"
        self.apiai_cliAccessToken = "1935fb3412a84fdfae962c4efac7ca47583"

        # export google authentication to environment
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "Dialogue_Manager_59010a9be645.json"

    def __del__(self):
        del self.file_name
        del self.num_seconds
        del self.recordAudio
        del self.apiai_devAccessToken
        del self.apiai_cliAccessToken

    def recognize(self):
        # Connect to Google Speech API by using Service Account access token
        # Instantiates a client
        client = speech.SpeechClient()

        # Start recording audio
        self.recordAudio.record_audio(self.num_seconds, self.file_name)

        # Read audio file created
        audio_loaded = self.recordAudio.read_audio(self.file_name)

        audio = types.RecognitionAudio(content=audio_loaded)

        config = types.RecognitionConfig(
            encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code='en-US')

        # Detects speech in the audio file
        response = client.recognize(config, audio)

        for result in response.results:
            print('Transcript: {}'.format(result.alternatives[0].transcript))

        return format(result.alternatives[0].transcript)


# This function is responsible for coordinating activities between the hand track module and main app
def speech_coordinator_worker():
    lock.acquire()
    print("Locked Worker")
    import time
    start_time = time.time()
    print(threading.current_thread().getName() + "Started in the background, will start to read commands")
    try:
        audio_rec = RecognizeSpeech("speech_input.wav", 4)
        command = audio_rec.recognize()

        print(command)
        if command is "" or command is None:
            command = "no_command"

        f = open("Dialogue_Manager/command_temp.txt", "w")
        f.write(command)
        f.close()
        print("Unlocking Worker")
    except KeyError:
        error("Command Not Recognized")
        print("Command Not Recognized")

    lock.release()
    print("Wit Time {}".format(time.time() - start_time))


l = RecognizeSpeech("speech_input.wav", 4)
l.recognize()
