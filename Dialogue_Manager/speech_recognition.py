from wit import Wit
import requests
import json
import threading
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
    api_endpoint = None
    access_token = None

    def __init__(self, filename, num_seconds):
        self.file_name = filename
        self.num_seconds = num_seconds
        self.api_endpoint = 'https://api.wit.ai/speech'
        self.access_token = "GNC4JAHOTTVJ54MJHGSU62BHZLOOAMWI"
        # Create an object from RecordAudio
        self.recordAudio = RecordAudio()

    def __del__(self):
        del self.file_name
        del self.num_seconds
        del self.recordAudio
        del self.api_endpoint
        del self.access_token
        # del self.recordAudio

    def recognize(self):
        # Connect to Wit by using private server access token
        client = Wit(access_token=self.access_token)

        # Start recording audio
        self.recordAudio.record_audio(self.num_seconds, self.file_name)
        # Read audio file created
        audio_loaded = self.recordAudio.read_audio(self.file_name)

        headers = {'authorization': 'Bearer ' + self.access_token,
                   'Content-Type': 'audio/wav'}

        # Make the HTTP post request to wit servers
        response = requests.post(self.api_endpoint, headers=headers, data=audio_loaded)

        # Convert to Json
        decoded = response.content.decode()
        print(decoded)

        # Extract the command from the speech input
        converted_data = json.loads(decoded)
        reply_phrase = converted_data['entities']

        # Extract the command from entities
        command_return = None
        for i in reply_phrase.keys():
            command_return = i

        return command_return


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
