from wit import Wit
import requests
import json
import threading
from Dialogue_Manager.audio_recorder import RecordAudio


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
    print("BS")
    print(threading.current_thread().getName())
