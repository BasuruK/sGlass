import io
import os

# Imports the Google Cloud client library
from google.cloud import speech
from google.cloud.speech import enums
from google.cloud.speech import types

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/basuruk/Desktop/Dialogue-Manager-fb07b4ddbfb8.json"

# Instantiates a client
client = speech.SpeechClient()
import time
start_time = time.time()
# The name of the audio file to transcribe
file_name = "/home/basuruk/Desktop/speech_input.wav"

# Loads the audio into memory
with io.open(file_name, 'rb') as audio_file:
    content = audio_file.read()
    audio = types.RecognitionAudio(content=content)

config = types.RecognitionConfig(
    encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
    sample_rate_hertz=16000,
    language_code='en-US')

# Detects speech in the audio file
response = client.recognize(config, audio)
alternatives = response.results[0].alternatives

for alternative in alternatives:
    print('Transcript: {}'.format(alternative.transcript))

print("Time {}".format(time.time() - start_time))
