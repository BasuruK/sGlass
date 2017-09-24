"""
Outputs the text received in voice format
"""
from gtts import gTTS
import os

class TextToSpeech:

    IMPORT_MANAGER = None
    sentence = None
    prediction = None

    def __init__(self, import_manager):
        self.IMPORT_MANAGER = import_manager

    # Read out the sentence
    def predict_speech(self, sentence="You pointed at", prediction="Cannot Identify"):

        processed_sentence = sentence + prediction
        print("Processed Sentence => {}".format(processed_sentence))

        tts = self.IMPORT_MANAGER.gTTS(text=processed_sentence, lang='en', slow=False)
        tts.save("command.mp3")

        self.IMPORT_MANAGER.os.system("mpg321 command.mp3")

    def speak(self, sentence):

        tts = self.IMPORT_MANAGER.gTTS(text=sentence, lang='en', slow=False)
        tts.save("speak.mp3")

        self.IMPORT_MANAGER.os.system("mpg321 speak.mp3")


def error(sentence):

    tts = gTTS(text=sentence, lang='en', slow=False)
    tts.save("speak.mp3")
    os.system("mpg321 speak.mp3")
