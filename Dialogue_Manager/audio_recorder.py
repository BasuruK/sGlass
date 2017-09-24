import pyaudio
import wave


class RecordAudio:

    format = None
    channels = None
    rate = None
    chunk = None

    def __init__(self, format=pyaudio.paInt16, channels=2, rate=44100, chunk=1024):
        self.format = format
        self.channels = channels
        self.rate = rate
        self.chunk = chunk

    def __del__(self):
        del self.format
        del self.channels
        del self.rate
        del self.chunk

    def record_audio(self, seconds_to, output_file_name):
        # Create pyaudio object
        audio = pyaudio.PyAudio()

        # Open a stream from microphone
        stream = audio.open(format=self.format, channels=self.channels,
                            rate=self.rate, input=True, frames_per_buffer=self.chunk)
        print("Listening")

        frames = []

        for i in range(int(self.rate/ self.chunk * seconds_to)):
            # Read from microphone
            data = stream.read(self.chunk)
            frames.append(data)

        print("Finished Listening")

        # Close microphone and release
        stream.stop_stream()
        stream.close()
        audio.terminate()

        # Get a new wave file
        wave_file = wave.open(output_file_name, 'wb')
        # Pre-process the wave file
        wave_file.setnchannels(self.channels)
        wave_file.setsampwidth(audio.get_sample_size(self.format))
        wave_file.setframerate(self.rate)
        wave_file.writeframes(b''.join(frames))

        # Close the wave file
        wave_file.close()

    # Read audio file
    #@staticmethod
    def read_audio(self, wave_filename):
        with open(wave_filename, 'rb') as f:
            audio = f.read()
        return audio

