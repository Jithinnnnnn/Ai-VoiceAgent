import sounddevice as sd

fs = 44100  # Sample rate (CD quality)
seconds = 3

print("Recording...")
# Record audio from your microphone into a NumPy array
my_recording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
sd.wait()  # Wait until recording is finished

print("Playing back...")
# Play the recorded array back through your speakers
sd.play(my_recording, fs)
sd.wait()
