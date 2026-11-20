import pyttsx3

engine = pyttsx3.init()

# Get all available voices on your system
voices = engine.getProperty('voices')

# Switch to the second voice (usually female on Windows)
engine.setProperty('voice', voices[1].id)

# Set volume to 80%
engine.setProperty('rate', 151.8)

engine.say("Testing a different voice and volume.")
engine.runAndWait()
