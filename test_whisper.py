# test_whisper.py
# ------------------------------------------------------------
# Minimal faster-whisper test — with an av 19 workaround.
# 1. Put your audio file in the SAME folder as this script.
# 2. Change AUDIO_FILE below to your file's name.
# 3. Run:  py test_whisper.py
# ------------------------------------------------------------

import numpy as np
import av
from faster_whisper import WhisperModel

# 👇 EDIT THIS ONE LINE — the name of your audio file
AUDIO_FILE = "test.wav"

# 👇 Model size: "tiny" (fastest) up to "large-v3" (best)
MODEL_SIZE = "tiny"

TARGET_SR = 16000  # Whisper always wants 16 kHz mono float32


def load_audio(path, target_sr=TARGET_SR):
    """Decode any audio file to mono float32 numpy array at target_sr.
    Works with av 18 AND av 19 (no metadata_errors kwarg)."""
    with av.open(path) as container:          # <-- no metadata_errors
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(
            format="flt",                     # float32 planar
            layout="mono",
            rate=target_sr,
        )
        chunks = []
        for frame in container.decode(stream):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray())
        # flush resampler
        for out in resampler.resample(None):
            chunks.append(out.to_ndarray())

    if not chunks:
        raise RuntimeError(f"No audio decoded from {path!r}")

    audio = np.concatenate(chunks, axis=1).reshape(-1).astype(np.float32)
    return audio


print(f"Loading model '{MODEL_SIZE}' (first run downloads it)...")
model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
print("Model loaded. Decoding audio...")

audio = load_audio(AUDIO_FILE)
print(f"Decoded {len(audio)/TARGET_SR:.2f}s of audio. Transcribing...\n")

# Pass the NumPy array — skips faster-whisper's broken decode_audio()
segments, info = model.transcribe(audio)

print(f"Detected language: {info.language} (probability {info.language_probability:.2f})\n")
print("----- TRANSCRIPT -----")
for segment in segments:
    print(f"[{segment.start:6.2f}s -> {segment.end:6.2f}s]  {segment.text}")
print("----------------------")
print("Done ✅")